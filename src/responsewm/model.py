"""Image-grounded stochastic world model with a disjoint source-only forecast API."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import torch
from torch import nn
from .legacy.encoder import PhaseStateEncoder, StateHeads, MaskedStatePredictor, PatientState
from .backbones import CoupledVelocity
from .temporal import AvailableConditioner, HistoryTransformer, TrajectoryPCRHead
from .contracts import ForecastInput, gather_visit
from .flow import integrate

@dataclass
class StateSequence:
    dense: torch.Tensor
    anatomy: torch.Tensor
    disease: torch.Tensor
    days: torch.Tensor
    mask: torch.Tensor
    generated: torch.Tensor
    @property
    def tokens(self):
        return torch.cat((self.dense, self.anatomy, self.disease),2)
    def append(self, state, day, valid, generated):
        return StateSequence(
            torch.cat((self.dense,state.dense[:,None]),1),
            torch.cat((self.anatomy,state.anatomy[:,None]),1),
            torch.cat((self.disease,state.disease[:,None]),1),
            torch.cat((self.days,day[:,None]),1),
            torch.cat((self.mask,valid[:,None]),1),
            torch.cat((self.generated,torch.full_like(valid,generated)[:,None]),1))

@dataclass
class Forecast:
    latent: torch.Tensor          # [B,K,F,24,D,H,W], standardized
    state: torch.Tensor           # [B,K,F,S,d]
    image_state: torch.Tensor     # E(generated latent), same shape
    logits: torch.Tensor          # [B,K]
    residuals: torch.Tensor        # [B,K]
    observed_logit: torch.Tensor   # [B]
    probability: torch.Tensor      # [B], mean(sigmoid(logits))
    memory: torch.Tensor           # [B,4,d]
    observed_states: StateSequence

class ResponseWorldModel(nn.Module):
    def __init__(self, cfg, clinical_dim, action_dim):
        super().__init__()
        if type(self) is ResponseWorldModel and cfg.schema != "responsewm_v1":
            raise ValueError("Use MultistageResponseWorldModel and train-v2 for v2 configurations")
        self.cfg = cfg
        self.clinical_dim, self.action_dim = clinical_dim, action_dim
        self.encoder = PhaseStateEncoder(cfg.encoder)
        self.target_encoder = copy.deepcopy(self.encoder).eval().requires_grad_(False)
        self.state_heads = StateHeads(cfg.encoder)
        # The legacy single-visit pCR auxiliary head is not used by the new model.
        self.masked_predictor = MaskedStatePredictor(cfg.encoder)
        d = cfg.encoder.dim
        self.pillar_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,2*d), nn.GELU(),
                                               nn.Linear(2*d,cfg.network.pillar_dim))
        self.conditions = AvailableConditioner(cfg,clinical_dim,action_dim)
        self.history = HistoryTransformer(cfg)
        self.velocity = CoupledVelocity(cfg)
        self.pcr = TrajectoryPCRHead(cfg,clinical_dim)
        self.register_buffer("representation_ready",torch.tensor(False))
        self.register_buffer("flow_ready",torch.tensor(False))
        self.register_buffer("readout_ready",torch.tensor(False))
        self.register_buffer("joint_ready",torch.tensor(False))
        self.stage = "uninitialized"

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        if self.stage != "representation":
            self.encoder.eval()
        # The ODE must be a deterministic vector field. There is no dropout in it.
        return self

    def configure_stage(self, stage):
        if stage not in {"representation","flow","readout","joint"}:
            raise ValueError("Unknown training stage")
        self.stage = stage
        self.requires_grad_(False)
        if stage == "representation":
            modules = (self.encoder,self.state_heads,self.masked_predictor,self.pillar_projection,
                       self.conditions,self.history,self.pcr)
        elif stage == "flow":
            modules = (self.velocity,self.conditions,self.history)
        elif stage == "readout":
            modules = (self.pcr,)
        else:
            modules = (self.velocity,self.conditions,self.history,self.pcr)
        for module in modules:
            module.requires_grad_(True)
        self.state_heads.pcr.requires_grad_(False)
        self.target_encoder.requires_grad_(False)
        if stage == "joint":
            scope = self.cfg.training.joint_image_scope
            if scope == "decoder":
                self.velocity.image.tune_decoder()
            elif scope == "frozen":
                self.velocity.image.requires_grad_(False)
        self.train(True)
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def update_target(self):
        if self.stage != "representation":
            return
        decay = self.cfg.training.ema_decay
        for dst,src in zip(self.target_encoder.parameters(),self.encoder.parameters()):
            dst.lerp_(src,1-decay)
        for dst,src in zip(self.target_encoder.buffers(),self.encoder.buffers()):
            dst.copy_(src)

    @torch.no_grad()
    def freeze_representation(self):
        # Same coordinates on both sides of grounding. No simultaneous teacher drift.
        self.target_encoder.load_state_dict(self.encoder.state_dict(),strict=True)
        self.target_encoder.eval().requires_grad_(False)
        self.encoder.eval().requires_grad_(False)
        self.representation_ready.fill_(True)

    def encode_sequence(self,z,days,mask,teacher=False,generated=False):
        e = self.target_encoder if teacher else self.encoder
        b,t = z.shape[:2]
        flat = z.reshape(b*t,*z.shape[2:])
        valid = mask.reshape(-1).nonzero(as_tuple=True)[0]
        cfg = self.cfg.encoder
        def empty(count):
            return z.new_zeros(b*t,count,cfg.dim)
        dense,anatomy,disease = empty(int(torch.tensor(cfg.token_grid).prod())),empty(cfg.anatomy_tokens),empty(cfg.disease_tokens)
        if len(valid):
            value = e(flat[valid])
            # index_copy is differentiable w.r.t. the encoded observations.
            dense = dense.to(value.dense).index_copy(0,valid,value.dense)
            anatomy = anatomy.to(value.anatomy).index_copy(0,valid,value.anatomy)
            disease = disease.to(value.disease).index_copy(0,valid,value.disease)
        return StateSequence(dense.reshape(b,t,*dense.shape[1:]),anatomy.reshape(b,t,*anatomy.shape[1:]),
                             disease.reshape(b,t,*disease.shape[1:]),days,mask,torch.full_like(mask,generated))

    def memory(self,inp,sequence):
        clinical = self.conditions.static_tokens(inp.clinical,inp.clinical_mask)
        plans = self.conditions.plan_tokens(inp.actions,inp.action_mask,inp.future_days,inp.future_mask)
        return self.history(sequence.tokens,sequence.days,sequence.mask,sequence.generated,
                            clinical,plans,inp.future_mask)

    def read_trajectory(self,inp,memory,sequence):
        return self.pcr(memory,sequence.disease,sequence.days,sequence.mask,sequence.generated,
                        inp.observed.shape[1],inp.clinical,inp.clinical_mask)

    def sample_interval(self,z,s,context,steps,method,generator=None,direction=1):
        ez = torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=generator)
        es = torch.randn(s.shape,device=s.device,dtype=s.dtype,generator=generator)
        if direction == 1:
            zi,si = torch.cat((ez,z),1),torch.cat((es,s),1)
        elif direction == -1:
            zi,si = torch.cat((z,ez),1),torch.cat((s,es),1)
        else:
            raise ValueError("Direction must be +/-1")
        zf,sf = integrate(self.velocity,zi,si,context,steps,method,direction)
        zs,ss = zf.chunk(2,1),sf.chunk(2,1)
        return zs[0 if direction == 1 else 1],ss[0 if direction == 1 else 1]

    def forecast(self,inp: ForecastInput,*,samples=None,steps=None,method=None,generator=None):
        """No targets, labels, patient identifiers, teacher-forcing switch or files.

        Independent sampling across trajectories; within each trajectory its own
        generated history is fed forward. Retaining K graphs still costs O(K).
        """
        inp.validate(self.clinical_dim,self.action_dim,self.cfg.network.max_visits)
        k = self.cfg.sampling.inference_samples if samples is None else samples
        steps = self.cfg.sampling.inference_steps if steps is None else steps
        method = self.cfg.sampling.method if method is None else method
        if k < 1:
            raise ValueError("At least one trajectory required")
        initial = self.encode_sequence(inp.observed,inp.observed_days,inp.observed_mask)
        h0 = self.memory(inp,initial)
        all_z,all_s,all_es,logits,residuals = [],[],[],[],[]
        b,f = inp.future_days.shape
        for _ in range(k):
            sequence = initial
            z = gather_visit(inp.observed,inp.observed_mask)
            states_z,states_s,image_states = [],[],[]
            for j in range(f):
                valid = inp.future_mask[:,j]
                h = self.memory(inp,sequence)
                day = gather_visit(sequence.days,sequence.mask)
                requested_day = torch.where(valid,inp.future_days[:,j],day)
                context = self.conditions(h,inp.actions[:,j],inp.action_mask[:,j],day,requested_day)
                previous_s = gather_visit(sequence.disease,sequence.mask)
                new_z,new_s = self.sample_interval(z,previous_s,context,steps,method,generator)
                z = torch.where(valid[:,None,None,None,None],new_z,z)
                # Fixed parameters, intentionally NOT torch.no_grad(): image gradients survive.
                image_state = self.target_encoder(z)
                s = image_state.disease if self.cfg.network.readout_source == "reencode" else new_s
                state = PatientState(image_state.dense,image_state.anatomy,s,self.cfg.encoder.token_grid)
                sequence = sequence.append(state,requested_day,valid,True)
                states_z.append(z)
                states_s.append(new_s)
                image_states.append(image_state.disease)
            if f:
                all_z.append(torch.stack(states_z,1))
                all_s.append(torch.stack(states_s,1))
                all_es.append(torch.stack(image_states,1))
            else:
                e = self.cfg.encoder
                all_z.append(inp.observed.new_empty(b,0,*inp.observed.shape[2:]))
                all_s.append(inp.observed.new_empty(b,0,e.disease_tokens,e.dim))
                all_es.append(all_s[-1])
            logit,residual,obs = self.read_trajectory(inp,h0,sequence)
            logits.append(logit)
            residuals.append(residual)
        logits = torch.stack(logits,1)
        return Forecast(torch.stack(all_z,1),torch.stack(all_s,1),torch.stack(all_es,1),logits,
                        torch.stack(residuals,1),obs,logits.float().sigmoid().mean(1),h0,initial)

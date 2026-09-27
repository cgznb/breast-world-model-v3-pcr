"""Stage B is the ONLY longitudinal predictor. It never calls A's SSL predictor."""
from __future__ import annotations
import copy
import torch
from torch import nn
import torch.nn.functional as F
from .conditioning import ClinicalEncoder
from .velocity import build_velocity
from .flow import make_path, estimate_endpoints, integrate
from .observation_encoder import ThreePhaseObservationEncoder


class StateAdapter(nn.Module):
    """Learned-query attentive pooling preserves a fixed context length for 3D U-Net."""
    def __init__(self, encoder_dim, context_dim, tokens, depth, heads):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, context_dim))
        self.queries = nn.Parameter(torch.randn(1, tokens, context_dim)*.02)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(context_dim, heads, context_dim*4,
                                      dropout=0., activation="gelu", batch_first=True, norm_first=True)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, features):
        memory = self.project(features.tokens)
        q = self.queries.expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            q = block(q, memory, memory_key_padding_mask=features.padding)
        return self.norm(q)


class ThreePhaseWorldModel(nn.Module):
    def __init__(self, cfg, observation_encoder, condition_schema):
        super().__init__()
        self.cfg = cfg
        self.observation_encoder = copy.deepcopy(observation_encoder).eval().requires_grad_(False)
        v = cfg.velocity
        self.condition_encoder = ClinicalEncoder(condition_schema, v.context_dim, v.attention_heads)
        self.state_adapter = StateAdapter(cfg.encoder.dim, v.context_dim, v.patient_tokens, v.adapter_depth, v.attention_heads)
        self.velocity_model = build_velocity(v, v.context_dim)

    def train(self, mode=True):
        super().train(mode)
        self.observation_encoder.eval()
        return self

    def current_features(self, latent, valid=None):
        # No no_grad here: semantic endpoint supervision needs dE/d(predicted latent).
        return self.observation_encoder(latent, valid, normalized=True,
               minimum_coverage=self.cfg.observation.minimum_patch_coverage, teacher_average=True)

    def build_context(self, source, conditions, valid=None, direction=1):
        clinical = self.condition_encoder(conditions, direction)
        if not self.cfg.velocity.source_tokens:
            return clinical
        # The real source is constant. No target/prior state argument exists.
        with torch.no_grad():
            features = self.current_features(source, valid)
        return torch.cat((clinical, self.state_adapter(features)), dim=1)

    def loss(self, batch, step=0, *, direction=None):
        norm = self.observation_encoder.normalize
        earlier, later = norm(batch["source_raw"]), norm(batch["target_raw"])
        if direction is None:
            direction = -1 if float(torch.rand(())) < self.cfg.velocity.reverse_probability else 1
        condition_source = earlier if direction == 1 else later
        source_valid = batch["source_valid"] if direction == 1 else batch["target_valid"]
        context = self.build_context(condition_source, batch["conditions"], source_valid, direction)
        path = make_path(earlier, later)
        predicted = self.velocity_model(path.joint, path.tau, context)
        if predicted.shape != path.velocity.shape:
            raise ValueError("Velocity shape violates original two-branch SymmFlow contract")
        vx, vy = predicted.chunk(2, 1)
        tx, ty = path.velocity.chunk(2, 1)
        lx, ly = F.mse_loss(vx.float(), tx.float()), F.mse_loss(vy.float(), ty.float())
        total = lx+ly
        parts = {"velocity_x": float(lx.detach()), "velocity_y": float(ly.detach()),
                 "forward_batches": int(direction == 1), "reverse_batches": int(direction == -1)}
        v = self.cfg.velocity
        if v.semantic_endpoint_weight and step >= v.semantic_start_step and step % v.semantic_every == 0:
            ehat, lhat = estimate_endpoints(path.joint, predicted, path.tau)
            estimate, target, valid = (lhat, later, batch["target_valid"]) if direction == 1 else (ehat, earlier, batch["source_valid"])
            pf = self.current_features(estimate, valid)
            with torch.no_grad():
                tf = self.current_features(target, valid)
            # Summary features avoid imposing nonexistent cross-visit voxel registration.
            semantic = F.smooth_l1_loss(pf.pooled().float(), tf.pooled().float())
            total = total+v.semantic_endpoint_weight*semantic
            parts["semantic_endpoint"] = float(semantic.detach())
        parts["loss"] = float(total.detach())
        return total, parts

    def sample(self, source, conditions, noise, *, valid=None, steps=None, method=None, direction=1):
        """source/noise are standardized VQ coordinates; target is NEVER an argument."""
        if source.shape != noise.shape or source.ndim != 5 or source.shape[1] != 24:
            raise ValueError("Sampling inputs must both be [B,24,D,H,W]")
        context = self.build_context(source, conditions, valid, direction)
        initial = torch.cat((noise, source), 1) if direction == 1 else torch.cat((source, noise), 1)
        final = integrate(self.velocity_model, initial, context,
                          steps=steps or self.cfg.sampling.steps,
                          method=method or self.cfg.sampling.method, direction=direction)
        return final[:, :24] if direction == 1 else final[:, 24:]

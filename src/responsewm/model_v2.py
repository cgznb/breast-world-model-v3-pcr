"""Stage-index multistage model, preserving the separate v1 experiment API."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import uuid
import torch
from .model import ResponseWorldModel
from .backbones import CoupledVelocityV2
from .conditioning import ConditionResampler, IntervalSpec
from .state import PatientBelief, Observation, EvidenceEvent, ModelEvent, ObservationSeedPool, ObservationAssimilator
from .temporal_v2 import HistoryContextBuilder, SharedPCRReadout
from .rollout import NoiseLedger, ForecastTrace, PCRQueryResult
from .flow import integrate
from .representation import DeepJEPAPredictor


def _hash_parts(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _repeat_samples(value, samples):
    return value[:, None].expand(len(value), samples, *value.shape[1:]).reshape(len(value)*samples, *value.shape[1:])


class ResponseWorldModelV2(ResponseWorldModel):
    def __init__(self, cfg, clinical_dim, action_dim):
        super().__init__(cfg, clinical_dim, action_dim)
        self.conditions = ConditionResampler(cfg, clinical_dim, action_dim)
        self.history = HistoryContextBuilder(cfg)
        self.velocity = CoupledVelocityV2(cfg)
        self.pcr = SharedPCRReadout(cfg, clinical_dim)
        self.seed_pool = ObservationSeedPool(cfg)
        self.assimilator = ObservationAssimilator(cfg)
        self.masked_predictor = DeepJEPAPredictor(cfg.encoder)
        self._lineage_versions = {}

    @property
    def deep_predictor(self):
        return self.masked_predictor

    @property
    def assimilation_projection(self):
        return self.assimilator.reconstruction

    def observed_logit(self, belief):
        return self.query_pcr(belief, mode="observed_landmark").logit_per_sample

    def configure_stage(self, stage):
        super().configure_stage(stage)
        if stage in {"representation", "flow", "joint"}:
            self.seed_pool.requires_grad_(True)
            self.assimilator.requires_grad_(True)
        if hasattr(self, "deep_predictor"):
            self.deep_predictor.requires_grad_(stage == "representation")
        return [p for p in self.parameters() if p.requires_grad]

    def _clinical_tokens(self, belief, samples=None):
        samples = belief.samples if samples is None else samples
        values = _repeat_samples(belief.clinical, samples)
        mask = _repeat_samples(belief.clinical_mask, samples)
        return self.conditions.static_tokens(values, mask)

    def _event(self, seed, disease, clinical_tokens, stage, origin, valid, version, b, k):
        raw = self.history.event_tokens(seed, clinical_tokens, stage, origin)
        return ModelEvent(stage, origin, raw.reshape(b, k, *raw.shape[1:]),
                          disease.reshape(b, k, *disease.shape[1:]), valid, version)

    def initialize(self, observed_prefix, mode="observed"):
        if mode != "observed":
            raise ValueError("Initialization accepts only real observed prefixes")
        inp = observed_prefix
        inp.validate(self.clinical_dim, self.action_dim, self.cfg.network.max_visits)
        b = len(inp.observed)
        stages = inp.observed_days.long()
        if bool((inp.observed_days != stages).logical_and(inp.observed_mask).any()):
            raise ValueError("R1 requires integer canonical stages")
        latest = stages.masked_fill(~inp.observed_mask, -1).max(1).values
        if not bool((latest == latest[0]).all()):
            raise ValueError("Batch patients by latest observed stage for v2 state operations")
        if not bool(((stages == 0) & inp.observed_mask).any(1).all()):
            raise ValueError("R1 initialization requires a real T0 MRI")
        if hasattr(inp, "as_of") and bool((inp.as_of != latest).any()):
            raise ValueError("Initialize at an observed MRI stage; use explicit advance for later query stages")
        if hasattr(inp, "observed_available_at") and bool(
                ((inp.observed_available_at != inp.observed_days) & inp.observed_mask).any()):
            raise ValueError("R1 initialization requires stage-aligned MRI availability; delayed arrival replay is unsupported")
        belief = None
        for stage in range(int(latest[0])+1):
            mask = inp.observed_mask & (stages == stage)
            valid = mask.any(1)
            if not bool(valid.any()):
                continue
            indices = mask.long().argmax(1)
            latent = inp.observed[torch.arange(b, device=indices.device), indices]
            clinical_mask = inp.clinical_mask
            if getattr(inp, "clinical_known_at", None) is not None:
                clinical_mask = clinical_mask & (inp.clinical_known_at <= stage)
            obs = Observation(stage, latent, tuple(f"mri:T{stage}" for _ in range(b)),
                              valid=valid, clinical=inp.clinical, clinical_mask=clinical_mask)
            if belief is None:
                belief = self._initialize_observation(obs)
            else:
                belief = self.observe(belief, obs)
        return belief

    def _initialize_observation(self, observation):
        z = observation.latent
        b = len(z)
        if observation.clinical is None or observation.clinical_mask is None:
            raise ValueError("Initialization requires an explicit clinical snapshot")
        clinical, mask = observation.clinical, observation.clinical_mask
        encoded = self.encoder(z)
        # Frozen teacher parameters preserve differentiability with respect to Z.
        teacher = self.target_encoder(z)
        ct = self.conditions.static_tokens(clinical, mask)
        seed = self.seed_pool(encoded, ct)
        valid = torch.ones(b, dtype=torch.bool, device=z.device)
        event = self._event(seed, teacher.disease, ct, observation.stage, "observed", valid, 0, b, 1)
        memory = self.history((event,))
        hashes = observation.hashes()
        evidence = EvidenceEvent(observation.stage, observation.event_ids, hashes, valid)
        physiology = tuple(_hash_parts(observation.stage, h) for h in hashes)
        lineage = uuid.uuid4().hex
        self._lineage_versions[lineage] = 0
        return PatientBelief(0, observation.stage, float(observation.stage), memory, z[:, None],
                             teacher.disease[:, None], torch.zeros(b, 1, dtype=torch.long, device=z.device),
                             clinical, mask, (evidence,), (event,),
                             physiology, tuple("root" for _ in range(b)), lineage, lineage)

    def _fork(self, belief, samples):
        if samples == belief.samples:
            return belief
        if belief.samples != 1:
            raise ValueError("Existing K branches must retain their sample identities")
        if samples < 1:
            raise ValueError("At least one sample is required")
        def expand(x):
            return x.expand(len(x), samples, *x.shape[2:])
        events = tuple(replace(e, raw_tokens=expand(e.raw_tokens), disease=expand(e.disease)) for e in belief.model_state_log)
        return replace(belief, memory=expand(belief.memory), anchor_latent=expand(belief.anchor_latent),
                       anchor_disease=expand(belief.anchor_disease), anchor_origin=expand(belief.anchor_origin),
                       model_state_log=events)

    def condition_for(self, belief, target_stage, interval=None):
        interval = IntervalSpec(belief.stage, target_stage) if interval is None else interval
        interval.validate(belief.stage)
        k = belief.samples
        routed = replace(interval,
                         actions=None if interval.actions is None else _repeat_samples(interval.actions, k),
                         action_mask=None if interval.action_mask is None else _repeat_samples(interval.action_mask, k))
        return self.conditions(belief.memory.flatten(0, 1), _repeat_samples(belief.clinical, k),
                               _repeat_samples(belief.clinical_mask, k), routed)

    def advance(self, belief, interval, noise_ledger=None, samples=None, steps=None, method=None):
        interval.validate(belief.stage)
        samples = belief.samples if samples is None else int(samples)
        source = self._fork(belief, samples)
        ledger = NoiseLedger(self.cfg.training.seed) if noise_ledger is None else noise_ledger
        steps = self.cfg.sampling.inference_steps if steps is None else steps
        method = self.cfg.sampling.method if method is None else method
        context = self.condition_for(source, interval.dst_stage, interval)
        controls = []
        for p, previous in enumerate(source.branch_prefix_hashes):
            if interval.actions is None or self.action_dim == 0:
                payload = "ACTION_UNOBSERVED"
            else:
                from .state import tensor_hash
                payload = tensor_hash(interval.actions[p].masked_fill(~interval.action_mask[p], 0))
                payload += tensor_hash(interval.action_mask[p])
            controls.append(_hash_parts(previous, interval.src_stage, interval.dst_stage, payload))
        control_hashes = tuple(controls)
        ez = ledger.noise(source.anchor_latent, source.physiology_hashes, control_hashes,
                          interval.src_stage, interval.dst_stage, "image")
        es = ledger.noise(source.anchor_disease, source.physiology_hashes, control_hashes,
                          interval.src_stage, interval.dst_stage, "semantic")
        z, s = source.anchor_latent.flatten(0, 1), source.anchor_disease.flatten(0, 1)
        zi, si = torch.cat((ez.flatten(0, 1), z), 1), torch.cat((es.flatten(0, 1), s), 1)
        zf, sf = integrate(self.velocity, zi, si, context, steps, method, 1)
        zpred, spred = zf.chunk(2, 1)[0], sf.chunk(2, 1)[0]
        encoded = self.target_encoder(zpred)
        disease = encoded.disease if self.cfg.network.readout_source == "reencode" else spred
        ct = self._clinical_tokens(source)
        seed = self.seed_pool(encoded, ct, disease=disease)
        b, k = source.memory.shape[:2]
        valid = torch.ones(b, dtype=torch.bool, device=z.device)
        event = self._event(seed, disease, ct, interval.dst_stage, "predicted", valid,
                            source.version+1, b, k)
        events = source.model_state_log + (event,)
        state = replace(source, version=source.version+1, stage=interval.dst_stage,
                        as_of=float(interval.dst_stage), memory=self.history(events),
                        anchor_latent=zpred.reshape(b, k, *zpred.shape[1:]),
                        anchor_disease=disease.reshape(b, k, *disease.shape[1:]),
                        anchor_origin=torch.ones(b, k, dtype=torch.long, device=z.device),
                        model_state_log=events, branch_prefix_hashes=control_hashes,
                        branch_id=_hash_parts(source.branch_id, ledger.seed, ledger.trajectory_id,
                                              interval.dst_stage, *control_hashes))
        risk = self.query_pcr(state, mode="forecasted_state")
        return ForecastTrace(source, (state,), (state.stage,),
                             (encoded.disease.reshape(b, k, *encoded.disease.shape[1:]),),
                             risk.logit_per_sample[:, :, None], ledger.trajectory_id,
                             assumed_plan_version=interval.plan_version)

    def observe(self, prior, observation):
        if not isinstance(observation, Observation):
            raise TypeError("observe accepts one input-only Observation")
        if isinstance(observation.stage, bool) or not isinstance(observation.stage, int):
            raise ValueError("R1 observations require integer canonical stages")
        if not 0 <= observation.stage <= 3 or observation.stage < prior.stage:
            raise ValueError("Observation cannot update a state from a later stage")
        b, k = prior.memory.shape[:2]
        if len(observation.event_ids) != b or len(observation.latent) != b:
            raise ValueError("Observation batch does not match belief")
        clinical = prior.clinical if observation.clinical is None else observation.clinical
        clinical_mask = prior.clinical_mask if observation.clinical_mask is None else observation.clinical_mask
        hashes = replace(observation, clinical=clinical, clinical_mask=clinical_mask).hashes()
        valid = (torch.ones(b, device=prior.memory.device, dtype=torch.bool)
                 if observation.valid is None else observation.valid)
        already = torch.zeros_like(valid)
        for old in prior.evidence_log:
            for patient in range(b):
                if bool(valid[patient] and old.valid[patient]) and old.event_ids[patient] == observation.event_ids[patient]:
                    if old.payload_hashes[patient] != hashes[patient] or old.stage != observation.stage:
                        raise ValueError("Observation event ID conflicts with an existing payload")
                    already[patient] = True
        valid = valid & ~already
        if not bool(valid.any()):
            return prior
        clinical = torch.where(valid[:, None], clinical, prior.clinical)
        clinical_mask = torch.where(valid[:, None], clinical_mask, prior.clinical_mask)
        ct = self.conditions.static_tokens(_repeat_samples(clinical, k), _repeat_samples(clinical_mask, k))
        encoded = self.encoder(observation.latent)
        teacher = self.target_encoder(observation.latent)
        observation_tokens = torch.cat((encoded.dense, encoded.anatomy, encoded.disease), 1)
        flat_valid = valid[:, None].expand(b, k).reshape(-1)
        gap = prior.memory.new_full((b*k,), observation.stage-prior.stage)
        quality = None if observation.quality is None else observation.quality[:, None].expand(b, k).reshape(-1)
        memory = self.assimilator(prior.memory.flatten(0, 1), _repeat_samples(observation_tokens, k),
                                  flat_valid, gap, quality)
        disease = _repeat_samples(teacher.disease, k)
        event = self._event(memory, disease, ct, observation.stage, "observed", valid, prior.version+1, b, k)
        # A predicted placeholder is audit history, never a second observed MRI.
        events = tuple(e for e in prior.model_state_log if not (e.stage == observation.stage and e.origin == "predicted")) + (event,)
        contextual = self.history(events)
        memory = torch.where(valid[:, None, None, None], contextual, prior.memory)
        real_z = observation.latent[:, None].expand(b, k, *observation.latent.shape[1:])
        real_s = teacher.disease[:, None].expand(b, k, *teacher.disease.shape[1:])
        physiology = tuple(_hash_parts(old, observation.stage, h) if bool(v) else old
                           for old, h, v in zip(prior.physiology_hashes, hashes, valid))
        evidence = EvidenceEvent(observation.stage, observation.event_ids, hashes, valid)
        revision = max(prior.evidence_version, self._lineage_versions.get(prior.lineage_id, 0))+1
        self._lineage_versions[prior.lineage_id] = revision
        return replace(prior, version=prior.version+1, stage=observation.stage, as_of=float(observation.stage),
                       memory=memory, anchor_latent=torch.where(valid[:, None, None, None, None, None], real_z, prior.anchor_latent),
                       anchor_disease=torch.where(valid[:, None, None, None], real_s, prior.anchor_disease),
                       anchor_origin=torch.where(valid[:, None], torch.zeros_like(prior.anchor_origin), prior.anchor_origin),
                       clinical=clinical, clinical_mask=clinical_mask,
                       evidence_log=prior.evidence_log+(evidence,), model_state_log=events,
                       physiology_hashes=physiology, evidence_version=revision,
                       branch_prefix_hashes=tuple("root" for _ in range(b)),
                       branch_id=_hash_parts(prior.lineage_id, revision), cache=None)

    def forecast(self, belief, output_stages=None, plan=None, noise_ledger=None, samples=None, steps=None, method=None,
                 through_stage=3):
        if not isinstance(belief, PatientBelief):
            raise TypeError("v2 forecast requires an initialized PatientBelief")
        if isinstance(through_stage, bool) or not isinstance(through_stage, int) or not belief.stage <= through_stage <= 3:
            raise ValueError("Invalid explicit rollout horizon")
        stages = tuple(range(belief.stage+1, through_stage+1))
        requested = stages if output_stages is None else tuple(output_stages)
        if len(set(requested)) != len(requested) or any(isinstance(s, bool) or not isinstance(s, int) or s not in stages for s in requested):
            raise ValueError("Output requests must be distinct remaining canonical stages")
        ledger = NoiseLedger(self.cfg.training.seed) if noise_ledger is None else noise_ledger
        k = (self.cfg.sampling.inference_samples if belief.samples == 1 else belief.samples) if samples is None else samples
        source, current = self._fork(belief, k), self._fork(belief, k)
        states, images, logits, plan_versions = [], [], [], []
        for dst in stages:
            interval = IntervalSpec(current.stage, dst)
            if plan is not None:
                if not isinstance(plan, (tuple, list, dict)):
                    raise TypeError("Plan must contain explicitly scoped IntervalSpec values")
                intervals = plan.values() if isinstance(plan, dict) else plan
                interval = next((x for x in intervals if x.src_stage == current.stage and x.dst_stage == dst), interval)
                if interval.known_at is not None and bool(torch.as_tensor(interval.known_at).gt(belief.stage).any()):
                    raise ValueError("Forecast cannot consume plans learned after its real source prefix")
            trace = self.advance(current, interval, ledger, samples=k, steps=steps, method=method)
            if interval.plan_version is not None:
                plan_versions.append(str(interval.plan_version))
            current = trace.final_state
            states.append(current)
            images.append(trace.image_states[0])
            logits.append(trace.stage_logits[:, :, 0])
        stage_logits = torch.stack(logits, -1) if logits else source.memory.new_empty(len(source.memory), k, 0)
        trace = ForecastTrace(source, tuple(states), requested, tuple(images), stage_logits, ledger.trajectory_id,
                              assumed_plan_version="|".join(plan_versions) if plan_versions else None)
        if through_stage == 3:
            trace = replace(trace, pcr_marginal=self.query_pcr(source, mode="marginal_current", future_trace=trace))
        return trace

    def query_pcr(self, belief, mode="observed_landmark", future_trace=None):
        if mode not in {"observed_landmark", "forecasted_state", "marginal_current"}:
            raise ValueError("Unknown pCR interpretation")
        if mode == "observed_landmark" and not bool((belief.anchor_origin == 0).all()):
            raise ValueError("A predicted state is not an observed landmark")
        if mode == "forecasted_state" and not bool((belief.anchor_origin == 1).all()):
            raise ValueError("forecasted_state requires a predicted state")
        state = belief
        events = belief.model_state_log
        if mode == "marginal_current":
            if future_trace is None:
                raise ValueError("Marginal readout requires the complete source-only trace")
            source = future_trace.source_state
            if (source.lineage_id != belief.lineage_id or source.evidence_version != belief.evidence_version
                    or source.physiology_hashes != belief.physiology_hashes
                    or source.branch_id != belief.branch_id or source.stage != belief.stage):
                raise ValueError("Forecast was not forked from this observed evidence version")
            if self._lineage_versions.get(belief.lineage_id, belief.evidence_version) != belief.evidence_version:
                raise ValueError("Forecast branch was invalidated by a newer real observation")
            if future_trace.stage_ids != tuple(range(belief.stage+1, 4)):
                raise ValueError("Marginal readout requires every remaining canonical transition")
            state = self._fork(belief, source.samples)
            if self.cfg.network.use_future:
                events = future_trace.final_state.model_state_log
        if mode == "observed_landmark":
            events = tuple(e for e in events if e.origin == "observed")
        ct = self.conditions.static_tokens(state.clinical, state.clinical_mask)
        logits = self.pcr(state.memory, ct, state.clinical, state.clinical_mask, events)
        return PCRQueryResult(logits, logits.float().sigmoid().mean(1), mode, belief.as_of,
                              belief.observed_stage_ids,
                              None if future_trace is None else future_trace.assumed_plan_version)


MultistageResponseWorldModel = ResponseWorldModelV2

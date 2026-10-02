"""Strict versioned configuration; small tests use the same architecture classes."""
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import json
import hashlib
import math
import yaml

@dataclass
class EncoderConfig:
    phase_channels: int = 8
    stem_width: int = 48
    widths: tuple = (96, 192, 384)
    depths: tuple = (2, 2, 4)
    heads: tuple = (3, 6, 12)
    window: tuple = (2, 4, 4)
    token_grid: tuple = (2, 4, 4)
    dim: int = 192
    query_heads: int = 6
    anatomy_tokens: int = 4
    disease_tokens: int = 8
    predictor_depth: int = 4
    phase_mixer_depth: int = 2
    drop_path: float = 0.0
    checkpoint_blocks: bool = True
    mask_ratio: float = 0.5
    phase_drop_probability: float = 0.25

@dataclass
class NetworkConfig:
    # MONAI is the actual pinned dependency, never silently replaced by native.
    backend: str = "monai"
    time_basis: str = "calendar_days"  # calendar_days | stage_index
    use_future: bool = True
    channels: tuple = (128, 256, 384)
    num_res_blocks: int = 2
    attention_heads: int = 8
    history_depth: int = 3
    semantic_depth: int = 4
    readout_depth: int = 3
    max_visits: int = 8
    dropout: float = 0.1  # readout only; vector field is deterministic
    coupling: bool = True
    readout_source: str = "joint"  # joint | reencode
    checkpoint_blocks: bool = True
    pillar_dim: int = 1152
    future_residual_limit: float = 3.0
    clinical_prior: bool = True

@dataclass
class LossConfig:
    reconstruction: float = 1.0
    phase_difference: float = 0.1
    masked_jepa: float = 1.0
    variance: float = 0.1
    covariance: float = 0.01
    anatomy: float = 0.01
    pillar: float = 0.1
    dense_teacher: float = 0.0
    segmentation: float = 0.0
    kinetics: float = 0.0
    biomarkers: float = 0.0
    fm_image: float = 1.0
    fm_state: float = 1.0
    repa: float = 0.05
    real_pcr: float = 1.0
    marginal_pcr: float = 1.0
    observed_pcr: float = 0.25
    grounding: float = 0.1
    energy: float = 0.05
    prediction_grounding: float = 0.02
    residual_l2: float = 0.01
    # Explicit ablation, NOT the default probabilistic objective.
    per_sample_bce: bool = False

@dataclass
class TrainConfig:
    seed: int = 20260928
    device: str = "cuda"
    precision: str = "bf16"
    threads: int = 4
    batch_size: int = 1
    accumulation: int = 4
    lr: float = 1e-4
    joint_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: int = 100
    ema_decay: float = 0.995
    representation_steps: int = 10000
    flow_steps: int = 30000
    readout_steps: int = 3000
    joint_steps: int = 5000
    validation_every: int = 100
    checkpoint_every: int = 100
    log_every: int = 10
    validation_cases: int = 32
    reverse_probability: float = 0.25
    joint_image_scope: str = "decoder"  # all | decoder | frozen
    strict_determinism: bool = False
    validate_samples: int = 4
    selection_generation_weight: float = 0.05
    allow_synthetic: bool = False

@dataclass
class SamplingConfig:
    train_samples: int = 2
    train_steps: int = 20
    inference_samples: int = 8
    inference_steps: int = 20
    method: str = "heun"

@dataclass
class MultistageConfig:
    canonical_stages: tuple = (0, 1, 2, 3)
    memory_tokens: int = 16
    global_tokens: int = 4
    assimilation_depth: int = 4
    assimilation_gate_bias: float = -2.0
    condition_queries: int = 4
    condition_depth: int = 2
    deep_jepa_mix: float = 0.5
    deep_jepa_weights: tuple = (0.2, 0.3, 1.0)
    deep_jepa_visible_weight: float = 0.1
    assimilation_reconstruction: float = 0.1
    distribution_stage_mix: float = 0.5
    distribution_joint_mix: float = 0.5
    edge_sampling: str = "balanced_edge_sample"
    bridge_auxiliary_weight: float = 0.25
    rollout_every: int = 8
    flow_phase_fractions: tuple = (0.2, 0.3, 0.5)
    marginal_pcr_start: float = 0.1
    marginal_pcr_end: float = 0.5
    representation_pcr_weight: float = 0.25
    joint_image_lr: float = 3e-6
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.001
    continuous_clinical_time_enabled: bool = False
    use_full_future_plan_in_physiology: bool = False
    truncate_bptt: bool = False

@dataclass
class DeploymentProtocolConfig:
    primary_origin: int = 0
    selection_metric: str = "t0_marginal_nll"
    validation_seed: int = 20261001
    validation_batch_size: int = 8
    representation_lr: float = 3e-5
    readout_lr: float = 1e-5
    readout_scope: str = "last_block"  # output | last_block | all
    joint_readout_scope: str = "last_block"
    readout_weight_decay: float = 0.05
    residual_logit_limit: float = 2.0
    residual_penalty: float = 0.01
    latent_mean_weight: float = 0.1
    latent_energy_weight: float = 0.05
    generation_guard_ratio: float = 1.10
    image_eval_cases: int = 8
    imaging_manifest: str | None = None
    codec_path: str | None = None
    checkpoint_on_stop: bool = True
    stage_batch_sizes: dict = field(default_factory=dict)
    stage_validation_every: dict = field(default_factory=dict)

@dataclass
class Config:
    schema: str = "responsewm_v1"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    multistage: MultistageConfig = field(default_factory=MultistageConfig)
    protocol: DeploymentProtocolConfig = field(default_factory=DeploymentProtocolConfig)

    def validate(self):
        e, n, t, s = self.encoder, self.network, self.training, self.sampling
        if self.schema not in {"responsewm_v1", "responsewm_v2", "responsewm_v3"}:
            raise ValueError("Unsupported configuration schema")
        if self.schema in {"responsewm_v2", "responsewm_v3"}:
            m = self.multistage
            if n.time_basis != "stage_index" or tuple(m.canonical_stages) != (0,1,2,3):
                raise ValueError("R1 requires canonical stage_index [0,1,2,3]")
            if m.continuous_clinical_time_enabled or m.use_full_future_plan_in_physiology or m.truncate_bptt:
                raise ValueError("R1 does not support continuous time, noncausal plans or truncated BPTT")
            if n.semantic_depth != 6 or len(n.channels) != 3:
                raise ValueError("V2 requires six semantic blocks and three image levels")
            if m.memory_tokens != e.anatomy_tokens+e.disease_tokens+m.global_tokens:
                raise ValueError("Memory slots must match anatomy + disease + global slots")
            if min(m.assimilation_depth,m.global_tokens,m.condition_queries,m.condition_depth,m.rollout_every) < 1:
                raise ValueError("Invalid state module depth or rollout cadence")
            if not 0 <= m.deep_jepa_mix <= 1 or len(m.deep_jepa_weights) != len(e.widths) or min(m.deep_jepa_weights) < 0 or sum(m.deep_jepa_weights) <= 0:
                raise ValueError("Invalid deep JEPA mixture")
            if m.edge_sampling not in {"balanced_edge_sample","all_edges"}:
                raise ValueError("Unknown edge objective estimator")
            if len(m.flow_phase_fractions) != 3 or min(m.flow_phase_fractions) <= 0 or abs(sum(m.flow_phase_fractions)-1)>1e-6:
                raise ValueError("Flow curriculum must contain three positive fractions summing to one")
            if abs(m.distribution_stage_mix+m.distribution_joint_mix-1)>1e-6 or min(m.distribution_stage_mix,m.distribution_joint_mix)<0:
                raise ValueError("Distribution mixture must sum to one")
            if t.reverse_probability != 0:
                raise ValueError("R1 production uses forward clinical transitions")
            nonnegative=(m.deep_jepa_visible_weight,m.assimilation_reconstruction,m.bridge_auxiliary_weight,
                         m.marginal_pcr_start,m.marginal_pcr_end,m.representation_pcr_weight,
                         m.joint_image_lr,m.early_stopping_patience,m.early_stopping_min_delta)
            if any(not math.isfinite(v) or v<0 for v in nonnegative) or m.bridge_auxiliary_weight>1:
                raise ValueError("Invalid multistage objective/optimizer settings")
            if not math.isfinite(m.assimilation_gate_bias):
                raise ValueError("Assimilation gate initialization must be finite")
        if self.schema == "responsewm_v3":
            p = self.protocol
            if p.primary_origin != 0 or p.selection_metric != "t0_marginal_nll":
                raise ValueError("V3 deployment protocol fixes T0 marginal NLL before training")
            if p.readout_scope not in {"output", "last_block", "all"} or p.joint_readout_scope not in {"output", "last_block", "all"}:
                raise ValueError("Invalid conservative readout update scope")
            if min(p.representation_lr,p.readout_lr,p.residual_logit_limit,p.validation_batch_size) <= 0:
                raise ValueError("Invalid deployment learning rate, residual limit or validation batch size")
            if min(p.residual_penalty,p.latent_mean_weight,p.latent_energy_weight,p.readout_weight_decay,p.image_eval_cases) < 0 or p.generation_guard_ratio < 1:
                raise ValueError("Invalid deployment regularization/evaluation settings")
            protocol_numbers = (p.representation_lr, p.readout_lr, p.residual_logit_limit,
                                p.residual_penalty, p.latent_mean_weight, p.latent_energy_weight,
                                p.readout_weight_decay, p.generation_guard_ratio)
            if any(not math.isfinite(v) for v in protocol_numbers):
                raise ValueError("Deployment protocol numbers must be finite")
            if (type(p.validation_batch_size) is not int or p.validation_batch_size < 1
                    or type(p.image_eval_cases) is not int or p.image_eval_cases < 0
                    or type(p.validation_seed) is not int or not 0 <= p.validation_seed < 2**32):
                raise ValueError("Invalid evaluation batch size, case count or seed")
            for values in (p.stage_batch_sizes, p.stage_validation_every):
                if (not isinstance(values, dict)
                        or any(k not in {"representation", "flow", "readout", "joint"}
                               or type(v) is not int or v < 1 for k, v in values.items())):
                    raise ValueError("Stage batch sizes/cadences must be positive integers for A/B/C/D")
        dimensions = (*e.widths,*e.depths,*e.heads,*e.window,*e.token_grid,*n.channels,
                      e.stem_width,e.dim,e.query_heads,e.anatomy_tokens,e.disease_tokens,
                      e.predictor_depth,e.phase_mixer_depth,n.attention_heads,n.num_res_blocks,n.pillar_dim)
        if any(not isinstance(v,int) or isinstance(v,bool) or v < 1 for v in dimensions):
            raise ValueError("Architecture sizes must be positive integers")
        if e.phase_channels != 8 or len(e.widths) < 2:
            raise ValueError("24-channel, three-phase VQ contract required")
        if len(e.widths) != len(e.depths) or len(e.widths) != len(e.heads):
            raise ValueError("Encoder stages do not match")
        if any(w % h for w, h in zip(e.widths, e.heads)) or e.dim % e.query_heads or e.stem_width % e.query_heads:
            raise ValueError("Invalid attention dimensions")
        if len(e.window) != 3 or len(e.token_grid) != 3:
            raise ValueError("3D grids required")
        if len(n.channels) < 2 or any(c % n.attention_heads for c in n.channels):
            raise ValueError("Invalid image backbone")
        if n.time_basis not in {"calendar_days", "stage_index"}:
            raise ValueError("Unknown time basis")
        if n.backend not in {"monai", "native"} or n.readout_source not in {"joint", "reencode"}:
            raise ValueError("Unknown backbone/readout")
        if min(n.history_depth, n.semantic_depth, n.readout_depth) < 2:
            raise ValueError("Use multi-block history/semantic/readout stacks")
        if n.semantic_depth % 2 or n.max_visits < 2:
            raise ValueError("Even semantic depth and >=2 visits required")
        if t.precision not in {"fp32", "bf16"} or (t.device == "cpu" and t.precision != "fp32"):
            raise ValueError("Use fp32 on CPU, fp32/bf16 on CUDA")
        if t.joint_image_scope not in {"all", "decoder", "frozen"}:
            raise ValueError("Unknown joint image tuning scope")
        if not 0 <= t.reverse_probability <= 1 or not 0 < t.ema_decay < 1:
            raise ValueError("Invalid reverse probability / EMA")
        if not 0 < e.mask_ratio < 1 or not 0 <= e.phase_drop_probability <= 1 or not 0 <= e.drop_path < 1:
            raise ValueError("Invalid encoder regularization")
        if not 0 <= n.dropout < 1 or n.future_residual_limit <= 0:
            raise ValueError("Invalid readout regularization")
        positive = (t.lr, t.joint_lr, t.grad_clip, t.batch_size, t.accumulation,
                    t.validation_every, t.checkpoint_every, t.log_every, t.threads,
                    t.validation_cases, s.train_steps, s.inference_steps, s.inference_samples,
                    t.validate_samples)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Positive finite training/sampling values required")
        if s.train_samples < 2 or s.method not in {"euler", "heun"}:
            raise ValueError("Training distribution scores need K>=2 and a supported ODE solver")
        if t.weight_decay < 0 or t.warmup < 0 or t.selection_generation_weight < 0:
            raise ValueError("Negative optimizer/selection hyperparameter")
        if any(v < 0 for k, v in asdict(t).items() if k.endswith("_steps")):
            raise ValueError("Negative training budget")
        if any(not isinstance(v, bool) and (not math.isfinite(v) or v < 0) for v in asdict(self.loss).values()):
            raise ValueError("Invalid loss weights")
        return self

    def to_dict(self):
        value = asdict(self)
        if self.schema == "responsewm_v1":
            value.pop("multistage")
        if self.schema != "responsewm_v3":
            value.pop("protocol")
        return value

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def from_dict(value):
    def build(cls, data):
        extra = set(data) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(extra)}")
        return cls(**data)
    v = dict(value)
    for key, cls in (("encoder", EncoderConfig), ("network", NetworkConfig), ("loss", LossConfig),
                     ("training", TrainConfig), ("sampling", SamplingConfig), ("multistage", MultistageConfig),
                     ("protocol", DeploymentProtocolConfig)):
        v[key] = build(cls, v.get(key, {}))
    cfg = build(Config, v)
    for obj, names in ((cfg.encoder, ("widths", "depths", "heads", "window", "token_grid")),
                       (cfg.network, ("channels",)),
                       (cfg.multistage, ("canonical_stages", "deep_jepa_weights", "flow_phase_fractions"))):
        for name in names:
            setattr(obj, name, tuple(getattr(obj, name)))
    return cfg.validate()


def load_config(path):
    return from_dict(yaml.safe_load(Path(path).read_text()))

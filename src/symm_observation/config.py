"""Versioned, fail-closed configuration for the A-observation/B-dynamics split."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import math
import yaml


@dataclass
class EncoderConfig:
    patch: tuple[int, int, int] = (2, 4, 4)
    phase_dim: int = 128
    phase_heads: int = 4
    phase_depth: int = 2
    dim: int = 384
    heads: int = 6
    depth: int = 12
    predictor_dim: int = 384
    predictor_heads: int = 6
    predictor_depth: int = 6
    readout_depth: int = 4
    teacher_last_k: int = 4
    checkpoint_blocks: bool = True
    max_tokens: int = 4096


@dataclass
class ObservationConfig:
    crop: tuple[int, int, int] = (16, 32, 32)
    mask_ratio: float = 0.55
    mask_blocks: int = 4
    minimum_patch_coverage: float = 0.95
    domain_enabled: bool = False
    local_weight: float = 1.0
    global_weight: float = 1.0
    reconstruction_weight: float = 0.25
    kinetics_weight: float = 0.0
    segmentation_weight: float = 0.0
    arcface_weight: float = 0.0
    dann_weight: float = 0.0
    variance_weight: float = 0.0
    phenotype_classes: int = 0
    phenotype_definition: str = ""
    domain_classes: int = 0
    domain_definition: str = ""
    supervised_start_step: int = 5000
    ema_start: float = 0.996
    ema_end: float = 0.9999
    require_phase_alignment: bool = False


@dataclass
class VelocityConfig:
    backend: str = "monai"
    channels: tuple[int, ...] = (64, 128, 256, 256)
    attention_levels: tuple[bool, ...] = (False, False, True, True)
    num_res_blocks: int = 2
    attention_heads: int = 8
    checkpoint_blocks: bool = True
    context_dim: int = 256
    patient_tokens: int = 32
    adapter_depth: int = 2
    source_tokens: bool = True
    semantic_endpoint_weight: float = 0.0
    semantic_start_step: int = 2000
    semantic_every: int = 4
    reverse_probability: float = 0.0


@dataclass
class TrainingConfig:
    seed: int = 20260920
    device: str = "cuda"
    precision: str = "bf16"
    stage_a_steps: int = 20000
    stage_b_steps: int = 60000
    batch_size: int = 1
    accumulation: int = 8
    lr_a: float = 0.0002
    lr_b: float = 0.0001
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    checkpoint_every: int = 1000
    validate_every: int = 1000
    validation_cases: int = 16
    log_every: int = 25
    cpu_threads: int = 2
    strict_determinism: bool = True
    stage_batches: dict[str, int] = field(default_factory=dict)
    reference_batch_size: int = 0
    preload_latents: bool = False
    patient_balanced_validation: bool = False


@dataclass
class SamplingConfig:
    steps: int = 20
    method: str = "heun"
    samples: int = 4


@dataclass
class Config:
    schema: str = "three_phase_observation_v3"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def validate(self):
        if self.schema != "three_phase_observation_v3":
            raise ValueError("Not an observation-v3 configuration; v2 checkpoints/configs are incompatible")
        e, a, v, t, s = self.encoder, self.observation, self.velocity, self.training, self.sampling
        for dim, heads in ((e.phase_dim, e.phase_heads), (e.dim, e.heads),
                           (e.predictor_dim, e.predictor_heads), (v.context_dim, v.attention_heads)):
            if heads < 1 or dim < 1 or dim % heads:
                raise ValueError("Attention widths must be positive multiples of their head counts")
        if len(e.patch) != 3 or len(a.crop) != 3 or any(p < 1 for p in e.patch):
            raise ValueError("patch and crop must be 3D")
        if any(n < p or n % p for n, p in zip(a.crop, e.patch)):
            raise ValueError("Observation crop must be divisible by patch size")
        if math.prod(n // p for n, p in zip(a.crop, e.patch)) < 2:
            raise ValueError("At least two patches are needed for masked prediction")
        if min(e.phase_depth, e.depth, e.predictor_depth, e.readout_depth, e.max_tokens) < 1:
            raise ValueError("Transformer depths/max_tokens must be positive")
        if not 1 <= e.teacher_last_k <= e.depth:
            raise ValueError("teacher_last_k must lie within encoder depth")
        if not 0 < a.mask_ratio < 1 or a.mask_blocks < 1 or not 0 < a.minimum_patch_coverage <= 1:
            raise ValueError("Invalid mask configuration")
        if not 0 <= a.ema_start <= a.ema_end < 1:
            raise ValueError("Invalid observation EMA schedule")
        for key, value in asdict(a).items():
            if key.endswith("_weight") and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid observation loss {key}")
        if a.local_weight + a.global_weight + a.reconstruction_weight <= 0:
            raise ValueError("A requires at least one observation learning objective")
        if a.arcface_weight and (a.phenotype_classes < 2 or not a.phenotype_definition):
            raise ValueError("ArcFace needs a declared current-visit phenotype and >=2 classes")
        if a.dann_weight and (a.domain_classes < 2 or a.domain_definition not in {"scanner", "site"}):
            raise ValueError("DANN only supports declared scanner/site labels, not phase/treatment/visit")
        if a.supervised_start_step < 0 or v.semantic_start_step < 0:
            raise ValueError("Start steps cannot be negative")
        if v.backend not in {"native", "monai"} or len(v.channels) < 2:
            raise ValueError("Use a native or MONAI multi-scale velocity U-Net")
        if len(v.attention_levels) != len(v.channels):
            raise ValueError("attention_levels must match velocity channels")
        if any(c < 4 or c % v.attention_heads for c in v.channels):
            raise ValueError("Velocity widths must be positive multiples of attention_heads")
        if min(v.num_res_blocks, v.patient_tokens, v.adapter_depth, v.semantic_every) < 1:
            raise ValueError("Invalid velocity/adaptor dimensions")
        if not 0 <= v.reverse_probability < 1 or not math.isfinite(v.semantic_endpoint_weight) or v.semantic_endpoint_weight < 0:
            raise ValueError("Invalid B sampling/loss setting")
        if t.precision not in {"fp32", "bf16"} or (t.device.startswith("cpu") and t.precision != "fp32"):
            raise ValueError("CPU requires fp32; GPU supports fp32/bf16")
        for key in ("stage_a_steps", "stage_b_steps", "warmup_steps"):
            if getattr(t, key) < 0:
                raise ValueError(f"training.{key} cannot be negative")
        for key in ("batch_size", "accumulation", "checkpoint_every", "validate_every", "validation_cases", "log_every", "cpu_threads"):
            if getattr(t, key) < 1:
                raise ValueError(f"training.{key} must be positive")
        if not all(math.isfinite(x) for x in (t.lr_a,t.lr_b,t.grad_clip,t.weight_decay)):
            raise ValueError("Optimizer hyperparameters must be finite")
        if not 0 <= t.ema_decay < 1 or min(t.lr_a, t.lr_b, t.grad_clip) <= 0 or t.weight_decay < 0:
            raise ValueError("Invalid optimizer/EMA setting")
        if t.reference_batch_size < 0 or set(t.stage_batches) - {"A", "B"}:
            raise ValueError("Invalid reference batch or stage names")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in t.stage_batches.values()):
            raise ValueError("Stage batch sizes must be positive integers")
        if s.steps < 1 or s.samples < 1 or s.method not in {"euler", "heun"}:
            raise ValueError("Invalid ODE sampling settings")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def from_dict(value):
    classes = {"encoder": EncoderConfig, "observation": ObservationConfig, "velocity": VelocityConfig,
               "training": TrainingConfig, "sampling": SamplingConfig}
    if not isinstance(value, dict) or set(value) - (set(classes) | {"schema"}):
        raise ValueError("Unknown top-level config keys")
    parts = {}
    for name, cls in classes.items():
        raw = dict(value.get(name, {}))
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} config keys: {sorted(unknown)}")
        for key in ("patch", "crop", "channels", "attention_levels"):
            if key in raw:
                raw[key] = tuple(raw[key])
        parts[name] = cls(**raw)
    return Config(schema=value.get("schema", "three_phase_observation_v3"), **parts).validate()


def stage_budget(cfg, stage):
    if stage not in {"A", "B"}:
        raise ValueError("Expected stage A or B")
    t = cfg.training
    batch = t.stage_batches.get(stage, t.batch_size)
    effective = batch * t.accumulation
    reference = t.reference_batch_size or effective
    reference_steps = t.stage_a_steps if stage == "A" else t.stage_b_steps
    samples = reference_steps * reference
    # Keep the original sample budget when changing physical batch size.
    if samples % effective:
        raise ValueError("Stage sample budget must be divisible by effective batch size")
    scale = reference / effective
    return {"batch": batch, "effective_batch": effective, "reference_batch": reference,
            "samples": samples, "steps": samples // effective,
            "warmup_steps": math.ceil(t.warmup_steps * scale),
            "validate_every": max(1, math.ceil(t.validate_every * scale)),
            "checkpoint_every": max(1, math.ceil(t.checkpoint_every * scale)),
            "ema_decay": t.ema_decay ** (effective / reference)}


def load_config(path):
    def read(p, seen):
        p = Path(p).resolve()
        if p in seen:
            raise ValueError("Cyclic YAML inheritance")
        value = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a YAML mapping")
        parent = value.pop("extends", None)
        if parent:
            base = read(p.parent / parent, seen | {p})
            for key, val in value.items():
                base[key] = {**base[key], **val} if isinstance(val, dict) and isinstance(base.get(key), dict) else val
            return base
        return value
    return from_dict(read(path, set()))

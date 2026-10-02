"""Stable per-physiology noise and full canonical forecast traces."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import torch
from .state import PatientBelief


class NoiseLedger:
    def __init__(self, seed=0, trajectory_id="default"):
        self.seed, self.trajectory_id = int(seed), str(trajectory_id)

    def noise(self, reference, physiology_hashes, branch_prefix_hashes, src_stage, dst_stage, component):
        b, k = reference.shape[:2]
        rows = []
        for patient in range(b):
            samples = []
            for sample in range(k):
                key = (self.seed, physiology_hashes[patient], self.trajectory_id, sample,
                       src_stage, dst_stage, branch_prefix_hashes[patient], component)
                digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).digest()
                generator = torch.Generator(device=reference.device).manual_seed(int.from_bytes(digest[:8], "big") % (2**63-1))
                samples.append(torch.randn(reference.shape[2:], device=reference.device,
                                           dtype=reference.dtype, generator=generator))
            rows.append(torch.stack(samples))
        return torch.stack(rows)

    def state_dict(self):
        return {"seed": self.seed, "trajectory_id": self.trajectory_id}


@dataclass(frozen=True)
class PCRQueryResult:
    logit_per_sample: torch.Tensor
    probability: torch.Tensor
    interpretation: str
    as_of: float
    observed_stage_ids: tuple[tuple[int, ...], ...]
    assumed_plan_version: str | None = None


@dataclass(frozen=True)
class ForecastTrace:
    source_state: PatientBelief
    internal_full_trace: tuple[PatientBelief, ...]
    requested_stages: tuple[int, ...]
    image_states: tuple[torch.Tensor, ...]
    stage_logits: torch.Tensor
    trajectory_id: str
    pcr_marginal: PCRQueryResult | None = None
    assumed_plan_version: str | None = None

    @property
    def all_states(self):
        return self.internal_full_trace

    @property
    def output_stages(self):
        return tuple(s for s in self.stage_ids if s in self.requested_stages)

    @property
    def final_state(self):
        return self.internal_full_trace[-1] if self.internal_full_trace else self.source_state

    @property
    def stage_ids(self):
        return tuple(s.stage for s in self.internal_full_trace)

    @property
    def origin_stage(self):
        return self.source_state.stage

    @property
    def stage_valid_mask(self):
        return torch.ones(len(self.source_state.memory), len(self.stage_ids), device=self.stage_logits.device, dtype=torch.bool)

    @property
    def stage_probs(self):
        return self.stage_logits.float().sigmoid().mean(1)

    @property
    def state_versions(self):
        return tuple(s.version for s in self.internal_full_trace)

    def _stack(self, field):
        values = [getattr(s, field) for s in self.internal_full_trace if s.stage in self.requested_stages]
        if values:
            return torch.stack(values, 2)
        source = getattr(self.source_state, field)
        return source.new_empty(*source.shape[:2], 0, *source.shape[2:])

    @property
    def latent(self):
        return self._stack("anchor_latent")

    @property
    def state(self):
        return self._stack("anchor_disease")

    @property
    def memory(self):
        return self._stack("memory")

    @property
    def image_state(self):
        values = [v for s, v in zip(self.internal_full_trace, self.image_states) if s.stage in self.requested_stages]
        return torch.stack(values, 2) if values else self.state

    @property
    def logits(self):
        if self.pcr_marginal is None:
            return self.stage_logits[:, :, -1]
        return self.pcr_marginal.logit_per_sample

    @property
    def probability(self):
        return self.logits.float().sigmoid().mean(1)

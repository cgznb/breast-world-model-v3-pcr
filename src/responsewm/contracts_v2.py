"""Stage-based, input-only patient prefixes and separate training supervision."""
from __future__ import annotations

from dataclasses import dataclass, fields
import torch

from .contracts import ForecastInput, Supervision


@dataclass(frozen=True)
class PatientPrefix(ForecastInput):
    as_of: torch.Tensor
    stage_valid: torch.Tensor
    output_query_mask: torch.Tensor
    observed_stage_ids: tuple[tuple[int, ...], ...]
    observed_available_at: torch.Tensor
    clinical_known_at: torch.Tensor | None = None

    @property
    def value_known_mask(self):
        return self.clinical_mask

    def to(self, device):
        return type(self)(**{f.name: (getattr(self, f.name).to(device)
                            if isinstance(getattr(self, f.name), torch.Tensor)
                            else getattr(self, f.name)) for f in fields(self)})

    def validate(self, clinical_dim, action_dim, max_visits=4):
        super().validate(clinical_dim, action_dim, max_visits)
        b = len(self.observed)
        if self.as_of.shape != (b,) or self.stage_valid.shape != (b, 4):
            raise ValueError("Invalid stage/landmark shapes")
        if self.output_query_mask.shape != (b, 4):
            raise ValueError("Output query mask must address the canonical stage grid")
        if self.stage_valid.dtype != torch.bool or self.output_query_mask.dtype != torch.bool:
            raise ValueError("Stage validity and output query masks must be boolean")
        if self.observed_available_at.shape != self.observed_days.shape:
            raise ValueError("Observed availability dimensions differ")
        if self.clinical_known_at is not None and self.clinical_known_at.shape != self.clinical.shape:
            raise ValueError("Clinical availability dimensions differ")
        if len(self.observed_stage_ids) != b:
            raise ValueError("One observed stage set per patient is required")
        for i in range(b):
            stage = int(self.as_of[i])
            if self.as_of[i] != stage or stage not in range(4):
                raise ValueError("R1 supports integer canonical stages 0, 1, 2, 3 only")
            valid = self.observed_mask[i]
            coordinates = torch.cat((self.observed_days[i, valid], self.future_days[i, self.future_mask[i]]))
            if ((coordinates != coordinates.round()) | (coordinates < 0) | (coordinates > 3)).any():
                raise ValueError("R1 accepts only canonical integer stages, never arbitrary calendar days")
            ids = tuple(int(v) for v in self.observed_days[i, valid])
            if ids != self.observed_stage_ids[i] or any(v > stage for v in ids):
                raise ValueError("Observed stage metadata disagrees with legal prefix")
            if (self.observed_available_at[i, valid] > stage).any():
                raise ValueError("Observed MRI was unavailable at this landmark")
            if (self.observed_available_at[i, valid] < self.observed_days[i, valid]).any():
                raise ValueError("MRI cannot be available before its acquisition stage")
            if self.clinical_known_at is not None:
                known = self.clinical_known_at[i, self.clinical_mask[i]]
                if not torch.isfinite(known).all() or ((known < 0) | (known > stage)).any():
                    raise ValueError("Clinical fields were unavailable at this landmark")
            expected = tuple(j for j in range(stage + 1, 4) if self.stage_valid[i, j])
            actual = tuple(int(v) for v in self.future_days[i, self.future_mask[i]])
            if actual != expected:
                raise ValueError("Future grid must contain all remaining canonical stages")
            if self.output_query_mask[i, :stage + 1].any() or (self.output_query_mask[i] & ~self.stage_valid[i]).any():
                raise ValueError("Output queries must be valid future stages")
        return self


PrefixInput = PatientPrefix


@dataclass
class TrajectorySupervision(Supervision):
    target_available_mask: torch.Tensor

    def to(self, device):
        base = super().to(device)
        return type(self)(**{f.name: getattr(base, f.name) for f in fields(Supervision)},
                          target_available_mask=self.target_available_mask.to(device))


@dataclass
class EdgeSupervision:
    latent: torch.Tensor
    source_stage: int
    target_stage: int
    kind: str
    anatomy_comparable: torch.Tensor
    auxiliary: list

    def to(self, device):
        return type(self)(self.latent.to(device), self.source_stage, self.target_stage,
                          self.kind, self.anatomy_comparable.to(device),
                          [{k: v.to(device) for k, v in item.items()} for item in self.auxiliary])

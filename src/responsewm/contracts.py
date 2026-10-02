"""Input-only forecast API, deliberately disjoint from supervision and IDs."""
from __future__ import annotations
from dataclasses import dataclass, fields
import torch

@dataclass(frozen=True)
class ForecastInput:
    observed: torch.Tensor      # [B,T,24,D,H,W], standardized with TRAIN statistics
    observed_mask: torch.Tensor  # [B,T] boolean, holes allowed, >=1 valid per patient
    observed_days: torch.Tensor  # [B,T]
    clinical: torch.Tensor       # [B,C], standardized; unknown values zero
    clinical_mask: torch.Tensor  # [B,C], explicit missingness
    future_days: torch.Tensor   # [B,F], scheduled/query times, not unknown realized dates
    future_mask: torch.Tensor   # [B,F], requested horizons, contiguous prefix
    actions: torch.Tensor       # [B,F,A], currently known/planned treatments
    action_mask: torch.Tensor   # [B,F,A]

    def to(self, device):
        return ForecastInput(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def validate(self, clinical_dim, action_dim, max_visits):
        z, m, d = self.observed, self.observed_mask, self.observed_days
        if z.ndim != 6 or z.shape[2] != 24:
            raise ValueError("Observed latent must be [B,T,24,D,H,W]")
        b, t = z.shape[:2]
        if b < 1 or t < 1 or self.future_days.ndim != 2 or min(z.shape[3:]) < 1:
            raise ValueError("Nonempty observed batch/history and rank-2 future times required")
        f = self.future_days.shape[1]
        shapes = {"observed_mask":(b,t), "observed_days":(b,t), "clinical":(b,clinical_dim),
                  "clinical_mask":(b,clinical_dim), "future_mask":(b,f),
                  "actions":(b,f,action_dim), "action_mask":(b,f,action_dim)}
        for key, shape in shapes.items():
            if getattr(self, key).shape != shape:
                raise ValueError(f"Invalid {key} shape")
        if self.future_days.shape != (b, f) or t+f > max_visits:
            raise ValueError("Invalid horizon / too many visits")
        for key in ("observed_mask", "clinical_mask", "future_mask", "action_mask"):
            if getattr(self, key).dtype != torch.bool:
                raise ValueError(f"{key} must be boolean")
        if not m.any(1).all():
            raise ValueError("At least one observed MRI per patient is required")
        for key in ("observed", "observed_days", "clinical", "future_days", "actions"):
            if not torch.isfinite(getattr(self,key)).all():
                raise ValueError(f"Nonfinite input {key}; impute unknowns explicitly, with mask")
        for i in range(b):
            od = d[i, m[i]]
            fd = self.future_days[i, self.future_mask[i]]
            if (od < 0).any() or (od[1:] <= od[:-1]).any():
                raise ValueError("Observed days must be nonnegative and increasing")
            if f and (self.future_mask[i, 1:] & ~self.future_mask[i, :-1]).any():
                raise ValueError("Requested future horizons must be a prefix")
            if len(fd) and (fd[0] <= od[-1] or (fd[1:] <= fd[:-1]).any()):
                raise ValueError("Future query days must follow observed history")
        return self

@dataclass
class Supervision:
    future: torch.Tensor       # [B,F,24,D,H,W], never accepted by forecast()
    future_mask: torch.Tensor  # actual availability, may have holes
    label: torch.Tensor        # [B], arbitrary finite placeholder when missing
    label_mask: torch.Tensor   # [B]
    # Optional sidecars: list of B lists, ordered observed slots then future slots.
    auxiliary: list
    anatomy_comparable: torch.Tensor  # [B,F], audited geometry/tissue policy

    def to(self, device):
        aux = [[{k:v.to(device) for k,v in visit.items()} for visit in patient]
               for patient in self.auxiliary]
        return Supervision(self.future.to(device), self.future_mask.to(device), self.label.to(device),
                           self.label_mask.to(device), aux, self.anatomy_comparable.to(device))


def last_valid(mask):
    """Index, not count-1: valid when intermediate observations are missing."""
    return torch.arange(mask.shape[1], device=mask.device).expand_as(mask).masked_fill(~mask, -1).amax(1)


def gather_visit(x, mask):
    return x[torch.arange(len(x), device=x.device), last_valid(mask)]

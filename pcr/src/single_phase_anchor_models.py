"""Fold-fitted clinical anchor, frozen T0 path and causal bounded corrections."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.single_phase_response_models import prefix_mask
from src.single_phase_temporal_models import FrozenFeatureTransform


class ClinicalAnchor(nn.Module):
    def __init__(self, state):
        super().__init__()
        for name in ("median", "mean", "scale"):
            self.register_buffer(name, torch.as_tensor(state["transform"][name]).float().clone())
        self.register_buffer("coefficient", torch.as_tensor(state["coefficient"]).float().clone())
        self.register_buffer("intercept", torch.tensor(float(state["intercept"])))

    def forward(self, clinical):
        x = torch.where(torch.isfinite(clinical), clinical, self.median)
        return ((x - self.mean) / self.scale) @ self.coefficient + self.intercept


class T0Anchor(nn.Module):
    def __init__(self, settings, statistics, clinical):
        super().__init__()
        self.clinical = ClinicalAnchor(clinical)
        self.features = FrozenFeatureTransform(settings, statistics)
        width = int(settings["hidden_dim"])
        self.limit = float(settings["residual_limit"])
        self.head = nn.Sequential(nn.Linear(self.features.output_dim, width), nn.GELU(),
                                  nn.Dropout(settings["dropout"]), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, mask, clinical):
        keep = prefix_mask(mask)
        # The baseline transform and forward both consume T0 only.
        z = self.features(features[:, :1], keep[:, :1])[:, 0]
        residual = self.limit * torch.tanh(self.head(z).squeeze(-1))
        residual = torch.where(keep[:, 0], residual, 0)
        return self.clinical(clinical) + residual, residual


class AnchoredResponse(nn.Module):
    def __init__(self, settings, kind, statistics, clinical, base_spec=None):
        super().__init__()
        if kind not in ("t0", "linear", "gated"):
            raise ValueError("Unknown anchored response head")
        self.kind, self.settings = kind, dict(settings)
        self.limit = float(settings["residual_limit"])
        if kind == "t0":
            if base_spec is not None:
                raise ValueError("A T0 model must be initialized independently")
            self.base = T0Anchor(settings, statistics, clinical)
            return
        if base_spec is None:
            raise ValueError("Follow-up fitting needs its exact training-partition T0 model")
        self.base = T0Anchor(settings, base_spec["statistics"], clinical)
        self.base.load_state_dict(base_spec["state"], strict=True)
        self.base.requires_grad_(False).eval()
        self.features = FrozenFeatureTransform(settings, statistics)
        size = 3 * self.features.output_dim + 2
        width = int(settings["hidden_dim"])
        if kind == "linear":
            self.outputs = nn.ModuleList([nn.Linear(size, 1) for _ in range(3)])
        else:
            self.encode = nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.Dropout(settings["dropout"]))
            self.quality = nn.Linear(width, 1)
            self.outputs = nn.ModuleList([nn.Linear(2 * width, 1) for _ in range(3)])
            self.gates = nn.ModuleList([nn.Linear(2 * width, 1) for _ in range(3)])
            initial = float(settings["gate_initial_probability"])
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, math.log(initial / (1 - initial)))
        for layer in self.outputs:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.kind != "t0":
            self.base.eval()
        return self

    def forward(self, features, mask, clinical, days):
        keep = prefix_mask(mask)
        base, t0_residual = self.base(features, mask, clinical)
        if self.kind == "t0":
            return base[:, None].expand(-1, 4), t0_residual[:, None].expand(-1, 4)
        z = self.features(torch.where(keep[..., None], features, 0), keep)
        times = torch.where(keep, days, 0) / 180.0
        outputs, residuals = [base], [torch.zeros_like(base)]
        last, last_residual = base, torch.zeros_like(base)
        pooled, mass = None, None
        for t in range(1, 4):
            value = torch.cat((z[:, t], z[:, t] - z[:, 0], z[:, t] - z[:, t - 1],
                               times[:, t, None], (times[:, t] - times[:, t - 1])[:, None]), -1)
            if self.kind == "gated":
                current = self.encode(value)
                quality = torch.sigmoid(self.quality(current)) * keep[:, t, None]
                pooled = quality * current if pooled is None else pooled + quality * current
                mass = quality if mass is None else mass + quality
                value = torch.cat((current, pooled / mass.clamp_min(1e-6)), -1)
                gate = torch.sigmoid(self.gates[t - 1](value).squeeze(-1))
            else:
                gate = 1.0
            correction = self.limit * gate * torch.tanh(self.outputs[t - 1](value).squeeze(-1))
            last = torch.where(keep[:, t], base + correction, last)
            last_residual = torch.where(keep[:, t], correction, last_residual)
            outputs.append(last)
            residuals.append(last_residual)
        return torch.stack(outputs, 1), torch.stack(residuals, 1)


def learning_mask(mask, kind):
    keep = prefix_mask(mask).clone()
    if kind == "t0":
        keep[:, 1:] = False
    else:
        keep[:, 0] = False
    return keep


def objective(logits, residual, labels, mask, kind, penalty=0.0):
    keep = learning_mask(mask, kind)
    eligible = keep.any(1)
    if not eligible.any():
        raise ValueError("No eligible observed prefix for this fitting stage")
    values = F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits), reduction="none")
    values = values + penalty * residual.square()
    per_patient = torch.where(keep, values, 0).sum(1) / keep.sum(1).clamp_min(1)
    return per_patient[eligible].mean()


def probe_inputs(features, days, kind, t):
    if kind == "current":
        return features[:, t]
    if kind == "pair":
        interval = days[:, t] - days[:, max(0, t - 1)]
        return np.concatenate((features[:, 0], features[:, t] - features[:, 0],
                               days[:, t, None] / 180, interval[:, None] / 180), axis=1)
    raise ValueError("Unknown single-visit or paired-visit probe")

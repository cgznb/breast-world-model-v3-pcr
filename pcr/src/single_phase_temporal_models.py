"""Fold-fitted frozen feature transforms and explicit longitudinal response heads."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from scripts.run_full978_independent_cv import _subset_split
from src.tdn import TDN


def window_split(split, ids, depth, effective):
    allowed = set(ids)
    result = _subset_split(split, [p in allowed for p in split["pids"]])
    if len(result["pids"]) != len(ids) or len(allowed) != len(ids):
        raise ValueError("Task patient identities are missing or repeated")
    limit = min(int(depth), int(effective.get("input_depth", depth)))
    if not 1 <= limit <= 4:
        raise ValueError("Invalid input window")
    keep = result["masks"] > 0
    policy = effective.get("visit_policy", "contiguous")
    if policy == "contiguous":
        keep = np.cumprod(keep, axis=1).astype(bool)
    elif policy == "available":
        keep &= keep[:, :1]
    else:
        raise ValueError("Unknown visit policy")
    keep[:, limit:] = False
    result["embs"] = np.where(keep[..., None], result["embs"], 0).astype(np.float32)
    result["masks"] = keep.astype(np.float32)
    result["days"] = np.where(keep, result["days"], 0).astype(np.float32)
    return result


def masked_window(embeddings, mask, days, depth, effective):
    limit = min(int(depth), int(effective.get("input_depth", depth)))
    keep = mask > 0
    if effective.get("visit_policy", "contiguous") == "contiguous":
        keep = torch.cumprod(keep.to(torch.int64), dim=1).bool()
    else:
        keep = keep & keep[:, :1]
    keep = keep & (torch.arange(mask.shape[1], device=mask.device)[None, :] < limit)
    days = torch.zeros_like(mask) if days is None else days
    return torch.where(keep[..., None], embeddings, 0), keep.to(mask.dtype), torch.where(keep, days, 0)


def fit_feature_statistics(train, effective):
    """Only the current training partition and allowed visits may enter PCA."""
    mode = effective.get("feature_transform", "identity")
    dim = int(effective["input_dim"])
    if mode == "identity":
        return {"mode": mode, "fitted_visits": 0}
    rows = train["embs"][train["masks"] > 0].astype(np.float64)
    if len(rows) < 2 or rows.shape[1] != dim or not np.isfinite(rows).all():
        raise ValueError("Invalid training features for fold transform")
    mean = rows.mean(axis=0)
    state = {"mode": mode, "mean": torch.as_tensor(mean, dtype=torch.float32),
             "fitted_visits": len(rows)}
    if mode == "standardize":
        deviation = rows.std(axis=0)
        floor = max(float(np.median(deviation)) * 0.1, 1e-5)
        state["scale"] = torch.as_tensor(1 / np.maximum(deviation, floor), dtype=torch.float32)
    elif mode == "pca":
        width = int(effective.get("pca_dim", 32))
        count = min(width, dim, len(rows) - 1)
        estimator = PCA(n_components=count, svd_solver="randomized", random_state=92026)
        estimator.fit(rows)
        components = np.zeros((dim, width), dtype=np.float32)
        components[:, :count] = estimator.components_.T
        scale = np.zeros(width, dtype=np.float32)
        floor = max(float(estimator.explained_variance_[0]) * 1e-4, 1e-8)
        scale[:count] = 1 / np.sqrt(np.maximum(estimator.explained_variance_, floor))
        state.update(components=torch.from_numpy(components), scale=torch.from_numpy(scale),
                     explained_variance_fraction=float(estimator.explained_variance_ratio_.sum()))
    else:
        raise ValueError("Unknown feature transform")
    return state


class FrozenFeatureTransform(nn.Module):
    def __init__(self, effective, state=None):
        super().__init__()
        self.mode = effective.get("feature_transform", "identity")
        dim = int(effective["input_dim"])
        self.output_dim = int(effective.get("pca_dim", 32)) if self.mode == "pca" else dim
        if self.mode != "identity":
            self.register_buffer("mean", torch.zeros(dim))
            self.register_buffer("scale", torch.ones(self.output_dim))
        if self.mode == "pca":
            self.register_buffer("components", torch.zeros(dim, self.output_dim))
        if state is not None:
            if state["mode"] != self.mode:
                raise ValueError("Feature transform state/config mismatch")
            for name in ("mean", "scale", "components"):
                if name in state:
                    getattr(self, name).copy_(state[name])

    def forward(self, embeddings, mask):
        if self.mode == "identity":
            result = embeddings
        elif self.mode == "standardize":
            result = ((embeddings - self.mean) * self.scale).clamp(-5, 5)
        else:
            result = (((embeddings - self.mean) @ self.components) * self.scale).clamp(-5, 5)
        return torch.where((mask > 0)[..., None], result, 0)


class FeatureTDN(TDN):
    def __init__(self, effective, depth, statistics=None):
        downstream = dict(effective)
        if effective.get("feature_transform") == "pca":
            downstream["input_dim"] = effective["pca_dim"]
        super().__init__({"downstream": downstream})
        self.features = FrozenFeatureTransform(effective, statistics)
        self.effective, self.depth = dict(effective), depth

    def forward(self, embeddings, mask, clinical, days=None, prior_logit=None, return_residual=False):
        embeddings, mask, days = masked_window(embeddings, mask, days, self.depth, self.effective)
        return super().forward(self.features(embeddings, mask), mask, clinical, days,
                               prior_logit, return_residual)


class DeltaHead(nn.Module):
    """Clinical prior plus a compact head over T0 and visit-specific changes."""
    def __init__(self, effective, depth, statistics=None):
        super().__init__()
        self.features = FrozenFeatureTransform(effective, statistics)
        self.effective, self.depth = dict(effective), depth
        dim = self.features.output_dim
        width = 4 * dim + 8
        hidden = int(effective.get("hidden_dim", 32))
        self.head = (nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                  nn.Dropout(float(effective["dropout"])), nn.Linear(hidden, 1))
                     if hidden else nn.Linear(width, 1))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def response_features(self, embeddings, mask, days):
        embeddings, mask, days = masked_window(embeddings, mask, days, self.depth, self.effective)
        z = self.features(embeddings, mask)
        changes = (z[:, 1:] - z[:, :1]) * mask[:, 1:, None]
        return torch.cat([z[:, 0], changes.flatten(1), mask, (days / 180).clamp(0, 2)], dim=1)

    def forward(self, embeddings, mask, clinical, days=None, prior_logit=None, return_residual=False):
        inputs = self.response_features(embeddings, mask, days)
        residual = self.alpha * self.head(inputs).squeeze(-1)
        logit = residual if prior_logit is None else prior_logit + residual
        return (logit, residual) if return_residual else logit


def build_model(effective, depth, statistics=None):
    architecture = effective.get("architecture", "tdn")
    if architecture == "tdn":
        return FeatureTDN(effective, depth, statistics)
    if architecture == "delta":
        return DeltaHead(effective, depth, statistics)
    raise ValueError("Unknown temporal architecture")

"""Frozen T0 reference with a bounded, patient-specific temporal correction."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from src.single_phase_temporal_models import FeatureTDN, masked_window
from src.tdn import Projection


class GatedTemporal(nn.Module):
    def __init__(self, effective, depth):
        super().__init__()
        self.effective, self.depth = dict(effective), int(depth)
        self.t0 = FeatureTDN(effective["t0_effective"], 1).requires_grad_(False).eval()
        width = int(effective["sig_dim"])
        if effective["proj_dim"] != width:
            raise ValueError("Gated projection and signature widths must match")
        self.limit = float(effective["temporal_logit_limit"])
        initial = float(effective["gate_initial_probability"])
        if self.limit <= 0 or not 0 < initial < 1:
            raise ValueError("Invalid temporal gate settings")
        self.proj = Projection(int(effective["input_dim"]), width)
        self.embedding_dropout = nn.Dropout(float(effective["embedding_dropout"]))
        self.change = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.GELU())
        self.pos = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.q = nn.Parameter(torch.randn(width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, int(effective["n_heads"]), dim_feedforward=2 * width,
            dropout=float(effective["dropout"]), batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, int(effective["n_layers"]))
        self.dropout = nn.Dropout(float(effective["dropout"]))
        self.head, self.gate = nn.Linear(width, 1), nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(initial / (1 - initial)))

    def train(self, mode=True):
        super().train(mode)
        self.t0.eval()
        return self

    def forward(self, embeddings, mask, clinical, days=None, prior_logit=None,
                return_residual=False, max_depth=None, return_components=False):
        depth = self.depth if max_depth is None else int(max_depth)
        if not 1 <= depth <= self.depth:
            raise ValueError("Requested prefix exceeds the trained window")
        embeddings, mask, days = masked_window(embeddings, mask, days, depth, self.effective)
        with torch.no_grad():
            base = self.t0(embeddings, mask, clinical, days, prior_logit)
        z = self.embedding_dropout(self.proj(embeddings))
        present = mask > 0
        z = torch.where(present[..., None], z, 0)
        delta = torch.where(present[..., None], z - z[:, :1], 0)
        tokens = self.change(torch.cat([z, delta], dim=-1))
        order = torch.arange(mask.shape[1], device=mask.device)[None, :].expand_as(mask) / 3
        elapsed = days.clamp_min(0)
        temporal = torch.stack([(elapsed / 180).clamp(0, 2), torch.log1p(elapsed) / 6, order], dim=-1)
        tokens = tokens + self.pos(temporal)
        query = self.q[None, None, :].expand(len(mask), 1, -1)
        padding = torch.cat([torch.zeros_like(present[:, :1]), ~present], dim=1)
        hidden = self.dropout(self.tr(torch.cat([query, tokens], dim=1), src_key_padding_mask=padding)[:, 0])
        has_followup = present[:, 0] & present[:, 1:].any(dim=1)
        gate = torch.sigmoid(self.gate(hidden).squeeze(-1)) * has_followup.to(hidden.dtype)
        correction = self.limit * gate * torch.tanh(self.head(hidden).squeeze(-1))
        logits = base + correction
        if return_components:
            return logits, correction, gate, base
        return (logits, correction) if return_residual else logits


def build_model(effective, depth):
    if effective.get("architecture", "tdn") == "tdn":
        return FeatureTDN(effective, depth)
    if effective["architecture"] == "gated":
        return GatedTemporal(effective, depth)
    raise ValueError("Unsupported v4 architecture")


def drop_followups(data, probability, generator):
    if not 0 <= probability <= 1:
        raise ValueError("Invalid follow-up dropout probability")
    if probability == 0:
        return data
    keep = torch.rand(data["masks"].shape, generator=generator, device=data["masks"].device) >= probability
    keep[:, 0] = True
    return {**data, "masks": data["masks"] * keep.to(data["masks"].dtype)}


@torch.no_grad()
def predict(model, data, depth):
    model.eval()
    rows = {k: [] for k in ("probability", "residual_logit", "temporal_gate", "reference_probability")}
    for start in range(0, len(data["labels"]), 256):
        part = {k: v[start:start + 256] for k, v in data.items()}
        args = (part["embs"], part["masks"], part["clinical"], part["days"], part["prior"])
        if isinstance(model, GatedTemporal):
            logits, residual, gate, reference = model(*args, max_depth=depth, return_components=True)
        else:
            if model.depth != depth:
                raise ValueError("A window-specific TDN cannot serve another prefix")
            logits, residual = model(*args, return_residual=True)
            gate, reference = torch.zeros_like(logits), part["prior"]
        for key, value in zip(rows, (logits.sigmoid(), residual, gate, reference.sigmoid())):
            rows[key].append(value.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in rows.items()}

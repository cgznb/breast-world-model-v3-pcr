"""Same-visit deep masked prediction using the existing observation encoder.

The multi-level objective is a task-specific adaptation of the dense supervision
idea in V-JEPA 2.1, not an implementation or checkpoint of its video model.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .legacy.encoder import MaskedStatePredictor, corrupt_latent
from .legacy.layers import position_3d


class DeepJEPAPredictor(MaskedStatePredictor):
    """Share the original masked predictor across fused and stage features."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.level_embedding = nn.Parameter(torch.randn(len(cfg.widths), 1, cfg.dim) * .02)
        self.level_outputs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 2 * cfg.dim),
                          nn.GELU(), nn.Linear(2 * cfg.dim, cfg.dim))
            for _ in cfg.widths
        ])

    def predict_level(self, tokens, level, grid=None):
        grid = self.cfg.token_grid if grid is None else grid
        q = self.mask_token.expand(len(tokens), math.prod(grid), -1)
        pos = position_3d(grid, self.cfg.dim, q.device, q.dtype)
        embedding = self.level_embedding[level][None]
        predicted = self.predictor(q + pos + embedding, tokens + pos + embedding)
        return self.level_outputs[level](predicted)

    def forward(self, context, pyramid=None):
        fused = super().forward(context)
        if pyramid is None:
            return fused
        return fused, tuple(self.predict_level(tokens, i, pyramid.grid)
                            for i, tokens in enumerate(pyramid.tokens))


def masked_token_loss(prediction, target, mask, masked_weight=1.0, visible_weight=0.1):
    """Average within each patient and region before weighting masked/visible."""
    error = (prediction.float() - target.detach().float()).square().mean(-1)
    mask = mask.to(dtype=torch.bool)
    masked = (error * mask).sum(1) / mask.sum(1).clamp_min(1)
    visible = (error * ~mask).sum(1) / (~mask).sum(1).clamp_min(1)
    return (masked_weight * masked + visible_weight * visible).mean()


def deep_jepa_loss(online_encoder, teacher_encoder, predictor, z, *, mix=0.5,
                   level_weights=(0.2, 0.3, 1.0), masked_weight=1.0, visible_weight=0.1,
                   masked_latent=None, mask=None):
    """Mask in latent space before convolutions and predict the same real visit.

    The caller owns EMA updates and all non-JEPA representation objectives.
    Frozen-teacher re-encoding of generated images elsewhere must stay outside
    this function: only these same-visit teacher targets use no_grad.
    """
    if not 0 <= mix <= 1 or min(masked_weight, visible_weight) < 0:
        raise ValueError("Invalid deep JEPA mixing or region weight")
    if (masked_latent is None) != (mask is None):
        raise ValueError("Supply both masked_latent and its canonical mask")
    if masked_latent is None:
        masked_latent, mask = corrupt_latent(z, online_encoder.cfg)
    online, online_pyramid = online_encoder(masked_latent, return_pyramid=True)
    with torch.no_grad():
        target, target_pyramid = teacher_encoder(z, return_pyramid=True)
    fused_prediction, level_predictions = predictor(online, online_pyramid)
    weights = z.new_tensor(level_weights, dtype=torch.float32)
    if len(weights) != len(level_predictions) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
        raise ValueError("One nonnegative deep JEPA weight per encoder level is required")
    weights = weights / weights.sum()
    fused = masked_token_loss(fused_prediction, target.dense, mask, masked_weight, visible_weight)
    levels = tuple(masked_token_loss(prediction, truth, mask, masked_weight, visible_weight)
                   for prediction, truth in zip(level_predictions, target_pyramid.tokens))
    deep = (torch.stack(levels) * weights).sum()
    loss = (1 - mix) * fused + mix * deep
    details = {"jepa_fused": fused, "jepa_deep": deep,
               **{f"jepa_level{i}": value for i, value in enumerate(levels)}}
    return loss, details

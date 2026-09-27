"""Spatial pretraining heads, never a future-state predictor or prognosis head."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .observation_encoder import TokenStack, sine_position


class ObservationPredictor(nn.Module):
    """Predict same-visit spatial tokens given relative coordinates and nuisance action."""
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.predictor_dim
        self.input = nn.Linear(cfg.dim, self.dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.dim) * .02)
        self.policy = nn.Sequential(nn.Linear(self.dim+4, self.dim), nn.GELU(),
                                    nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim))
        self.stack = TokenStack(self.dim, cfg.predictor_heads, cfg.predictor_depth, cfg.checkpoint_blocks)
        self.output = nn.Linear(self.dim, cfg.dim)

    def forward(self, context, query_positions, action, query_padding=None):
        b, nq, _ = query_positions.shape
        if action.shape != (b, 4):
            raise ValueError("Observation action is [B,4]: gain offset/noise/blur, NOT clinical treatment")
        if query_padding is None:
            query_padding = torch.zeros(b, nq, dtype=torch.bool, device=query_positions.device)
        src = self.input(context.tokens) + sine_position(context.positions, self.dim).to(context.tokens.dtype)
        query = self.mask_token + sine_position(query_positions, self.dim).to(src.dtype)
        query = self.policy(torch.cat((query, action[:, None].expand(-1, nq, -1).to(query.dtype)), -1))
        both = torch.cat((src, query), 1)
        padding = torch.cat((context.padding, query_padding), 1)
        return self.output(self.stack(both, padding)[:, src.shape[1]:])


class SpatialReadout(nn.Module):
    """Deep Transformer readout with patch unprojection; no raw-input skip."""
    def __init__(self, cfg, channels):
        super().__init__()
        self.patch, self.channels = tuple(cfg.patch), channels
        self.stack = TokenStack(cfg.dim, cfg.heads, cfg.readout_depth, cfg.checkpoint_blocks)
        self.output = nn.Linear(cfg.dim, channels*math.prod(self.patch))

    def forward(self, features):
        tokens = self.stack(features.tokens, features.padding)
        patchvalues = self.output(tokens).masked_fill(features.padding[..., None], 0.)
        b, _, c = patchvalues.shape
        full = patchvalues.new_zeros(b, math.prod(features.grid), c)
        full.scatter_add_(1, features.indices.clamp_min(0)[..., None].expand(-1, -1, c), patchvalues)
        gd, gh, gw = features.grid
        pd, ph, pw = self.patch
        full = full.reshape(b, gd, gh, gw, self.channels, pd, ph, pw)
        return full.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, self.channels, gd*pd, gh*ph, gw*pw)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class ArcFaceHead(nn.Module):
    def __init__(self, dim, classes, margin=.3, scale=16.):
        super().__init__()
        self.projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.weight = nn.Parameter(torch.empty(classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, pooled, labels):
        cosine = F.linear(F.normalize(self.projector(pooled).float(), dim=-1), F.normalize(self.weight.float(), dim=-1))
        cosine = cosine.clamp(-1+1e-6, 1-1e-6)
        sine = (1-cosine.square()).clamp_min(0).sqrt()
        phi = cosine*math.cos(self.margin) - sine*math.sin(self.margin)
        phi = torch.where(cosine > math.cos(math.pi-self.margin), phi, cosine-math.sin(math.pi-self.margin)*self.margin)
        one_hot = F.one_hot(labels, cosine.shape[1]).to(cosine)
        return self.scale * (cosine*(1-one_hot) + phi*one_hot)


class DomainHead(nn.Module):
    """Token-wise DANN on a separate projection; this differs from AlphaCell's exact head."""
    def __init__(self, dim, classes):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.classifier = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, classes))

    def forward(self, tokens, reversal=1.):
        return self.classifier(GradientReverse.apply(self.project(tokens), reversal))

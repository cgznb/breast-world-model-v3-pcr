"""Preserves PhaseStateEncoder/StateHeads parameter names from V2.
Only imports and formatting adapted. See docs/SOURCES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .layers import ConvNeXt3D, SwinBlock3D, PatchMerging3D, QueryPool, TokenPredictor, position_3d, maybe_checkpoint

@dataclass
class PatientState:
    dense: torch.Tensor
    anatomy: torch.Tensor
    disease: torch.Tensor
    grid: tuple
    @property
    def summary(self):
        return torch.cat((self.anatomy.mean(1), self.disease.mean(1)), -1)

@dataclass(frozen=True)
class EncoderPyramid:
    """Pre-pool stage features and projected tokens on the canonical grid.

    Pooling a singleton depth to two repeats coordinates; it adds no resolution.
    Online tensors intentionally retain their computation graphs.
    """
    features: tuple[torch.Tensor, ...]
    tokens: tuple[torch.Tensor, ...]
    grid: tuple

class PhaseMixer(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        self.phase_embedding = nn.Parameter(torch.randn(1, 3, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
    def forward(self, x):
        b, p, c, d, h, w = x.shape
        seq = x.permute(0, 3, 4, 5, 1, 2).reshape(-1, p, c)
        seq = torch.cat([self.layers(part + self.phase_embedding)
                         for part in seq.split(32768, dim=0)], dim=0)
        return seq.reshape(b, d, h, w, p*c).movedim(-1, 1)

class PhaseStateEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, s = cfg.phase_channels, cfg.stem_width
        self.register_buffer("latent_mean", torch.zeros(1, 24, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, 24, 1, 1, 1))
        self.stem = nn.Sequential(nn.Conv3d(c, s, 3, stride=2, padding=1), ConvNeXt3D(s), ConvNeXt3D(s))
        self.phase_mixer = PhaseMixer(s, cfg.query_heads, cfg.phase_mixer_depth)
        self.fusion = nn.Conv3d(s*3, cfg.widths[0], 1)
        self.difference_stem = nn.Sequential(nn.Conv3d(24, cfg.widths[0], 3, stride=2, padding=1),
                                             ConvNeXt3D(cfg.widths[0]))
        self.merges, self.stages, self.projections = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for level, (width, depth, heads) in enumerate(zip(cfg.widths, cfg.depths, cfg.heads)):
            self.merges.append(nn.Identity() if level == 0 else PatchMerging3D(cfg.widths[level-1], width))
            self.stages.append(nn.ModuleList([SwinBlock3D(width, heads, cfg.window, shifted=bool(j%2),
                                                            drop_path=cfg.drop_path) for j in range(depth)]))
            self.projections.append(nn.Conv3d(width, cfg.dim, 1))
        self.multiscale = nn.Sequential(nn.Linear(cfg.dim*len(cfg.widths), cfg.dim*2), nn.GELU(),
                                         nn.Linear(cfg.dim*2, cfg.dim), nn.LayerNorm(cfg.dim))
        self.anatomy_pool = QueryPool(cfg.dim, cfg.query_heads, cfg.anatomy_tokens)
        self.disease_pool = QueryPool(cfg.dim, cfg.query_heads, cfg.disease_tokens)
    @torch.no_grad()
    def set_normalization(self, mean, std):
        mean = torch.as_tensor(mean).reshape_as(self.latent_mean)
        std = torch.as_tensor(std).reshape_as(self.latent_std)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid latent normalization")
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)
    def raw_differences(self, z):
        raw = z * self.latent_std + self.latent_mean
        p, e, l = raw.split(8, 1)
        scale = self.latent_std.reshape(1, 3, 8, 1, 1, 1).mean(1).clamp_min(1e-5)
        return torch.cat(((e-p)/scale, (l-p)/scale, (l-e)/scale), 1)
    def forward(self, z, return_pyramid=False):
        if z.ndim != 5 or z.shape[1] != 24:
            raise ValueError("Expected [B,24,D,H,W] standardized continuous VQ latent")
        b, _, d, h, w = z.shape
        phase = maybe_checkpoint(self.stem, z.reshape(b*3, 8, d, h, w), enabled=self.cfg.checkpoint_blocks)
        phase = phase.reshape(b, 3, *phase.shape[1:])
        mixed = maybe_checkpoint(self.phase_mixer, phase, enabled=self.cfg.checkpoint_blocks)
        x = self.fusion(mixed) + self.difference_stem(self.raw_differences(z))
        levels, features = [], []
        for merge, stage, project in zip(self.merges, self.stages, self.projections):
            x = merge(x)
            for block in stage:
                x = maybe_checkpoint(block, x, enabled=self.cfg.checkpoint_blocks)
            if return_pyramid:
                features.append(x)
            levels.append(F.adaptive_avg_pool3d(project(x), self.cfg.token_grid).flatten(2).transpose(1, 2))
        dense = self.multiscale(torch.cat(levels, -1))
        dense = dense + position_3d(self.cfg.token_grid, self.cfg.dim, dense.device, dense.dtype)
        state = PatientState(dense, self.anatomy_pool(dense), self.disease_pool(dense), self.cfg.token_grid)
        if return_pyramid:
            return state, EncoderPyramid(tuple(features), tuple(levels), self.cfg.token_grid)
        return state

class StateHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        def dense_head(out):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, out))
        self.reconstruction = dense_head(24)
        self.latent_delta = dense_head(24)
        self.kinetics = dense_head(3)
        self.segmentation = dense_head(1)
        self.biomarker = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, 4))
        self.external_projection = dense_head(d)
        self.pcr = nn.Sequential(nn.LayerNorm(d*3), nn.Linear(d*3, d*2), nn.GELU(),
                                 nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
    def volume(self, kind, state):
        value = getattr(self, kind)(state.dense).transpose(1, 2)
        return value.reshape(len(value), -1, *state.grid)

class MaskedStatePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.dim) * .02)
        self.predictor = TokenPredictor(cfg.dim, cfg.query_heads, cfg.predictor_depth)
    def forward(self, context):
        q = self.mask_token.expand(len(context.dense), math.prod(context.grid), -1)
        q = q + position_3d(context.grid, self.cfg.dim, q.device, q.dtype)
        return self.predictor(q, context.dense)

def corrupt_latent(z, cfg):
    b, n = len(z), math.prod(cfg.token_grid)
    count = max(1, min(n-1, round(n*cfg.mask_ratio))) if n > 1 else 1
    mask = torch.zeros((b, n), device=z.device, dtype=torch.bool)
    for row in mask:
        row[torch.randperm(n, device=z.device)[:count]] = True
    high = F.interpolate(mask.reshape(b, 1, *cfg.token_grid).float(), size=z.shape[2:], mode="nearest").bool()
    out = z.masked_fill(high, 0)
    if cfg.phase_drop_probability:
        out = out.clone().reshape(b, 3, 8, *z.shape[2:])
        for i in range(b):
            if torch.rand((), device=z.device) < cfg.phase_drop_probability:
                out[i, int(torch.randint(3, (), device=z.device))] = 0
        out = out.reshape_as(z)
    return out, mask

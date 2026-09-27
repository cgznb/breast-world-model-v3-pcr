"""3D, phase-aware ViT with strict visible-token encoding.

Architecture adaptation of the CheXWorld/JEPA design, not copied pretrained
weights. Patch embedding has kernel=stride: no new spatial mixing occurs before
hidden locations have been removed. The original frozen VQ receptive fields
can nevertheless overlap; this is latent-space, not raw-voxel independent SSL.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def positions_3d(grid, device):
    return torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32)
                                       for n in grid), indexing="ij"), -1).reshape(-1, 3)


def sine_position(coords, dim):
    """Sin/cos encoding for real-valued 3D patch coordinates, including negatives."""
    count = math.ceil(dim / 6)
    frequencies = torch.exp(-math.log(10000.) * torch.arange(count, device=coords.device).float() / max(1, count))
    phase = coords.float()[..., :, None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)[..., :dim]


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x, padding):
        q = self.norm1(x)
        x = x + self.attn(q, q, q, key_padding_mask=padding, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x.masked_fill(padding[..., None], 0.)


class TokenStack(nn.Module):
    def __init__(self, dim, heads, depth, checkpoint_blocks=False):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, tokens, padding, last_k=1):
        results = []
        for i, block in enumerate(self.blocks):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, padding, use_reentrant=False)
            else:
                tokens = block(tokens, padding)
            if i >= len(self.blocks) - last_k:
                results.append(tokens)
        # A teacher can average its last K transformer blocks before normalizing.
        return self.norm(torch.stack(results).mean(0)).masked_fill(padding[..., None], 0.)


@dataclass
class ObservationFeatures:
    tokens: torch.Tensor
    padding: torch.Tensor
    positions: torch.Tensor
    indices: torch.Tensor
    grid: tuple[int, int, int]

    def pooled(self):
        keep = (~self.padding).to(self.tokens.dtype)
        return (self.tokens * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp_min(1.)

    def dense(self):
        b, _, c = self.tokens.shape
        out = self.tokens.new_zeros(b, math.prod(self.grid), c)
        out.scatter_add_(1, self.indices.clamp_min(0)[..., None].expand(-1, -1, c),
                         self.tokens.masked_fill(self.padding[..., None], 0.))
        return out


def pack_tokens(tokens, positions, keep):
    """Ragged visible sequences with key-padding masks; never encode hidden tokens."""
    lengths = keep.sum(1)
    if (lengths < 1).any():
        raise ValueError("Every observation needs at least one visible, valid patch")
    nmax = int(lengths.max())
    packed, coords, indexes, pads = [], [], [], []
    for row, mask in enumerate(keep):
        ids = mask.nonzero(as_tuple=False).flatten()
        n = len(ids)
        packed.append(F.pad(tokens[row, ids], (0, 0, 0, nmax-n)))
        coords.append(F.pad(positions[ids], (0, 0, 0, nmax-n)))
        indexes.append(F.pad(ids, (0, nmax-n), value=-1))
        pads.append(torch.arange(nmax, device=tokens.device) >= n)
    return torch.stack(packed), torch.stack(coords), torch.stack(indexes), torch.stack(pads)


class ThreePhaseObservationEncoder(nn.Module):
    def __init__(self, cfg, statistics=None):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv3d(8, cfg.phase_dim, kernel_size=cfg.patch, stride=cfg.patch)
        self.phase_embedding = nn.Parameter(torch.randn(1, 3, cfg.phase_dim) * .02)
        self.phase_mixer = TokenStack(cfg.phase_dim, cfg.phase_heads, cfg.phase_depth, cfg.checkpoint_blocks)
        self.joint_project = nn.Sequential(nn.LayerNorm(3*cfg.phase_dim), nn.Linear(3*cfg.phase_dim, cfg.dim))
        self.spatial = TokenStack(cfg.dim, cfg.heads, cfg.depth, cfg.checkpoint_blocks)
        self.register_buffer("latent_mean", torch.zeros(1, 24, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, 24, 1, 1, 1))
        if statistics is not None:
            self.set_statistics(statistics)

    @torch.no_grad()
    def set_statistics(self, statistics):
        mean = torch.as_tensor(statistics["mean"], dtype=torch.float32)
        std = torch.as_tensor(statistics["std"], dtype=torch.float32)
        if mean.shape != (24,) or std.shape != (24,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Expected 24 finite train-fitted means and positive std values")
        if statistics.get("fit_split") != "train":
            raise ValueError("Latent normalization must be fitted on train only")
        self.latent_mean.copy_(mean.reshape_as(self.latent_mean))
        self.latent_std.copy_(std.reshape_as(self.latent_std))

    def normalize(self, raw):
        return (raw - self.latent_mean) / self.latent_std

    def denormalize(self, normalized):
        return normalized * self.latent_std + self.latent_mean

    def patch_validity(self, valid, spatial_shape):
        if valid.ndim != 5 or valid.shape[1] not in (1, 3) or tuple(valid.shape[2:]) != tuple(spatial_shape):
            raise ValueError("valid must be [B,1|3,D,H,W] on the latent lattice")
        intersection = valid.float().amin(1, keepdim=True)
        return F.avg_pool3d(intersection, self.cfg.patch, self.cfg.patch).flatten(1)

    def forward(self, latent, valid=None, visible=None, *, normalized=False,
                minimum_coverage=.95, teacher_average=False):
        if latent.ndim != 5 or latent.shape[1] != 24 or any(n % p for n, p in zip(latent.shape[2:], self.cfg.patch)):
            raise ValueError("Expected [B,24,D,H,W], spatial dimensions divisible by encoder.patch")
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite observation latent")
        b, _, d, h, w = latent.shape
        grid = tuple(n // p for n, p in zip((d, h, w), self.cfg.patch))
        n = math.prod(grid)
        if n > self.cfg.max_tokens:
            raise ValueError(f"{n} patches exceeds max_tokens={self.cfg.max_tokens}; use audited crop/patch config")
        if valid is None:
            valid = torch.ones(b, 1, d, h, w, device=latent.device)
        keep = self.patch_validity(valid, (d, h, w)) >= minimum_coverage
        if visible is not None:
            if visible.shape != keep.shape or visible.dtype != torch.bool:
                raise ValueError("visible must be a [B,N] boolean mask on the patch grid")
            keep = keep & visible
        z = latent if normalized else self.normalize(latent)
        phase = self.patch_embed(z.reshape(b*3, 8, d, h, w))
        phase = phase.flatten(2).transpose(1, 2).reshape(b, 3, n, self.cfg.phase_dim)
        phase = phase.permute(0, 2, 1, 3).reshape(b, n, 3*self.cfg.phase_dim)
        # Remove all three phases of every hidden spatial token BEFORE phase/spatial attention.
        coords = positions_3d(grid, latent.device)
        phase, pos, ids, padding = pack_tokens(phase, coords, keep)
        k = phase.shape[1]
        phase = phase.reshape(b*k, 3, self.cfg.phase_dim) + self.phase_embedding
        phase_padding = torch.zeros(b*k, 3, dtype=torch.bool, device=latent.device)
        # Each row is an independent spatial location. Bound the CUDA attention
        # launch grid at large patient batches without changing its attention.
        phase = torch.cat([self.phase_mixer(x, p) for x, p in
                           zip(phase.split(8192), phase_padding.split(8192))], dim=0)
        tokens = self.joint_project(phase.reshape(b, k, -1))
        tokens = tokens + sine_position(pos, self.cfg.dim).to(tokens.dtype)
        tokens = self.spatial(tokens, padding, last_k=self.cfg.teacher_last_k if teacher_average else 1)
        return ObservationFeatures(tokens, padding, pos, ids, grid)

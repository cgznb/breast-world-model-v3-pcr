"""Keep the MONAI 3D diffusion U-Net option; ship a full torch-only alternative.

The native alternative uses ADM-style residual/FiLM conditioning, multi-scale
skip connections, 3D shifted-window self attention and cross attention to the
clinical/patient tokens. It is not represented as pretrained MONAI weights.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .layers import TimeEmbedding, SwinBlock3D, maybe_checkpoint
from .utils import group_count


class FiLMResBlock3D(nn.Module):
    def __init__(self, cin, cout, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(cin, min(32, cin//2)), cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(cout, min(32, cout//2)), cout)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2*cout))
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(c).to(h.dtype).chunk(2, -1)
        h = self.norm2(h)*(1+scale[..., None, None, None]) + shift[..., None, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels, context_dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context = nn.Linear(context_dim, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0)
        self.output = nn.Linear(channels, channels)
    def forward(self, x, context):
        seq = x.flatten(2).transpose(1, 2)
        c = self.context(context)
        h = self.attention(self.norm(seq), c, c, need_weights=False)[0]
        return (seq + self.output(h)).transpose(1, 2).reshape_as(x)


class NativeVelocityUNet(nn.Module):
    def __init__(self, cfg, context_dim):
        super().__init__()
        self.cfg = cfg
        widths = cfg.channels
        td = widths[0]*4
        self.time = TimeEmbedding(td)
        self.pooled_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, td))
        self.input = nn.Conv3d(48, widths[0], 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        previous = widths[0]
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(FiLMResBlock3D(previous, width, td))
                previous = width
            self.down.append(blocks)
            if level < len(widths)-1:
                self.downsample.append(nn.Conv3d(width, width, 3, stride=2, padding=1))
        self.mid1 = FiLMResBlock3D(widths[-1], widths[-1], td)
        self.mid_self = nn.Sequential(SwinBlock3D(widths[-1], cfg.attention_heads, (2,4,4)),
                                       SwinBlock3D(widths[-1], cfg.attention_heads, (2,4,4), shifted=True))
        self.mid_cross = SpatialCrossAttention(widths[-1], context_dim, cfg.attention_heads)
        self.mid2 = FiLMResBlock3D(widths[-1], widths[-1], td)
        self.up = nn.ModuleList()
        previous = widths[-1]
        for width in reversed(widths[:-1]):
            blocks = nn.ModuleList([FiLMResBlock3D(previous+width, width, td)])
            blocks.extend(FiLMResBlock3D(width, width, td) for _ in range(cfg.num_res_blocks-1))
            self.up.append(blocks)
            previous = width
        self.out_norm = nn.GroupNorm(group_count(widths[0], min(32, widths[0]//2)), widths[0])
        self.out = nn.Conv3d(widths[0], 48, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, joint, tau, context):
        if joint.ndim != 5 or joint.shape[1] != 48:
            raise ValueError("Joint SymmFlow state must be [B,48,D,H,W]")
        conditioning = self.time(tau) + self.pooled_context(context.mean(1))
        x, skips = self.input(joint), []
        for level, blocks in enumerate(self.down):
            for block in blocks:
                x = maybe_checkpoint(block, x, conditioning, enabled=self.cfg.checkpoint_blocks)
            skips.append(x)
            if level < len(self.downsample):
                x = self.downsample[level](x)
        x = maybe_checkpoint(self.mid1, x, conditioning, enabled=self.cfg.checkpoint_blocks)
        x = maybe_checkpoint(self.mid_self, x, enabled=self.cfg.checkpoint_blocks)
        x = maybe_checkpoint(self.mid_cross, x, context, enabled=self.cfg.checkpoint_blocks)
        x = maybe_checkpoint(self.mid2, x, conditioning, enabled=self.cfg.checkpoint_blocks)
        for blocks, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat((x, skip), 1)
            for block in blocks:
                x = maybe_checkpoint(block, x, conditioning, enabled=self.cfg.checkpoint_blocks)
        return self.out(F.silu(self.out_norm(x)))


class MonaiVelocityUNet(nn.Module):
    def __init__(self, cfg, context_dim):
        super().__init__()
        try:
            from monai.networks.nets import DiffusionModelUNet
        except ImportError as exc:
            raise ImportError("Install monai==1.5.1 or select the explicitly documented native backend. No silent fallback is performed.") from exc
        self.network = DiffusionModelUNet(
            spatial_dims=3, in_channels=48, out_channels=48,
            channels=tuple(cfg.channels), num_res_blocks=cfg.num_res_blocks,
            attention_levels=tuple(cfg.attention_levels),
            num_head_channels=tuple(c//cfg.attention_heads if cfg.attention_levels[i] else 0
                                    for i, c in enumerate(cfg.channels)),
            norm_num_groups=group_count(cfg.channels[0]), with_conditioning=True,
            cross_attention_dim=context_dim, transformer_num_layers=1,
            upcast_attention=True, use_flash_attention=False)
        self.cfg = cfg
    def forward(self, joint, tau, context):
        # Match the original repository's tau convention for this backend.
        if self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda x, t, c: self.network(x=x, timesteps=t, context=c),
                              joint, tau, context, use_reentrant=False)
        return self.network(x=joint, timesteps=tau, context=context)


def build_velocity(cfg, context_dim):
    return (MonaiVelocityUNet if cfg.backend == "monai" else NativeVelocityUNet)(cfg, context_dim)

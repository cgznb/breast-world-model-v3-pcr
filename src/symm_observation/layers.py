"""3D Swin/ConvNeXt/Transformer building blocks.

Shifted-window partitioning, cyclic masks and 3D relative biases follow the
Video Swin implementation in torchvision/models/video/swin_transformer.py
(blob 1a198142874224a6766f321d9e0dfc97a01ecb43, BSD-3-Clause).
This adaptation adds explicit padding-key masks and true 3D patch merging.
No torchvision dependency or pretrained-weight compatibility is implied.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .utils import group_count


def maybe_checkpoint(module, *args, enabled=False):
    if enabled and torch.is_grad_enabled() and module.training:
        return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=True)
    return module(*args)


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)
    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((len(x),) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvNeXt3D(nn.Module):
    """Depthwise 3D convolution, channels-last LN, 4x MLP, layer scale."""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.depthwise = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, 4 * dim)
        self.project = nn.Linear(4 * dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 1e-6))
        self.drop = DropPath(drop_path)
    def forward(self, x):
        h = self.depthwise(x).movedim(1, -1)
        h = self.project(F.gelu(self.expand(self.norm(h)))) * self.scale
        return x + self.drop(h.movedim(-1, 1))


def partition_windows(x: Tensor, window: tuple[int, int, int]) -> Tensor:
    b, d, h, w, c = x.shape
    wd, wh, ww = window
    return x.reshape(b, d // wd, wd, h // wh, wh, w // ww, ww, c).permute(
        0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wd * wh * ww, c)


def reverse_windows(x, window, padded_shape, batch):
    wd, wh, ww = window
    d, h, w = padded_shape
    return x.reshape(batch, d // wd, h // wh, w // ww, wd, wh, ww, -1).permute(
        0, 1, 4, 2, 5, 3, 6, 7).reshape(batch, d, h, w, -1)


def _window_mask(shape, window, shift, device):
    """Prevent attention across the artificial cyclic wrap boundary."""
    labels = torch.zeros((1, *shape, 1), device=device)
    slices = []
    for size, win, sh in zip(shape, window, shift):
        slices.append((slice(0, -win), slice(-win, -sh), slice(-sh, None))
                      if sh else (slice(0, size),))
    count = 0
    for sd in slices[0]:
        for sh in slices[1]:
            for sw in slices[2]:
                labels[:, sd, sh, sw] = count
                count += 1
    windows = partition_windows(labels, window).squeeze(-1)
    return windows.unsqueeze(1) != windows.unsqueeze(2)


class ShiftedWindowAttention3D(nn.Module):
    def __init__(self, dim, heads, window=(2, 4, 4), shifted=False):
        super().__init__()
        if dim % heads:
            raise ValueError("Swin dimension must be divisible by heads")
        self.heads, self.window, self.shifted = heads, tuple(window), shifted
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        count = math.prod(2 * n - 1 for n in self.window)
        self.relative_bias = nn.Parameter(torch.zeros(count, heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def _relative_bias(self, actual, device):
        coords = torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in actual], indexing="ij"))
        coords = coords.flatten(1)
        diff = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        diff = diff + diff.new_tensor([n - 1 for n in self.window])
        index = diff[..., 0] * ((2*self.window[1]-1)*(2*self.window[2]-1))
        index = index + diff[..., 1] * (2*self.window[2]-1) + diff[..., 2]
        return self.relative_bias[index.long()].permute(2, 0, 1).unsqueeze(0)

    def forward(self, value):
        b, d, h, w, c = value.shape
        actual = tuple(min(n, win) for n, win in zip((d, h, w), self.window))
        shift = tuple(win//2 if self.shifted and size > win else 0
                      for size, win in zip((d, h, w), actual))
        pads = tuple((-size) % win for size, win in zip((d, h, w), actual))
        x = F.pad(value, (0, 0, 0, pads[2], 0, pads[1], 0, pads[0]))
        shape = tuple(x.shape[1:4])
        valid = torch.zeros((1, *shape, 1), dtype=torch.bool, device=value.device)
        valid[:, :d, :h, :w] = True
        if any(shift):
            x = torch.roll(x, tuple(-s for s in shift), (1, 2, 3))
            valid = torch.roll(valid, tuple(-s for s in shift), (1, 2, 3))
        windows = partition_windows(x, actual)
        tokens = windows.shape[1]
        nw = len(windows) // b
        qkv = self.qkv(windows).reshape(-1, tokens, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        invalid_keys = ~partition_windows(valid, actual).squeeze(-1)
        mask = invalid_keys[:, None, :].expand(-1, tokens, -1).clone()
        if any(shift):
            mask |= _window_mask(shape, actual, shift, value.device)
        # Finite sentinel also makes entirely padded query rows well-defined.
        bias = self._relative_bias(actual, value.device).to(q.dtype)
        bias = bias + mask.repeat(b, 1, 1)[:, None].to(q.dtype) * -10000.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        out = self.proj(out.transpose(1, 2).reshape(-1, tokens, c))
        out = reverse_windows(out, actual, shape, b)
        if any(shift):
            out = torch.roll(out, shift, (1, 2, 3))
        return out[:, :d, :h, :w].contiguous()


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, window, shifted=False, drop_path=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = ShiftedWindowAttention3D(dim, heads, window, shifted)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.drop = DropPath(drop_path)
    def forward(self, x):
        x = x.movedim(1, -1)
        x = x + self.drop(self.attention(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x.movedim(-1, 1).contiguous()


class PatchMerging3D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * input_dim)
        self.reduction = nn.Linear(8 * input_dim, output_dim, bias=False)
    def forward(self, x):
        _, _, d, h, w = x.shape
        x = F.pad(x, (0, w % 2, 0, h % 2, 0, d % 2))
        chunks = [x[:, :, zd::2, yh::2, xw::2] for zd in (0, 1) for yh in (0, 1) for xw in (0, 1)]
        x = torch.cat(chunks, 1).movedim(1, -1)
        return self.reduction(self.norm(x)).movedim(-1, 1).contiguous()


def position_3d(grid, dim, device, dtype):
    """Fixed physical-axis-aware Fourier coordinates, padded to any dimension."""
    axes = [torch.linspace(-1, 1, n, device=device) for n in grid]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    bands = max(1, math.ceil(dim / 6))
    frequency = 2.0 ** torch.arange(bands, device=device).float()
    pos = coords[..., None] * frequency * math.pi
    pos = torch.cat((pos.sin(), pos.cos()), -1).flatten(1)[:, :dim]
    return pos.to(dtype)[None]


class QueryPool(nn.Module):
    def __init__(self, dim, heads, count):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, count, dim))
        nn.init.trunc_normal_(self.queries, std=.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, tokens):
        q = self.queries.expand(len(tokens), -1, -1)
        h = self.attention(q, tokens, tokens, need_weights=False)[0] + q
        return self.norm(h + self.ffn(h))


class TokenPredictor(nn.Module):
    """Multi-layer pre-norm transformer decoder; not a linear latent mapper."""
    def __init__(self, dim, heads, depth):
        super().__init__()
        layer = nn.TransformerDecoderLayer(dim, heads, dim*4, dropout=0.0,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, dim)
    def forward(self, query, memory):
        return self.output(self.decoder(query, memory))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    def forward(self, time):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=time.device).float() / max(half-1, 1))
        arg = time.float()[:, None] * 1000 * f[None]
        emb = torch.cat((arg.cos(), arg.sin()), -1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim-emb.shape[-1]))
        return self.mlp(emb)

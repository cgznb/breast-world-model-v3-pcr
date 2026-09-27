"""Checkpoint-compatible inference subset of the user's single-channel VQ codec.

Adapted from cgznb/symm-fm at 92265d3b1749ae3b686f2089843c49da129fd4d2,
workflows/first_post_three_phase/mewm_ispy2/vqgan.py (MeWM-derived, CC BY-NC 4.0).
Changes: no Lightning/discriminator dependency, frozen codebook, explicit
straight-through decode for input gradients, safe checkpoint loading.
See licenses/CODEC_NOTICE.md. External pretrained weights are not distributed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class VQConfig:
    image_channels: int = 1
    hidden_channels: int = 16
    embedding_dim: int = 8
    n_codes: int = 16384
    downsample_factor: int = 4
    bottleneck_blocks: int = 0
    num_groups: int = 32
    commitment_weight: float = .25
    ema_decay: float = .99
    ema_smoothing: float = 1e-7
    restart_threshold: float = 1.0
    nearest_chunk_size: int = 4096
    def __post_init__(self):
        if self.image_channels != 1 or self.embedding_dim != 8 or self.downsample_factor != 4:
            raise ValueError("Expected the original single-channel, 8-dimensional, 4x codec")
        if self.hidden_channels*2 % self.num_groups or self.bottleneck_blocks < 0:
            raise ValueError("Invalid VQ architecture")


def triple(x):
    return (x,x,x) if isinstance(x, int) else tuple(x)


def same_padding(kernel, stride):
    pads = [(k-s)//2 for k,s in zip(kernel,stride)]
    full = [(k-s)-p for k,s,p in zip(kernel,stride,pads)]
    return (full[2],pads[2],full[1],pads[1],full[0],pads[0])


class SamePadConv3d(nn.Conv3d):
    def __init__(self, cin, cout, kernel, stride=1):
        kernel, stride = triple(kernel), triple(stride)
        super().__init__(cin, cout, kernel, stride=stride, padding=0)
        self.pad_input = same_padding(kernel,stride)
    def forward(self, x):
        return super().forward(F.pad(x,self.pad_input,mode="replicate") if any(self.pad_input) else x)


class SamePadConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, cin, cout, kernel, stride=1):
        kernel, stride = triple(kernel), triple(stride)
        super().__init__(cin,cout,kernel,stride=stride,padding=tuple(k-1 for k in kernel))
        self.pad_input = same_padding(kernel,stride)
    def forward(self,x,output_size=None):
        x = F.pad(x,self.pad_input,mode="replicate") if any(self.pad_input) else x
        return super().forward(x,output_size=output_size)


class ResidualBlock3D(nn.Module):
    def __init__(self,c,groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups,c,eps=1e-6)
        self.conv1 = SamePadConv3d(c,c,3)
        self.norm2 = nn.GroupNorm(groups,c,eps=1e-6)
        self.conv2 = SamePadConv3d(c,c,3)
    def forward(self,x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x+self.conv2(F.silu(self.norm2(h)))


def bottleneck(count,c,groups):
    blocks = nn.ModuleList([ResidualBlock3D(c,groups) for _ in range(count)])
    for block in blocks:
        nn.init.zeros_(block.conv2.weight); nn.init.zeros_(block.conv2.bias)
    return blocks


class MRIEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        h,g = cfg.hidden_channels,cfg.num_groups
        self.input = SamePadConv3d(1,h,3)
        self.downsamples = nn.ModuleList([SamePadConv3d(h,h*2,4,2),SamePadConv3d(h*2,h*4,4,2)])
        self.residuals = nn.ModuleList([ResidualBlock3D(h*2,g),ResidualBlock3D(h*4,g)])
        self.bottleneck = bottleneck(cfg.bottleneck_blocks,h*4,g)
        self.final_norm = nn.GroupNorm(g,h*4,eps=1e-6)
        self.output_channels = h*4
    def forward(self,x):
        x = self.input(x)
        for down,res in zip(self.downsamples,self.residuals):
            x = res(down(x))
        for block in self.bottleneck:
            x = block(x)
        return F.silu(self.final_norm(x))


class MRIDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        h,g = cfg.hidden_channels,cfg.num_groups
        self.final_norm = nn.GroupNorm(g,h*4,eps=1e-6)
        self.bottleneck = bottleneck(cfg.bottleneck_blocks,h*4,g)
        self.upsamples = nn.ModuleList([SamePadConvTranspose3d(h*4,h*4,4,2),SamePadConvTranspose3d(h*4,h*2,4,2)])
        self.residual_one = nn.ModuleList([ResidualBlock3D(h*4,g),ResidualBlock3D(h*2,g)])
        self.residual_two = nn.ModuleList([ResidualBlock3D(h*4,g),ResidualBlock3D(h*2,g)])
        self.output = SamePadConv3d(h*2,1,3)
    def forward(self,x):
        x = F.silu(self.final_norm(x))
        for block in self.bottleneck:
            x = block(x)
        for up,a,b in zip(self.upsamples,self.residual_one,self.residual_two):
            x = b(a(up(x)))
        return self.output(x)


class FrozenQuantizer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        emb = F.normalize(torch.randn(cfg.n_codes,cfg.embedding_dim),dim=1)
        self.register_buffer("embeddings",emb)
        self.register_buffer("cluster_size",torch.ones(cfg.n_codes))
        self.register_buffer("embedding_sum",emb.clone())
        self.embedding_dim = cfg.embedding_dim
        self.chunk_size = cfg.nearest_chunk_size
    def forward(self,z):
        # Exact nearest-code forward; identity straight-through derivative.
        # No EMA codebook mutation is allowed in either train() or eval().
        flat = z.movedim(1,-1).reshape(-1,self.embedding_dim)
        with torch.no_grad():
            norms = self.embeddings.float().square().sum(1)
            indices = []
            for x in flat.detach().float().split(self.chunk_size):
                distances = x.square().sum(1,keepdim=True)-2*x@self.embeddings.float().T+norms[None]
                indices.append(distances.argmin(1))
            idx = torch.cat(indices)
            q = F.embedding(idx,self.embeddings).reshape(z.shape[0],*z.shape[2:],-1).movedim(-1,1)
        return z+(q.to(z)-z).detach()


class MRILevelVQGAN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.config = cfg
        self.encoder = MRIEncoder(cfg)
        self.pre_quant = SamePadConv3d(self.encoder.output_channels,cfg.embedding_dim,1)
        self.quantizer = FrozenQuantizer(cfg)
        self.post_quant = SamePadConv3d(cfg.embedding_dim,self.encoder.output_channels,1)
        self.decoder = MRIDecoder(cfg)
    def encode_continuous(self,x):
        return self.pre_quant(self.encoder(x))
    def decode(self,z):
        return self.decoder(self.post_quant(z))


class FrozenThreePhaseCodec(nn.Module):
    def __init__(self,codec):
        super().__init__()
        self.codec = codec.eval().requires_grad_(False)
    def train(self,mode=True):
        super().train(False)
        return self
    @torch.no_grad()
    def encode(self,images):
        if images.ndim != 5 or images.shape[1] != 3 or any(n%4 for n in images.shape[2:]):
            raise ValueError("Codec images must be [B,3,D,H,W] with spatial dimensions divisible by 4")
        return torch.cat([self.codec.encode_continuous(images[:,i:i+1]) for i in range(3)],1)
    def decode(self,raw_latent):
        if raw_latent.ndim != 5 or raw_latent.shape[1] != 24:
            raise ValueError("Codec decode expects unstandardized [B,24,D,H,W] latent")
        # Do not wrap this function in torch.no_grad: prediction gradients are
        # required for decoded-kinetics supervision. Parameters remain frozen.
        with torch.autocast(raw_latent.device.type,enabled=False):
            return torch.cat([self.codec.decode(self.codec.quantizer(z.float()))
                              for z in raw_latent.split(8,1)],1)


def load_codec(path,device="cpu",trusted_legacy=False):
    try:
        payload = torch.load(path,map_location="cpu",weights_only=True)
    except Exception:
        if not trusted_legacy:
            raise ValueError("Checkpoint is not weights-only readable. Convert your own trusted old checkpoint with --trusted-legacy; never enable this for unknown downloads.") from None
        payload = torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema") in {"first_post_unregistered_tumor_roi_v1","symm_world_codec_v2"}:
        cfg,state = payload["model_config"],payload["codec_state"]
    else:
        marker = payload.get("mewm_ispy2_vqgan_identity",{})
        if (marker.get("numeric_contract") != "ispy2_first_post_unregistered_train_zscore_v1"
                or marker.get("phase_index") != 1 or marker.get("registered") is not False):
            raise ValueError("Not the audited first-post codec format; explicitly convert the matching architecture and numeric contract")
        cfg = marker["architecture_contract"]["config"]
        state = {k.removeprefix("autoencoder."):v for k,v in payload["state_dict"].items() if k.startswith("autoencoder.")}
    codec = MRILevelVQGAN(VQConfig(**cfg))
    codec.load_state_dict(state,strict=True)
    return FrozenThreePhaseCodec(codec).to(device)

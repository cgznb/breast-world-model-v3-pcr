"""Two explicit image backends and the coupled semantic velocity network.

Native retains the V2 image parameter layout. MONAI calls its actual 1.5.1
modules with an explicit bottleneck boundary, rather than mutable forward hooks.
The semantic branch adapts adaLN-Zero conditioning (DiT) to state tokens.
The bidirectional bridge is new task-specific code, not a released pretrained model.
"""
# DiT-derived adaLN blocks: Copyright (c) Meta Platforms, Inc. and affiliates.
# MONAI split traversal: Copyright (c) MONAI Consortium.
# Retained license notices and pinned sources: docs/MULTISTAGE_NETWORK_SOURCES.md.
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .legacy.layers import TimeEmbedding, SwinBlock3D, maybe_checkpoint
from .legacy.utils import group_count

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

class NativeImageBackbone(nn.Module):
    """Full V2 residual/Swin/cross-attention U-Net, not a smoke-only mock."""
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

    def encode(self, joint, tau, context):
        c = self.time(tau) + self.pooled_context(context.mean(1))
        x, skips = self.input(joint), []
        for level, blocks in enumerate(self.down):
            x = self.encode_level(level, x, c)
            skips.append(x)
            if level < len(self.downsample):
                x = self.downsample[level](x)
        x = self.middle(x, c, context)
        return x, (c, skips)

    def encode_level(self, level, x, condition):
        for block in self.down[level]:
            x = block(x, condition)
        return x

    def middle(self, x, condition, context):
        return self.mid2(self.mid_cross(self.mid_self(self.mid1(x, condition)), context), condition)

    def decode(self, x, cache, context):
        c, skips = cache
        for blocks, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat((x, skip), 1)
            for block in blocks:
                x = block(x, c)
        return self.out(F.silu(self.out_norm(x)))

    def forward(self, joint, tau, context):
        h, cache = self.encode(joint, tau, context)
        return self.decode(h, cache, context)

    def tune_decoder(self):
        self.requires_grad_(False)
        for module in (self.up, self.mid1, self.mid2, self.mid_cross, self.mid_self, self.out_norm, self.out):
            module.requires_grad_(True)

class MonaiImageBackbone(nn.Module):
    """Parameter-compatible with V2 MonaiVelocityUNet.network (MONAI 1.5.1).

    The split forward is adapted from MONAI's Apache-2.0 forward implementation.
    Class conditioning and ControlNet residuals are intentionally not exposed.
    Test test_monai_split_parity checks native MONAI output equivalence when installed.
    """
    def __init__(self, cfg, context_dim):
        super().__init__()
        try:
            import monai
            from monai.networks.nets import DiffusionModelUNet
        except ImportError as exc:
            raise ImportError("Install monai==1.5.1 or explicitly choose network.backend=native") from exc
        if monai.__version__ != "1.5.1":
            raise RuntimeError("Pinned MONAI 1.5.1 required; audit split-forward before upgrading")
        self.network = DiffusionModelUNet(
            spatial_dims=3, in_channels=48, out_channels=48, channels=tuple(cfg.channels),
            num_res_blocks=cfg.num_res_blocks,
            attention_levels=tuple(i == len(cfg.channels)-1 for i in range(len(cfg.channels))),
            num_head_channels=tuple(c//cfg.attention_heads if i == len(cfg.channels)-1 else 0
                                    for i, c in enumerate(cfg.channels)),
            norm_num_groups=group_count(cfg.channels[0]), with_conditioning=True,
            cross_attention_dim=context_dim, transformer_num_layers=1,
            upcast_attention=True, use_flash_attention=False)
        self.cfg = cfg

    def encode(self, joint, tau, context):
        from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
        n = self.network
        emb = n.time_embed(get_timestep_embedding(tau, n.block_out_channels[0]).to(joint.dtype))
        h = n.conv_in(joint)
        skips = [h]
        for block in n.down_blocks:
            h, residuals = block(hidden_states=h, temb=emb, context=context)
            skips.extend(residuals)
        h = n.middle_block(hidden_states=h, temb=emb, context=context)
        return h, (emb, skips)

    def decode(self, h, cache, context):
        emb, skips = cache
        for block in self.network.up_blocks:
            idx = -len(block.resnets)
            residuals, skips = skips[idx:], skips[:idx]
            h = block(hidden_states=h, res_hidden_states_list=residuals, temb=emb, context=context)
        return self.network.out(h)

    def forward(self, joint, tau, context):
        h, cache = self.encode(joint, tau, context)
        return self.decode(h, cache, context)

    def tune_decoder(self):
        self.requires_grad_(False)
        for module in (self.network.middle_block, self.network.up_blocks, self.network.out):
            module.requires_grad_(True)

class StateDiTBlock(nn.Module):
    """adaLN-Zero self attention + conditioned cross attention + gated FFN."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.memory_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(approximate="tanh"), nn.Linear(4*dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9*dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    def forward(self, s, condition, memory):
        values = self.modulation(condition).chunk(9, -1)
        a, b, g, ac, bc, gc, af, bf, gf = [v[:, None] for v in values]
        q = self.norm_self(s)*(1+b)+a
        s = s + g*self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norm_cross(s)*(1+bc)+ac
        m = self.memory_norm(memory)
        s = s + gc*self.cross_attn(q, m, m, need_weights=False)[0]
        return s + gf*self.ff(self.norm_ff(s)*(1+bf)+af)

class BidirectionalBridge(nn.Module):
    """Both updates use the same pre-update pair; no hooks or mutable caches."""
    def __init__(self, channels, dim, heads):
        super().__init__()
        self.image_in = nn.Conv3d(channels, dim, 1)
        self.image_out = nn.Conv3d(dim, channels, 1)
        self.image_norm = nn.LayerNorm(dim)
        self.state_norm = nn.LayerNorm(dim)
        self.to_image = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.to_state = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.image_gate = nn.Parameter(torch.zeros(()))
        self.state_gate = nn.Parameter(torch.zeros(()))
    def forward(self, h, s):
        spatial = self.image_in(h)
        z = self.image_norm(spatial.flatten(2).transpose(1, 2))
        sn = self.state_norm(s)
        dz = self.to_image(z, sn, sn, need_weights=False)[0]
        ds = self.to_state(sn, z, z, need_weights=False)[0]
        dh = self.image_out(dz.transpose(1, 2).reshape_as(spatial))
        return h + self.image_gate.tanh()*dh, s + self.state_gate.tanh()*ds

class CoupledVelocity(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, n = cfg.encoder, cfg.network
        self.cfg = cfg
        self.image = (MonaiImageBackbone if n.backend == "monai" else NativeImageBackbone)(n, e.dim)
        self.semantic_in = nn.Linear(e.dim, e.dim)
        self.token_position = nn.Parameter(torch.randn(1, 2*e.disease_tokens, e.dim)*0.02)
        self.time = TimeEmbedding(e.dim)
        self.context_condition = nn.Sequential(nn.LayerNorm(e.dim), nn.Linear(e.dim, e.dim))
        self.blocks = nn.ModuleList([StateDiTBlock(e.dim, e.query_heads) for _ in range(n.semantic_depth)])
        # Multiple residual exchanges at the bottleneck, not just output concatenation.
        self.bridges = nn.ModuleList([BidirectionalBridge(n.channels[-1], e.dim, e.query_heads)
                                      for _ in range(n.semantic_depth//2)])
        self.semantic_out = nn.Sequential(nn.LayerNorm(e.dim), nn.Linear(e.dim, e.dim))
        nn.init.zeros_(self.semantic_out[-1].weight)
        nn.init.zeros_(self.semantic_out[-1].bias)
        self.spatial_projection = nn.Conv3d(n.channels[-1], e.dim, 3, padding=1)

    def _forward(self, z, s, tau, context):
        h, cache = self.image.encode(z, tau, context)
        s = self.semantic_in(s) + self.token_position
        c = self.time(tau) + self.context_condition(context.mean(1))
        for index, block in enumerate(self.blocks):
            s = block(s, c, context)
            if index % 2 == 1 and self.cfg.network.coupling:
                h, s = self.bridges[index//2](h, s)
        vz = self.image.decode(h, cache, context)
        vs = self.semantic_out(s)
        grid = self.cfg.encoder.token_grid
        dense = F.adaptive_avg_pool3d(self.spatial_projection(h), grid).flatten(2).transpose(1, 2)
        return vz, vs, dense

    def forward(self, z, s, tau, context):
        if z.ndim != 5 or z.shape[1] != 48:
            raise ValueError("Expected joint image [B,48,D,H,W]")
        if s.shape != (len(z), 2*self.cfg.encoder.disease_tokens, self.cfg.encoder.dim):
            raise ValueError("Joint semantic shape mismatch")
        # Checkpoint the full coupled evaluation, without forward hooks or mutable state.
        if self.cfg.network.checkpoint_blocks and torch.is_grad_enabled() and (
            self.training or z.requires_grad or s.requires_grad
        ):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward, z, s, tau, context, use_reentrant=False, preserve_rng_state=True)
        return self._forward(z, s, tau, context)


class SemanticTransitionBlock(nn.Module):
    """Four independently modulated residuals: self, history, action and FFN.

    Extends the existing DiT-derived block (CC BY-NC 4.0 attribution retained
    in docs/SOURCES.md); action conditioning is a separate attention operation.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_action = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.action_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.memory_norm = nn.LayerNorm(dim)
        self.action_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, state, condition, memory, actions):
        values = [value[:, None] for value in self.modulation(condition).chunk(12, -1)]
        shift, scale, gate, hs, hc, hg, acs, acc, acg, fs, fc, fg = values
        q = self.norm_self(state) * (1 + scale) + shift
        state = state + gate * self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norm_cross(state) * (1 + hc) + hs
        memory = self.memory_norm(memory)
        state = state + hg * self.cross_attn(q, memory, memory, need_weights=False)[0]
        q = self.norm_action(state) * (1 + acc) + acs
        actions = self.action_norm(actions)
        state = state + acg * self.action_attn(q, actions, actions, need_weights=False)[0]
        return state + fg * self.ff(self.norm_ff(state) * (1 + fc) + fs)


class CoupledVelocityV2(nn.Module):
    """Three genuinely interleaved image/semantic exchanges on one vector field.

    Native image module names and parameter shapes remain exactly those of v1.
    ConditionContext contains only information legal at the interval source.
    """
    def __init__(self, cfg):
        super().__init__()
        e, n = cfg.encoder, cfg.network
        if len(n.channels) != 3 or n.semantic_depth != 6:
            raise ValueError("Multistage velocity requires three image levels and six semantic blocks")
        self.cfg = cfg
        self.image = (MonaiImageBackbone if n.backend == "monai" else NativeImageBackbone)(n, e.dim)
        self.semantic_in = nn.Linear(e.dim, e.dim)
        self.token_position = nn.Parameter(torch.randn(1, 2 * e.disease_tokens, e.dim) * .02)
        self.endpoint_role = nn.Parameter(torch.zeros(1, 2, 1, e.dim))
        self.time = TimeEmbedding(e.dim)
        self.blocks = nn.ModuleList([SemanticTransitionBlock(e.dim, e.query_heads) for _ in range(6)])
        self.bridges = nn.ModuleList([
            BidirectionalBridge(width, e.dim, e.query_heads)
            for width in (n.channels[1], n.channels[2], n.channels[2])
        ])
        self.semantic_out = nn.Sequential(nn.LayerNorm(e.dim), nn.Linear(e.dim, e.dim))
        nn.init.zeros_(self.semantic_out[-1].weight)
        nn.init.zeros_(self.semantic_out[-1].bias)
        self.spatial_projection = nn.Conv3d(n.channels[-1], e.dim, 3, padding=1)

    def _semantic_pair(self, state, condition, memory, actions, index):
        for block in self.blocks[2 * index:2 * index + 2]:
            state = block(state, condition, memory, actions)
        return state

    def _exchange(self, image, state, index):
        if self.cfg.network.coupling:
            return self.bridges[index](image, state)
        return image, state

    def _native(self, z, state, tau, tokens, image_global, condition, memory, actions):
        image = self.image
        routed = image.pooled_context(tokens.mean(1)) if image_global is None else image_global
        film = image.time(tau) + routed
        h = image.encode_level(0, image.input(z), film)
        skips = [h]
        h = image.encode_level(1, image.downsample[0](h), film)
        state = self._semantic_pair(state, condition, memory, actions, 0)
        h, state = self._exchange(h, state, 0)
        skips.append(h)
        h = image.encode_level(2, image.downsample[1](h), film)
        state = self._semantic_pair(state, condition, memory, actions, 1)
        h, state = self._exchange(h, state, 1)
        skips.append(h)
        h = image.middle(h, film, tokens)
        state = self._semantic_pair(state, condition, memory, actions, 2)
        h, state = self._exchange(h, state, 2)
        return image.decode(h, (film, skips), tokens), state, h

    def _monai(self, z, state, tau, tokens, image_global, condition, memory, actions):
        from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
        image = self.image.network
        emb = image.time_embed(get_timestep_embedding(tau, image.block_out_channels[0]).to(z.dtype))
        if image_global is not None:
            emb = emb + image_global
        h = image.conv_in(z)
        skips = [h]
        # Traverse the actual MONAI 1.5.1 blocks before each downsampler so an
        # early bridge changes both the matching skip and all subsequent levels.
        for level, block in enumerate(image.down_blocks):
            attentions = getattr(block, "attentions", None)
            for index, resnet in enumerate(block.resnets):
                h = resnet(h, emb)
                if attentions is not None:
                    h = attentions[index](h, context=tokens).contiguous()
                skips.append(h)
            if level in (1, 2):
                bridge_index = level - 1
                state = self._semantic_pair(state, condition, memory, actions, bridge_index)
                h, state = self._exchange(h, state, bridge_index)
                skips[-1] = h
            if block.downsampler is not None:
                h = block.downsampler(h, emb)
                skips.append(h)
        h = image.middle_block(hidden_states=h, temb=emb, context=tokens)
        state = self._semantic_pair(state, condition, memory, actions, 2)
        h, state = self._exchange(h, state, 2)
        return self.image.decode(h, (emb, skips), tokens), state, h

    def _forward(self, z, state, tau, tokens, memory, actions, semantic_global, image_global=None):
        role = self.endpoint_role.expand(1, 2, self.cfg.encoder.disease_tokens, -1).flatten(1, 2)
        state = self.semantic_in(state) + self.token_position + role
        condition = self.time(tau) + semantic_global
        evaluate = self._native if self.cfg.network.backend == "native" else self._monai
        vz, state, h = evaluate(z, state, tau, tokens, image_global, condition, memory, actions)
        dense = F.adaptive_avg_pool3d(self.spatial_projection(h), self.cfg.encoder.token_grid)
        return vz, self.semantic_out(state), dense.flatten(2).transpose(1, 2)

    def forward(self, z, state, tau, context):
        if z.ndim != 5 or z.shape[1] != 48:
            raise ValueError("Expected joint image [B,48,D,H,W]")
        if state.shape != (len(z), 2 * self.cfg.encoder.disease_tokens, self.cfg.encoder.dim):
            raise ValueError("Joint semantic shape mismatch")
        def field(name, default=None):
            return context.get(name, default) if isinstance(context, dict) else getattr(context, name, default)
        tokens, actions = field("cond_tokens"), field("interval_action_tokens")
        state_tokens, clinical = field("state_tokens"), field("clinical_tokens")
        if tokens is None or actions is None or state_tokens is None:
            raise ValueError("Velocity v2 requires routed condition, state and action tokens")
        memory = state_tokens if clinical is None else torch.cat((state_tokens, clinical), 1)
        args = (z, state, tau, tokens, memory, actions, field("semantic_global"), field("cond_global"))
        if args[-2] is None:
            raise ValueError("Velocity v2 requires routed semantic_global")
        if self.cfg.network.checkpoint_blocks and torch.is_grad_enabled() and (
            self.training or z.requires_grad or state.requires_grad
        ):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward, *args, use_reentrant=False, preserve_rng_state=True)
        return self._forward(*args)

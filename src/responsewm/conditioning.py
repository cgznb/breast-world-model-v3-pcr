"""Source-visible, typed conditions for a single canonical transition."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .temporal import NumericTokens


@dataclass(frozen=True)
class IntervalSpec:
    src_stage: int
    dst_stage: int
    actions: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    known_at: float | torch.Tensor | None = None
    plan_version: str | None = None
    mode: str = "sequential"

    def validate(self, stage):
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (self.src_stage, self.dst_stage)):
            raise ValueError("R1 interval stages must be integers")
        if self.mode != "sequential":
            raise ValueError("advance only accepts sequential transition mode")
        if self.src_stage != stage or self.dst_stage != stage + 1 or not 0 <= stage < 3:
            raise ValueError("advance requires the next canonical interval")
        if self.known_at is not None and bool(torch.as_tensor(self.known_at).gt(stage).any()):
            raise ValueError("Interval control was unavailable at the source stage")


@dataclass(frozen=True)
class ConditionBundle:
    state_tokens: torch.Tensor
    clinical_tokens: torch.Tensor
    interval_action_tokens: torch.Tensor
    source_stage_token: torch.Tensor
    target_stage_token: torch.Tensor
    interval_token: torch.Tensor
    mode_token: torch.Tensor

    @property
    def tokens(self):
        return torch.cat(tuple(self.__dict__.values()), dim=1)


@dataclass(frozen=True)
class ConditionContext:
    cond_tokens: torch.Tensor
    state_tokens: torch.Tensor
    clinical_tokens: torch.Tensor
    interval_action_tokens: torch.Tensor
    semantic_global: torch.Tensor
    cond_global: torch.Tensor | None = None


class ResamplerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_query, self.norm_context = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, q, context):
        kv = self.norm_context(context)
        q = q + self.cross(self.norm_query(q), kv, kv, need_weights=False)[0]
        x = self.norm_self(q)
        q = q + self.self_attention(x, x, x, need_weights=False)[0]
        return q + self.ff(q)


class ConditionResampler(nn.Module):
    def __init__(self, cfg, clinical_dim, action_dim):
        super().__init__()
        d, h = cfg.encoder.dim, cfg.encoder.query_heads
        m = cfg.multistage
        self.clinical = NumericTokens(clinical_dim, d)
        self.actions = NumericTokens(action_dim, d)
        self.null_clinical = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.action_unobserved = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.stage = nn.Embedding(4, d)
        self.mode = nn.Embedding(2, d)
        self.interval = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.queries = nn.Parameter(torch.randn(1, m.condition_queries, d)*.02)
        self.blocks = nn.ModuleList(ResamplerBlock(d, h) for _ in range(m.condition_depth))
        self.semantic_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def static_tokens(self, values, mask):
        return self.clinical(values, mask) if self.clinical.count else self.null_clinical.expand(len(values), -1, -1)

    def forward(self, memory, clinical, clinical_mask, interval):
        if interval.mode not in {"sequential", "direct_aux"}:
            raise ValueError("Unsupported transition conditioning mode")
        if any(isinstance(x, bool) or not isinstance(x, int) or x not in range(4)
               for x in (interval.src_stage, interval.dst_stage)):
            raise ValueError("Condition stages must be canonical integers")
        if interval.known_at is not None and bool(torch.as_tensor(interval.known_at).gt(interval.src_stage).any()):
            raise ValueError("Interval control was unavailable at the source stage")
        if self.actions.count == 0 and interval.actions is not None and interval.actions.shape[-1] != 0:
            raise ValueError("This checkpoint has action_dim=0")
        b = len(memory)
        ct = self.static_tokens(clinical, clinical_mask)
        if self.actions.count and interval.actions is not None:
            if interval.action_mask is None:
                raise ValueError("Actions require explicit availability masks")
            at = self.actions(interval.actions, interval.action_mask)
        else:
            at = self.action_unobserved.expand(b, -1, -1)
        def token(value):
            return value.reshape(1, 1, -1).expand(b, -1, -1)
        gap = memory.new_full((b, 1), interval.dst_stage-interval.src_stage)
        bundle = ConditionBundle(memory, ct, at, token(self.stage.weight[interval.src_stage]),
                                 token(self.stage.weight[interval.dst_stage]), self.interval(gap)[:, None],
                                 token(self.mode.weight[{"sequential": 0, "direct_aux": 1}[interval.mode]]))
        q = self.queries.expand(b, -1, -1)
        for block in self.blocks:
            q = block(q, bundle.tokens)
        return ConditionContext(q, memory, ct, at, self.semantic_projection(q.mean(1)))

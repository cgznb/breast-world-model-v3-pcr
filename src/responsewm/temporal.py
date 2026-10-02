"""Irregular-time history, available-feature conditioning and trajectory readout."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .legacy.layers import maybe_checkpoint


def time_features(days, mask, basis="calendar_days"):
    """Calendar days, NOT flow time; gaps measured from the last valid visit."""
    valid_days = days.masked_fill(~mask, -1)
    previous = valid_days.cummax(1).values
    previous = F.pad(previous[:, :-1], (1, 0), value=0).clamp_min(0)
    gaps = (days-previous).clamp_min(0)
    absolute,interval,logscale = (365.,180.,6.) if basis == "calendar_days" else (3.,1.,1.38629436112)
    x = torch.stack((days/absolute, torch.log1p(days.clamp_min(0))/logscale,
                     gaps/interval, torch.log1p(gaps)/logscale,
                     torch.sin(days/interval), torch.cos(days/interval)), -1)
    return x.masked_fill(~mask[..., None], 0)


class TemporalBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4*dim, dim), nn.Dropout(dropout))
    def forward(self, x, padding):
        q = self.norm1(x)
        x = x + self.attention(q, q, q, key_padding_mask=padding, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class NumericTokens(nn.Module):
    """Feature-identity, numeric value and explicit missingness embeddings.

    Fields are ordered by the checkpoint schema. This is not an LLM embedding of
    unsanitized patient records. Unknown numeric values are never negative labels.
    """
    def __init__(self, count, dim):
        super().__init__()
        self.count = count
        self.weight = nn.Parameter(torch.randn(count, dim)*0.02)
        self.identity = nn.Parameter(torch.randn(count, dim)*0.02)
        self.missing = nn.Parameter(torch.randn(count, dim)*0.02)
        self.norm = nn.LayerNorm(dim)
    def forward(self, values, mask):
        values = values.masked_fill(~mask, 0)
        return self.norm(values[..., None]*self.weight + self.identity + (~mask)[..., None]*self.missing)


class AvailableConditioner(nn.Module):
    def __init__(self, cfg, clinical_dim, action_dim):
        super().__init__()
        d = cfg.encoder.dim
        self.basis = cfg.network.time_basis
        self.clinical = NumericTokens(clinical_dim, d)
        self.actions = NumericTokens(action_dim, d)
        self.null_clinical = nn.Parameter(torch.zeros(1, 1, d))
        self.null_action = nn.Parameter(torch.zeros(1, 1, d))
        self.plan_time = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.interval = nn.Sequential(nn.Linear(5, d), nn.SiLU(), nn.Linear(d, d))
        self.direction = nn.Embedding(2, d)
        self.norm = nn.LayerNorm(d)

    def static_tokens(self, clinical, mask):
        return self.clinical(clinical, mask) if self.clinical.count else self.null_clinical.expand(len(clinical), -1, -1)

    def plan_tokens(self, actions, mask, days, valid):
        if days.shape[1] == 0:
            return days.new_empty((len(days), 0, self.norm.normalized_shape[0]))
        if self.actions.count:
            token = self.actions(actions, mask).mean(-2)
        else:
            token = self.null_action.expand(len(days), days.shape[1], -1)
        scale,logscale = (365.,6.) if self.basis == "calendar_days" else (3.,1.38629436112)
        tf = torch.stack((days/scale, torch.log1p(days.clamp_min(0))/logscale), -1)
        return self.norm(token+self.plan_time(tf)).masked_fill(~valid[..., None], 0)

    def forward(self, memory, action, mask, source_day, target_day, direction=1):
        if direction not in (-1, 1):
            raise ValueError("Direction must be +/-1")
        d = target_day-source_day
        scale,gapscale,logscale = (365.,180.,6.) if self.basis == "calendar_days" else (3.,1.,1.38629436112)
        tf = torch.stack((source_day/scale, target_day/scale, d/gapscale,
                          torch.log1p(d.abs())/logscale, torch.sign(d)), -1)
        interval = self.interval(tf) + self.direction.weight[int(direction < 0)]
        at = self.actions(action, mask) if self.actions.count else self.null_action.expand(len(memory), -1, -1)
        return self.norm(torch.cat((memory, at, interval[:, None]), 1))


class HistoryTransformer(nn.Module):
    """Only tokens in the supplied observed/generated prefix are visible.

    Bidirectional attention inside this already truncated prefix is legal. No
    causal mask is claimed over a full untruncated patient record.
    """
    def __init__(self, cfg):
        super().__init__()
        d, heads = cfg.encoder.dim, cfg.encoder.query_heads
        self.cfg = cfg
        self.time = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.role = nn.Embedding(3, d)  # observed MRI, generated MRI, known future plan
        self.query = nn.Parameter(torch.randn(1, 4, d)*0.02)
        self.blocks = nn.ModuleList([TemporalBlock(d, heads) for _ in range(cfg.network.history_depth)])
        self.norm = nn.LayerNorm(d)
    def forward(self, tokens, days, mask, generated, clinical, plans, plan_mask):
        b, t, n, d = tokens.shape
        x = tokens + self.time(time_features(days, mask, self.cfg.network.time_basis))[:, :, None] + self.role(generated.long())[:, :, None]
        x = x.reshape(b, t*n, d)
        query = self.query.expand(b, -1, -1)
        plans = plans+self.role.weight[2] if plans.shape[1] else plans
        x = torch.cat((query, clinical, plans, x), 1)
        padding = torch.cat((torch.zeros(b, 4+clinical.shape[1], dtype=torch.bool, device=x.device),
                             ~plan_mask, ~mask.repeat_interleave(n, 1)), 1)
        for block in self.blocks:
            x = maybe_checkpoint(block, x, padding, enabled=self.cfg.network.checkpoint_blocks)
        return self.norm(x[:, :4])


class ClinicalPrior(nn.Module):
    def __init__(self, clinical_dim, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("coefficient", torch.zeros(clinical_dim*2))
        self.register_buffer("intercept", torch.zeros(()))
        self.register_buffer("fitted", torch.tensor(False))
    def forward(self, clinical, mask):
        if not self.enabled:
            return clinical.new_zeros(len(clinical))
        values = torch.cat((clinical.masked_fill(~mask, 0), mask.to(clinical)), -1)
        return values @ self.coefficient + self.intercept


class TrajectoryPCRHead(nn.Module):
    """Shared landmark head: observed evidence + bounded future residual.

    Each trajectory is read separately. Probabilities, never embeddings/images/
    logits, are averaged across the stochastic sample axis by the caller.
    """
    def __init__(self, cfg, clinical_dim):
        super().__init__()
        d, h, n = cfg.encoder.dim, cfg.encoder.query_heads, cfg.network
        self.cfg = cfg
        self.prior = ClinicalPrior(clinical_dim, n.clinical_prior)
        self.time = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.delta = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.provenance = nn.Embedding(2, d)
        self.q = nn.Parameter(torch.randn(1, 1, d)*0.02)
        self.blocks = nn.ModuleList([TemporalBlock(d, h, n.dropout) for _ in range(n.readout_depth)])
        self.norm = nn.LayerNorm(d)
        self.observed = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(),
                                      nn.Dropout(n.dropout), nn.Linear(2*d, 1))
        self.future = nn.Linear(d, 1)
        # Bounded image residual reduces unrestrained classifier/generator drift.
        self.scale_logit = nn.Parameter(torch.tensor(-3.0))

    def observed_logit(self, memory, clinical, clinical_mask):
        return self.prior(clinical, clinical_mask)+self.observed(memory.mean(1)).squeeze(-1)

    def forward(self, memory, disease, days, mask, generated, observed_slots, clinical, clinical_mask):
        b, t, n, d = disease.shape
        if not self.cfg.network.use_future:
            mask = mask.clone()
            mask[:,observed_slots:] = False
        previous = torch.zeros_like(disease[:, 0])
        deltas = []
        for j in range(t):
            deltas.append((disease[:,j]-previous).masked_fill(~mask[:,j,None,None], 0))
            previous = torch.where(mask[:,j,None,None], disease[:,j], previous)
        delta = torch.stack(deltas, 1)
        x = disease + self.delta(delta) + self.time(time_features(days,mask,self.cfg.network.time_basis))[:,:,None] + self.provenance(generated.long())[:,:,None]
        x = torch.cat((self.q.expand(b,-1,-1), memory, x.reshape(b,t*n,d)), 1)
        padding = torch.cat((torch.zeros(b,1+memory.shape[1],device=x.device,dtype=torch.bool),
                             ~mask.repeat_interleave(n,1)),1)
        for block in self.blocks:
            x = maybe_checkpoint(block, x, padding, enabled=self.cfg.network.checkpoint_blocks)
        raw = self.future(self.norm(x[:,0])).squeeze(-1)
        has_future = mask[:,observed_slots:].any(1).to(raw)
        scale = self.cfg.network.future_residual_limit*self.scale_logit.sigmoid()
        residual = scale*raw.tanh()*has_future
        obs = self.observed_logit(memory,clinical,clinical_mask)
        return obs+residual, residual, obs

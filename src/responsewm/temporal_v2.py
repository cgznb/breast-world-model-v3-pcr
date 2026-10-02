"""Block-causal raw event replay and one readout for every legal prefix."""
from __future__ import annotations
import torch
from torch import nn
from .temporal import ClinicalPrior, TemporalBlock
from .legacy.layers import maybe_checkpoint


class CausalEventBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x, causal_mask, padding):
        q = self.norm1(x)
        x = x + self.attention(q, q, q, attn_mask=causal_mask,
                               key_padding_mask=padding, need_weights=False)[0]
        return x + self.ff(x)


class HistoryContextBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.encoder.dim, cfg.encoder.query_heads
        self.event_type = nn.Embedding(2, d)
        self.stage = nn.Embedding(4, d)
        self.clinical_summary = nn.Linear(d, 2*d)
        self.blocks = nn.ModuleList(CausalEventBlock(d, h) for _ in range(cfg.network.history_depth))
        self.norm = nn.LayerNorm(d)

    def event_tokens(self, memory, clinical_tokens, stage, origin):
        b = len(memory)
        et = self.event_type.weight[int(origin == "predicted")].view(1, 1, -1).expand(b, 1, -1)
        time = self.stage.weight[stage].view(1, 1, -1).expand(b, 1, -1)
        clinical = self.clinical_summary(clinical_tokens.mean(1)).reshape(b, 2, -1)
        return torch.cat((et, time, clinical, memory), 1)

    def forward(self, events, return_all=False):
        if not events:
            raise ValueError("Causal history requires an event")
        b, k, n, d = events[-1].raw_tokens.shape
        x = torch.cat([e.raw_tokens for e in events], dim=2).reshape(b*k, -1, d)
        event_index = torch.arange(len(events), device=x.device).repeat_interleave(n)
        causal_mask = event_index[None, :] > event_index[:, None]
        valid = torch.stack([e.valid for e in events], dim=1)
        padding = (~valid).repeat_interleave(n, 1)[:, None].expand(b, k, -1).reshape(b*k, -1)
        for block in self.blocks:
            x = maybe_checkpoint(block, x, causal_mask, padding,
                                 enabled=self.cfg.network.checkpoint_blocks)
        x = self.norm(x).reshape(b, k, len(events), n, d)
        if return_all:
            return x[:, :, :, 4:]
        return x[:, :, -1, 4:]


class SharedPCRReadout(nn.Module):
    def __init__(self, cfg, clinical_dim):
        super().__init__()
        self.cfg = cfg
        d, h, n = cfg.encoder.dim, cfg.encoder.query_heads, cfg.network
        self.prior = ClinicalPrior(clinical_dim, n.clinical_prior)
        self.query = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.stage = nn.Embedding(4, d)
        self.provenance = nn.Embedding(2, d)
        self.delta = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.generated_gate = nn.Parameter(torch.tensor(-2.))
        self.blocks = nn.ModuleList(TemporalBlock(d, h, n.dropout) for _ in range(n.readout_depth))
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, memory, clinical_tokens, clinical, clinical_mask, events):
        b, k, m, d = memory.shape
        ct = clinical_tokens[:, None].expand(b, k, -1, d).reshape(b*k, -1, d)
        memory = memory.reshape(b*k, m, d)
        parts = [self.query.expand(b*k, -1, -1), memory, ct]
        masks = [torch.zeros(b*k, 1+m+ct.shape[1], device=memory.device, dtype=torch.bool)]
        previous = None
        for event in events:
            disease = event.disease
            if disease.shape[1] == 1 and k > 1:
                disease = disease.expand(b, k, -1, -1)
            delta = torch.zeros_like(disease) if previous is None else disease-previous
            token = disease + self.delta(delta) + self.stage.weight[event.stage] + self.provenance.weight[int(event.origin == "predicted")]
            if event.origin == "predicted":
                token = token*self.generated_gate.sigmoid()
            parts.append(token.reshape(b*k, -1, d))
            masks.append((~event.valid)[:, None, None].expand(b, k, disease.shape[2]).reshape(b*k, -1))
            previous = disease if previous is None else torch.where(event.valid[:, None, None, None], disease, previous)
        x, padding = torch.cat(parts, 1), torch.cat(masks, 1)
        for block in self.blocks:
            x = maybe_checkpoint(block, x, padding, enabled=self.cfg.network.checkpoint_blocks)
        logit = self.output(x[:, 0]).reshape(b, k)
        return logit + self.prior(clinical, clinical_mask)[:, None]

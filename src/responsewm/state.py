"""Persistent input-only patient beliefs and nondegenerate observation updates."""
from __future__ import annotations
from dataclasses import dataclass, fields
import hashlib
import torch
from torch import nn


def tensor_hash(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class Observation:
    stage: int
    latent: torch.Tensor
    event_ids: tuple[str, ...]
    payload_hashes: tuple[str, ...] | None = None
    valid: torch.Tensor | None = None
    clinical: torch.Tensor | None = None
    clinical_mask: torch.Tensor | None = None
    quality: torch.Tensor | None = None

    def hashes(self):
        hashes = []
        for b, latent in enumerate(self.latent):
            payload = [str(self.stage), tensor_hash(latent)]
            if self.clinical is not None:
                if self.clinical_mask is None:
                    raise ValueError("Clinical observation values require availability masks")
                payload.extend((tensor_hash(self.clinical[b].masked_fill(~self.clinical_mask[b], 0)),
                                tensor_hash(self.clinical_mask[b])))
            if self.quality is not None:
                payload.append(tensor_hash(self.quality[b:b+1]))
            hashes.append(hashlib.sha256("|".join(payload).encode()).hexdigest())
        hashes = tuple(hashes)
        if self.payload_hashes is not None and self.payload_hashes != hashes:
            raise ValueError("Observation payload hash does not match the MRI tensor")
        return hashes


@dataclass(frozen=True)
class EvidenceEvent:
    stage: int
    event_ids: tuple[str, ...]
    payload_hashes: tuple[str, ...]
    valid: torch.Tensor


@dataclass(frozen=True)
class ModelEvent:
    stage: int
    origin: str
    raw_tokens: torch.Tensor
    disease: torch.Tensor
    valid: torch.Tensor
    version: int


@dataclass(frozen=True)
class PatientBelief:
    version: int
    stage: int
    as_of: float
    memory: torch.Tensor
    anchor_latent: torch.Tensor
    anchor_disease: torch.Tensor
    anchor_origin: torch.Tensor  # [B,K], 0=observed, 1=predicted
    clinical: torch.Tensor
    clinical_mask: torch.Tensor
    evidence_log: tuple[EvidenceEvent, ...]
    model_state_log: tuple[ModelEvent, ...]
    physiology_hashes: tuple[str, ...]
    branch_prefix_hashes: tuple[str, ...]
    branch_id: str
    lineage_id: str
    evidence_version: int = 0
    cache: None = None

    @property
    def samples(self):
        return self.memory.shape[1]

    @property
    def observed_stage_ids(self):
        return tuple(tuple(e.stage for e in self.evidence_log if bool(e.valid[b]))
                     for b in range(len(self.memory)))

    def state_dict(self):
        """Only primitive containers and tensors, accepted by weights_only loading."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["evidence_log"] = [{f.name: getattr(e, f.name) for f in fields(e)} for e in self.evidence_log]
        out["model_state_log"] = [{f.name: getattr(e, f.name) for f in fields(e)} for e in self.model_state_log]
        return out

    @classmethod
    def from_state_dict(cls, value):
        value = dict(value)
        value["evidence_log"] = tuple(EvidenceEvent(**x) for x in value["evidence_log"])
        value["model_state_log"] = tuple(ModelEvent(**x) for x in value["model_state_log"])
        return cls(**value)


class ObservationSeedPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, m = cfg.encoder.dim, cfg.encoder.query_heads, cfg.multistage
        required = cfg.encoder.anatomy_tokens + cfg.encoder.disease_tokens + m.global_tokens
        if m.memory_tokens != required:
            raise ValueError("Memory slots must equal anatomy+disease+global token counts")
        self.queries = nn.Parameter(torch.randn(1, m.global_tokens, d)*.02)
        self.attention = nn.MultiheadAttention(d, h, dropout=0, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.slots = nn.Parameter(torch.randn(1, m.memory_tokens, d)*.02)
        self.clinical_projection = nn.Linear(d, d)

    def forward(self, observation, clinical_tokens, disease=None):
        tokens = torch.cat((observation.dense, observation.anatomy, observation.disease), dim=1)
        q = self.queries.expand(len(tokens), -1, -1)
        global_tokens = q + self.attention(q, tokens, tokens, need_weights=False)[0]
        global_tokens = global_tokens + self.ff(global_tokens)
        disease = observation.disease if disease is None else disease
        return (torch.cat((observation.anatomy, disease, global_tokens), 1) + self.slots +
                .1*self.clinical_projection(clinical_tokens.mean(1))[:, None])


class AssimilationBlock(nn.Module):
    def __init__(self, dim, heads, bias):
        super().__init__()
        self.memory_norm, self.obs_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
        self.gates = nn.Linear(2*dim+2, 3*dim)
        nn.init.constant_(self.gates.bias, bias)

    def forward(self, memory, observation, gap, quality):
        gate = self.gates(torch.cat((memory.mean(1), observation.mean(1), gap[:, None], quality[:, None]), -1)).sigmoid()
        obs_gate, self_gate, ff_gate = gate.chunk(3, -1)
        kv = self.obs_norm(observation)
        memory = memory + obs_gate[:, None]*self.cross(self.memory_norm(memory), kv, kv, need_weights=False)[0]
        x = self.self_norm(memory)
        memory = memory + self_gate[:, None]*self.self_attention(x, x, x, need_weights=False)[0]
        return memory + ff_gate[:, None]*self.ff(memory)


class ObservationAssimilator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList(AssimilationBlock(cfg.encoder.dim, cfg.encoder.query_heads,
                                   cfg.multistage.assimilation_gate_bias)
                                   for _ in range(cfg.multistage.assimilation_depth))
        d = cfg.encoder.dim
        self.reconstruction = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))

    def forward(self, memory, observation_tokens, valid, gap, quality=None):
        original = memory
        quality = memory.new_ones(len(memory)) if quality is None else quality.to(memory)
        for block in self.blocks:
            memory = block(memory, observation_tokens, gap.to(memory), quality)
        return torch.where(valid[:, None, None], memory, original)

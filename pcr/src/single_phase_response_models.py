"""Causal prefix heads and a fold-local adapter for the last FM-BCMRI block."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_clinical_transform(clinical):
    value = np.asarray(clinical, np.float64)
    median = np.asarray([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in value.T])
    filled = np.where(np.isfinite(value), value, median)
    return dict(median=median.astype(np.float32), mean=filled.mean(0).astype(np.float32),
                scale=np.maximum(filled.std(0), 0.1).astype(np.float32))


def apply_clinical_transform(clinical, state):
    x = np.where(np.isfinite(clinical), clinical, state["median"])
    return ((x - state["mean"]) / state["scale"]).astype(np.float32)


def prefix_mask(mask):
    return torch.cumprod((mask > 0).long(), dim=1).bool()


def patient_prefix_loss(logits, labels, mask):
    keep = prefix_mask(mask)
    eligible = keep[:, 0]
    if not eligible.any():
        raise ValueError("No labeled patient with an observed T0 in the training batch")
    loss = F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits), reduction="none")
    per_patient = (torch.where(keep, loss, 0).sum(1) / keep.sum(1).clamp_min(1))
    return per_patient[eligible].mean()


class ResponseHead(nn.Module):
    def __init__(self, arm, settings, prevalence=0.5):
        super().__init__()
        self.arm = dict(arm)
        self.width = int(settings["hidden_dim"])
        self.use_clinical = bool(arm.get("clinical", False))
        self.kind = arm["model"]
        drop = float(settings["dropout"])
        if self.kind == "linear":
            self.output = nn.Linear(3 * 768 + 2, 1)
        elif self.kind == "gru":
            self.projection = nn.Sequential(nn.Linear(768, self.width), nn.GELU(), nn.Dropout(drop))
            self.gru = nn.GRUCell(3 * self.width + 2, self.width)
            self.dropout = nn.Dropout(drop)
            self.output = nn.Linear(self.width + (17 if self.use_clinical else 0), 1)
        else:
            raise ValueError("Unexpected neural head")
        self.register_buffer("fallback_logit", torch.tensor(float(np.log(prevalence / (1 - prevalence)))))
        self.register_buffer("fallback_clinical_coef", torch.zeros(17))
        self.register_buffer("fallback_clinical_intercept", self.fallback_logit.clone())
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, float(self.fallback_logit))

    def forward(self, features, mask, clinical, days):
        keep = prefix_mask(mask)
        x = torch.where(keep[..., None], features, 0)
        x = F.normalize(x, dim=-1) * (768 ** 0.5)
        if self.kind == "gru":
            z = self.projection(x)
            hidden = torch.zeros(len(x), self.width, device=x.device, dtype=x.dtype)
        else:
            z = x
        time = torch.where(keep, days, 0) / 180.0
        previous = z[:, 0]
        previous_time = time[:, 0]
        fallback = (clinical @ self.fallback_clinical_coef + self.fallback_clinical_intercept
                    if self.use_clinical else self.fallback_logit.expand(len(x)))
        outputs, last_logit = [], fallback
        for t in range(4):
            baseline_delta = z[:, t] - z[:, 0]
            previous_delta = z[:, t] - previous
            interval = time[:, t] - previous_time
            inputs = torch.cat((z[:, t], baseline_delta, previous_delta,
                                time[:, t, None], interval[:, None]), dim=-1)
            if self.kind == "gru":
                candidate = self.gru(inputs, hidden)
                hidden = torch.where(keep[:, t, None], candidate, hidden)
                state = self.dropout(hidden)
                if self.use_clinical:
                    state = torch.cat((state, clinical), dim=-1)
                logit = self.output(state).squeeze(-1)
            else:
                logit = self.output(inputs).squeeze(-1)
            last_logit = torch.where(keep[:, t], logit, last_logit)
            outputs.append(last_logit)
            previous = torch.where(keep[:, t, None], z[:, t], previous)
            previous_time = torch.where(keep[:, t], time[:, t], previous_time)
        return torch.stack(outputs, dim=1)


class LoRALinear(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.base = copy.deepcopy(original).requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, original.in_features))
        self.b = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = float(alpha) / rank

    def forward(self, value):
        return self.base(value) + F.linear(F.linear(value, self.a), self.b) * self.scale


class LastBlockAdapter(nn.Module):
    def __init__(self, encoder, settings):
        super().__init__()
        if settings["last_blocks"] != 1:
            raise ValueError("This cache binds exactly eleven frozen blocks and one adapted block")
        self.block = copy.deepcopy(encoder.blocks[-1]).requires_grad_(False)
        self.norm = copy.deepcopy(encoder.norm).requires_grad_(False)
        for name in ("qkv", "proj"):
            setattr(self.block.attn, name, LoRALinear(getattr(self.block.attn, name), settings["rank"], settings["alpha"]))
        self.base_parameter_count = sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        # Adapter matrices train; pretrained stochastic layers stay in their original eval behavior.
        self.block.eval()
        self.norm.eval()
        return self

    def forward(self, tokens):
        return F.normalize(self.norm(self.block(tokens))[:, 0], dim=-1)

    def adapter_state(self):
        return {k: p.detach().cpu().clone() for k, p in self.named_parameters() if p.requires_grad}

    def load_adapter_state(self, state):
        expected = {k: p for k, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(expected):
            raise ValueError("Adapter parameter names changed")
        with torch.no_grad():
            for k, p in expected.items():
                p.copy_(state[k])


class ResponseModel(nn.Module):
    def __init__(self, arm, settings, prevalence, encoder=None, adaptation=None):
        super().__init__()
        self.head = ResponseHead(arm, settings, prevalence)
        self.adapter = LastBlockAdapter(encoder, adaptation) if arm.get("lora") else None

    def forward(self, features, mask, clinical, days, tokens=None):
        if self.adapter is not None:
            if tokens is None:
                raise ValueError("An adapted model requires tokens from the frozen prefix encoder")
            indices = prefix_mask(mask).nonzero(as_tuple=True)
            if len(tokens):
                encoded = self.adapter(tokens)
                features = features.clone()
                features[indices] = encoded
        return self.head(features, mask, clinical, days)

    def portable_state(self):
        return dict(head={k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()},
                    adapter=self.adapter.adapter_state() if self.adapter is not None else None)

    def load_portable_state(self, state):
        self.head.load_state_dict(state["head"], strict=True)
        if self.adapter is not None:
            self.adapter.load_adapter_state(state["adapter"])
        elif state["adapter"] is not None:
            raise ValueError("An adapter checkpoint cannot be loaded into a frozen-feature model")

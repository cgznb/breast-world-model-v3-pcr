"""Deployment-aligned v3 model with a conservative clinical-prior residual."""
from __future__ import annotations

import torch
from torch import nn

from .model_v2 import ResponseWorldModelV2
from .temporal_v2 import SharedPCRReadout


class BoundedPCRReadout(SharedPCRReadout):
    """Keep the architecture and state keys; bound only the learned logit residual."""

    def __init__(self, cfg, clinical_dim):
        super().__init__(cfg, clinical_dim)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, memory, clinical_tokens, clinical, clinical_mask, events):
        prior = self.prior(clinical, clinical_mask)[:, None]
        residual = super().forward(memory, clinical_tokens, clinical, clinical_mask, events) - prior
        limit = self.cfg.protocol.residual_logit_limit
        return prior + limit * torch.tanh(residual / limit)


class ResponseWorldModelV3(ResponseWorldModelV2):
    def __init__(self, cfg, clinical_dim, action_dim):
        if cfg.schema != "responsewm_v3":
            raise ValueError("ResponseWorldModelV3 requires responsewm_v3 configuration")
        super().__init__(cfg, clinical_dim, action_dim)
        self.pcr = BoundedPCRReadout(cfg, clinical_dim)

    def _configure_readout(self, scope):
        if scope not in {"output", "last_block", "all"}:
            raise ValueError("Unknown readout update scope")
        self.pcr.requires_grad_(scope == "all")
        self.pcr.output.requires_grad_(True)
        if scope == "last_block":
            self.pcr.blocks[-1].requires_grad_(True)

    def configure_stage(self, stage):
        super().configure_stage(stage)
        if stage in {"readout", "joint"}:
            scope = (self.cfg.protocol.readout_scope if stage == "readout"
                     else self.cfg.protocol.joint_readout_scope)
            self._configure_readout(scope)
        self.train(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        # A frozen readout is a deterministic supervisor. Frozen early readout
        # blocks also remain deterministic when only its final block is tuned.
        if hasattr(self, "pcr"):
            if not any(p.requires_grad for p in self.pcr.parameters()):
                self.pcr.eval()
            elif hasattr(self.pcr, "blocks"):
                for block in self.pcr.blocks:
                    if not any(p.requires_grad for p in block.parameters()):
                        block.eval()
        return self


MultistageResponseWorldModel = ResponseWorldModelV3

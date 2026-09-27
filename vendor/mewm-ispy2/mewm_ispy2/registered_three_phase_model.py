"""The ROI32 SymmFlow architecture with three shared-codec latent groups."""

from __future__ import annotations

import torch
from torch import nn

from .registered_roi32_fm import bridge
from .registered_three_phase_data import LATENT_SHAPE, PHASES


def split_latents(value):
    if value.ndim != 5 or tuple(value.shape[1:]) != LATENT_SHAPE:
        raise ValueError("Expected ordered [B,24,8,32,32] three-phase latents")
    return value.split(8, dim=1)


class SharedPhaseCodec(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec.eval().requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, images):
        if images.ndim != 5 or tuple(images.shape[1:]) != (3, 32, 128, 128):
            raise ValueError("Expected [B,3,32,128,128] registered MRI")
        value = torch.cat([self.codec.encode_continuous(images[:, i:i + 1]).float() for i in range(3)], dim=1)
        split_latents(value)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite three-phase encoding")
        return value

    @torch.no_grad()
    def decode(self, normalized, statistics):
        mean = normalized.new_tensor(statistics["mean"], dtype=torch.float32).reshape(1, 24, 1, 1, 1)
        std = normalized.new_tensor(statistics["std"], dtype=torch.float32).reshape(1, 24, 1, 1, 1)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("Invalid phase/channel normalization")
        continuous = normalized.float() * std + mean
        decoded = []
        for phase in split_latents(continuous):
            quantized, _ = self.codec.quantizer(phase)
            decoded.append(self.codec.decode(quantized).float())
        result = torch.cat(decoded, dim=1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite decoded phases")
        return result


class ThreePhaseROI32Flow(nn.Module):
    def __init__(self, config, records, backbone=None):
        super().__init__()
        bridge(config)
        from ispy2_symmflow.flow.path import SymmetricFlowObjective
        from ispy2_symmflow.models.conditioning import StructuredConditionEncoder
        from ispy2_symmflow.models.velocity import build_velocity_model_from_config
        from ispy2_symmflow.training.schema import fit_condition_schema

        fm = config["fm"]
        if fm["velocity"]["latent_channels"] != 24:
            raise ValueError("Three phases require 24 branch and 48 joint channels")
        schema, provenance = fit_condition_schema(records, fm["conditions"])
        self.condition_encoder = StructuredConditionEncoder(schema)
        self.velocity_model = build_velocity_model_from_config(fm["velocity"], backbone=backbone)
        self.objective = SymmetricFlowObjective(sigma_min=0.0, loss_weight_x=1.0, loss_weight_y=1.0)
        self.description = {"data_interface": config["schema"], "phase_order": list(PHASES),
                            "latent_shape_czyx": list(LATENT_SHAPE), "joint_branch_order": ["later", "earlier"],
                            "condition_schema": schema.to_dict(), "fit_split": provenance["fit_split"],
                            "velocity_configuration": fm["velocity"]}

    def loss(self, batch):
        split_latents(batch["later_latent"])
        split_latents(batch["earlier_latent"])
        tokens = self.condition_encoder(batch["conditions"], batch_size=len(batch["later_latent"]))
        return self.objective(self.velocity_model, batch["later_latent"], batch["earlier_latent"], tokens)

    @torch.no_grad()
    def sample(self, source, conditions, noise, *, steps=25, solver="heun"):
        from ispy2_symmflow.flow.solver import integrate_ode

        split_latents(source)
        if source.shape != noise.shape or source.device != noise.device or source.dtype != noise.dtype:
            raise ValueError("Sampling noise must match the source phases")
        tokens = self.condition_encoder(conditions, batch_size=len(source))
        joint = torch.cat((noise, source), dim=1)
        solution = integrate_ode(lambda state, tau: self.velocity_model(state, tau, tokens),
                                 joint, t0=0.0, t1=1.0, steps=steps, method=solver)
        return solution.final_state[:, :24].float()

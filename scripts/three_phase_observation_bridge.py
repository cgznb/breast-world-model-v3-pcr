"""Drop-in model for the original three_phase_all_pairs_training entry point.

Installed next to three_phase_symmflow.py. The original VQ/model/data contracts
remain in effect; Stage A is loaded only as a frozen current-visit encoder.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[3] / 'workflows' / 'observation_v3' / 'src'
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from symm_observation.training import load_observation
from symm_observation.world_model import StateAdapter
from symm_observation.utils import file_digest
from .three_phase_symmflow import ThreePhaseSymmFlow, split_phase_latents, JOINT_LATENT_SHAPE


class ObservationConditionedSymmFlow(ThreePhaseSymmFlow):
    def __init__(self, records, *, source, observation_checkpoint, statistics,
                 codec_checkpoint, patient_splits, backbone=None):
        super().__init__(records, source=source, backbone=backbone)
        encoder, metadata = load_observation(observation_checkpoint)
        if metadata['codec_id'] != 'sha256:'+file_digest(codec_checkpoint):
            raise ValueError('A observation checkpoint and original three-phase VQ checkpoint differ')
        for pid, split in patient_splits.items():
            previous = metadata['patient_splits'].get(pid)
            if (split in {'val','test'} and previous == 'train') or (split == 'test' and previous == 'val'):
                raise ValueError('Original B holdout patient was exposed to A training/selection')
        self.observation_encoder = encoder
        self.register_buffer('upstream_mean', torch.tensor(statistics['mean']).float().view(1,24,1,1,1))
        self.register_buffer('upstream_std', torch.tensor(statistics['std']).float().view(1,24,1,1,1))
        if not torch.isfinite(self.upstream_mean).all() or not torch.isfinite(self.upstream_std).all() or (self.upstream_std <= 0).any():
            raise ValueError('Invalid original three-phase latent normalization')
        width = self.condition_encoder.schema.token_dim
        self.state_adapter = StateAdapter(encoder.cfg.dim, width, 32, 2, 8)
        # Upstream public() removes hash-named fields and long hex strings.
        # Store the content identity as integer bytes so resume contracts still detect A changes.
        asset = Path(observation_checkpoint).stat()
        content_identity = list(bytes.fromhex(file_digest(observation_checkpoint)))
        self.description.update(observation_asset={"size_bytes": asset.st_size, "mtime_ns": asset.st_mtime_ns,
                                                    "content_identity_bytes": content_identity},
            observation_encoder=metadata['config']['encoder'], conditioning='current_visit_only_v3',
            A_longitudinal_predictor=False, codec_id=metadata['codec_id'],
            upstream_normalization=statistics, observation_support='original_loader_has_no_valid_mask')

    def train(self, mode=True):
        super().train(mode)
        self.observation_encoder.eval()
        return self

    def build_context(self, source, records):
        # The original loader standardizes with its own statistics; restore RAW
        # VQ coordinates before applying A's train-fitted normalization.
        raw = source*self.upstream_std + self.upstream_mean
        with torch.no_grad():
            features = self.observation_encoder(raw, teacher_average=True)
        patient = self.state_adapter(features)
        clinical = super().tokens(records)
        return torch.cat((clinical, patient), dim=1)

    def loss(self, batch):
        source, target = batch['source'], batch['target']
        split_phase_latents(source); split_phase_latents(target)
        result = self.objective(self.velocity_model, target, source, self.build_context(source, batch['records']))
        return result.total, {'x': float(result.x.detach()), 'y': float(result.y.detach())}

    def sample(self, source, records, noise, steps):
        from ispy2_symmflow.flow.solver import integrate_ode
        split_phase_latents(source)
        if source.shape != noise.shape or source.device != noise.device or source.dtype != noise.dtype:
            raise ValueError('Source/noise must match')
        context = self.build_context(source, records)
        result = integrate_ode(lambda z,tau: self.velocity_model(z,tau,context),
                               torch.cat((noise,source),1), t0=0.,t1=1.,steps=steps,method='euler')
        return result.final_state[:,:JOINT_LATENT_SHAPE[0]]

import copy
import json

import pytest
import torch

from responsewm.io import digest
from responsewm.model import ResponseWorldModel
from responsewm.model_v2 import ResponseWorldModelV2
from responsewm.migration_v2 import MigrationError, migrate_checkpoint


def migration_fixture(cfg, tmp_path):
    source_cfg = copy.deepcopy(cfg)
    source = ResponseWorldModel(source_cfg, 3, 0)
    target_cfg = copy.deepcopy(cfg)
    target_cfg.schema = "responsewm_v2"
    target_cfg.network.time_basis = "stage_index"
    target_cfg.network.semantic_depth = 6
    target_cfg.multistage.global_tokens = 2
    target_cfg.multistage.memory_tokens = 8
    target_cfg.multistage.deep_jepa_weights = (.3, .7)
    target = ResponseWorldModelV2(target_cfg, 3, 0)
    with torch.no_grad():
        source.encoder.latent_mean.fill_(99)
        source.encoder.latent_std.fill_(17)
        for i, block in enumerate(source.velocity.blocks):
            width = cfg.encoder.dim
            for chunk in range(9):
                block.modulation[-1].weight[chunk * width:(chunk + 1) * width].fill_(i + chunk / 10)
                block.modulation[-1].bias[chunk * width:(chunk + 1) * width].fill_(i + chunk / 10)
    payload = {"schema": "responsewm_checkpoint_v1", "config": source_cfg.to_dict(),
               "stage": "joint", "step": 10, "model": source.state_dict(),
               "metadata": {"manifest_digest": "original-manifest",
                            "data_exposure": "train/validation used in prior development"}}
    path = tmp_path / "v1.pt"
    torch.save(payload, path)
    metadata = {"manifest_digest": "new-manifest", "statistics": {
        "latent_mean": [2.] * 24, "latent_std": [3.] * 24}}
    return source, target, payload, path, metadata


def test_migration_maps_semantic_roles_and_preserves_new_state_modules(cfg, tmp_path):
    source, target, _, path, metadata = migration_fixture(cfg, tmp_path)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    report = migrate_checkpoint(target, path, metadata)
    assert report["status"] == "applied" and report["source_sha256"] == digest(path)
    assert not report["errors"] and report["counts"]["mismatch"] == 0
    assert report["data_exposure"]["source_declaration"] == "train/validation used in prior development"
    assert report["source_manifest_digest"] == "original-manifest"
    assert report["destination_manifest_digest"] == "new-manifest"
    assert torch.equal(target.velocity.image.input.weight, source.velocity.image.input.weight)
    assert torch.equal(target.encoder.stem[0].weight, source.encoder.stem[0].weight)
    for i in range(4):
        d = cfg.encoder.dim
        old, new = source.velocity.blocks[i].modulation[-1], target.velocity.blocks[i].modulation[-1]
        for name in ("weight", "bias"):
            old_value, new_value = getattr(old, name), getattr(new, name)
            assert torch.equal(old_value[:6 * d], new_value[:6 * d])
            assert torch.equal(old_value[6 * d:], new_value[9 * d:])
            assert torch.count_nonzero(new_value[6 * d:9 * d]) == 0
    for key, value in target.state_dict().items():
        if key.startswith(("history.", "pcr.", "conditions.", "assimilator.", "velocity.bridges.",
                           "velocity.blocks.4.", "velocity.blocks.5.")):
            assert torch.equal(value, before[key]), key
    for encoder in (target.encoder, target.target_encoder):
        assert torch.equal(encoder.latent_mean, torch.full_like(encoder.latent_mean, 2))
        assert torch.equal(encoder.latent_std, torch.full_like(encoder.latent_std, 3))
    assert all(torch.equal(a, b) for a, b in zip(target.encoder.state_dict().values(), target.target_encoder.state_dict().values()))
    assert not target.representation_ready
    counted = sum(report["counts"][key] for key in ("loaded", "new", "mismatch", "derived"))
    assert counted == len(target.state_dict())
    json.dumps(report, allow_nan=False)


def test_required_shape_mismatch_rejects_without_partial_mutation(cfg, tmp_path):
    _, target, payload, path, metadata = migration_fixture(cfg, tmp_path)
    payload["model"]["velocity.image.input.weight"] = torch.ones(1)
    torch.save(payload, path)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(MigrationError) as failure:
        migrate_checkpoint(target, path, metadata)
    assert failure.value.report["status"] == "rejected"
    assert failure.value.report["counts"]["mismatch"] == 1
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())


def test_optional_semantic_mismatch_is_audited_and_keeps_fresh_tensor(cfg, tmp_path):
    _, target, payload, path, metadata = migration_fixture(cfg, tmp_path)
    key = "velocity.semantic_out.1.weight"
    payload["model"][key] = torch.ones(1)
    torch.save(payload, path)
    before = target.state_dict()[key].clone()
    report = migrate_checkpoint(target, path, metadata)
    assert report["counts"]["mismatch"] == 1
    assert torch.equal(before, target.state_dict()[key])
    assert any(item["key"] == key for item in report["unused_source"])


def test_rejects_resume_schema_and_invalid_current_statistics(cfg, tmp_path):
    _, target, payload, path, metadata = migration_fixture(cfg, tmp_path)
    metadata["statistics"]["latent_std"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        migrate_checkpoint(target, path, metadata)
    payload["schema"] = "responsewm_checkpoint_v2"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="responsewm_checkpoint_v1"):
        migrate_checkpoint(target, path, metadata)

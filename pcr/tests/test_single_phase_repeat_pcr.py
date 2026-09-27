import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from src import single_phase_repeat_pcr as repeat
from src.first_post_optimization import effective_config
from src.first_post_pcr_data import identity, read_json, save_tensor, write_json


def test_registered_reads_only_acquisition_zero_and_preserves_preprocessing(tmp_path, monkeypatch):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path, meta = tmp_path / "input_dce_aqc_0.nii.gz", tmp_path / "geometry.json"
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
    write_json(meta, {"shape": list(values.shape)})
    seen = []

    def channel(roi, spacing):
        seen.append((roi.copy(), spacing))
        return torch.from_numpy(roi).permute(1, 2, 0).contiguous()

    monkeypatch.setattr(repeat, "pillar_channel", channel)
    monkeypatch.setattr(repeat, "PILLAR_SHAPE", (3, 3, 4, 2))
    visit = {"input_path": str(path), "input_identity": identity(path), "meta_path": str(meta),
             "meta_identity": identity(meta), "shape_zyx": list(values.shape), "spacing_zyx": [2, .7, .7]}
    volume, audit = repeat.build_volume({"branch": "registered_dce0", "input_mode": "registered_dce0_repeat"}, visit)
    assert len(seen) == 1 and seen[0][1] == [2, .7, .7]
    np.testing.assert_array_equal(seen[0][0], values)
    assert torch.equal(volume[0], volume[1]) and torch.equal(volume[1], volume[2])
    assert audit["identical_channels"]
    changed = copy.deepcopy(visit)
    changed["input_identity"]["size_bytes"] += 1
    with pytest.raises(ValueError, match="changed"):
        repeat.build_volume({"branch": "registered_dce0"}, changed)


def test_first_post_requires_no_pre_or_late_and_keeps_each_visit(tmp_path, monkeypatch):
    source_cfg = tmp_path / "source.yaml"
    source_cfg.write_text("world_repo: /unused\n")
    native = tmp_path / "first_post.nii.gz"
    native.write_bytes(b"native-placeholder")
    cached = tmp_path / "crop.npy"
    roi = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 30
    normalized = np.zeros_like(roi)
    nonzero = roi != 0
    normalized[nonzero] = (roi[nonzero] - 142.555409) / 283.804038
    np.save(cached, normalized)
    seen = []
    monkeypatch.setattr(repeat, "world_imports", lambda cfg: None)
    monkeypatch.setattr(repeat, "IMAGE_SHAPE", roi.shape)
    monkeypatch.setattr(repeat, "PILLAR_SHAPE", (3, 3, 4, 2))

    def resample(path, visit):
        seen.append((path, visit["timepoint"]))
        return roi

    monkeypatch.setattr(repeat, "resample_native_roi", resample)
    monkeypatch.setattr(repeat, "pillar_channel", lambda value, spacing: torch.from_numpy(value).permute(1, 2, 0))
    visit = {"native_first_post": str(native), "native_first_post_identity": identity(native),
             "image_path": str(cached), "crop_identity": identity(cached), "timepoint": 2,
             "crop_geometry": {"spacing_xyz_mm": [.7, .7, 2]}}
    cfg = {"branch": "first_post", "input_mode": "native_first_post_repeat", "source_config": str(source_cfg)}
    volume, audit = repeat.build_volume(cfg, visit)
    assert seen == [(str(native), 2)]
    assert audit["vq_crop_max_absolute_difference"] == 0
    assert torch.equal(volume[0], volume[1]) and torch.equal(volume[1], volume[2])


def test_feature_resume_rejects_changed_embedding_and_wrong_phase(tmp_path):
    cfg = {"output_dir": str(tmp_path), "input_mode": "native_first_post_repeat"}
    visit = {"canonical_patient_id": "P1", "timepoint": 1, "native_first_post_identity": {"size_bytes": 10, "mtime_ns": 20}}
    path = repeat.embedding_path(cfg, visit)
    save_tensor(path, torch.ones(1152))
    audit_path = tmp_path / "feature_audits/P1_T1.json"
    audit = {"input_mode": cfg["input_mode"], "identical_channels": True,
             "source_identity": visit["native_first_post_identity"], "embedding_identity": identity(path)}
    write_json(audit_path, audit)
    assert repeat.feature_complete(cfg, visit)
    write_json(audit_path, {**audit, "input_mode": "three_phase"})
    with pytest.raises(ValueError, match="different"):
        repeat.feature_complete(cfg, visit)
    write_json(audit_path, audit)
    save_tensor(path, torch.zeros(1152))
    with pytest.raises(ValueError, match="changed|Invalid"):
        repeat.feature_complete(cfg, visit)


def test_completion_marker_is_stable_on_resume(tmp_path):
    marker = tmp_path / "COMPLETE.json"
    record = {"complete": True, "visits": 2, "input_mode": "native_first_post_repeat"}
    repeat.unchanged_json(marker, record)
    before = identity(marker)
    repeat.unchanged_json(marker, record)
    assert identity(marker) == before
    with pytest.raises(ValueError, match="changed"):
        repeat.unchanged_json(marker, {**record, "visits": 3})


def test_registered_recipes_preserve_frozen_depth_settings(tmp_path):
    cfg = repeat.load_config("configs/single_phase_repeat_pcr_v1.yaml", "registered_dce0")
    cfg["output_dir"] = str(tmp_path)
    paths = repeat.recipe_configs(cfg)
    expected = {"T0": (32, 16, 200), "T0-T1": (32, 16, 200),
                "T0-T2": (32, 16, 80), "T0-T3": (64, 32, 150)}
    for depth, path in paths.items():
        config = yaml.safe_load(Path(path).read_text())
        ds = config["downstream"]
        assert (ds["proj_dim"], ds["sig_dim"], ds["epochs"]) == expected[depth]
        assert ds["input_dim"] == 1152 and ds["n_layers"] == 1 and ds["n_heads"] == 4
        assert ds["lr"] == 3e-4 and ds["batch_size"] == 64
        assert config["data"]["train_embeddings_dir"].startswith(str(tmp_path))


def test_first_post_search_is_fixed_to_two_candidates_and_new_features(tmp_path):
    cfg = repeat.load_config("configs/single_phase_repeat_pcr_v1.yaml", "first_post")
    cfg["output_dir"] = str(tmp_path)
    repeat.recipe_configs(cfg)
    study = repeat.optimization_config(cfg)
    assert set(study["candidates"]) == {"baseline", "unbounded_l2"}
    assert study["feature_sources"] == ["physical_roi"]
    assert study["baseline_run"] == str(tmp_path)
    ds = effective_config(study, "unbounded_l2", 4)
    assert (ds["proj_dim"], ds["sig_dim"]) == (32, 16)
    assert ds["residual_logit_limit"] is None and ds["residual_l2_weight"] == .1
    assert ds["dropout"] == .35 and ds["embedding_dropout"] == .2
    assert ds["epoch_selection"] == "logloss" and ds["patience"] == 12
    assert study["formal_seeds"] == list(range(42, 52))


def test_extract_only_requested_partition_and_resume_does_not_rewrite_marker(tmp_path, monkeypatch):
    cfg = {"output_dir": str(tmp_path), "branch": "first_post", "gpu": 1,
           "input_mode": "native_first_post_repeat", "preprocess_workers": 1}
    cohort = {"visits": [{"canonical_patient_id": "DEV", "timepoint": 0, "split": "train"},
                          {"canonical_patient_id": "HOLD", "timepoint": 0, "split": "val"}]}
    seen = []
    monkeypatch.setattr(repeat, "feature_complete", lambda cfg, visit: seen.append(visit["canonical_patient_id"]) or True)
    monkeypatch.setattr(repeat, "load_frozen_pillar", lambda: pytest.fail("No inference needed for completed features"))
    repeat.extract(cfg, cohort, "train")
    marker = tmp_path / "REAL_FEATURES_COMPLETE.json"
    before = identity(marker)
    repeat.extract(cfg, cohort, "train")
    assert seen == ["DEV", "DEV"] and identity(marker) == before
    assert read_json(marker)["visits"] == 1
    assert not (tmp_path / "HOLDOUT_FEATURES_COMPLETE.json").exists()

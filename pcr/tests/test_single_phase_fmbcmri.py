from __future__ import annotations

import copy
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_full978_independent_cv import _canonical_split
from src.first_post_optimization import nested_folds, select_patients
from src.first_post_pcr_data import identity, save_tensor
from src.single_phase_fmbcmri_data import DEPTHS, encode, feature_complete, load_config, prepare_volume, recipes, store_feature
from src.single_phase_fmbcmri_evaluation import SOURCES, average_folds, metric_tables, overlay
from src.single_phase_fmbcmri_generated import deny_real_pixels, read_first_post
from src.single_phase_fmbcmri_training import fit, formal_tasks, inner_tasks, metrics


def sample_split(n=12, dim=768):
    rng = np.random.default_rng(123)
    return {"pids": [f"p{i}" for i in range(n)], "dim": dim,
            "embs": rng.normal(size=(n, 4, dim)).astype(np.float32), "masks": np.ones((n, 4), np.float32),
            "clinical": rng.normal(size=(n, 17)).astype(np.float32), "labels": np.arange(n, dtype=np.float32) % 2,
            "days": np.tile(np.array([0, 40, 90, 160], np.float32), (n, 1))}


def effective():
    return {"input_dim": 768, "clinical_dim": 17, "proj_dim": 8, "sig_dim": 8, "n_layers": 1,
            "n_heads": 2, "dropout": 0.2, "embedding_dropout": 0.1, "use_prior": True,
            "use_clinical_token": True, "optimizer": "adamw_layerwise", "lr": 0.001,
            "weight_decay": 0.003, "projection_weight_decay": 0.01, "grad_clip": 1.0,
            "residual_l2_weight": 0.1, "positive_class_weight": "balanced", "batch_size": 4,
            "epochs": 4, "scheduler_t_max": 4, "patience": 20, "min_delta": 1e-4,
            "prefix_dropout_probability": 0.3}


def test_single_phase_preprocessing_and_constant_roi(monkeypatch):
    monkeypatch.setattr("src.single_phase_fmbcmri_data.ROI_SHAPE", (8, 12, 16))
    roi = np.random.default_rng(7).normal(size=(8, 12, 16)).astype(np.float32)
    volume = prepare_volume(roi)
    assert volume.shape == (1, 48, 48, 48)
    assert abs(volume.mean().item()) < 1e-6
    assert abs(volume.std().item() - 1) < 1e-6
    assert torch.isfinite(prepare_volume(np.zeros_like(roi))).all()
    with pytest.raises(RuntimeError):
        prepare_volume(np.stack([roi] * 3))


def test_encoder_freeze_dimension_and_cls():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(768))

        def forward_features(self, x):
            return self.weight.expand(len(x), 217, 768)
    model = Encoder()
    batch = torch.zeros(2, 1, 48, 48, 48)
    with pytest.raises(ValueError, match="frozen"):
        encode(model, batch)
    model.eval().requires_grad_(False)
    assert torch.equal(encode(model, batch), torch.ones(2, 768))
    with pytest.raises(ValueError, match="single-channel"):
        encode(model, batch.expand(2, 3, 48, 48, 48))


def test_features_fail_closed_on_missing_or_changed_cache(tmp_path):
    cfg = {"output_dir": str(tmp_path), "phase": "dce0"}
    visit = {"canonical_patient_id": "p1", "timepoint": 1}
    provenance = {"input": {"size_bytes": 12, "mtime_ns": 3}}
    assert not feature_complete(cfg, "real", visit, provenance)
    store_feature(cfg, "real", visit, torch.ones(768), provenance)
    assert feature_complete(cfg, "real", visit, provenance)
    with pytest.raises(ValueError, match="provenance"):
        feature_complete(cfg, "real", visit, {"input": {"size_bytes": 15, "mtime_ns": 3}})


def test_real_first_post_read_does_not_touch_other_phases(tmp_path, monkeypatch):
    from src.single_phase_fmbcmri_data import real_roi
    path = tmp_path / "first_post.npy"
    value = np.arange(8 * 12 * 16, dtype=np.float32).reshape(8, 12, 16)
    np.save(path, value)
    monkeypatch.setattr("src.single_phase_fmbcmri_data.ROI_SHAPE", value.shape)
    def forbidden(*args, **kwargs):
        raise AssertionError("another phase was read")
    monkeypatch.setattr("src.single_phase_fmbcmri_data.nib.load", forbidden)
    result = real_roi({"phase": "first_post"}, {"input_path": str(path), "input_identity": identity(path)})
    np.testing.assert_array_equal(result, value)


def test_generated_read_guard_blocks_target_pixels(tmp_path, monkeypatch):
    route = {"patient_id": "p1", "source_stage": 0, "target_stage": 1}
    path = tmp_path / "prediction.pt"
    snapshot = {"identity": {"size_bytes": 4, "mtime_ns": 3}, "optimizer_step": 80000}
    monkeypatch.setattr("src.single_phase_fmbcmri_generated.ROI_SHAPE", (8, 12, 16))
    payload = {"schema": "first_post_unregistered_tumor_roi_pcr_v1", "route": route,
               "checkpoint_identity": snapshot["identity"], "optimizer_step": 80000, "solver_steps": 20,
               "image_space": "fixed_vqgan_first_post_global_zscore",
               "image": torch.randn(8, 12, 16)}
    save_tensor(path, payload)
    with deny_real_pixels():
        assert read_first_post(path, route, snapshot).shape == (8, 12, 16)
        for filename in ("target.nii.gz", "other_phase.nii", "target_mask.npy", "roi_cache/real.pt"):
            with pytest.raises(PermissionError):
                Path(tmp_path / filename).open("rb")
    with pytest.raises(ValueError, match="provenance"):
        read_first_post(path, {**route, "source_stage": 1}, snapshot)


@pytest.mark.parametrize("fixed", [False, True])
def test_optimizer_scheduler_rng_exact_resume(tmp_path, fixed):
    data = sample_split()
    train = select_patients(data, data["pids"][:8], 4)
    val = None if fixed else select_patients(data, data["pids"][8:], 4)
    kwargs = {"fixed_epochs": 4} if fixed else {}
    _, continuous = fit(train, val, effective(), 42, 4, "cpu", tmp_path / "continuous", **kwargs)
    with pytest.raises(InterruptedError):
        fit(train, val, effective(), 42, 4, "cpu", tmp_path / "resumed", pause_after=2, **kwargs)
    _, recovered = fit(train, val, effective(), 42, 4, "cpu", tmp_path / "resumed", **kwargs)
    assert continuous["history"] == recovered["history"]
    assert continuous["selected_epochs"] == recovered["selected_epochs"]
    assert all(torch.equal(continuous["model_state"][k], v) for k, v in recovered["model_state"].items())
    assert all(recovered["gradient_audit"].values())
    assert recovered["updated_parameter_tensors"] > 0
    before = identity(tmp_path / "resumed/recovery.pt")
    fit(train, val, effective(), 42, 4, "cpu", tmp_path / "resumed", **kwargs)
    assert identity(tmp_path / "resumed/recovery.pt") == before


def test_outer_refit_rejects_validation_and_missing_t0_excluded(tmp_path):
    data = sample_split()
    data["masks"][0, 0] = 0
    train = _canonical_split(data, 4)
    assert not train["masks"][0].any()
    with pytest.raises(ValueError, match="cannot receive"):
        fit(train, train, effective(), 42, 4, "cpu", tmp_path / "bad", fixed_epochs=1)
    _, result = fit(train, None, effective(), 42, 4, "cpu", tmp_path / "good", fixed_epochs=1)
    assert result["neural_loss_patients"] == 11
    assert result["clinical_prior_patients"] == 12
    assert not any(k.startswith("validation_") for k in result["history"][0])


def test_minimum_logloss_selection_is_distinct_from_patience(tmp_path, monkeypatch):
    import src.single_phase_fmbcmri_training as module
    data = sample_split()
    train, val = select_patients(data, data["pids"][:8], 4), select_patients(data, data["pids"][8:], 4)
    original = module.metrics
    count = 0
    def controlled(labels, probability):
        nonlocal count
        result = original(labels, probability)
        if len(labels) == 4:
            result["logloss"] = [0.5, 0.49999, 0.51, 0.52][count]
            count += 1
        return result
    monkeypatch.setattr(module, "metrics", controlled)
    _, result = fit(train, val, {**effective(), "patience": 2}, 42, 4, "cpu", tmp_path)
    assert result["selected_epochs"] == 2
    assert result["epochs_trained"] == 3


def test_prefix_gap_and_t0_overlay(tmp_path):
    data = sample_split(2)
    data["masks"][0, 1] = 0
    for pid in data["pids"]:
        for stage in range(1, 4):
            if pid == data["pids"][0] and stage == 1:
                continue
            save_tensor(tmp_path / pid / f"{pid}_T{stage}.pt", torch.full((768,), float(stage)))
    replaced = overlay(data, tmp_path)
    assert np.array_equal(data["embs"][:, 0], replaced["embs"][:, 0])
    prefix = _canonical_split(replaced, 4)
    assert prefix["masks"][0].tolist() == [1, 0, 0, 0]
    assert not prefix["embs"][0, 1:].any()
    assert not prefix["days"][0, 1:].any()


def fake_fold_predictions():
    return pd.DataFrame([{"source": source, "window": window, "seed": seed, "patient_id": pid,
                          "label": label, "outer": fold, "complete_window": True, "available_prefix": depth,
                          "probability": 0.2 + label * 0.4 + fold * 0.01 + (seed - 42) * 0.02}
                         for source in SOURCES for window, depth in DEPTHS.items() for seed in (42, 43)
                         for pid, label in (("p0", 0), ("p1", 1)) for fold in range(5)])


def test_five_fold_mean_then_seed_metrics():
    frame = fake_fold_predictions()
    means = average_folds(frame, ["p0", "p1"], [42, 43])
    assert means.iloc[0].probability == pytest.approx(0.22)
    per_seed, summary = metric_tables(means, "holdout")
    assert set(summary.seeds) == {2}
    rows = per_seed[(per_seed.source == "real") & (per_seed.window == "T0") & (per_seed.subset == "all_prefixes")]
    item = summary[(summary.source == "real") & (summary.window == "T0") & (summary.subset == "all_prefixes")].iloc[0]
    assert item.logloss_mean == pytest.approx(rows.logloss.mean())
    assert item.logloss_std == pytest.approx(rows.logloss.std(ddof=1))
    with pytest.raises(ValueError, match="five distinct"):
        average_folds(frame.iloc[1:], ["p0", "p1"], [42, 43])
    damaged = frame.copy()
    damaged.loc[0, "probability"] += 0.1
    with pytest.raises(ValueError, match="T0"):
        average_folds(damaged, ["p0", "p1"], [42, 43])


def test_prespecified_four_window_recipes_and_counts():
    cfg = load_config("configs/single_phase_fmbcmri_pcr_v1.yaml", "registered_dce0")
    assert cfg["phase"] == "dce0" and cfg["gpu"] == 0
    resolved = recipes(cfg)
    assert [r["epochs"] for r in resolved.values()] == [200, 200, 80, 150]
    assert all(r["input_dim"] == 768 and r["patience"] == 20 and r["min_delta"] == 1e-4 for r in resolved.values())
    ids = [f"p{i}" for i in range(60)]
    outer = [{"fold": k, "train_ids": [p for i, p in enumerate(ids) if i % 5 != k],
              "val_ids": [p for i, p in enumerate(ids) if i % 5 == k]} for k in range(5)]
    folds = nested_folds(ids, ["heldout"], outer, {p: i % 2 for i, p in enumerate(ids)}, 3, 92026)
    inner = inner_tasks(cfg, folds, resolved)
    assert len(inner) == 120
    selection = [{"window": w, "outer": k, "fixed_epochs": 7} for w in DEPTHS for k in range(5)]
    formal = formal_tasks(cfg, folds, resolved, selection)
    assert len(formal) == 200 and {t["seed"] for t in formal} == set(range(42, 52))
    for task in inner:
        assert (set(task["train_ids"]) | set(task["val_ids"])) <= set(outer[task["outer"]]["train_ids"])


def test_recovery_rejects_changed_training_contract(tmp_path):
    data = sample_split()
    with pytest.raises(InterruptedError):
        fit(data, None, effective(), 42, 4, "cpu", tmp_path, fixed_epochs=4, pause_after=1)
    with pytest.raises(ValueError, match="Recovery"):
        fit(data, None, {**effective(), "lr": 0.01}, 42, 4, "cpu", tmp_path, fixed_epochs=4)


def test_generated_preprocessing_spawn_workers(tmp_path):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from src.single_phase_fmbcmri_generated import preprocess_forecast
    route = {"patient_id": "p0", "source_stage": 0, "target_stage": 1}
    snapshot = {"identity": {"size_bytes": 2, "mtime_ns": 3}, "optimizer_step": 80000}
    path = tmp_path / "generated.pt"
    save_tensor(path, {"schema": "first_post_unregistered_tumor_roi_pcr_v1", "route": route,
                      "checkpoint_identity": snapshot["identity"], "optimizer_step": 80000,
                      "solver_steps": 20, "image_space": "fixed_vqgan_first_post_global_zscore",
                      "image": torch.randn(96, 256, 256)})
    args = ("first_post", "symm", route, str(path), snapshot)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        values = list(pool.map(preprocess_forecast, [args, args]))
    assert values[0].shape == (1, 48, 48, 48)
    assert torch.equal(values[0], values[1])

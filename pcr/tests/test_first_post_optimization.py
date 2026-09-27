import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import first_post_optimization as opt
from src.first_post_roi_optimization import replacement_roi, resize_roi_channel


def synthetic_split(n=24, dim=12):
    rng = np.random.default_rng(9)
    return {"pids": [f"P{i}" for i in range(n)], "dim": dim,
            "embs": rng.normal(size=(n, 4, dim)).astype(np.float32),
            "masks": np.ones((n, 4), np.float32), "days": np.tile([0, 25, 70, 140], (n, 1)).astype(np.float32),
            "clinical": rng.normal(size=(n, 17)).astype(np.float32),
            "labels": (np.arange(n) % 2).astype(np.float32)}


def small_config():
    return {"kind": "tdn", "input_dim": 12, "clinical_dim": 17, "proj_dim": 8,
            "sig_dim": 8, "n_heads": 4, "n_layers": 1, "use_prior": True,
            "use_clinical_token": False, "dropout": 0.1, "embedding_dropout": 0.1,
            "residual_logit_limit": 1.0, "residual_l2_weight": 0.1,
            "optimizer": "adamw_layerwise", "lr": 0.001, "weight_decay": 0.01,
            "projection_weight_decay": 0.01, "batch_size": 8, "grad_clip": 1.0,
            "epochs": 3, "patience": 3, "epoch_selection": "logloss", "positive_class_weight": 1.0}


def test_nested_validation_is_disjoint_and_does_not_use_outer_labels():
    ids = [f"P{i}" for i in range(24)]
    labels = {pid: i % 2 for i, pid in enumerate(ids)}
    outer = [{"fold": i, "train_ids": ids[:i * 8] + ids[(i + 1) * 8:],
              "val_ids": ids[i * 8:(i + 1) * 8]} for i in range(3)]
    folds = opt.nested_folds(ids, ["H1", "H2"], outer, labels, 2, 2026)
    changed = dict(labels)
    changed.update({pid: 1 - changed[pid] for pid in outer[0]["val_ids"]})
    again = opt.nested_folds(ids, ["H1", "H2"], outer, changed, 2, 2026)
    assert folds[0]["inner_folds"] == again[0]["inner_folds"]
    for fold in folds:
        for inner in fold["inner_folds"]:
            assert not set(inner["train_ids"]) & set(inner["val_ids"])
            assert not (set(inner["train_ids"]) | set(inner["val_ids"])) & set(fold["val_ids"])
    with pytest.raises(ValueError, match="overlap"):
        opt.validate_partitions(ids, [ids[0]], outer)


def test_selector_cannot_win_by_underfitting_or_small_training_gap():
    rows = [{"candidate": "good", "source": "physical_roi", "auroc": .79, "logloss": .52, "parameters": 40000, "gap": .09},
            {"candidate": "underfit", "source": "physical_roi", "auroc": .73, "logloss": .50, "parameters": 18, "gap": 0},
            {"candidate": "regularized", "source": "resized_roi", "auroc": .787, "logloss": .51, "parameters": 20000, "gap": .03}]
    assert opt.choose_candidate(rows, .005)["candidate"] == "regularized"
    assert opt.choose_candidate(rows[:2], .005)["candidate"] == "good"


def test_canonical_depth_removes_future_values_and_gaps():
    split = synthetic_split()
    split["masks"][0, 1] = 0
    selected = opt.select_patients(split, ["P0", "P1"], 3)
    assert selected["masks"][0].tolist() == [1, 0, 0, 0]
    assert not selected["embs"][0, 1:].any()
    assert not selected["days"][:, 3:].any()
    assert not selected["embs"][:, 3:].any()


def test_outer_fit_refuses_validation_and_never_records_validation_metrics():
    torch.set_num_threads(1)
    train = synthetic_split()
    val = synthetic_split(12)
    with pytest.raises(ValueError, match="must not receive"):
        opt.fit_tdn(train, val, small_config(), 7, 4, "cpu", fixed_epochs=2)
    model, state = opt.fit_tdn(train, None, small_config(), 7, 4, "cpu", fixed_epochs=2)
    assert state["selected_epochs"] == state["epochs_trained"] == 2
    assert all(not any(k.startswith("validation_") for k in row) for row in state["history"])
    prior = opt._prior_from_state(train["clinical"], state["clinical_prior"])
    original, residual = opt.predict(model, opt.tensor_split(train, prior, "cpu"))
    reloaded = opt.TDN({"downstream": small_config()})
    reloaded.load_state_dict(state["model_state"])
    replay, _ = opt.predict(reloaded, opt.tensor_split(train, prior, "cpu"))
    np.testing.assert_allclose(original, replay, atol=1e-7)
    assert np.abs(residual).max() < 1.0


def test_inner_checkpoint_matches_selected_epoch_loss():
    torch.set_num_threads(1)
    split = synthetic_split(36)
    train = opt.select_patients(split, split["pids"][:24], 3)
    val = opt.select_patients(split, split["pids"][24:], 3)
    model, state = opt.fit_tdn(train, val, small_config(), 9, 3, "cpu")
    prior = opt._prior_from_state(val["clinical"], state["clinical_prior"])
    probability, _ = opt.predict(model, opt.tensor_split(val, prior, "cpu"))
    loss = opt.metric_values(val["labels"], probability)["logloss"]
    assert loss == pytest.approx(state["history"][state["selected_epochs"] - 1]["validation_logloss"], abs=1e-6)


def test_completed_artifacts_reject_corruption_and_resume(tmp_path, monkeypatch):
    split = synthetic_split(36)
    monkeypatch.setattr(opt, "load_development", lambda *_: split)
    task = {"stage": "outer", "source": "physical_roi", "depth": 2, "candidate": "compact32",
            "outer": 0, "inner": -1, "seed": 42, "effective": small_config(), "fixed_epochs": 2,
            "train_ids": split["pids"][:24], "val_ids": split["pids"][24:]}
    opt.run_task(task, tmp_path, tmp_path, device="cpu")
    path = opt.task_path(tmp_path, task)
    assert opt.task_complete(path, task, verify_weights=True)
    assert opt.run_task(task, tmp_path, tmp_path, device="cpu")["skipped"]
    altered = copy.deepcopy(task)
    altered["fixed_epochs"] = 3
    assert not opt.task_complete(path, altered)
    frame = pd.read_csv(path / "predictions.csv")
    frame.loc[0, "probability"] = np.nan
    frame.to_csv(path / "predictions.csv", index=False)
    assert not opt.task_complete(path, task)
    opt.run_task(task, tmp_path, tmp_path, device="cpu")
    assert opt.task_complete(path, task, verify_weights=True)
    (path / "history.csv").unlink()
    assert not opt.task_complete(path, task)
    opt.run_task(task, tmp_path, tmp_path, device="cpu")
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    checkpoint["selected_epochs"] = 999
    torch.save(checkpoint, path / "model.pt")
    assert not opt.task_complete(path, task)


def test_roi_resize_preserves_axis_direction_bounds_and_shape():
    roi = np.arange(6, dtype=np.float32)[:, None, None] * np.ones((1, 8, 10), np.float32)
    tensor = resize_roi_channel(roi, (12, 16, 20))
    assert tensor.shape == (16, 20, 12)
    assert tensor.min() >= 0 and tensor.max() <= 1
    assert torch.all(tensor[..., 1:] >= tensor[..., :-1])
    assert torch.equal(tensor[0], tensor[-1])


def test_real_and_generated_roi_use_identical_support_and_normalization():
    shape = (96, 256, 256)
    real = np.full(shape, 250, dtype=np.float32)
    real[:30] = 0
    real[60:] = 500
    support = (real > 0).astype(np.uint8)
    generated = (real - 142.555409) / 283.804038
    restored = replacement_roi(generated, support)
    np.testing.assert_allclose(real, restored, atol=0.0001)
    torch.testing.assert_close(resize_roi_channel(real, (12, 16, 20)),
                               resize_roi_channel(restored, (12, 16, 20)), rtol=1e-5, atol=1e-6)
    valid = np.ones(shape, dtype=bool)
    valid[60:] = False
    assert not replacement_roi(generated, support, valid)[60:].any()


def test_nested_selection_refit_and_report_end_to_end(tmp_path, monkeypatch):
    from scripts.report_first_post_optimization import report
    from src.first_post_pcr_data import write_json
    split = synthetic_split(48)
    ids = split["pids"]
    outer = [{"fold": 0, "train_ids": ids[:24], "val_ids": ids[24:]},
             {"fold": 1, "train_ids": ids[24:], "val_ids": ids[:24]}]
    labels = dict(zip(ids, split["labels"].astype(int)))
    folds = opt.nested_folds(ids, ["HELDOUT"], outer, labels, 2, 2026)
    write_json(tmp_path / "nested_folds.json", folds)
    cfg = {"output_dir": str(tmp_path), "depths": [2], "tuning_seeds": [42],
           "formal_seeds": [42, 43], "candidates": {"baseline": {}, "compact32": {}, "clinical_only": {}},
           "auroc_tolerance": .005, "bootstrap_samples": 16}
    monkeypatch.setattr(opt, "load_development", lambda *_: split)

    def effective(_cfg, candidate, _depth):
        if candidate == "clinical_only":
            return {"kind": "clinical", "epochs": 0}
        result = small_config()
        result.update(epochs=2, patience=2)
        if candidate == "compact32":
            result["dropout"] = .5
        return result

    monkeypatch.setattr(opt, "effective_config", effective)
    for task in opt.inner_tasks(cfg, folds, ["physical_roi"]):
        opt.run_task(task, tmp_path, tmp_path, "cpu")
    selection = opt.select_models(cfg, folds, ["physical_roi"], "unit_integration")
    frozen = copy.deepcopy(selection)
    refits = opt.formal_tasks(cfg, folds, selection)
    for task in refits:
        opt.run_task(task, tmp_path, tmp_path, "cpu")
    assert opt.select_models(cfg, folds, ["physical_roi"], "unit_integration") == frozen
    result = report(cfg, "unit_integration")
    assert result["development_patients"] == 48
    assert result["holdout_evaluated"] is False
    assert len(result["summary"]) == 2
    directory = tmp_path / "reports/unit_integration"
    assert (directory / "comparison.png").stat().st_size > 1000
    frame = pd.read_csv(directory / "outer_oof_predictions.csv")
    assert len(frame) == 48 * 2 * 2
    assert "HELDOUT" not in set(frame.patient_id)
    for task in opt.inner_tasks(cfg, folds, ["resized_roi"]):
        assert task["candidate"] != "clinical_only"
        opt.run_task(task, tmp_path, tmp_path, "cpu")
    joint = opt.select_models(cfg, folds, ["physical_roi", "resized_roi"], "unit_joint")
    for task in opt.formal_tasks(cfg, folds, joint):
        opt.run_task(task, tmp_path, tmp_path, "cpu")
    assert report(cfg, "unit_joint")["holdout_evaluated"] is False

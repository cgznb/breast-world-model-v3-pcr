import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import first_post_optimization as opt
from src import single_phase_repeat_retraining as retrain
from test_first_post_optimization import small_config, synthetic_split


def test_all_candidates_use_logloss_and_original_patient_folds():
    for branch in ("registered_dce0", "first_post"):
        cfg = retrain.load_config("configs/single_phase_repeat_pcr_v2.yaml", branch)
        assert cfg["formal_seeds"] == list(range(42, 52))
        assert cfg["inner_folds"] == 3
        for candidate in cfg["candidates"]:
            for depth in cfg["depths"]:
                effective = opt.effective_config(cfg, candidate, depth)
                assert effective["epoch_selection"] == "logloss"
                assert effective["proj_dim"] <= 32
                assert effective["use_clinical_token"] is False


def test_no_t0_is_excluded_from_neural_loss_but_retained_in_prior(monkeypatch):
    torch.set_num_threads(1)
    raw = synthetic_split(36)
    raw["masks"][0, 0] = 0
    raw = opt.select_patients(raw, raw["pids"], 4)
    assert not raw["embs"][0].any()
    assert not raw["masks"][0].any()
    seen_prior, neural_batch_sizes = [], []
    original_prior, original_bce = retrain._fit_prior, retrain.F.binary_cross_entropy_with_logits

    def prior(train, val, seed):
        seen_prior.append(len(train["labels"]))
        return original_prior(train, val, seed)

    def bce(logits, labels, **kwargs):
        neural_batch_sizes.append(len(labels))
        return original_bce(logits, labels, **kwargs)

    monkeypatch.setattr(retrain, "_fit_prior", prior)
    monkeypatch.setattr(retrain.F, "binary_cross_entropy_with_logits", bce)
    model, state = retrain.fit_tdn(raw, None, small_config(), 7, 4, "cpu", fixed_epochs=2)
    assert seen_prior == [36]
    assert sum(neural_batch_sizes) == 35 * 2
    assert state["clinical_prior_train_count"] == 36
    assert state["neural_train_count"] == 35
    assert not any(k.startswith("validation_") for row in state["history"] for k in row)
    prior_logits = opt._prior_from_state(raw["clinical"], state["clinical_prior"])
    probability, residual = opt.predict(model, opt.tensor_split(raw, prior_logits, "cpu"))
    assert len(probability) == 36 and np.isfinite(probability).all()
    assert np.abs(residual).max() < 1.0


def test_no_t0_extension_preserves_original_complete_t0_fit():
    torch.set_num_threads(1)
    raw = synthetic_split(36)
    train = opt.select_patients(raw, raw["pids"][:24], 4)
    val = opt.select_patients(raw, raw["pids"][24:], 4)
    _, old = opt.fit_tdn(train, val, small_config(), 17, 4, "cpu")
    _, new = retrain.fit_tdn(train, val, small_config(), 17, 4, "cpu")
    assert old["selected_epochs"] == new["selected_epochs"]
    for name in old["model_state"]:
        torch.testing.assert_close(old["model_state"][name], new["model_state"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="must not receive validation"):
        retrain.fit_tdn(train, val, small_config(), 17, 4, "cpu", fixed_epochs=2)


def test_outer_weights_precede_validation_and_resume_replays(tmp_path, monkeypatch):
    raw = synthetic_split(36)
    raw["masks"][0, 0] = 0
    task = {"stage": "outer", "source": "physical_roi", "depth": 4, "candidate": "compact16",
            "outer": 0, "inner": -1, "seed": 42, "effective": small_config(), "fixed_epochs": 2,
            "train_ids": raw["pids"][:24], "val_ids": raw["pids"][24:]}
    path = opt.task_path(tmp_path, task)
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)
    original = opt.select_patients

    def select(split, ids, depth):
        if ids == task["val_ids"]:
            assert (path / "model.pt").exists()
        return original(split, ids, depth)

    monkeypatch.setattr(opt, "select_patients", select)
    retrain.run_task(task, tmp_path, tmp_path, "cpu")
    assert opt.task_complete(path, task)
    assert retrain.run_task(task, tmp_path, tmp_path, "cpu")["skipped"]
    frame = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model = opt.TDN({"downstream": small_config()})
    model.load_state_dict(state["model_state"])
    for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
        split = original(raw, ids, 4)
        prior = opt._prior_from_state(split["clinical"], state["clinical_prior"])
        actual, _ = opt.predict(model, opt.tensor_split(split, prior, "cpu"))
        expected = frame[frame.role == role].set_index("patient_id").loc[split["pids"]].probability.to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    altered = copy.deepcopy(task)
    altered["fixed_epochs"] = 3
    assert not opt.task_complete(path, altered)
    frame.loc[0, "probability"] = np.nan
    frame.to_csv(path / "predictions.csv", index=False)
    assert not opt.task_complete(path, task)


def test_nested_selection_and_new_report_cover_development_once(tmp_path, monkeypatch):
    raw = synthetic_split(48)
    raw["masks"][0, 0] = 0
    ids = raw["pids"]
    outer = [{"fold": 0, "train_ids": ids[:24], "val_ids": ids[24:]},
             {"fold": 1, "train_ids": ids[24:], "val_ids": ids[:24]}]
    folds = opt.nested_folds(ids, ["HELDOUT"], outer, dict(zip(ids, raw["labels"])), 2, 2026)
    opt.write_json(tmp_path / "nested_folds.json", folds)
    cfg = {"output_dir": str(tmp_path), "branch": "registered_dce0", "depths": [4],
           "tuning_seeds": [42], "formal_seeds": [42, 43], "auroc_tolerance": .005,
           "candidates": {"baseline": {}, "compact16": {}}}
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)

    def effective(_cfg, candidate, _depth):
        result = small_config()
        result.update(epochs=2, patience=2, dropout=.45 if candidate == "compact16" else .1)
        return result

    monkeypatch.setattr(opt, "effective_config", effective)
    for task in opt.inner_tasks(cfg, folds, ["physical_roi"]):
        retrain.run_task(task, tmp_path, tmp_path, "cpu")
    selected = opt.select_models(cfg, folds, ["physical_roi"], "regularized")
    for task in opt.formal_tasks(cfg, folds, selected):
        retrain.run_task(task, tmp_path, tmp_path, "cpu")
    report = retrain.report(cfg)
    assert report["development_patients"] == 48
    assert report["holdout_evaluated"] is False
    assert len(report["summary"]) == 2
    frame = pd.read_csv(tmp_path / "reports/outer_oof_predictions.csv")
    assert "HELDOUT" not in set(frame.patient_id)
    assert len(frame) == 48 * 2 * 2
    assert "registered_dce0" in (tmp_path / "README.md").read_text()

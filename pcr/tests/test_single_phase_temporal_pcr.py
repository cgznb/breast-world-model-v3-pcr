import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import first_post_optimization as opt
from src import single_phase_repeat_retraining as v2
from src import single_phase_temporal_training as train
from src.single_phase_temporal_models import build_model, fit_feature_statistics, window_split
from src.single_phase_temporal_reporting import report_development, seed_group_summary
from test_first_post_optimization import small_config, synthetic_split


def config(architecture="delta", transform="pca"):
    result = small_config()
    result.update(architecture=architecture, feature_transform=transform, visit_policy="available",
                  pca_dim=8, hidden_dim=8, residual_logit_limit=None, epoch_selection="logloss")
    return result


def test_available_visits_preserve_gaps_without_unanchored_or_future_images():
    raw = synthetic_split()
    raw["masks"][0, 1] = 0
    raw["masks"][1, 0] = 0
    data = window_split(raw, raw["pids"], 3, config())
    assert data["masks"][0].tolist() == [1, 0, 1, 0]
    assert not data["masks"][1].any()
    assert not data["embs"][1].any()
    assert not data["embs"][:, 3:].any()
    contiguous = window_split(raw, raw["pids"], 3, {**config(), "visit_policy": "contiguous"})
    assert contiguous["masks"][0].tolist() == [1, 0, 0, 0]


@pytest.mark.parametrize("architecture,transform", [("tdn", "identity"), ("tdn", "standardize"),
                                                     ("tdn", "pca"), ("delta", "pca")])
def test_prediction_ignores_unavailable_and_future_values(architecture, transform):
    torch.set_num_threads(1)
    raw = synthetic_split()
    raw["masks"][0, 1] = 0
    effective = config(architecture, transform)
    selected = window_split(raw, raw["pids"], 2, effective)
    state = fit_feature_statistics(selected, effective)
    model = build_model(effective, 2, state).eval()
    data = opt.tensor_split(raw, np.zeros(len(raw["pids"])), "cpu")
    expected, _ = opt.predict(model, data)
    changed = {k: v.clone() for k, v in data.items()}
    changed["embs"][:, 2:] = 1e6
    changed["days"][:, 2:] = 1e6
    changed["embs"][0, 1] = 1e6
    changed["days"][0, 1] = 1e6
    actual, _ = opt.predict(model, changed)
    np.testing.assert_array_equal(expected, actual)


def test_fold_transform_excludes_validation_and_cached_state_is_bound(tmp_path):
    raw = synthetic_split(48)
    effective = config()
    ids = raw["pids"][:32]
    data = window_split(raw, ids, 3, effective)
    task = {"effective": effective, "depth": 3, "outer": 0, "inner": 0}
    first = train.cached_statistics(task, data, tmp_path)
    raw["embs"][32:] = 10000
    raw["labels"][32:] = 1 - raw["labels"][32:]
    second = fit_feature_statistics(window_split(raw, ids, 3, effective), effective)
    for name in ("mean", "scale", "components"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    assert first["fitted_visits"] == 32 * 3
    different = window_split(raw, raw["pids"][1:33], 3, effective)
    with pytest.raises(ValueError, match="cache identity"):
        train.cached_statistics(task, different, tmp_path)


def test_delta_features_preserve_t0_and_encode_each_followup_difference():
    raw = synthetic_split(4)
    effective = config("delta", "identity")
    model = build_model(effective, 3)
    data = opt.tensor_split(raw, np.zeros(4), "cpu")
    features = model.response_features(data["embs"], data["masks"], data["days"])
    dim = raw["dim"]
    torch.testing.assert_close(features[:, :dim], data["embs"][:, 0])
    torch.testing.assert_close(features[:, dim:2*dim], data["embs"][:, 1] - data["embs"][:, 0])
    assert not features[:, 3*dim:4*dim].any()
    control = build_model({**effective, "input_depth": 1}, 4)
    values = control.response_features(data["embs"], data["masks"], data["days"])
    assert not values[:, dim:4*dim].any()


def test_baseline_training_matches_v2_and_outer_stopping_is_isolated():
    torch.set_num_threads(1)
    raw = synthetic_split(36)
    effective = config("tdn", "identity")
    effective["visit_policy"] = "contiguous"
    data = window_split(raw, raw["pids"][:24], 4, effective)
    val = window_split(raw, raw["pids"][24:], 4, effective)
    _, old = v2.fit_tdn(data, val, effective, 42, 4, "cpu")
    _, new = train.fit_model(data, val, effective, 42, 4, "cpu")
    for name in old["model_state"]:
        torch.testing.assert_close(old["model_state"][name], new["model_state"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="must not receive validation"):
        train.fit_model(data, val, effective, 42, 4, "cpu", fixed_epochs=2)


def test_real_training_artifacts_replay_transform_and_missing_t0(tmp_path, monkeypatch):
    raw = synthetic_split(48)
    raw["masks"][0, 0] = 0
    effective = config()
    task = {"stage": "outer", "source": "physical_roi", "depth": 4, "candidate": "delta32",
            "outer": 0, "inner": -1, "seed": 42, "effective": effective, "fixed_epochs": 2,
            "train_ids": raw["pids"][:32], "val_ids": raw["pids"][32:]}
    path = opt.task_path(tmp_path, task)
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)
    original = train.window_split

    def guarded(split, ids, depth, effective):
        if ids == task["val_ids"]:
            assert (path / "model.pt").exists()
        return original(split, ids, depth, effective)

    monkeypatch.setattr(train, "window_split", guarded)
    train.run_task(task, tmp_path, tmp_path, "cpu")
    assert opt.task_complete(path, task)
    assert train.run_task(task, tmp_path, tmp_path, "cpu")["skipped"]
    saved = torch.load(path / "model.pt", weights_only=False, map_location="cpu")
    assert saved["clinical_prior_train_count"] == 32 and saved["neural_train_count"] == 31
    assert saved["feature_statistics_training_visits"] == 31 * 4
    assert all(not k.startswith("validation_") for k in pd.read_csv(path / "history.csv").columns)
    model = build_model(effective, 4)
    model.load_state_dict(saved["model_state"])
    expected = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
    for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
        split = original(raw, ids, 4, effective)
        prior = opt._prior_from_state(split["clinical"], saved["clinical_prior"])
        prediction, _ = opt.predict(model, opt.tensor_split(split, prior, "cpu"))
        recorded = expected[expected.role == role].set_index("patient_id").loc[split["pids"]].probability.to_numpy()
        np.testing.assert_allclose(prediction, recorded, rtol=0, atol=1e-7)
    altered = copy.deepcopy(task)
    altered["effective"]["pca_dim"] = 6
    assert not opt.task_complete(path, altered)


def test_seed_groups_keep_original_and_added_seeds_separate():
    frame = pd.DataFrame({"depth": [1] * 30, "seed": range(42, 72), "auroc": np.arange(30) / 100})
    result = seed_group_summary(frame, ["depth"]).set_index("seed_group")
    assert result.loc["original_10", "seed_count"] == 10
    assert result.loc["additional_20", "seed_count"] == 20
    assert result.loc["all_30", "seed_count"] == 30
    assert result.loc["original_10", "auroc_mean"] == pytest.approx(.045)


def test_nested_models_report_without_holdout_data(tmp_path, monkeypatch):
    raw = synthetic_split(48)
    ids = raw["pids"]
    outer = [{"fold": 0, "train_ids": ids[:24], "val_ids": ids[24:]},
             {"fold": 1, "train_ids": ids[24:], "val_ids": ids[:24]}]
    folds = opt.nested_folds(ids, ["HELDOUT"], outer, dict(zip(ids, raw["labels"])), 2, 2026)
    opt.write_json(tmp_path / "nested_folds.json", folds)
    cfg = {"output_dir": str(tmp_path), "branch": "unit", "depths": [4], "tuning_seeds": [42],
           "formal_seeds": [42, 52], "auroc_tolerance": .005, "candidates": {"baseline": {}, "delta32": {}}}
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)

    def effective(_cfg, candidate, _depth):
        result = config("tdn", "identity") if candidate == "baseline" else config()
        result.update(epochs=2, patience=2)
        return result

    monkeypatch.setattr(opt, "effective_config", effective)
    for task in opt.inner_tasks(cfg, folds, ["physical_roi"]):
        train.run_task(task, tmp_path, tmp_path, "cpu")
    selected = opt.select_models(cfg, folds, ["physical_roi"], "temporal")
    for task in opt.formal_tasks(cfg, folds, selected):
        train.run_task(task, tmp_path, tmp_path, "cpu")
    result = report_development(cfg)
    assert result["development_patients"] == 48 and result["holdout_evaluated"] is False
    predictions = pd.read_csv(tmp_path / "reports/outer_oof_predictions.csv")
    assert len(predictions) == 48 * 2 * 2 and "HELDOUT" not in set(predictions.patient_id)

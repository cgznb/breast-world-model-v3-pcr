import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import first_post_optimization as opt
from src import registered_three_phase_budget as study
from src.first_post_pcr_data import read_json, repo_path
from test_first_post_optimization import small_config, synthetic_split


@pytest.mark.skipif(
    not repo_path("results/registered_three_phase_roi32_pcr_v1_20260917/folds.json").exists(),
    reason="Requires authorized historical patient folds and baseline recipe artifacts",
)
def test_all_250_fits_are_matched_and_only_t3_residual_penalty_differs():
    cfg = study.load_config("configs/registered_three_phase_pcr_300_50.yaml")
    folds = read_json(repo_path("results/registered_three_phase_roi32_pcr_v1_20260917/folds.json"))
    tasks = study.build_tasks(cfg, folds)
    assert len(tasks) == 250
    assert len({str(study.task_path(cfg["output_dir"], t)) for t in tasks}) == 250
    for task in tasks:
        effective = task["effective"]
        assert effective["epochs"] == effective["scheduler_t_max"] == 300
        assert effective["patience"] == 50 and effective["epoch_selection"] == "auroc"
        assert task.get("fixed_epochs") is None and effective["min_delta"] == 0
        assert not set(task["train_ids"]) & set(task["val_ids"])
    v1, v4 = (study.effective_config(cfg, arm, 4) for arm in ("v1", "v4"))
    assert {k for k in v1 if v1[k] != v4[k]} == {"residual_l2_weight"}
    assert (v1["residual_l2_weight"], v4["residual_l2_weight"]) == (0.0, 0.1)


@pytest.mark.parametrize("patience,expected_epochs", [(50, 51), (400, 300)])
def test_real_training_plateau_stops_at_patience_or_300_cap(patience, expected_epochs):
    torch.set_num_threads(1)
    raw = synthetic_split(36)
    train, val = opt.select_patients(raw, raw["pids"][:24], 3), opt.select_patients(raw, raw["pids"][24:], 3)
    effective = {**small_config(), "epochs": 300, "patience": patience, "scheduler_t_max": 300,
                 "epoch_selection": "auroc", "min_delta": 0.0, "lr": 0.0}
    _, state = opt.fit_tdn(train, val, effective, 9, 3, "cpu")
    assert state["epochs_trained"] == expected_epochs and state["selected_epochs"] == 1
    study.validate_history(pd.DataFrame(state["history"]), effective, 1)


def test_ties_do_not_reset_patience_and_best_weights_are_restored(monkeypatch):
    torch.set_num_threads(1)
    raw = synthetic_split(36)
    train, val = opt.select_patients(raw, raw["pids"][:24], 3), opt.select_patients(raw, raw["pids"][24:], 3)
    effective = {**small_config(), "epochs": 20, "patience": 2, "scheduler_t_max": 20,
                 "epoch_selection": "auroc", "min_delta": 0.0}
    metrics, predict, states, values = opt.metric_values, opt.predict, [], iter([.6, .7, .7, .65])

    def controlled(labels, probability):
        result = metrics(labels, probability)
        if len(labels) == 12:
            result["auroc"] = next(values)
        return result

    def capture(model, data):
        if len(data["labels"]) == 12:
            states.append(copy.deepcopy(model.state_dict()))
        return predict(model, data)

    monkeypatch.setattr(opt, "metric_values", controlled)
    monkeypatch.setattr(opt, "predict", capture)
    model, state = opt.fit_tdn(train, val, effective, 9, 3, "cpu")
    assert state["selected_epochs"] == 2 and state["epochs_trained"] == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)
    assert any(not torch.equal(states[1][k], states[3][k]) for k in states[1])
    study.validate_history(pd.DataFrame(state["history"]), effective, 2)
    with pytest.raises(ValueError, match="Selected epoch"):
        study.validate_history(pd.DataFrame(state["history"]), effective, 4)


def test_saved_fit_replays_and_resumes_without_fitting(tmp_path, monkeypatch):
    raw = synthetic_split(36)
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)
    effective = {**small_config(), "epochs": 3, "patience": 2, "scheduler_t_max": 3,
                 "epoch_selection": "auroc", "min_delta": 0.0}
    task = {"stage": "cv_checkpoint_selection", "source": "physical_roi", "depth": 3,
            "candidate": "v1", "outer": 0, "inner": -1, "seed": 42, "effective": effective,
            "train_ids": raw["pids"][:24], "val_ids": raw["pids"][24:]}
    study.run_task(task, tmp_path, tmp_path)
    path = study.task_path(tmp_path, task)
    assert study.task_complete(path, task)
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    assert state["validation_used_for_checkpoint_selection"]
    assert not state["holdout_labels_or_embeddings_loaded"]
    monkeypatch.setattr(opt, "fit_tdn", lambda *_: pytest.fail("Completed fit retrained"))
    assert study.run_task(task, tmp_path, tmp_path)["skipped"]
    corrupt = pd.read_csv(path / "history.csv")
    corrupt.loc[0, "lr"] = 99
    corrupt.to_csv(path / "history.csv", index=False)
    assert not study.task_complete(path, task)


def test_two_auc_statistics_and_each_seed_remain_distinct():
    from src import registered_three_phase_budget_report as report
    from src import registered_three_phase_fixed_report as old_report
    from test_registered_three_phase_fixed_pcr import raw_predictions

    raw = raw_predictions().replace({"model": {"primary": "v4", "reference": "v1"}})
    raw["depth"] = 4
    predictions = old_report.mc4_within_model(raw)
    metrics = old_report.fold_metrics(predictions, [f"P{i}" for i in range(4)])
    cfg = {"windows": {"v1": [4], "v4": [4]}, "formal_seeds": [42, 43],
           "bootstrap_seed": 26, "bootstrap_samples": 30}
    summary = report.summarize_folds(metrics, cfg, report.SOURCES)
    for row in summary.itertuples():
        assert row.auroc_mean == pytest.approx(.4 if row.seed == 42 else .6)
    _, ensemble = report.ensemble_metrics(predictions)
    np.testing.assert_allclose(ensemble.auroc, 1)
    assert len(summary) == len(ensemble) == 16
    intervals = report.matched_intervals(cfg, predictions)
    assert set(intervals.statistic) == {"fold_mean", "probability_ensemble"}
    np.testing.assert_allclose(intervals[["v4_minus_v1", "ci95_low", "ci95_high"]], 0, atol=1e-12)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        report.summarize_folds(metrics.iloc[1:], cfg, report.SOURCES)
    with pytest.raises(ValueError, match="five fold"):
        report.ensemble_metrics(predictions.iloc[1:])

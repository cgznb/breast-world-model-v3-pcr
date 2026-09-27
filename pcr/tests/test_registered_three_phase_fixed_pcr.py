import copy

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src import first_post_optimization as opt
from src import registered_three_phase_fixed_pcr as fixed
from src import registered_three_phase_fixed_report as report
from src.first_post_pcr_data import read_json, repo_path
from test_first_post_optimization import small_config, synthetic_split


def configured_tasks(tmp_path):
    cfg = fixed.load_config("configs/registered_three_phase_fixed_pcr_v3.yaml")
    cfg["output_dir"] = str(tmp_path)
    folds = read_json(repo_path("results/registered_three_phase_roi32_pcr_v1_20260917/folds.json"))
    if isinstance(folds, dict):
        folds = list(folds.values())
    tasks, refs = fixed.build_tasks(cfg, folds)
    return cfg, folds, tasks, refs


def test_all_five_folds_and_ten_seeds_have_exact_same_recipe(tmp_path):
    cfg, folds, tasks, refs = configured_tasks(tmp_path)
    assert len(tasks) == 350 and len(refs) == 400
    fixed.validate_uniformity(cfg, tasks, refs, folds)
    for arm in fixed.ARMS:
        for depth in cfg["depths"]:
            recipe = fixed.effective_config(cfg, arm, depth)
            assert recipe["epochs"] == recipe["scheduler_t_max"] == 40
            assert recipe["epoch_selection"] == "fixed"
            assert recipe["proj_dim"] == (64 if depth == 4 else 32)
            assert recipe["sig_dim"] == (32 if depth == 4 else 16)
            if arm == "primary":
                assert recipe["positive_class_weight"] == 1


@pytest.mark.parametrize("change", ["dropout", "duration", "seed"])
def test_one_fold_cannot_silently_change_configuration(tmp_path, change):
    cfg, folds, tasks, refs = configured_tasks(tmp_path)
    tasks = copy.deepcopy(tasks)
    if change == "dropout":
        tasks[0]["effective"]["dropout"] += .01
    elif change == "duration":
        tasks[0]["fixed_epochs"] -= 1
    else:
        refs = copy.deepcopy(refs)
        refs[0]["seed"] = 99
    with pytest.raises(ValueError):
        fixed.validate_uniformity(cfg, tasks, refs, folds)


def raw_predictions():
    labels = [0, 0, 1, 1]
    good, bad = [.1, .2, .8, .9], [.85, .95, .65, .70]
    rows = []
    for model in fixed.ARMS:
        for seed in [42, 43]:
            values = [good, bad, good, bad, bad] if seed == 42 else [good, good, bad, good, bad]
            for fold, probabilities in enumerate(values):
                for i, value in enumerate(probabilities):
                    sources = [("real", value)]
                    for mode in ("direct", "rollout", "previous_real"):
                        sources.extend((f"{mode}_draw_{d}", value + (d - 1.5) * .01) for d in range(4))
                    for source, probability in sources:
                        rows.append({"model": model, "depth": 1, "seed": seed, "outer": fold,
                                     "patient_id": f"P{i}", "label": labels[i],
                                     "source": source, "probability": probability})
    return pd.DataFrame(rows)


def test_mc4_then_each_auc_then_fold_mean_keeps_seeds_separate():
    raw = raw_predictions()
    mc4 = report.mc4_within_model(raw)
    metrics = report.fold_metrics(mc4, [f"P{i}" for i in range(4)])
    summary = report.fivefold_summary(metrics, list(range(5)), [42, 43], [1], fixed.SOURCES)
    for row in summary.itertuples():
        assert row.auroc_mean == pytest.approx(.4 if row.seed == 42 else .6)
    first = mc4[(mc4.model == "primary") & (mc4.seed == 42) & (mc4.source == "real")]
    ensemble = first.groupby("patient_id").agg(label=("label", "first"), probability=("probability", "mean"))
    assert roc_auc_score(ensemble.label, ensemble.probability) == 1
    assert len(summary) == 16


def test_missing_draw_patient_or_fold_is_rejected():
    raw = raw_predictions()
    with pytest.raises(ValueError, match="Every fold/patient"):
        report.mc4_within_model(raw.iloc[1:])
    predictions = report.mc4_within_model(raw)
    with pytest.raises(ValueError, match="populations differ"):
        report.fold_metrics(predictions.iloc[1:], [f"P{i}" for i in range(4)])
    metrics = report.fold_metrics(predictions, [f"P{i}" for i in range(4)])
    with pytest.raises(ValueError, match="same five folds"):
        report.fivefold_summary(metrics.iloc[1:], list(range(5)), [42, 43], [1], fixed.SOURCES)


def test_outer_validation_is_accessed_after_weights_are_saved(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    raw = synthetic_split(48)
    effective = {**small_config(), "epoch_selection": "fixed", "epochs": 2, "scheduler_t_max": 2}
    task = {"stage": "outer", "source": "physical_roi", "depth": 3, "candidate": "fixed_regularized",
            "outer": 0, "inner": -1, "seed": 42, "effective": effective, "fixed_epochs": 2,
            "train_ids": raw["pids"][:32], "val_ids": raw["pids"][32:]}
    path = opt.task_path(tmp_path, task)
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)
    select = opt.select_patients

    def guarded(split, ids, depth):
        if ids == task["val_ids"]:
            assert (path / "model.pt").exists()
        return select(split, ids, depth)

    monkeypatch.setattr(opt, "select_patients", guarded)
    opt.run_task(task, tmp_path, tmp_path, "cpu")
    assert opt.run_task(task, tmp_path, tmp_path, "cpu")["skipped"]
    history = pd.read_csv(path / "history.csv")
    assert history.epoch.tolist() == [1, 2]
    assert not any(c.startswith("validation_") for c in history)


def test_bootstrap_compares_five_models_within_each_seed():
    predictions = report.mc4_within_model(raw_predictions())
    cfg = {"bootstrap_seed": 26, "bootstrap_samples": 30}
    intervals = report.paired_intervals(cfg, predictions)
    assert len(intervals) == 14 and set(intervals.seed) == {42, 43}
    np.testing.assert_allclose(intervals[["mean_fold_auc_difference", "ci95_low", "ci95_high"]], 0, atol=1e-12)

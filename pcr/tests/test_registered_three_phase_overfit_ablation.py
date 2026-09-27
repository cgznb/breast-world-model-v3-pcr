import copy

import numpy as np
import pandas as pd
import pytest

from src import registered_three_phase_overfit_ablation as study
from src.first_post_pcr_data import read_json, repo_path


def configured_tasks(tmp_path):
    cfg = study.load_config("configs/registered_three_phase_overfit_ablation_v4.yaml")
    cfg["output_dir"] = str(tmp_path)
    folds = read_json(repo_path("results/registered_three_phase_roi32_pcr_v1_20260917/folds.json"))
    tasks, refs = study.build_tasks(cfg, folds)
    return cfg, folds, tasks, refs


def test_only_declared_factor_changes_in_every_seed_and_fold(tmp_path):
    cfg, _, tasks, refs = configured_tasks(tmp_path)
    assert len(tasks) == 200 and len(refs) == 250
    reference = {(r["outer"], r["seed"]): r["task"] for r in refs if r["arm"] == "reference"}
    for task in tasks:
        expected = copy.deepcopy(reference[(task["outer"], task["seed"])])
        expected["candidate"] = task["candidate"]
        expected["effective"].update(cfg["arms"][task["candidate"]])
        assert task == expected
        assert task["fixed_epochs"] == task["effective"]["scheduler_t_max"] == 40
        assert task["effective"]["positive_class_weight"] == "balanced"


def test_cannot_change_class_weight_or_duration_as_an_unreported_factor(tmp_path):
    cfg, folds, _, _ = configured_tasks(tmp_path)
    cfg["arms"]["residual_penalty"]["positive_class_weight"] = 1.0
    with pytest.raises(ValueError, match="matching conditions"):
        study.build_tasks(cfg, folds)


def oof_predictions():
    rows = []
    for seed in [42, 43]:
        for fold in range(5):
            for arm in ["reference", "same", "better"]:
                for patient in range(8):
                    label = patient % 2
                    probability = .9 if (label and arm == "better") else .1 if arm == "better" else patient / 10
                    rows.append({"arm": arm, "seed": seed, "outer": fold,
                                 "patient_id": f"P{fold}_{patient}", "label": label, "probability": probability})
    return pd.DataFrame(rows)


def test_paired_bootstrap_uses_same_patients_and_keeps_seeds_separate():
    cfg = {"bootstrap_seed": 4, "bootstrap_samples": 100}
    intervals = study.paired_intervals(cfg, oof_predictions())
    assert len(intervals) == 4 and set(intervals.seed) == {42, 43}
    same = intervals[intervals.arm == "same"]
    np.testing.assert_allclose(same[["delta_validation_auroc", "conditional_ci95_low", "conditional_ci95_high"]], 0)
    better = intervals[intervals.arm == "better"]
    np.testing.assert_allclose(better.delta_validation_auroc, .375)
    assert (better.conditional_ci95_low > 0).all()
    with pytest.raises(ValueError, match="patients or labels"):
        study.paired_intervals(cfg, oof_predictions().iloc[1:])


def test_fold_summary_rejects_missing_fold_and_preserves_each_seed():
    rows = []
    for seed in [42, 43]:
        for fold in range(5):
            score = .70 if seed == 42 else .75
            rows.append({"arm": "reference", "seed": seed, "outer": fold, "train_auroc": .8,
                         "validation_auroc": score, "auroc_gap": .8 - score,
                         "validation_logloss": .5, "validation_brier": .2})
    frame = pd.DataFrame(rows)
    result = study.summarize_folds(frame, ["reference"], [42, 43], range(5))
    np.testing.assert_allclose(result.validation_auroc_mean, [.70, .75])
    with pytest.raises(ValueError, match="five folds"):
        study.summarize_folds(frame.iloc[1:], ["reference"], [42, 43], range(5))

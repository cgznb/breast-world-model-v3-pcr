from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.run_full978_independent_cv as independent_cv
import scripts.run_full978_t0_t1_optimized_refit as optimized_refit
from scripts.analyze_full978_t0_t1_optimization import select_oof_threshold
from src.tdn import TDN


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/mewm_ispy2_full978_locked102_t0_t1_optimization.yaml"
FIXED_CONFIG = REPO_ROOT / "configs/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch.yaml"
REFIT_CONFIG = REPO_ROOT / "configs/mewm_ispy2_full978_locked102_t0_t1_optimized_refit.yaml"
FIVEFOLD_EVAL_CONFIG = (
    REPO_ROOT
    / "configs/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch12_fivefold_evaluation.yaml"
)
T0_OPTIMIZATION_CONFIG = (
    REPO_ROOT
    / "configs/mewm_ispy2_full978_locked102_t0_fivefold_optimization.yaml"
)


def test_t0_t1_optimization_config_is_isolated_and_test_blind():
    config = yaml.safe_load(CONFIG.read_text())
    section = config["independent_cv"]

    assert section["output_dir"].endswith("_t0_t1_optimization")
    assert section["temporal_depths"] == [{"name": "T0-T1", "max_tp": 2}]
    assert section["folds"] == 5
    assert section["selection_tolerance"] == 0.005
    assert section["selection_prauc_tolerance"] == 0.010
    assert section["selection_tiebreak"] == "gap_then_parameters"
    assert len(section["formal_seeds"]) == 10
    assert all(
        float(variant.get("residual_l2_weight", config["downstream"]["residual_l2_weight"]))
        == 0.0
        for variant in config["variants"].values()
    )


def test_t0_t1_fixed_epoch_comparison_changes_only_checkpoint_policy():
    config = yaml.safe_load(FIXED_CONFIG.read_text())
    section = config["independent_cv"]

    assert section["output_dir"].endswith("_t0_t1_fixed_epoch")
    assert section["temporal_depths"] == [{"name": "T0-T1", "max_tp": 2}]
    assert list(config["variants"]) == [
        "validation_peak_control",
        "fixed_epoch12",
        "fixed_epoch18",
        "fixed_epoch24",
        "fixed_epoch30",
    ]
    assert [
        config["variants"][name].get("epochs")
        for name in list(config["variants"])[1:]
    ] == [12, 18, 24, 30]
    assert all(
        config["variants"][name]["checkpoint_policy"] == "final_epoch"
        for name in list(config["variants"])[1:]
    )
    assert config["downstream"]["proj_dim"] == 32
    assert config["downstream"]["sig_dim"] == 16
    assert config["downstream"]["use_clinical_token"] is False
    assert config["downstream"]["positive_class_weight"] == 1.0
    assert config["downstream"]["residual_l2_weight"] == 0.0


def test_t0_t1_final_refit_is_single_model_per_seed():
    config = yaml.safe_load(REFIT_CONFIG.read_text())
    section = config["refit"]

    assert optimized_refit._depth(config) == {"name": "T0-T1", "max_tp": 2}
    assert section["selected_variant"] == "fixed_epoch12"
    assert section["checkpoint_policy"] == "final_epoch"
    assert section["expected_epochs"] == 12
    assert section["expected_parameters"] == 40418
    assert section["expected_tdn_training_patients"] == 875
    assert len(section["seeds"]) == 10
    assert set(config["evaluation_sources"]) == {
        "real", "direct_t0", "rollout_generated_dce0_real_ser"
    }


def test_t0_t1_final_refit_binds_the_test_blind_development_selection():
    config = yaml.safe_load(REFIT_CONFIG.read_text())
    effective, selection, path = optimized_refit._development_config(config)

    assert path.is_file()
    assert selection["test_data_used"] is False
    assert selection["test_embeddings_or_labels_loaded"] is False
    assert effective["checkpoint_policy"] == "final_epoch"
    assert effective["epochs"] == 12
    assert effective["use_clinical_token"] is False
    assert effective["positive_class_weight"] == 1.0
    assert effective["residual_l2_weight"] == 0.0


def test_t0_t1_fivefold_evaluation_uses_the_exact_fixed12_training_config():
    training = yaml.safe_load(FIXED_CONFIG.read_text())
    evaluation = yaml.safe_load(FIVEFOLD_EVAL_CONFIG.read_text())

    assert evaluation["data"] == training["data"]
    assert evaluation["downstream"] == training["downstream"]
    assert evaluation["variants"] == training["variants"]
    training_cv = dict(training["independent_cv"])
    evaluation_cv = dict(evaluation["independent_cv"])
    assert training_cv.pop("output_dir").endswith("_t0_t1_fixed_epoch")
    assert evaluation_cv.pop("output_dir").endswith(
        "_t0_t1_fixed_epoch12_fivefold_evaluation"
    )
    assert evaluation_cv == training_cv
    assert set(evaluation["evaluation_sources"]) == {
        "real",
        "direct_t0",
        "rollout_generated_dce0_real_ser",
        "adjacent_real",
        "zero_pre",
    }


def test_t0_fivefold_optimization_is_test_blind_and_has_fixed_epoch_candidates():
    config = yaml.safe_load(T0_OPTIMIZATION_CONFIG.read_text())
    section = config["independent_cv"]

    assert section["temporal_depths"] == [{"name": "T0", "max_tp": 1}]
    assert section["folds"] == 5
    assert section["selection_tolerance"] == 0.005
    assert section["selection_prauc_tolerance"] == 0.010
    assert section["selection_tiebreak"] == "gap_then_parameters"
    assert len(section["formal_seeds"]) == 10
    assert list(config["evaluation_sources"]) == ["real"]
    assert [
        config["variants"][name]["epochs"]
        for name in (
            "ultra_bounded_unweighted_fixed6",
            "ultra_bounded_unweighted_fixed12",
            "ultra_bounded_unweighted_fixed18",
        )
    ] == [6, 12, 18]
    assert all(
        float(
            variant.get(
                "residual_l2_weight", config["downstream"]["residual_l2_weight"]
            )
        )
        == 0.0
        for variant in config["variants"].values()
    )


def test_oof_threshold_averages_seeds_before_balanced_accuracy_selection():
    frame = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D"] * 2,
            "label": [0, 0, 1, 1] * 2,
            "probability": [0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.4, 0.9],
            "seed": [42] * 4 + [43] * 4,
        }
    )

    threshold, score, averaged = select_oof_threshold(frame)

    assert len(averaged) == 4
    assert threshold == pytest.approx(0.35)
    assert score == pytest.approx(0.75)


def test_oof_threshold_rejects_inconsistent_labels():
    frame = pd.DataFrame(
        {
            "patient_id": ["A", "B", "A", "B"],
            "label": [0, 1, 1, 1],
            "probability": [0.1, 0.8, 0.2, 0.9],
            "seed": [42, 42, 43, 43],
        }
    )
    with pytest.raises(ValueError, match="labels disagree"):
        select_oof_threshold(frame)


def test_tdn_can_keep_prior_without_a_duplicate_clinical_token():
    common = {
        "input_dim": 16,
        "clinical_dim": 3,
        "proj_dim": 8,
        "sig_dim": 8,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "use_prior": True,
    }
    with_token = TDN({"downstream": common})
    without_token = TDN(
        {"downstream": {**common, "use_clinical_token": False}}
    )

    assert with_token.clin is not None
    assert without_token.clin is None
    assert without_token.use_prior is True
    assert sum(p.numel() for p in without_token.parameters()) < sum(
        p.numel() for p in with_token.parameters()
    )


def test_positive_class_weight_supports_balanced_and_unweighted():
    labels = np.asarray([0, 0, 0, 1], dtype=np.float32)
    assert independent_cv._positive_class_weight({}, labels) == 3.0
    assert independent_cv._positive_class_weight(
        {"positive_class_weight": 1.0}, labels
    ) == 1.0
    with pytest.raises(ValueError, match="positive_class_weight"):
        independent_cv._positive_class_weight(
            {"positive_class_weight": "invalid"}, labels
        )


def test_selection_can_guard_prauc_and_prioritize_generalization_gap():
    summary = pd.DataFrame(
        [
            {
                "variant": "best_auc_bad_pr",
                "oof_auroc_mean": 0.750,
                "oof_prauc_mean": 0.580,
                "parameters": 40_000,
                "positive_train_oof_gap": 0.01,
            },
            {
                "variant": "large_gap",
                "oof_auroc_mean": 0.748,
                "oof_prauc_mean": 0.602,
                "parameters": 30_000,
                "positive_train_oof_gap": 0.03,
            },
            {
                "variant": "small_gap",
                "oof_auroc_mean": 0.747,
                "oof_prauc_mean": 0.600,
                "parameters": 90_000,
                "positive_train_oof_gap": 0.01,
            },
        ]
    )

    selected, decorated = independent_cv._select_variant(
        summary,
        list(summary["variant"]),
        tolerance=0.005,
        prauc_tolerance=0.010,
        tiebreak="gap_then_parameters",
    )

    assert selected == "small_gap"
    indexed = decorated.set_index("variant")
    assert not bool(indexed.loc["best_auc_bad_pr", "within_prauc_tolerance"])

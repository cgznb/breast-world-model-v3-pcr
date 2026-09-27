from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.run_full978_independent_cv as independent_cv


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "mewm_ispy2_full978_locked102_independent_cv_regularization_v2.yaml"
)
OLD_OUTPUT = "results/mewm_ispy2_full978_locked102_independent_cv"
NEW_OUTPUT = "results/mewm_ispy2_full978_locked102_independent_cv_regularization_v2"
VARIANTS = [
    "current_control",
    "layerwise_adamw",
    "residual_3e4",
    "residual_1e3",
]


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_regularization_v2_config_is_isolated_and_development_selected():
    config = _config()
    section = config["independent_cv"]

    assert section["output_dir"] == NEW_OUTPUT
    assert section["output_dir"] != OLD_OUTPUT
    assert section["tune_seeds"] == [2026, 2027]
    assert section["formal_seeds"] == list(range(42, 52))
    assert section["folds"] == 5
    assert section["selection_tolerance"] == pytest.approx(0.005)
    assert section["temporal_depths"] == [
        {"name": "T0-T2", "max_tp": 3},
        {"name": "T0-T3", "max_tp": 4},
    ]
    assert list(config["variants"]) == VARIANTS


@pytest.mark.parametrize(
    ("depth", "expected_parameters"),
    [("T0-T2", 40_978), ("T0-T3", 88_610)],
)
def test_regularization_v2_candidates_have_equal_capacity_within_depth(
    depth, expected_parameters
):
    config = _config()
    counts = {
        variant: independent_cv._parameter_count(config, variant, depth)
        for variant in VARIANTS
    }

    assert set(counts.values()) == {expected_parameters}


def test_regularization_v2_t2_optimizer_and_residual_contract():
    config = _config()
    effective = {
        variant: independent_cv._effective_config(config, variant, "T0-T2")
        for variant in VARIANTS
    }

    assert all(row["proj_dim"] == 32 for row in effective.values())
    assert all(row["sig_dim"] == 16 for row in effective.values())
    assert all(row["optimizer"] == "adamw_layerwise" for row in effective.values())
    assert effective["current_control"]["projection_weight_decay"] == pytest.approx(
        3.0e-3
    )
    for variant in VARIANTS[1:]:
        assert effective[variant]["projection_weight_decay"] == pytest.approx(2.0e-3)
    assert [effective[variant]["residual_l2_weight"] for variant in VARIANTS] == [
        0.0,
        0.0,
        pytest.approx(3.0e-4),
        pytest.approx(1.0e-3),
    ]
    assert all(row["prefix_dropout_probability"] == 0.0 for row in effective.values())


def test_regularization_v2_t3_optimizer_and_residual_contract():
    config = _config()
    effective = {
        variant: independent_cv._effective_config(config, variant, "T0-T3")
        for variant in VARIANTS
    }

    assert all(row["proj_dim"] == 64 for row in effective.values())
    assert all(row["sig_dim"] == 32 for row in effective.values())
    assert effective["current_control"]["optimizer"] == "adam"
    for variant in VARIANTS[1:]:
        assert effective[variant]["optimizer"] == "adamw_layerwise"
        assert effective[variant]["projection_weight_decay"] == pytest.approx(2.0e-3)
    assert [effective[variant]["residual_l2_weight"] for variant in VARIANTS] == [
        0.0,
        0.0,
        pytest.approx(3.0e-4),
        pytest.approx(1.0e-3),
    ]
    assert all(row["prefix_dropout_probability"] == 0.0 for row in effective.values())


def test_equal_capacity_makes_selector_prefer_gap_then_auroc():
    summary = pd.DataFrame(
        [
            {
                "variant": "current_control",
                "oof_auroc_mean": 0.800,
                "parameters": 40_978,
                "positive_train_oof_gap": 0.030,
            },
            {
                "variant": "layerwise_adamw",
                "oof_auroc_mean": 0.799,
                "parameters": 40_978,
                "positive_train_oof_gap": 0.020,
            },
            {
                "variant": "residual_3e4",
                "oof_auroc_mean": 0.798,
                "parameters": 40_978,
                "positive_train_oof_gap": 0.010,
            },
            {
                "variant": "residual_1e3",
                "oof_auroc_mean": 0.790,
                "parameters": 40_978,
                "positive_train_oof_gap": 0.001,
            },
        ]
    )

    selected, decorated = independent_cv._select_variant(
        summary, VARIANTS, tolerance=0.005
    )

    assert selected == "residual_3e4"
    eligible = decorated[decorated["within_oof_tolerance"]]
    assert set(eligible["variant"]) == set(VARIANTS[:3])

    tied_gap = summary.iloc[:2].copy()
    tied_gap["positive_train_oof_gap"] = 0.020
    selected, _ = independent_cv._select_variant(
        tied_gap, VARIANTS[:2], tolerance=0.005
    )
    assert selected == "current_control"

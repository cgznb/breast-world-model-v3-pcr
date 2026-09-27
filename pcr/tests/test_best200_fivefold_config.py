from pathlib import Path

import yaml

from scripts import run_full978_independent_cv as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mewm_ispy2_full978_locked102_all_depths_best200_fivefold.yaml"


def test_best200_trains_every_depth_for_all_epochs_and_selects_validation_peak():
    config = yaml.safe_load(CONFIG.read_text())

    assert config["independent_cv"]["temporal_depths"] == [
        {"name": "T0", "max_tp": 1},
        {"name": "T0-T1", "max_tp": 2},
        {"name": "T0-T2", "max_tp": 3},
        {"name": "T0-T3", "max_tp": 4},
    ]
    assert config["independent_cv"]["folds"] == 5
    assert len(config["independent_cv"]["formal_seeds"]) == 10
    assert list(config["variants"]) == ["best200"]

    effective = {
        depth: runner._effective_config(config, "best200", depth)
        for depth in ("T0", "T0-T1", "T0-T2", "T0-T3")
    }
    assert all(values["checkpoint_policy"] == "best_validation" for values in effective.values())
    assert all(values["epochs"] == 200 for values in effective.values())
    assert all(values["patience"] > values["epochs"] for values in effective.values())
    assert all(values["min_delta"] == 0.0 for values in effective.values())
    assert all(values["scheduler_t_max"] == 200 for values in effective.values())
    assert all(values["residual_l2_weight"] == 0.0 for values in effective.values())


def test_best200_preserves_the_current_depth_specific_model_recipes():
    config = yaml.safe_load(CONFIG.read_text())
    effective = {
        depth: runner._effective_config(config, "best200", depth)
        for depth in ("T0", "T0-T1", "T0-T2", "T0-T3")
    }

    assert (effective["T0"]["proj_dim"], effective["T0"]["sig_dim"]) == (32, 16)
    assert (effective["T0-T1"]["proj_dim"], effective["T0-T1"]["sig_dim"]) == (32, 16)
    assert (effective["T0-T2"]["proj_dim"], effective["T0-T2"]["sig_dim"]) == (32, 16)
    assert (effective["T0-T3"]["proj_dim"], effective["T0-T3"]["sig_dim"]) == (64, 32)
    assert effective["T0"]["use_clinical_token"] is False
    assert effective["T0-T1"]["use_clinical_token"] is False
    assert effective["T0-T2"]["use_clinical_token"] is True
    assert effective["T0-T3"]["use_clinical_token"] is True
    assert effective["T0"]["positive_class_weight"] == 1.0
    assert effective["T0-T1"]["positive_class_weight"] == 1.0
    assert effective["T0-T2"]["positive_class_weight"] == "balanced"
    assert effective["T0-T3"]["positive_class_weight"] == "balanced"
    assert effective["T0"]["residual_logit_limit"] == 1.0
    assert effective["T0-T1"]["residual_logit_limit"] == 1.0
    assert effective["T0-T2"]["residual_logit_limit"] is None
    assert effective["T0-T3"]["residual_logit_limit"] is None


def test_best200_declares_every_locked102_input_scheme():
    config = yaml.safe_load(CONFIG.read_text())

    assert list(config["evaluation_sources"]) == [
        "real",
        "direct_t0",
        "rollout_generated_dce0_real_ser",
        "adjacent_real",
        "zero_pre",
    ]

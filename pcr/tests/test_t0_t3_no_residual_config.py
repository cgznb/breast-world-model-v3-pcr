from pathlib import Path

import yaml

from scripts.run_full978_independent_cv import _effective_config


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_t0_t3_no_residual_is_strict_single_field_ablation():
    baseline = _load(
        "mewm_ispy2_full978_locked102_independent_cv_regularization_v2.yaml"
    )
    ablation = _load(
        "mewm_ispy2_full978_locked102_t0_t3_layerwise_adamw_no_residual_10seed.yaml"
    )

    selected = _effective_config(baseline, "residual_1e3", "T0-T3")
    no_residual = _effective_config(ablation, "layerwise_adamw", "T0-T3")
    assert selected["residual_l2_weight"] == 1.0e-3
    assert no_residual["residual_l2_weight"] == 0.0
    selected["residual_l2_weight"] = 0.0
    assert no_residual == selected

    base_cv = baseline["independent_cv"]
    ablation_cv = ablation["independent_cv"]
    for key in ("fold_seed", "folds", "formal_seeds", "jobs", "prefix_policy", "fold_stratification"):
        assert ablation_cv[key] == base_cv[key]
    assert ablation_cv["temporal_depths"] == [{"name": "T0-T3", "max_tp": 4}]
    assert list(ablation["variants"]) == ["layerwise_adamw"]
    assert ablation["data"] == baseline["data"]
    assert ablation["evaluation_sources"] == baseline["evaluation_sources"]

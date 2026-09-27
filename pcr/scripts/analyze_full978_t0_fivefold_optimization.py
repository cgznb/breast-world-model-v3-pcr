"""Audit and summarize the development-selected T0 five-fold optimization."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_full978_t0_t1_optimization import select_oof_threshold
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_json, _atomic_text
from src.metrics import METRIC_KEYS, compute_metrics


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/mewm_ispy2_full978_locked102_t0_fivefold_optimization"
PREVIOUS = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_frozen_mixed_no_residual_biflow_trajectories"
)
OUTPUT = RESULT / "analysis"
SEEDS = tuple(range(42, 52))


def _load_oof():
    frames = []
    root = RESULT / "oof/formal/t0/variant_ultra_bounded_unweighted_fixed6"
    for seed in SEEDS:
        frame = pd.read_csv(
            root / f"seed_{seed}/oof_predictions.csv", dtype={"patient_id": str}
        )
        if (
            len(frame) != 876
            or frame["patient_id"].duplicated().any()
            or set(frame["seed"]) != {seed}
            or int(frame["label"].sum()) != 282
        ):
            raise RuntimeError("formal T0 OOF contract changed")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _test_metrics(threshold):
    rows = []
    prior_rows = []
    for seed in SEEDS:
        frame = pd.read_csv(
            RESULT / f"evaluation/real/t0/seed_{seed}/test_predictions.csv",
            dtype={"patient_id": str},
        )
        if (
            len(frame) != 102
            or frame["patient_id"].duplicated().any()
            or int(frame["label"].sum()) != 32
            or set(frame["seed"]) != {seed}
            or set(frame["source"]) != {"real"}
            or set(frame["split"]) != {"test_fold_ensemble"}
            or not np.isfinite(
                frame[["probability", "residual_logit", "clinical_prior_probability"]]
            ).all().all()
        ):
            raise RuntimeError("optimized T0 test-prediction contract changed")
        for policy, cutoff in (
            ("fixed_0.5", 0.5),
            ("development_oof_bacc", threshold),
        ):
            rows.append(
                {
                    "seed": seed,
                    "threshold_policy": policy,
                    "threshold": cutoff,
                    "brier": float(
                        np.mean((frame["probability"] - frame["label"]) ** 2)
                    ),
                    **compute_metrics(
                        frame["label"], frame["probability"], threshold=cutoff
                    ),
                }
            )
        prior_rows.append(
            {
                "seed": seed,
                "threshold": 0.5,
                "brier": float(
                    np.mean(
                        (frame["clinical_prior_probability"] - frame["label"]) ** 2
                    )
                ),
                **compute_metrics(
                    frame["label"],
                    frame["clinical_prior_probability"],
                    threshold=0.5,
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(prior_rows)


def _summarize(frame, group_keys):
    rows = []
    for values, selected in frame.groupby(group_keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(group_keys, values))
        row["n_seeds"] = len(selected)
        row["folds_per_seed"] = 5
        for metric in (*METRIC_KEYS, "brier"):
            row[f"{metric}_mean"] = float(selected[metric].mean())
            row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _previous_comparison(optimized):
    previous = pd.read_csv(PREVIOUS / "summary.csv")
    previous = previous[
        (previous["source"] == "real") & (previous["temporal_depth"] == "T0")
    ].iloc[0]
    current = optimized[optimized["threshold_policy"] == "fixed_0.5"].iloc[0]
    row = {
        "previous_protocol": str(previous["protocol"]),
        "optimized_protocol": "five_fold_probability_ensemble",
        "previous_parameters": 203842,
        "optimized_parameters_per_fold": 40418,
        "previous_folds_per_seed": 1,
        "optimized_folds_per_seed": 5,
    }
    for metric in METRIC_KEYS:
        old_value = float(previous[f"{metric}_mean"])
        new_value = float(current[f"{metric}_mean"])
        row[f"previous_{metric}_mean"] = old_value
        row[f"optimized_{metric}_mean"] = new_value
        row[f"optimized_minus_previous_{metric}"] = new_value - old_value
        row[f"previous_{metric}_std"] = float(previous[f"{metric}_std"])
        row[f"optimized_{metric}_std"] = float(current[f"{metric}_std"])
    return pd.DataFrame([row])


def main():
    selection = json.loads((RESULT / "selected_variants.json").read_text())
    selected = selection.get("depths", {}).get("T0", {})
    if (
        selection.get("test_data_used") is not False
        or selection.get("test_embeddings_or_labels_loaded") is not False
        or selected.get("variant") != "ultra_bounded_unweighted_fixed6"
        or selected.get("checkpoint_policy") != "final_epoch"
    ):
        raise RuntimeError("development-selected T0 contract changed")
    oof = _load_oof()
    threshold, oof_bacc, averaged = select_oof_threshold(oof)
    metrics, prior_metrics = _test_metrics(float(threshold))
    summary = _summarize(metrics, ["threshold_policy", "threshold"])
    prior_summary = _summarize(prior_metrics.assign(policy="clinical_prior_0.5"), ["policy", "threshold"])
    comparison = _previous_comparison(summary)

    threshold_payload = {
        "schema": "pillar_full978_t0_development_threshold_v1",
        "source": "ten_seed_mean_five_fold_oof_predictions",
        "development_patients": len(averaged),
        "development_positive_patients": int(averaged["label"].sum()),
        "criterion": "maximum_balanced_accuracy",
        "tie_break": "closest_to_0.5_then_lower_threshold",
        "threshold": float(threshold),
        "oof_balanced_accuracy": float(oof_bacc),
        "test_data_loaded_for_threshold_selection": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT / "development_threshold.json", threshold_payload)
    _atomic_csv(OUTPUT / "test_metrics_per_seed.csv", metrics)
    _atomic_csv(OUTPUT / "test_summary.csv", summary)
    _atomic_csv(OUTPUT / "clinical_prior_test_metrics_per_seed.csv", prior_metrics)
    _atomic_csv(OUTPUT / "clinical_prior_test_summary.csv", prior_summary)
    _atomic_csv(OUTPUT / "previous_vs_optimized.csv", comparison)

    fixed = summary[summary["threshold_policy"] == "fixed_0.5"].iloc[0]
    tuned = summary[summary["threshold_policy"] == "development_oof_bacc"].iloc[0]
    prior = prior_summary.iloc[0]
    delta = comparison.iloc[0]
    report = f"""# T0 Five-Fold Optimization Report

## Development-selected recipe

- Five fold probability ensemble per seed; ten seeds 42-51.
- 40,418 parameters per fold, projection/signature 32/16, fixed 6 epochs.
- Layerwise AdamW, unweighted BCE, bounded imaging residual, no duplicated clinical token, and no residual L2.
- Formal development OOF AUROC/PR-AUC/gap: 0.71229 / 0.53830 / 0.03142.
- Clinical-prior development OOF AUROC/PR-AUC: 0.71019 / 0.53442.
- Development-OOF balanced-accuracy threshold: {threshold:.9f}.

## Descriptive locked102 result

- Optimized AUROC/PR-AUC: {fixed['auroc_mean']:.5f} +/- {fixed['auroc_std']:.5f} / {fixed['prauc_mean']:.5f} +/- {fixed['prauc_std']:.5f}.
- Previous AUROC/PR-AUC: {delta['previous_auroc_mean']:.5f} +/- {delta['previous_auroc_std']:.5f} / {delta['previous_prauc_mean']:.5f} +/- {delta['previous_prauc_std']:.5f}.
- Optimized sensitivity/specificity/balanced accuracy at the OOF threshold: {tuned['sens_mean']:.5f} / {tuned['spec_mean']:.5f} / {tuned['bacc_mean']:.5f}.
- Fold-clinical-prior AUROC/PR-AUC: {prior['auroc_mean']:.5f} / {prior['prauc_mean']:.5f}.

T0 is identical for real, Direct-T0, rollout, adjacent-real, and Zero-pre inputs because only future visits are replaced. The locked102 cohort has already been inspected, so these test metrics are descriptive rather than untouched confirmation.
"""
    _atomic_text(OUTPUT / "report.md", report)
    _atomic_json(
        OUTPUT / "ANALYSIS_COMPLETE.json",
        {
            "schema": "pillar_full978_t0_fivefold_optimization_analysis_v1",
            "seeds": list(SEEDS),
            "folds_per_seed": 5,
            "threshold": float(threshold),
            "test_used_for_model_or_threshold_selection": False,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()

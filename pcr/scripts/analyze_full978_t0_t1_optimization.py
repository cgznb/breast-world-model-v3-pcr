"""Audit and summarize the development-selected T0-T1 optimization."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_json, _atomic_text
from src.metrics import METRIC_KEYS, compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION = REPO_ROOT / "results/mewm_ispy2_full978_locked102_t0_t1_optimization"
FIXED = REPO_ROOT / "results/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch"
REFIT = REPO_ROOT / "results/mewm_ispy2_full978_locked102_t0_t1_optimized_refit"
FROZEN = REPO_ROOT / "results/mewm_ispy2_full978_locked102_frozen_mixed_no_residual_biflow_trajectories"
OUTPUT = REFIT / "analysis"
SEEDS = tuple(range(42, 52))
SOURCES = ("real", "direct_t0", "rollout_generated_dce0_real_ser")


def select_oof_threshold(frame):
    required = {"patient_id", "label", "probability", "seed"}
    if required - set(frame.columns):
        raise ValueError("OOF predictions are missing required columns")
    if frame.duplicated(["patient_id", "seed"]).any():
        raise ValueError("OOF predictions contain duplicate patient-seed rows")
    counts = frame.groupby("patient_id")["seed"].nunique()
    if counts.nunique() != 1:
        raise ValueError("OOF patients do not have equal seed coverage")
    labels = frame.groupby("patient_id")["label"].nunique()
    if not labels.eq(1).all():
        raise ValueError("OOF labels disagree across seeds")
    averaged = (
        frame.groupby(["patient_id", "label"], as_index=False, sort=True)["probability"]
        .mean()
    )
    y_true = averaged["label"].to_numpy(dtype=int)
    probability = averaged["probability"].to_numpy(dtype=float)
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    scored = [
        (
            float(balanced_accuracy_score(y_true, probability >= threshold)),
            abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    best = max(scored, key=lambda row: (row[0], -row[1], -row[2]))
    return best[2], best[0], averaged


def _load_formal_oof():
    frames = []
    root = FIXED / "oof/formal/t0_t1/variant_fixed_epoch12"
    for seed in SEEDS:
        frame = pd.read_csv(
            root / f"seed_{seed}/oof_predictions.csv", dtype={"patient_id": str}
        )
        if set(frame["seed"]) != {seed} or len(frame) != 876:
            raise RuntimeError("formal fixed-epoch OOF contract changed")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _test_metrics(thresholds):
    rows = []
    prediction_frames = {}
    for source in SOURCES:
        source_frames = []
        for seed in SEEDS:
            path = REFIT / f"evaluation/{source}/seed_{seed}/test_predictions.csv"
            frame = pd.read_csv(path, dtype={"patient_id": str})
            if (
                len(frame) != 102
                or frame["patient_id"].duplicated().any()
                or set(frame["label"]) != {0, 1}
                or int(frame["label"].sum()) != 32
                or set(frame["seed"]) != {seed}
                or set(frame["source"]) != {source}
            ):
                raise RuntimeError("optimized test-prediction contract changed")
            source_frames.append(frame)
            for threshold_name, threshold in thresholds.items():
                metrics = compute_metrics(
                    frame["label"], frame["probability"], threshold=float(threshold)
                )
                rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "threshold_policy": threshold_name,
                        "threshold": float(threshold),
                        "brier": float(
                            np.mean(
                                (
                                    frame["probability"].to_numpy(dtype=float)
                                    - frame["label"].to_numpy(dtype=float)
                                )
                                ** 2
                            )
                        ),
                        **metrics,
                    }
                )
        prediction_frames[source] = pd.concat(source_frames, ignore_index=True)
    direct = prediction_frames["direct_t0"].sort_values(["seed", "patient_id"])
    rollout = prediction_frames["rollout_generated_dce0_real_ser"].sort_values(
        ["seed", "patient_id"]
    )
    if (
        direct[["seed", "patient_id", "label"]].reset_index(drop=True).equals(
            rollout[["seed", "patient_id", "label"]].reset_index(drop=True)
        )
        is False
        or not np.array_equal(
            direct["probability"].to_numpy(), rollout["probability"].to_numpy()
        )
    ):
        raise RuntimeError("T0-T1 direct and rollout predictions should be identical")
    return pd.DataFrame(rows)


def _summarize(metrics):
    rows = []
    for (source, policy, threshold), frame in metrics.groupby(
        ["source", "threshold_policy", "threshold"], sort=False
    ):
        row = {
            "source": source,
            "threshold_policy": policy,
            "threshold": float(threshold),
            "n_seeds": len(frame),
            "models_per_seed": 1,
        }
        for metric in (*METRIC_KEYS, "brier"):
            values = frame[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison(optimized):
    old = pd.read_csv(FROZEN / "summary.csv")
    source_map = {
        "real": "real",
        "direct_t0": "direct_t0",
        "rollout_generated_dce0_real_ser": "rollout_generated_dce0_real_ser",
    }
    rows = []
    for source, old_source in source_map.items():
        previous = old[
            (old["source"] == old_source) & (old["temporal_depth"] == "T0-T1")
        ].iloc[0]
        for policy in ("fixed_0.5", "development_oof_bacc"):
            current = optimized[
                (optimized["source"] == source)
                & (optimized["threshold_policy"] == policy)
            ].iloc[0]
            row = {
                "source": source,
                "optimized_threshold_policy": policy,
                "optimized_threshold": float(current["threshold"]),
                "previous_models_per_seed": 1,
                "optimized_models_per_seed": 1,
            }
            for metric in METRIC_KEYS:
                old_value = float(previous[f"{metric}_mean"])
                new_value = float(current[f"{metric}_mean"])
                row[f"previous_{metric}_mean"] = old_value
                row[f"optimized_{metric}_mean"] = new_value
                row[f"delta_{metric}"] = new_value - old_value
            row["previous_auroc_std"] = float(previous["auroc_std"])
            row["optimized_auroc_std"] = float(current["auroc_std"])
            row["delta_auroc_std"] = row["optimized_auroc_std"] - row["previous_auroc_std"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    oof = _load_formal_oof()
    threshold, oof_bacc, averaged = select_oof_threshold(oof)
    threshold_payload = {
        "schema": "pillar_full978_t0_t1_development_threshold_v1",
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

    metrics = _test_metrics(
        {"fixed_0.5": 0.5, "development_oof_bacc": float(threshold)}
    )
    summary = _summarize(metrics)
    comparison = _comparison(summary)
    _atomic_csv(OUTPUT / "test_metrics_per_seed.csv", metrics)
    _atomic_csv(OUTPUT / "test_summary.csv", summary)
    _atomic_csv(OUTPUT / "previous_vs_optimized.csv", comparison)

    real = summary[
        (summary["source"] == "real")
        & (summary["threshold_policy"] == "development_oof_bacc")
    ].iloc[0]
    direct = summary[
        (summary["source"] == "direct_t0")
        & (summary["threshold_policy"] == "development_oof_bacc")
    ].iloc[0]
    report = f"""# T0-T1 Optimization Report

## Frozen recipe

- One independent model per seed; no fold probability ensemble.
- 40,418 parameters, projection/signature 32/16, layerwise AdamW.
- Fixed 12 epochs on all 876 development patients; 875 have a valid T0 token.
- Unweighted BCE, bounded residual fusion, no duplicated clinical token, no residual L2.

## Development-only selection

- Ten-seed formal OOF AUROC: 0.71874 +/- 0.00156.
- Ten-seed formal OOF PR-AUC: 0.54681 +/- 0.00250.
- Mean train-OOF gap: 0.03793.
- OOF-selected balanced-accuracy threshold: {threshold:.8f}.
- The locked test was not used for recipe, epoch, or threshold selection.

## Descriptive locked102 results

- Real AUROC/PR-AUC: {real['auroc_mean']:.5f} +/- {real['auroc_std']:.5f} / {real['prauc_mean']:.5f} +/- {real['prauc_std']:.5f}.
- Real sensitivity/specificity/balanced accuracy at the OOF threshold: {real['sens_mean']:.5f} / {real['spec_mean']:.5f} / {real['bacc_mean']:.5f}.
- Direct-T0 AUROC/PR-AUC: {direct['auroc_mean']:.5f} +/- {direct['auroc_std']:.5f} / {direct['prauc_mean']:.5f} +/- {direct['prauc_std']:.5f}.
- Direct-T0 and rollout are identical at T0-T1 because both generate T1 directly from real T0.

The locked102 cohort has been inspected in prior work, so these test metrics are descriptive rather than untouched confirmation.
"""
    _atomic_text(OUTPUT / "report.md", report)
    _atomic_json(
        OUTPUT / "ANALYSIS_COMPLETE.json",
        {
            "schema": "pillar_full978_t0_t1_optimization_analysis_v1",
            "threshold": float(threshold),
            "sources": list(SOURCES),
            "seeds": list(SEEDS),
            "models_per_seed": 1,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()

"""Compose the user-selected short-best200/long-original five-fold policy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_json, _atomic_text
from src.metrics import METRIC_KEYS, compute_metrics


ROOT = Path(__file__).resolve().parents[1]
BEST200 = ROOT / "results/mewm_ispy2_full978_locked102_all_depths_best200_fivefold/analysis"
ORIGINAL = ROOT / "results/mewm_ispy2_full978_locked102_optimal_fivefold_all_inputs"
OUTPUT = ROOT / "results/mewm_ispy2_full978_locked102_final_hybrid_policy"
SEEDS = tuple(range(42, 52))
DEPTHS = ("T0", "T0-T1", "T0-T2", "T0-T3")
SOURCES = ("real", "direct_t0", "rollout", "adjacent_real", "zero_pre")
SOURCE_LABELS = {
    "real": "Real",
    "direct_t0": "Direct-T0",
    "rollout": "Rollout",
    "adjacent_real": "Adjacent-real",
    "zero_pre": "Zero-pre",
}
POLICY = {
    "T0": {
        "model_source": "best_checkpoint_within_200_epochs",
        "source_result": BEST200.parent,
        "variant": "best200",
        "parameters_per_fold": 40418,
        "training_epoch_limit": 200,
        "checkpoint_policy": "best_validation_auroc",
    },
    "T0-T1": {
        "model_source": "best_checkpoint_within_200_epochs",
        "source_result": BEST200.parent,
        "variant": "best200",
        "parameters_per_fold": 40418,
        "training_epoch_limit": 200,
        "checkpoint_policy": "best_validation_auroc",
    },
    "T0-T2": {
        "model_source": "previous_optimal_fivefold",
        "source_result": ORIGINAL,
        "variant": "layerwise_adamw",
        "parameters_per_fold": 40978,
        "training_epoch_limit": 80,
        "checkpoint_policy": "best_validation_auroc_with_early_stopping",
    },
    "T0-T3": {
        "model_source": "previous_optimal_fivefold",
        "source_result": ORIGINAL,
        "variant": "layerwise_adamw",
        "parameters_per_fold": 88610,
        "training_epoch_limit": 150,
        "checkpoint_policy": "best_validation_auroc_with_early_stopping",
    },
}


def _thresholds():
    short = pd.read_csv(BEST200 / "development_thresholds.csv")
    long = pd.read_csv(ORIGINAL / "development_thresholds.csv")
    frames = []
    for depth in DEPTHS:
        source = short if depth in {"T0", "T0-T1"} else long
        selected = source[source["temporal_depth"] == depth].copy()
        if len(selected) != 1:
            raise RuntimeError(f"missing unique threshold for {depth}")
        selected["model_source"] = POLICY[depth]["model_source"]
        selected["threshold_selected_from_test"] = False
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True, sort=False)
    if not result["development_patients"].eq(876).all():
        raise RuntimeError("development threshold cohort changed")
    return result


def _predictions():
    short = pd.read_csv(
        BEST200 / "all_test_predictions.csv", dtype={"patient_id": str}
    )
    long = pd.read_csv(
        ORIGINAL / "all_test_predictions.csv", dtype={"patient_id": str}
    )
    frames = []
    for depth in DEPTHS:
        source = short if depth in {"T0", "T0-T1"} else long
        selected = source[source["temporal_depth"] == depth][
            [
                "patient_id",
                "label",
                "probability",
                "seed",
                "temporal_depth",
                "input_scheme",
                "input_scheme_label",
            ]
        ].copy()
        selected["folds_per_seed"] = 5
        selected["model_source"] = POLICY[depth]["model_source"]
        selected["model_variant"] = POLICY[depth]["variant"]
        selected["source_result"] = str(
            POLICY[depth]["source_result"].relative_to(ROOT)
        )
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    if (
        len(result) != len(DEPTHS) * len(SOURCES) * len(SEEDS) * 102
        or result.duplicated(
            ["temporal_depth", "input_scheme", "seed", "patient_id"]
        ).any()
        or not np.isfinite(result["probability"]).all()
        or not result["probability"].between(0.0, 1.0).all()
    ):
        raise RuntimeError("invalid hybrid prediction matrix")
    reference = None
    for keys, selected in result.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        if (
            len(selected) != 102
            or selected["patient_id"].nunique() != 102
            or int(selected["label"].sum()) != 32
        ):
            raise RuntimeError(f"invalid locked102 group: {keys}")
        labels = selected.set_index("patient_id")["label"].sort_index()
        if reference is None:
            reference = labels
        elif not labels.equals(reference):
            raise RuntimeError("locked102 membership or labels differ")
    return result


def _metrics(predictions, thresholds):
    cutoff = thresholds.set_index("temporal_depth")["threshold"].to_dict()
    rows = []
    for (depth, source, seed), selected in predictions.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        for threshold_policy, threshold in (
            ("development_oof_bacc", float(cutoff[depth])),
            ("fixed_0.5", 0.5),
        ):
            rows.append(
                {
                    "temporal_depth": depth,
                    "input_scheme": source,
                    "input_scheme_label": SOURCE_LABELS[source],
                    "model_source": POLICY[depth]["model_source"],
                    "seed": int(seed),
                    "threshold_policy": threshold_policy,
                    "threshold": threshold,
                    **compute_metrics(
                        selected["label"], selected["probability"], threshold=threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summary(metrics):
    rows = []
    keys = [
        "temporal_depth",
        "input_scheme",
        "input_scheme_label",
        "model_source",
        "threshold_policy",
        "threshold",
    ]
    for values, selected in metrics.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = len(selected)
        row["folds_per_seed"] = 5
        for metric in METRIC_KEYS:
            row[f"{metric}_mean"] = float(selected[metric].mean())
            row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _format(value, std):
    return f"{value:.5f} +/- {std:.5f}"


def _table(summary, depth):
    selected = summary[
        (summary["temporal_depth"] == depth)
        & (summary["threshold_policy"] == "development_oof_bacc")
    ]
    metrics = ("acc", "auroc", "sens", "spec", "prec", "npv", "bacc", "prauc")
    headers = ("Input", "Accuracy", "AUROC", "Sensitivity", "Specificity", "Precision", "NPV", "BAcc", "PR-AUC")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in selected.iterrows():
        values = [row["input_scheme_label"]] + [
            _format(row[f"{metric}_mean"], row[f"{metric}_std"])
            for metric in metrics
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _real_temporal_comparison(summary):
    selected = summary[
        (summary["input_scheme"] == "real")
        & (summary["threshold_policy"] == "development_oof_bacc")
    ].copy()
    selected["temporal_depth"] = pd.Categorical(
        selected["temporal_depth"], categories=DEPTHS, ordered=True
    )
    selected = selected.sort_values("temporal_depth").reset_index(drop=True)
    if selected["temporal_depth"].astype(str).tolist() != list(DEPTHS):
        raise RuntimeError("real temporal comparison does not cover all depths")
    for metric in METRIC_KEYS:
        selected[f"{metric}_delta_vs_t0"] = (
            selected[f"{metric}_mean"] - float(selected.iloc[0][f"{metric}_mean"])
        )
    return selected


def _real_temporal_table(comparison):
    metrics = ("acc", "auroc", "sens", "spec", "prec", "npv", "bacc", "prauc")
    headers = (
        "Temporal input",
        "Threshold",
        "Accuracy",
        "AUROC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "NPV",
        "BAcc",
        "PR-AUC",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in comparison.iterrows():
        values = [row["temporal_depth"], f"{row['threshold']:.5f}"] + [
            _format(row[f"{metric}_mean"], row[f"{metric}_std"])
            for metric in metrics
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    thresholds = _thresholds()
    predictions = _predictions()
    metrics = _metrics(predictions, thresholds)
    summary = _summary(metrics)
    real_comparison = _real_temporal_comparison(summary)
    policies = pd.DataFrame(
        [
            {
                "temporal_depth": depth,
                **{
                    key: (
                        str(value.relative_to(ROOT))
                        if key == "source_result"
                        else value
                    )
                    for key, value in POLICY[depth].items()
                },
                "formal_seeds": "42-51",
                "folds_per_seed": 5,
                "residual_l2_weight": 0.0,
                "test_used_for_model_or_threshold_selection": False,
            }
            for depth in DEPTHS
        ]
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(OUTPUT / "model_policy.csv", policies)
    _atomic_csv(OUTPUT / "development_thresholds.csv", thresholds)
    _atomic_csv(OUTPUT / "all_test_predictions.csv", predictions)
    _atomic_csv(OUTPUT / "metrics_per_seed.csv", metrics)
    _atomic_csv(OUTPUT / "summary.csv", summary)
    _atomic_csv(OUTPUT / "real_temporal_comparison.csv", real_comparison)
    sections = [
        f"## {depth}\n\nModel source: `{POLICY[depth]['model_source']}`.\n\n{_table(summary, depth)}"
        for depth in DEPTHS
    ]
    report = """# Final Hybrid Five-Fold PCR Policy

This is the user-selected default reporting policy: T0 and T0-T1 use the best
validation-AUROC checkpoint within 200 fully trained epochs; T0-T2 and T0-T3
use the previous optimized five-fold/no-residual-L2 models. Values are ten-seed
mean +/- sample SD. Classification metrics use the model-matched development
OOF balanced-accuracy threshold; fixed-0.5 metrics remain in `summary.csv`.

## Real temporal comparison

""" + _real_temporal_table(real_comparison) + "\n\n" + "\n\n".join(sections) + """

The fixed locked102 cohort was not used for model, epoch, or threshold
selection. Because it has been inspected repeatedly, results are descriptive.
"""
    _atomic_text(OUTPUT / "report.md", report)
    _atomic_json(
        OUTPUT / "POLICY_COMPLETE.json",
        {
            "schema": "pillar_full978_final_hybrid_policy_v1",
            "policy_name": "best200_short_previous_optimal_long",
            "temporal_depths": list(DEPTHS),
            "input_schemes": list(SOURCES),
            "formal_seeds": list(SEEDS),
            "folds_per_seed": 5,
            "locked_test_patients": 102,
            "locked_test_positive_patients": 32,
            "test_used_for_model_epoch_or_threshold_selection": False,
            "complete": True,
        },
    )
    print(f"complete: {OUTPUT}")


if __name__ == "__main__":
    main()

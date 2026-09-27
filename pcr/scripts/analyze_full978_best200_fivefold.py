"""Audit and summarize the all-depth best-within-200-epochs experiment."""

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
RESULT = ROOT / "results/mewm_ispy2_full978_locked102_all_depths_best200_fivefold"
BASELINE = ROOT / "results/mewm_ispy2_full978_locked102_optimal_fivefold_all_inputs"
OUTPUT = RESULT / "analysis"
SEEDS = tuple(range(42, 52))
DEPTHS = ("T0", "T0-T1", "T0-T2", "T0-T3")
SOURCE_DIRS = {
    "real": "real",
    "direct_t0": "direct_t0",
    "rollout": "rollout_generated_dce0_real_ser",
    "adjacent_real": "adjacent_real",
    "zero_pre": "zero_pre",
}
SOURCE_LABELS = {
    "real": "Real",
    "direct_t0": "Direct-T0",
    "rollout": "Rollout",
    "adjacent_real": "Adjacent-real",
    "zero_pre": "Zero-pre",
}


def _slug(value):
    return value.lower().replace("-", "_")


def _load_oof(depth):
    frames = []
    root = RESULT / f"oof/formal/{_slug(depth)}/variant_best200"
    for seed in SEEDS:
        path = root / f"seed_{seed}/oof_predictions.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str})
        if (
            len(frame) != 876
            or frame["patient_id"].duplicated().any()
            or set(frame["seed"].astype(int)) != {seed}
            or set(frame["temporal_depth"]) != {depth}
            or int(frame["label"].sum()) != 282
            or not np.isfinite(frame["probability"]).all()
        ):
            raise RuntimeError(f"invalid OOF predictions: {path}")
        frames.append(frame[["patient_id", "label", "probability", "seed"]])
    combined = pd.concat(frames, ignore_index=True)
    reference = set(frames[0]["patient_id"])
    if any(set(frame["patient_id"]) != reference for frame in frames[1:]):
        raise RuntimeError(f"OOF membership differs across seeds at {depth}")
    return combined


def _thresholds():
    rows = []
    for depth in DEPTHS:
        threshold, bacc, averaged = select_oof_threshold(_load_oof(depth))
        metrics = compute_metrics(
            averaged["label"], averaged["probability"], threshold=threshold
        )
        rows.append(
            {
                "temporal_depth": depth,
                "threshold": float(threshold),
                "criterion": "maximum_balanced_accuracy",
                "tie_break": "closest_to_0.5_then_lower_threshold",
                "development_patients": len(averaged),
                "development_positive_patients": int(averaged["label"].sum()),
                "oof_balanced_accuracy_at_selection": float(bacc),
                **{f"oof_{metric}": value for metric, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def _load_predictions():
    frames = []
    for source, source_dir in SOURCE_DIRS.items():
        path = RESULT / f"evaluation/{source_dir}/all_test_predictions.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str})
        if len(frame) != len(DEPTHS) * len(SEEDS) * 102:
            raise RuntimeError(f"unexpected prediction count: {path}")
        frame["input_scheme"] = source
        frame["input_scheme_label"] = SOURCE_LABELS[source]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if (
        len(combined) != len(DEPTHS) * len(SOURCE_DIRS) * len(SEEDS) * 102
        or combined.duplicated(
            ["temporal_depth", "input_scheme", "seed", "patient_id"]
        ).any()
        or not np.isfinite(combined["probability"]).all()
        or not combined["probability"].between(0.0, 1.0).all()
    ):
        raise RuntimeError("invalid consolidated test predictions")
    for keys, selected in combined.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        if (
            len(selected) != 102
            or selected["patient_id"].nunique() != 102
            or int(selected["label"].sum()) != 32
        ):
            raise RuntimeError(f"locked102 contract changed: {keys}")
    reference = combined[
        (combined["temporal_depth"] == "T0")
        & (combined["input_scheme"] == "real")
        & (combined["seed"] == SEEDS[0])
    ].set_index("patient_id")["label"]
    for _, selected in combined.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        labels = selected.set_index("patient_id")["label"].sort_index()
        if not labels.equals(reference.sort_index()):
            raise RuntimeError("locked102 membership or labels differ")

    t0 = combined[combined["temporal_depth"] == "T0"].pivot(
        index=["patient_id", "seed"], columns="input_scheme", values="probability"
    )
    if any(
        not np.array_equal(t0["real"].to_numpy(), t0[source].to_numpy())
        for source in SOURCE_DIRS
        if source != "real"
    ):
        raise RuntimeError("T0 must be identical for every future-input scheme")
    t01 = combined[combined["temporal_depth"] == "T0-T1"].pivot(
        index=["patient_id", "seed"], columns="input_scheme", values="probability"
    )
    if not np.array_equal(t01["direct_t0"].to_numpy(), t01["rollout"].to_numpy()):
        raise RuntimeError("Direct-T0 and rollout must be identical at T0-T1")
    return combined


def _audit_fold_ensembles(predictions):
    for (depth, source, seed), selected in predictions.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        source_dir = SOURCE_DIRS[source]
        run_dir = RESULT / f"evaluation/{source_dir}/{_slug(depth)}/seed_{seed}"
        folds = pd.read_csv(
            run_dir / "fold_predictions.csv", dtype={"patient_id": str}
        )
        if len(folds) != 510 or folds.groupby("patient_id")["fold"].nunique().ne(5).any():
            raise RuntimeError(f"invalid fold predictions: {run_dir}")
        averaged = folds.groupby("patient_id", sort=True)["probability"].mean()
        saved = selected.set_index("patient_id")["probability"].sort_index()
        # The runner averages float32 tensors before CSV serialization.
        if not np.allclose(averaged, saved, rtol=0.0, atol=5e-7):
            raise RuntimeError(f"fold averaging mismatch: {depth}/{source}/{seed}")


def _epoch_audit():
    rows = []
    for depth in DEPTHS:
        root = RESULT / f"train/formal/{_slug(depth)}/variant_best200"
        for seed in SEEDS:
            for fold in range(5):
                run_dir = root / f"seed_{seed}/fold_{fold}"
                history = pd.read_csv(run_dir / "history.csv")
                summary = json.loads((run_dir / "summary.json").read_text())
                selection = summary["selection"]
                if (
                    len(history) != 200
                    or history["epoch"].astype(int).tolist() != list(range(200))
                    or int(selection["epochs_trained"]) != 200
                    or selection["checkpoint_policy"] != "best_validation"
                    or not np.isfinite(history[["train_loss", "validation_auroc"]]).all().all()
                ):
                    raise RuntimeError(f"best200 epoch contract changed: {run_dir}")
                best_epoch = int(selection["best_epoch_zero_based"])
                scores = history["validation_auroc"].to_numpy(dtype=float)
                if not np.isclose(scores[best_epoch], scores.max()):
                    raise RuntimeError(f"saved checkpoint is not validation peak: {run_dir}")
                rows.append(
                    {
                        "temporal_depth": depth,
                        "seed": seed,
                        "fold": fold,
                        "epochs_trained": 200,
                        "selected_epoch_one_based": best_epoch + 1,
                        "best_validation_auroc": float(selection["best_score"]),
                        "parameters": int(summary["parameters"]),
                    }
                )
    return pd.DataFrame(rows)


def _metrics(predictions, thresholds):
    threshold_map = thresholds.set_index("temporal_depth")["threshold"].to_dict()
    rows = []
    for (depth, source, seed), selected in predictions.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        for policy, threshold in (
            ("development_oof_bacc", float(threshold_map[depth])),
            ("fixed_0.5", 0.5),
        ):
            rows.append(
                {
                    "temporal_depth": depth,
                    "input_scheme": source,
                    "input_scheme_label": SOURCE_LABELS[source],
                    "seed": int(seed),
                    "threshold_policy": policy,
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


def _epoch_summary(epochs):
    return (
        epochs.groupby("temporal_depth", sort=False)
        .agg(
            fold_models=("selected_epoch_one_based", "size"),
            selected_epoch_mean=("selected_epoch_one_based", "mean"),
            selected_epoch_std=("selected_epoch_one_based", "std"),
            selected_epoch_median=("selected_epoch_one_based", "median"),
            selected_epoch_min=("selected_epoch_one_based", "min"),
            selected_epoch_max=("selected_epoch_one_based", "max"),
            validation_auroc_mean=("best_validation_auroc", "mean"),
            parameters_per_fold=("parameters", "first"),
        )
        .reset_index()
    )


def _baseline_comparison(summary):
    previous = pd.read_csv(BASELINE / "summary.csv")
    previous = previous[previous["threshold_policy"] == "development_oof_bacc"].copy()
    current = summary[summary["threshold_policy"] == "development_oof_bacc"].copy()
    merged = current.merge(
        previous,
        on=["temporal_depth", "input_scheme"],
        suffixes=("_best200", "_previous"),
        validate="one_to_one",
    )
    rows = merged[["temporal_depth", "input_scheme"]].copy()
    for metric in METRIC_KEYS:
        rows[f"previous_{metric}_mean"] = merged[f"{metric}_mean_previous"]
        rows[f"best200_{metric}_mean"] = merged[f"{metric}_mean_best200"]
        rows[f"best200_minus_previous_{metric}"] = (
            merged[f"{metric}_mean_best200"] - merged[f"{metric}_mean_previous"]
        )
    return rows


def _format(value, std):
    return f"{value:.5f} +/- {std:.5f}"


def _table(summary, depth, policy):
    selected = summary[
        (summary["temporal_depth"] == depth)
        & (summary["threshold_policy"] == policy)
    ]
    headers = ["Input", "Accuracy", "AUROC", "Sensitivity", "Specificity", "Precision", "NPV", "BAcc", "PR-AUC"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    names = ("acc", "auroc", "sens", "spec", "prec", "npv", "bacc", "prauc")
    for _, row in selected.iterrows():
        values = [row["input_scheme_label"]] + [
            _format(row[f"{metric}_mean"], row[f"{metric}_std"])
            for metric in names
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    completion = json.loads((RESULT / "EXPERIMENT_COMPLETE.json").read_text())
    if (
        completion.get("complete") is not True
        or completion.get("test_used_for_selection") is not False
        or completion.get("folds_per_seed") != 5
    ):
        raise RuntimeError("formal experiment completion contract changed")
    thresholds = _thresholds()
    predictions = _load_predictions()
    _audit_fold_ensembles(predictions)
    epochs = _epoch_audit()
    metrics = _metrics(predictions, thresholds)
    summary = _summary(metrics)
    epoch_summary = _epoch_summary(epochs)
    comparison = _baseline_comparison(summary)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(OUTPUT / "development_thresholds.csv", thresholds)
    _atomic_csv(OUTPUT / "all_test_predictions.csv", predictions)
    _atomic_csv(OUTPUT / "metrics_per_seed.csv", metrics)
    _atomic_csv(OUTPUT / "summary.csv", summary)
    _atomic_csv(OUTPUT / "selected_epochs_per_fold.csv", epochs)
    _atomic_csv(OUTPUT / "selected_epoch_summary.csv", epoch_summary)
    _atomic_csv(OUTPUT / "previous_vs_best200.csv", comparison)

    sections = []
    for depth in DEPTHS:
        sections.append(
            f"## {depth}: development-OOF threshold\n\n"
            f"{_table(summary, depth, 'development_oof_bacc')}"
        )
    report = """# Best Checkpoint Within 200 Epochs: Five-Fold Results

Each fold trained for exactly 200 epochs. The checkpoint with the highest fold
validation AUROC was retained, five fold probabilities were averaged per seed,
and results are mean +/- sample SD across seeds 42-51. Classification metrics
below use a depth-specific threshold selected only from development OOF
predictions. Fixed-0.5 results are also retained in `summary.csv`.

""" + "\n\n".join(sections) + """

The locked102 cohort was not used for checkpoint or threshold selection, but it
has been inspected in earlier work, so these remain descriptive results.
"""
    _atomic_text(OUTPUT / "report.md", report)
    _atomic_json(
        OUTPUT / "ANALYSIS_COMPLETE.json",
        {
            "schema": "pillar_full978_best200_fivefold_analysis_v1",
            "formal_fold_models": 200,
            "epochs_trained_per_fold": 200,
            "formal_seeds": list(SEEDS),
            "folds_per_seed": 5,
            "locked_test_patients": 102,
            "locked_test_positive_patients": 32,
            "test_used_for_checkpoint_or_threshold_selection": False,
            "complete": True,
        },
    )
    print(f"complete: {OUTPUT}")


if __name__ == "__main__":
    main()

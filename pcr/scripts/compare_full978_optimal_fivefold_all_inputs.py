"""Consolidate every locked102 input arm under the frozen five-fold policies.

Threshold-independent and threshold-dependent metrics are recomputed from the
per-patient probabilities.  Classification thresholds are selected only from
the ten-seed mean development OOF probabilities for each temporal depth.
"""

from __future__ import annotations

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
OUTPUT = ROOT / "results/mewm_ispy2_full978_locked102_optimal_fivefold_all_inputs"
SEEDS = tuple(range(42, 52))
DEPTHS = ("T0", "T0-T1", "T0-T2", "T0-T3")
SOURCES = ("real", "direct_t0", "rollout", "adjacent_real", "zero_pre")

T0_ROOT = ROOT / "results/mewm_ispy2_full978_locked102_t0_fivefold_optimization"
T01_ROOT = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch12_fivefold_evaluation"
)
MIXED_ROOT = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_frozen_mixed_no_residual_biflow_trajectories"
)
T2_ADJACENT_ROOT = (
    ROOT / "results/mewm_ispy2_full978_locked102_independent_cv_regularization_v2"
)
T3_ADJACENT_ROOT = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_t0_t3_layerwise_adamw_no_residual_10seed"
)
ZERO_ROOT = (
    ROOT / "results/mewm_ispy2_full978_locked102_generated_pre_zero_ablation"
)

OOF_SPECS = {
    "T0": T0_ROOT / "oof/formal/t0/variant_ultra_bounded_unweighted_fixed6",
    "T0-T1": (
        ROOT
        / "results/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch"
        / "oof/formal/t0_t1/variant_fixed_epoch12"
    ),
    "T0-T2": (
        T2_ADJACENT_ROOT / "oof/formal/t0_t2/variant_layerwise_adamw"
    ),
    "T0-T3": (
        T3_ADJACENT_ROOT / "oof/formal/t0_t3/variant_layerwise_adamw"
    ),
}

MODEL_POLICIES = {
    "T0": {
        "variant": "ultra_bounded_unweighted_fixed6",
        "parameters_per_fold": 40418,
        "epochs": 6,
        "checkpoint_policy": "final_epoch",
        "residual_l2_weight": 0.0,
    },
    "T0-T1": {
        "variant": "fixed_epoch12",
        "parameters_per_fold": 40418,
        "epochs": 12,
        "checkpoint_policy": "final_epoch",
        "residual_l2_weight": 0.0,
    },
    "T0-T2": {
        "variant": "layerwise_adamw",
        "parameters_per_fold": 40978,
        "epochs": 80,
        "checkpoint_policy": "best_validation",
        "residual_l2_weight": 0.0,
    },
    "T0-T3": {
        "variant": "layerwise_adamw",
        "parameters_per_fold": 88610,
        "epochs": 150,
        "checkpoint_policy": "best_validation",
        "residual_l2_weight": 0.0,
    },
}

SOURCE_LABELS = {
    "real": "Real",
    "direct_t0": "Direct-T0",
    "rollout": "Rollout",
    "adjacent_real": "Adjacent-real",
    "zero_pre": "Zero-pre",
}


def _spec(path, original_source):
    return {"path": Path(path), "original_source": original_source}


PREDICTION_SPECS = {
    "T0": {
        source: _spec(
            T0_ROOT / "evaluation/real/all_test_predictions.csv", "real"
        )
        for source in SOURCES
    },
    "T0-T1": {
        "real": _spec(T01_ROOT / "evaluation/real/all_test_predictions.csv", "real"),
        "direct_t0": _spec(
            T01_ROOT / "evaluation/direct_t0/all_test_predictions.csv", "direct_t0"
        ),
        "rollout": _spec(
            T01_ROOT
            / "evaluation/rollout_generated_dce0_real_ser/all_test_predictions.csv",
            "rollout_generated_dce0_real_ser",
        ),
        "adjacent_real": _spec(
            T01_ROOT / "evaluation/adjacent_real/all_test_predictions.csv",
            "adjacent_real",
        ),
        "zero_pre": _spec(
            T01_ROOT / "evaluation/zero_pre/all_test_predictions.csv", "zero_pre"
        ),
    },
    "T0-T2": {
        "real": _spec(MIXED_ROOT / "all_test_predictions.csv", "real"),
        "direct_t0": _spec(MIXED_ROOT / "all_test_predictions.csv", "direct_t0"),
        "rollout": _spec(
            MIXED_ROOT / "all_test_predictions.csv",
            "rollout_generated_dce0_real_ser",
        ),
        "adjacent_real": _spec(
            T2_ADJACENT_ROOT / "evaluation/generated/all_test_predictions.csv",
            "generated",
        ),
        "zero_pre": _spec(ZERO_ROOT / "all_test_predictions.csv", "zero_pre"),
    },
    "T0-T3": {
        "real": _spec(MIXED_ROOT / "all_test_predictions.csv", "real"),
        "direct_t0": _spec(MIXED_ROOT / "all_test_predictions.csv", "direct_t0"),
        "rollout": _spec(
            MIXED_ROOT / "all_test_predictions.csv",
            "rollout_generated_dce0_real_ser",
        ),
        "adjacent_real": _spec(
            T3_ADJACENT_ROOT / "evaluation/generated/all_test_predictions.csv",
            "generated",
        ),
        "zero_pre": _spec(ZERO_ROOT / "all_test_predictions.csv", "zero_pre"),
    },
}


def _load_oof(depth):
    frames = []
    for seed in SEEDS:
        path = OOF_SPECS[depth] / f"seed_{seed}/oof_predictions.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str})
        if (
            len(frame) != 876
            or frame["patient_id"].duplicated().any()
            or set(frame["seed"].astype(int)) != {seed}
            or int(frame["label"].sum()) != 282
            or set(frame["temporal_depth"]) != {depth}
            or not np.isfinite(frame["probability"]).all()
        ):
            raise RuntimeError(f"invalid development OOF predictions: {path}")
        frames.append(frame[["patient_id", "label", "probability", "seed"]])
    combined = pd.concat(frames, ignore_index=True)
    expected = set(frames[0]["patient_id"])
    for seed, frame in combined.groupby("seed"):
        if set(frame["patient_id"]) != expected:
            raise RuntimeError(f"development OOF patient mismatch at {depth}, seed {seed}")
    return combined


def _development_thresholds():
    rows = []
    for depth in DEPTHS:
        oof = _load_oof(depth)
        threshold, bacc, averaged = select_oof_threshold(oof)
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
                **{f"oof_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def _load_test_prediction(depth, source):
    spec = PREDICTION_SPECS[depth][source]
    frame = pd.read_csv(spec["path"], dtype={"patient_id": str})
    frame = frame[
        (frame["temporal_depth"] == depth)
        & (frame["source"] == spec["original_source"])
    ].copy()
    if (
        len(frame) != 1020
        or frame.duplicated(["patient_id", "seed"]).any()
        or set(frame["seed"].astype(int)) != set(SEEDS)
        or not np.isfinite(frame["probability"]).all()
        or not frame["probability"].between(0.0, 1.0).all()
    ):
        raise RuntimeError(f"invalid test predictions for {depth}/{source}")
    for seed, selected in frame.groupby("seed"):
        if (
            len(selected) != 102
            or selected["patient_id"].duplicated().any()
            or int(selected["label"].sum()) != 32
        ):
            raise RuntimeError(f"invalid locked102 cohort for {depth}/{source}/{seed}")
    frame["input_scheme"] = source
    frame["input_scheme_label"] = SOURCE_LABELS[source]
    frame["original_source"] = spec["original_source"]
    frame["source_file"] = str(spec["path"].relative_to(ROOT))
    frame["folds_per_seed"] = 5
    frame["model_variant"] = MODEL_POLICIES[depth]["variant"]
    return frame[
        [
            "patient_id",
            "label",
            "probability",
            "seed",
            "temporal_depth",
            "input_scheme",
            "input_scheme_label",
            "folds_per_seed",
            "model_variant",
            "original_source",
            "source_file",
        ]
    ]


def _load_all_test_predictions():
    frames = [
        _load_test_prediction(depth, source)
        for depth in DEPTHS
        for source in SOURCES
    ]
    canonical = frames[0].sort_values(["seed", "patient_id"])
    canonical_ids = canonical.groupby("seed")["patient_id"].apply(tuple).iloc[0]
    canonical_labels = dict(zip(canonical["patient_id"], canonical["label"]))
    for frame in frames:
        ordered = frame.sort_values(["seed", "patient_id"])
        for _, selected in ordered.groupby("seed", sort=True):
            if tuple(selected["patient_id"]) != canonical_ids:
                raise RuntimeError("locked102 membership differs across inputs")
            if any(
                canonical_labels[patient_id] != label
                for patient_id, label in zip(selected["patient_id"], selected["label"])
            ):
                raise RuntimeError("locked102 labels differ across inputs")
    combined = pd.concat(frames, ignore_index=True)

    t0 = combined[combined["temporal_depth"] == "T0"].pivot(
        index=["patient_id", "seed"], columns="input_scheme", values="probability"
    )
    if not all(np.array_equal(t0["real"].to_numpy(), t0[source].to_numpy()) for source in SOURCES[1:]):
        raise RuntimeError("T0 predictions must be identical for every future-input arm")
    t01 = combined[combined["temporal_depth"] == "T0-T1"].pivot(
        index=["patient_id", "seed"], columns="input_scheme", values="probability"
    )
    if not np.array_equal(t01["direct_t0"].to_numpy(), t01["rollout"].to_numpy()):
        raise RuntimeError("Direct-T0 and rollout must be identical at T0-T1")
    return combined


def _metrics(predictions, thresholds):
    cutoff = thresholds.set_index("temporal_depth")["threshold"].to_dict()
    rows = []
    for (depth, source, seed), selected in predictions.groupby(
        ["temporal_depth", "input_scheme", "seed"], sort=False
    ):
        for policy, threshold in (
            ("development_oof_bacc", float(cutoff[depth])),
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


def _summarize(metrics):
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


def _all_seed_ensemble(predictions, thresholds):
    cutoff = thresholds.set_index("temporal_depth")["threshold"].to_dict()
    rows = []
    for (depth, source), selected in predictions.groupby(
        ["temporal_depth", "input_scheme"], sort=False
    ):
        averaged = selected.groupby(
            ["patient_id", "label"], as_index=False, sort=True
        )["probability"].mean()
        for policy, threshold in (
            ("development_oof_bacc", float(cutoff[depth])),
            ("fixed_0.5", 0.5),
        ):
            rows.append(
                {
                    "temporal_depth": depth,
                    "input_scheme": source,
                    "input_scheme_label": SOURCE_LABELS[source],
                    "threshold_policy": policy,
                    "threshold": threshold,
                    "models_ensembled": 50,
                    **compute_metrics(
                        averaged["label"], averaged["probability"], threshold=threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def _formatted(value, std):
    return f"{value:.5f} +/- {std:.5f}"


def _markdown_table(summary, metrics):
    selected = summary[summary["threshold_policy"] == "development_oof_bacc"]
    headers = ["Temporal input", "Input scheme", *(metric.upper() for metric in metrics)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in selected.iterrows():
        values = [
            row["temporal_depth"],
            row["input_scheme_label"],
            *(
                _formatted(row[f"{metric}_mean"], row[f"{metric}_std"])
                for metric in metrics
            ),
        ]
        lines.append(
            "| " + " | ".join(str(value) for value in values) + " |"
        )
    return "\n".join(lines)


def main():
    thresholds = _development_thresholds()
    predictions = _load_all_test_predictions()
    metrics = _metrics(predictions, thresholds)
    summary = _summarize(metrics)
    all_seed = _all_seed_ensemble(predictions, thresholds)

    policies = pd.DataFrame(
        [
            {
                "temporal_depth": depth,
                "folds_per_seed": 5,
                "formal_seeds": "42-51",
                **MODEL_POLICIES[depth],
                "test_used_for_model_or_threshold_selection": False,
            }
            for depth in DEPTHS
        ]
    )
    source_manifest = pd.DataFrame(
        [
            {
                "temporal_depth": depth,
                "input_scheme": source,
                "input_scheme_label": SOURCE_LABELS[source],
                "source_file": str(PREDICTION_SPECS[depth][source]["path"].relative_to(ROOT)),
                "original_source": PREDICTION_SPECS[depth][source]["original_source"],
            }
            for depth in DEPTHS
            for source in SOURCES
        ]
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(OUTPUT / "development_thresholds.csv", thresholds)
    _atomic_csv(OUTPUT / "model_policies.csv", policies)
    _atomic_csv(OUTPUT / "source_manifest.csv", source_manifest)
    _atomic_csv(OUTPUT / "all_test_predictions.csv", predictions)
    _atomic_csv(OUTPUT / "metrics_per_seed.csv", metrics)
    _atomic_csv(OUTPUT / "summary.csv", summary)
    _atomic_csv(OUTPUT / "all_seed_ensemble_metrics.csv", all_seed)

    report = f"""# Optimal Five-Fold Input Comparison

Every row uses ten seeds (42-51), with five fold-model probabilities averaged
within each seed. Values are mean +/- sample SD across the ten seed-level test
metrics. Classification metrics use a temporal-depth-specific threshold chosen
from development OOF predictions only. AUROC and PR-AUC are threshold-free.

## Discrimination

{_markdown_table(summary, ("auroc", "prauc"))}

## Threshold-dependent metrics

{_markdown_table(summary, ("acc", "sens", "spec", "prec", "npv", "bacc"))}

T0 is intentionally identical across all five schemes because only future
visits are modified. Direct-T0 and rollout are also identical at T0-T1 because
both generate the first future visit from real T0. The fixed-0.5 results are
retained in `summary.csv` as a secondary operating-point analysis.

The locked102 cohort has already been inspected repeatedly. These results are
descriptive comparisons and are not an untouched confirmatory evaluation.
"""
    _atomic_text(OUTPUT / "report.md", report)
    _atomic_json(
        OUTPUT / "EXPERIMENT_COMPLETE.json",
        {
            "schema": "pillar_full978_optimal_fivefold_all_inputs_v1",
            "temporal_depths": list(DEPTHS),
            "input_schemes": list(SOURCES),
            "formal_seeds": list(SEEDS),
            "folds_per_seed": 5,
            "locked_test_patients": 102,
            "locked_test_positive_patients": 32,
            "threshold_source": "ten_seed_mean_development_oof_predictions",
            "test_used_for_model_or_threshold_selection": False,
            "complete": True,
        },
    )
    print(f"complete: {OUTPUT}")


if __name__ == "__main__":
    main()

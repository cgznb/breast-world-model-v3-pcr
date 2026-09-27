"""Build the final full978 optimization comparison from saved artifacts only.

The fixed-epoch policy decision is loaded and validated before any locked-test
artifact is opened. This script does not train, select, or run inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pillar_full978_final_optimization_comparison_v1"
SEEDS = tuple(range(42, 52))
DEPTHS = (("T0", 1), ("T0-T1", 2), ("T0-T2", 3), ("T0-T3", 4))
DEPTH_TO_MAX_TP = dict(DEPTHS)
METRICS = ("acc", "auroc", "sens", "spec", "prec", "npv", "bacc", "prauc")

DEFAULT_RESULT_DIRS = {
    "independent_no_cv": ROOT / "results/mewm_ispy2_full978_locked102_dual_strategy",
    "independent_old_cv": ROOT / "results/mewm_ispy2_full978_locked102_independent_cv",
    "independent_regularization_v2": (
        ROOT / "results/mewm_ispy2_full978_locked102_independent_cv_regularization_v2"
    ),
    "independent_fixed_epoch": (
        ROOT / "results/mewm_ispy2_full978_locked102_independent_cv_fixed_epoch"
    ),
    "shared_random_prefix_cv": ROOT / "results/mewm_ispy2_full978_locked102_anti_overfit",
    "shared_contiguous_validation_peak": (
        ROOT / "results/mewm_ispy2_full978_locked102_shared_contiguous_cv"
    ),
    "shared_contiguous_fixed12": (
        ROOT / "results/mewm_ispy2_full978_locked102_shared_contiguous_cv_fixed_epoch12"
    ),
}
DEFAULT_POLICY_PATH = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_fixed_epoch_policy_selection"
    / "analysis/fixed_epoch_policy_selection/checkpoint_policy_selection.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_fixed_epoch_policy_selection"
    / "analysis/final_optimization_comparison"
)
OUTPUT_NAMES = (
    "final_optimization_comparison.csv",
    "final_optimization_comparison.json",
    "final_optimization_comparison.md",
)


STRATEGY_SPECS: dict[str, dict[str, Any]] = {
    "independent_no_cv": {
        "display_name": "Independent fixed split (no CV)",
        "family": "independent",
        "weight_sharing": "separate_parameter_set_per_depth",
        "depths": tuple(name for name, _ in DEPTHS),
        "development_protocol": "fixed_778_train_98_validation",
        "development_estimate": "heldout_validation_not_oof",
        "development_n": 98,
        "folds": 1,
        "fold_ensemble": False,
        "checkpoint_policy": "best_validation",
        "prefix_training_policy": "one_target_depth_per_parameter_set",
        "selection_status": "historical_no_cv_baseline_outside_fixed_epoch_decision",
        "frozen_policy_selected": None,
        "test_evaluated": True,
        "test_subdir": "independent/evaluation",
        "expected_parameters": {
            "T0": 203842,
            "T0-T1": 203842,
            "T0-T2": 203842,
            "T0-T3": 88610,
        },
    },
    "independent_old_cv": {
        "display_name": "Independent old five-fold CV",
        "family": "independent",
        "weight_sharing": "separate_parameter_set_per_depth",
        "depths": ("T0-T2", "T0-T3"),
        "development_protocol": "development_876_five_fold_oof",
        "development_estimate": "five_fold_oof",
        "development_n": 876,
        "folds": 5,
        "fold_ensemble": True,
        "checkpoint_policy": "best_validation",
        "prefix_training_policy": "one_target_depth_per_parameter_set",
        "selection_status": "historical_pre_v2_cv_reference",
        "frozen_policy_selected": None,
        "test_evaluated": True,
        "test_subdir": "evaluation",
        "expected_parameters": {"T0-T2": 40978, "T0-T3": 88610},
    },
    "independent_regularization_v2": {
        "display_name": "Independent regularization v2 (selected)",
        "family": "independent",
        "weight_sharing": "separate_parameter_set_per_depth",
        "depths": ("T0-T2", "T0-T3"),
        "development_protocol": "development_876_five_fold_oof",
        "development_estimate": "five_fold_oof",
        "development_n": 876,
        "folds": 5,
        "fold_ensemble": True,
        "checkpoint_policy": "best_validation",
        "prefix_training_policy": "one_target_depth_per_parameter_set",
        "selection_status": "frozen_independent_winner",
        "frozen_policy_selected": True,
        "test_evaluated": True,
        "test_subdir": "evaluation",
        "expected_parameters": {"T0-T2": 40978, "T0-T3": 88610},
    },
    "independent_fixed_epoch": {
        "display_name": "Independent fixed epoch (development only)",
        "family": "independent",
        "weight_sharing": "separate_parameter_set_per_depth",
        "depths": ("T0-T2", "T0-T3"),
        "development_protocol": "development_876_five_fold_oof",
        "development_estimate": "five_fold_oof",
        "development_n": 876,
        "folds": 5,
        "fold_ensemble": True,
        "checkpoint_policy": "final_epoch",
        "prefix_training_policy": "one_target_depth_per_parameter_set",
        "selection_status": "fixed_epoch_candidate_rejected_by_frozen_development_rule",
        "frozen_policy_selected": False,
        "test_evaluated": False,
        "test_subdir": None,
        "expected_parameters": {"T0-T2": 40978, "T0-T3": 88610},
    },
    "shared_random_prefix_cv": {
        "display_name": "Shared old random-prefix five-fold CV",
        "family": "shared",
        "weight_sharing": "one_parameter_set_shared_across_all_depths",
        "depths": tuple(name for name, _ in DEPTHS),
        "development_protocol": "development_876_five_fold_oof",
        "development_estimate": "five_fold_oof",
        "development_n": 876,
        "folds": 5,
        "fold_ensemble": True,
        "checkpoint_policy": "best_validation_macro",
        "prefix_training_policy": "full_sequence_plus_one_random_prefix_per_batch",
        "selection_status": "historical_random_prefix_reference_outside_fixed_epoch_decision",
        "frozen_policy_selected": None,
        "test_evaluated": True,
        "test_subdir": "evaluation",
        "expected_parameters": {name: 203842 for name, _ in DEPTHS},
    },
    "shared_contiguous_validation_peak": {
        "display_name": "Shared contiguous validation peak",
        "family": "shared",
        "weight_sharing": "one_parameter_set_shared_across_all_depths",
        "depths": tuple(name for name, _ in DEPTHS),
        "development_protocol": "development_876_five_fold_oof",
        "development_estimate": "five_fold_oof",
        "development_n": 876,
        "folds": 5,
        "fold_ensemble": True,
        "checkpoint_policy": "best_validation_macro",
        "prefix_training_policy": "each_unique_contiguous_prefix_once_equal_depth_loss",
        "selection_status": "validation_peak_candidate_not_selected_by_frozen_rule",
        "frozen_policy_selected": False,
        "test_evaluated": False,
        "test_subdir": None,
        "expected_parameters": {name: 203842 for name, _ in DEPTHS},
    },
    "shared_contiguous_fixed12": {
        "display_name": "Shared contiguous fixed epoch 12 (selected)",
        "family": "shared",
        "weight_sharing": "one_parameter_set_shared_across_all_depths",
        "depths": tuple(name for name, _ in DEPTHS),
        "development_protocol": "development_876_five_fold_oof",
        "development_estimate": "five_fold_oof",
        "development_n": 876,
        "folds": 5,
        "fold_ensemble": True,
        "checkpoint_policy": "final_epoch",
        "prefix_training_policy": "each_unique_contiguous_prefix_once_equal_depth_loss",
        "selection_status": "frozen_shared_winner",
        "frozen_policy_selected": True,
        "test_evaluated": True,
        "test_subdir": "evaluation",
        "expected_parameters": {name: 203842 for name, _ in DEPTHS},
    },
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _require_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str, context: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{context} has non-finite {column}")
    return values.astype(float)


def _metric_value(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be finite and in [0, 1]")
    return result


def _read_csv(path: Path, inputs_read: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing input artifact: {path}")
    inputs_read.append(str(path.resolve()))
    return pd.read_csv(path, dtype={"patient_id": str})


def _read_json(path: Path, inputs_read: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing input artifact: {path}")
    inputs_read.append(str(path.resolve()))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_frozen_policy(
    payload: Mapping[str, Any], result_dirs: Mapping[str, Path]
) -> None:
    if payload.get("complete") is not True:
        raise ValueError("fixed-epoch policy selection is not complete")
    if (
        payload.get("test_data_used") is not False
        or payload.get("test_embeddings_or_labels_loaded") is not False
        or payload.get("evaluation_files_read") is not False
    ):
        raise ValueError("frozen policy selection must be development-only")
    if "test_metrics" not in set(payload.get("selection_inputs_not_used", [])):
        raise ValueError("frozen policy must explicitly exclude test metrics")

    expected = {
        "independent/T0-T2": (
            "best_validation", "independent_regularization_v2", "layerwise_adamw"
        ),
        "independent/T0-T3": (
            "best_validation", "independent_regularization_v2", "residual_1e3"
        ),
        "shared_contiguous/macro_T0_to_T0-T3": (
            "final_epoch", "shared_contiguous_fixed12", "moderate"
        ),
    }
    chosen = payload.get("chosen")
    if not isinstance(chosen, dict) or set(chosen) != set(expected):
        raise ValueError("frozen policy has an unexpected decision scope")
    for scope, (policy, strategy, variant) in expected.items():
        record = chosen[scope]
        if (
            record.get("policy") != policy
            or record.get("variant") != variant
            or Path(record.get("result_dir", "")).resolve()
            != result_dirs[strategy].resolve()
        ):
            raise ValueError(f"frozen policy winner mismatch for {scope}")


def _empty_test_metrics(row: dict[str, Any]) -> None:
    for source in ("real", "generated"):
        row[f"{source}_test_evaluated_n"] = None
        for metric in METRICS:
            row[f"{source}_{metric}_mean"] = None
            row[f"{source}_{metric}_std"] = None


def _base_row(
    strategy: str,
    spec: Mapping[str, Any],
    depth: str,
    variant: str,
    parameters: int,
) -> dict[str, Any]:
    row = {
        "strategy": strategy,
        "display_name": spec["display_name"],
        "family": spec["family"],
        "weight_sharing": spec["weight_sharing"],
        "temporal_depth": depth,
        "max_tp": DEPTH_TO_MAX_TP[depth],
        "variant": variant,
        "parameters_per_model": int(parameters),
        "development_protocol": spec["development_protocol"],
        "development_estimate": spec["development_estimate"],
        "development_n": int(spec["development_n"]),
        "folds": int(spec["folds"]),
        "fold_ensemble": bool(spec["fold_ensemble"]),
        "checkpoint_policy": spec["checkpoint_policy"],
        "prefix_training_policy": spec["prefix_training_policy"],
        "selection_status": spec["selection_status"],
        "frozen_policy_selected": spec["frozen_policy_selected"],
        "test_target_cohort": "locked102_all_patients_no_exclusion",
        "test_target_n": 102,
        "test_evaluated": bool(spec["test_evaluated"]),
        "test_evaluation_status": (
            "evaluated_from_saved_predictions"
            if spec["test_evaluated"]
            else "not_evaluated"
        ),
        "test_models_per_seed": int(spec["folds"]) if spec["test_evaluated"] else None,
        "test_metrics_used_for_frozen_selection": False,
        "n_formal_seeds": len(SEEDS),
        "development_auroc_mean": None,
        "development_auroc_std": None,
        "development_prauc_mean": None,
        "development_prauc_std": None,
        "positive_train_development_gap": None,
        "gap_definition": None,
    }
    _empty_test_metrics(row)
    return row


def _load_no_cv_development(
    result_dir: Path, spec: Mapping[str, Any], inputs_read: list[str]
) -> list[dict[str, Any]]:
    formal = _read_csv(
        result_dir / "analysis/formal_development_metrics.csv", inputs_read
    )
    summary = _read_csv(
        result_dir / "analysis/new_generalization_summary.csv", inputs_read
    )
    selected = _read_json(result_dir / "selected_variants.json", inputs_read)
    formal = formal[formal["strategy"].astype(str) == "independent"].copy()
    summary = summary[summary["strategy"].astype(str) == "independent"].copy()
    _require_columns(
        formal,
        {"seed", "temporal_depth", "validation_auroc"},
        "independent_no_cv/formal_development_metrics",
    )
    _require_columns(
        summary,
        {
            "temporal_depth", "validation_auroc_mean", "validation_auroc_std",
            "train_validation_gap",
        },
        "independent_no_cv/new_generalization_summary",
    )
    if set(formal["temporal_depth"].astype(str)) != set(spec["depths"]):
        raise ValueError("independent_no_cv development depths mismatch")
    if set(summary["temporal_depth"].astype(str)) != set(spec["depths"]):
        raise ValueError("independent_no_cv summary depths mismatch")

    selected_depths = selected.get("independent")
    if not isinstance(selected_depths, dict):
        raise ValueError("independent_no_cv selected variants are missing")
    rows = []
    for depth in spec["depths"]:
        depth_formal = formal[formal["temporal_depth"].astype(str) == depth]
        seeds = set(pd.to_numeric(depth_formal["seed"], errors="coerce").astype(int))
        if len(depth_formal) != len(SEEDS) or seeds != set(SEEDS):
            raise ValueError(f"independent_no_cv/{depth} requires seeds 42-51")
        validation_values = _numeric(
            depth_formal, "validation_auroc", f"independent_no_cv/{depth}"
        )
        summary_row = summary[summary["temporal_depth"].astype(str) == depth]
        if len(summary_row) != 1:
            raise ValueError(f"independent_no_cv/{depth} summary must have one row")
        summary_row = summary_row.iloc[0]
        mean = _metric_value(
            summary_row["validation_auroc_mean"], f"independent_no_cv/{depth}/mean"
        )
        std = _metric_value(
            summary_row["validation_auroc_std"], f"independent_no_cv/{depth}/std"
        )
        if not np.isclose(mean, validation_values.mean(), atol=1e-12) or not np.isclose(
            std, validation_values.std(ddof=1), atol=1e-12
        ):
            raise ValueError(f"independent_no_cv/{depth} development summary mismatch")
        selected_row = selected_depths.get(depth)
        if not isinstance(selected_row, dict) or not selected_row.get("variant"):
            raise ValueError(f"independent_no_cv/{depth} selected variant is missing")
        row = _base_row(
            "independent_no_cv",
            spec,
            depth,
            str(selected_row["variant"]),
            spec["expected_parameters"][depth],
        )
        row.update(
            {
                "development_auroc_mean": mean,
                "development_auroc_std": std,
                "positive_train_development_gap": float(
                    summary_row["train_validation_gap"]
                ),
                "gap_definition": "mean_train_minus_fixed_validation_auroc",
            }
        )
        rows.append(row)
    return rows


def _load_cv_development(
    strategy: str,
    result_dir: Path,
    spec: Mapping[str, Any],
    inputs_read: list[str],
) -> list[dict[str, Any]]:
    frame = _read_csv(result_dir / "formal_oof_metrics_all.csv", inputs_read)
    _require_columns(
        frame,
        {"seed", "temporal_depth", "max_tp", "variant", "auroc", "prauc"},
        f"{strategy}/formal_oof_metrics_all",
    )
    if set(frame["temporal_depth"].astype(str)) != set(spec["depths"]):
        raise ValueError(f"{strategy} development depths mismatch")
    rows = []
    for depth in spec["depths"]:
        selected = frame[frame["temporal_depth"].astype(str) == depth].copy()
        seeds = set(pd.to_numeric(selected["seed"], errors="coerce").astype(int))
        if len(selected) != len(SEEDS) or seeds != set(SEEDS):
            raise ValueError(f"{strategy}/{depth} requires seeds 42-51")
        if set(pd.to_numeric(selected["max_tp"], errors="coerce").astype(int)) != {
            DEPTH_TO_MAX_TP[depth]
        }:
            raise ValueError(f"{strategy}/{depth} max_tp mismatch")
        variants = set(selected["variant"].astype(str))
        if len(variants) != 1:
            raise ValueError(f"{strategy}/{depth} has multiple formal variants")
        if "checkpoint_policy" in selected:
            policies = set(selected["checkpoint_policy"].astype(str))
            if policies != {spec["checkpoint_policy"]}:
                raise ValueError(f"{strategy}/{depth} checkpoint policy mismatch")
        elif spec["checkpoint_policy"] == "final_epoch":
            raise ValueError(f"{strategy}/{depth} must record final_epoch explicitly")

        expected_parameters = int(spec["expected_parameters"][depth])
        if "parameters" in selected:
            parameters = set(
                pd.to_numeric(selected["parameters"], errors="coerce").astype(int)
            )
            if parameters != {expected_parameters}:
                raise ValueError(f"{strategy}/{depth} parameter count mismatch")
        auroc = _numeric(selected, "auroc", f"{strategy}/{depth}")
        prauc = _numeric(selected, "prauc", f"{strategy}/{depth}")
        row = _base_row(
            strategy,
            spec,
            depth,
            next(iter(variants)),
            expected_parameters,
        )
        row.update(
            {
                "development_auroc_mean": float(auroc.mean()),
                "development_auroc_std": float(auroc.std(ddof=1)),
                "development_prauc_mean": float(prauc.mean()),
                "development_prauc_std": float(prauc.std(ddof=1)),
            }
        )
        if "train_oof_gap" in selected:
            gaps = _numeric(selected, "train_oof_gap", f"{strategy}/{depth}")
            row["positive_train_development_gap"] = float(gaps.clip(lower=0).mean())
            row["gap_definition"] = "mean_positive_fold_train_minus_oof_auroc"
        rows.append(row)
    return rows


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    predictions = (probabilities >= 0.5).astype(int)
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(predictions[positives] == 1))
    fn = int(np.sum(predictions[positives] == 0))
    tn = int(np.sum(predictions[negatives] == 0))
    fp = int(np.sum(predictions[negatives] == 1))
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return {
        "acc": float(np.mean(predictions == labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "sens": float(sens),
        "spec": float(spec),
        "prec": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "bacc": float((sens + spec) / 2.0),
        "prauc": float(average_precision_score(labels, probabilities)),
    }


def _validate_test_predictions(
    frame: pd.DataFrame,
    *,
    strategy: str,
    source: str,
    depths: tuple[str, ...],
    folds: int,
    reference_labels: dict[str, int] | None,
) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    context = f"{strategy}/{source}/all_test_predictions"
    _require_columns(
        frame,
        {"patient_id", "label", "probability", "seed", "split", "temporal_depth", "max_tp", "source"},
        context,
    )
    if set(frame["temporal_depth"].astype(str)) != set(depths):
        raise ValueError(f"{context} depths mismatch")
    if set(frame["source"].astype(str)) != {source}:
        raise ValueError(f"{context} source mismatch")
    expected_split = "test" if folds == 1 else "test_fold_ensemble"
    if set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError(f"{context} split mismatch")
    probabilities = _numeric(frame, "probability", context)
    if not probabilities.between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{context} probabilities outside [0, 1]")

    metric_rows: dict[str, list[dict[str, float]]] = {depth: [] for depth in depths}
    observed_reference = reference_labels
    for depth in depths:
        max_tp = DEPTH_TO_MAX_TP[depth]
        for seed in SEEDS:
            selected = frame[
                (frame["temporal_depth"].astype(str) == depth)
                & (pd.to_numeric(frame["seed"], errors="coerce") == seed)
            ].copy()
            if len(selected) != 102 or selected["patient_id"].duplicated().any():
                raise ValueError(f"{context}/{depth}/seed{seed} is not exactly locked102")
            if set(pd.to_numeric(selected["max_tp"], errors="coerce").astype(int)) != {max_tp}:
                raise ValueError(f"{context}/{depth}/seed{seed} max_tp mismatch")
            labels = pd.to_numeric(selected["label"], errors="coerce")
            if labels.isna().any() or set(labels.astype(int)) - {0, 1}:
                raise ValueError(f"{context}/{depth}/seed{seed} labels are invalid")
            current = dict(zip(selected["patient_id"].astype(str), labels.astype(int)))
            if len(current) != 102 or sum(current.values()) != 32:
                raise ValueError(f"{context}/{depth}/seed{seed} label counts mismatch")
            if observed_reference is None:
                observed_reference = current
            elif current != observed_reference:
                raise ValueError(f"{context}/{depth}/seed{seed} locked102 membership mismatch")
            metric_rows[depth].append(
                _binary_metrics(
                    labels.to_numpy(dtype=int),
                    pd.to_numeric(selected["probability"]).to_numpy(dtype=float),
                )
            )
    assert observed_reference is not None
    aggregate = {}
    for depth, records in metric_rows.items():
        aggregate[depth] = {}
        for metric in METRICS:
            values = np.asarray([record[metric] for record in records], dtype=float)
            aggregate[depth][f"{metric}_mean"] = float(values.mean())
            aggregate[depth][f"{metric}_std"] = float(values.std(ddof=1))
    return observed_reference, aggregate


def _load_test_metrics(
    strategy: str,
    result_dir: Path,
    spec: Mapping[str, Any],
    inputs_read: list[str],
    reference_labels: dict[str, int] | None,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, int] | None]:
    if not spec["test_evaluated"]:
        if (result_dir / "evaluation").exists():
            raise ValueError(
                f"{strategy} is declared not evaluated but has a top-level evaluation directory"
            )
        return {}, reference_labels

    values: dict[str, dict[str, dict[str, float]]] = {}
    test_root = result_dir / str(spec["test_subdir"])
    for source in ("real", "generated"):
        source_dir = test_root / source
        summary = _read_csv(source_dir / "summary.csv", inputs_read)
        predictions = _read_csv(source_dir / "all_test_predictions.csv", inputs_read)
        reference_labels, recomputed = _validate_test_predictions(
            predictions,
            strategy=strategy,
            source=source,
            depths=spec["depths"],
            folds=int(spec["folds"]),
            reference_labels=reference_labels,
        )
        _require_columns(
            summary,
            {"source", "temporal_depth", "max_tp", "n_seeds"}
            | {f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")},
            f"{strategy}/{source}/summary",
        )
        if len(summary) != len(spec["depths"]) or set(
            summary["temporal_depth"].astype(str)
        ) != set(spec["depths"]):
            raise ValueError(f"{strategy}/{source} summary depths mismatch")
        if set(summary["source"].astype(str)) != {source}:
            raise ValueError(f"{strategy}/{source} summary source mismatch")
        if set(pd.to_numeric(summary["n_seeds"], errors="coerce").astype(int)) != {
            len(SEEDS)
        }:
            raise ValueError(f"{strategy}/{source} summary seed count mismatch")
        if "folds_per_seed" in summary and set(
            pd.to_numeric(summary["folds_per_seed"], errors="coerce").astype(int)
        ) != {int(spec["folds"])}:
            raise ValueError(f"{strategy}/{source} folds_per_seed mismatch")

        values[source] = {}
        for depth in spec["depths"]:
            record = summary[summary["temporal_depth"].astype(str) == depth]
            if len(record) != 1:
                raise ValueError(f"{strategy}/{source}/{depth} summary must have one row")
            record = record.iloc[0]
            if int(record["max_tp"]) != DEPTH_TO_MAX_TP[depth]:
                raise ValueError(f"{strategy}/{source}/{depth} summary max_tp mismatch")
            values[source][depth] = {}
            for metric in METRICS:
                for suffix in ("mean", "std"):
                    key = f"{metric}_{suffix}"
                    observed = _metric_value(record[key], f"{strategy}/{source}/{depth}/{key}")
                    if not np.isclose(observed, recomputed[depth][key], atol=1e-10):
                        raise ValueError(
                            f"{strategy}/{source}/{depth} saved and recomputed {key} mismatch"
                        )
                    values[source][depth][key] = observed
    return values, reference_labels


def _attach_test_metrics(
    rows: list[dict[str, Any]], values: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> None:
    for row in rows:
        depth = row["temporal_depth"]
        if not row["test_evaluated"]:
            continue
        for source in ("real", "generated"):
            row[f"{source}_test_evaluated_n"] = 102
            for metric in METRICS:
                row[f"{source}_{metric}_mean"] = values[source][depth][f"{metric}_mean"]
                row[f"{source}_{metric}_std"] = values[source][depth][f"{metric}_std"]


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for raw in frame.to_dict(orient="records"):
        record = {}
        for key, value in raw.items():
            if value is None or pd.isna(value):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def _format_metric(mean: Any, std: Any, unavailable: str) -> str:
    if mean is None or pd.isna(mean):
        return unavailable
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _development_display(row: Mapping[str, Any], metric: str) -> str:
    value = _format_metric(
        row[f"development_{metric}_mean"],
        row[f"development_{metric}_std"],
        "not reported",
    )
    if metric == "auroc" and row["development_estimate"] == "heldout_validation_not_oof":
        return f"{value} (98-val; not OOF)"
    if value != "not reported":
        return f"{value} (876 OOF)"
    return value


def _markdown(frame: pd.DataFrame, policy: Mapping[str, Any], generated_at: str) -> str:
    lines = [
        "# Final Full978 Optimization Comparison",
        "",
        f"Generated at (UTC): `{generated_at}`",
        "",
        "All reported values are mean +/- sample standard deviation over formal seeds 42-51. "
        "For five-fold test runs, the five fold probabilities are averaged per patient before "
        "each seed-level metric is calculated.",
        "",
        "The fixed-epoch choice was frozen from development data at "
        f"`{policy['frozen_at_utc']}`. Locked102 metrics were not used to alter that decision.",
        "",
        "## AUROC",
        "",
        "| Family | Strategy | Depth | Parameters | Development AUROC | Real locked102 | Generated locked102 | Checkpoint | Frozen decision |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {family} | {display_name} | {temporal_depth} | {parameters_per_model:,} | "
            "{development} | {real} | {generated} | {checkpoint_policy} | {selection_status} |".format(
                **row,
                development=_development_display(row, "auroc"),
                real=_format_metric(
                    row["real_auroc_mean"], row["real_auroc_std"], "not evaluated"
                ),
                generated=_format_metric(
                    row["generated_auroc_mean"],
                    row["generated_auroc_std"],
                    "not evaluated",
                ),
            )
        )
    lines.extend(
        [
            "",
            "## PR-AUC",
            "",
            "| Strategy | Depth | Development PR-AUC | Real locked102 | Generated locked102 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {display_name} | {temporal_depth} | {development} | {real} | {generated} |".format(
                **row,
                development=_development_display(row, "prauc"),
                real=_format_metric(
                    row["real_prauc_mean"], row["real_prauc_std"], "not evaluated"
                ),
                generated=_format_metric(
                    row["generated_prauc_mean"],
                    row["generated_prauc_std"],
                    "not evaluated",
                ),
            )
        )

    decisions = {item["temporal_depth"]: item for item in policy["decisions"]}
    t2 = decisions["T0-T2"]
    t3 = decisions["T0-T3"]
    shared = decisions["macro_T0_to_T0-T3"]
    lines.extend(
        [
            "",
            "## Frozen Development Decision",
            "",
            f"- Independent T0-T2 retained regularization v2 validation-peak: fixed epoch lost {t2['fixed_auroc_drop']:.5f} OOF AUROC, exceeding the {t2['auroc_tolerance']:.3f} tolerance.",
            f"- Independent T0-T3 retained regularization v2 validation-peak: fixed epoch lost {t3['fixed_auroc_drop']:.5f} OOF AUROC, exceeding the {t3['auroc_tolerance']:.3f} tolerance.",
            f"- Shared contiguous selected fixed epoch 12: its macro OOF loss was {shared['fixed_auroc_drop']:.5f}, within tolerance, and its positive train-OOF gap decreased from {shared['validation_positive_train_oof_gap']:.5f} to {shared['fixed_positive_train_oof_gap']:.5f}.",
            "- These choices are copied from the frozen policy artifact; this comparison never re-ranks them using real or generated locked102 results.",
            "",
            "## Comparability Notes",
            "",
            "- `Independent fixed split (no CV)` uses the same 98-patient validation set for checkpoint selection and reporting. Its development number is validation AUROC, not OOF AUROC, and is not directly comparable to the 876-patient five-fold OOF estimates.",
            "- Independent models use a separate parameter set at each listed depth. Shared models use one parameter set across all four depth views within a seed/fold.",
            "- The old shared random-prefix model trains on the full sequence plus one sampled prefix per batch; the contiguous models enumerate each unique valid T0-starting prefix once and use equal depth loss weighting.",
            "- `not evaluated` is a genuine missing result, not zero. No test value is imputed for independent fixed epoch or shared contiguous validation peak.",
            "- Every reported test result uses all 102 locked patients. No post-hoc patient deletion is included. Generated T0 equals real T0 because only future DCE0 visits are replaced.",
            "- Locked102 has been inspected in prior experiments, so these are retrospective comparisons rather than untouched prospective confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def run_comparison(
    *,
    result_dirs: Mapping[str, str | Path] | None = None,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    generated_at: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    supplied = DEFAULT_RESULT_DIRS if result_dirs is None else result_dirs
    if set(supplied) != set(STRATEGY_SPECS):
        raise ValueError("result_dirs must contain exactly the seven comparison strategies")
    resolved_dirs = {name: Path(path).expanduser().resolve() for name, path in supplied.items()}
    if len(set(resolved_dirs.values())) != len(resolved_dirs):
        raise ValueError("comparison result directories must be distinct")

    inputs_read: list[str] = []
    resolved_policy_path = Path(policy_path).expanduser().resolve()
    policy = _read_json(resolved_policy_path, inputs_read)
    _validate_frozen_policy(policy, resolved_dirs)

    rows: list[dict[str, Any]] = []
    reference_labels: dict[str, int] | None = None
    for strategy, spec in STRATEGY_SPECS.items():
        result_dir = resolved_dirs[strategy]
        if not result_dir.is_dir():
            raise FileNotFoundError(f"missing result directory: {result_dir}")
        if strategy == "independent_no_cv":
            strategy_rows = _load_no_cv_development(result_dir, spec, inputs_read)
        else:
            strategy_rows = _load_cv_development(
                strategy, result_dir, spec, inputs_read
            )
        test_values, reference_labels = _load_test_metrics(
            strategy,
            result_dir,
            spec,
            inputs_read,
            reference_labels,
        )
        _attach_test_metrics(strategy_rows, test_values)
        rows.extend(strategy_rows)

    if len(rows) != 22:
        raise ValueError(f"expected 22 strategy-depth rows, found {len(rows)}")
    if reference_labels is None or len(reference_labels) != 102 or sum(reference_labels.values()) != 32:
        raise ValueError("could not validate the common 102-patient locked test cohort")

    frame = pd.DataFrame(rows)
    timestamp = _normalize_timestamp(generated_at)
    output = Path(output_dir).expanduser().resolve()
    if output.name != "final_optimization_comparison" or output.parent.name != "analysis":
        raise ValueError(
            "output_dir must end with analysis/final_optimization_comparison"
        )
    if output in set(resolved_dirs.values()):
        raise ValueError("output_dir must be isolated from experiment result directories")
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "comparison artifacts already exist; pass overwrite=True for explicit replacement"
        )

    payload = {
        "schema": SCHEMA,
        "complete": True,
        "generated_at_utc": timestamp,
        "frozen_policy_path": str(resolved_policy_path),
        "frozen_policy_timestamp_utc": policy["frozen_at_utc"],
        "policy_selection_was_development_only": True,
        "test_metrics_used_to_change_frozen_decision": False,
        "development_cohort": "876 patients for CV; fixed 98-patient validation for no-CV",
        "locked_test_cohort": {
            "patients": 102,
            "positive": 32,
            "negative": 70,
            "patient_exclusions": 0,
        },
        "aggregation": (
            "mean_and_sample_std_over_seeds_42_to_51; five_fold_test_"
            "probabilities_are_patientwise_averaged_before_seed_metric"
        ),
        "missing_value_policy": "JSON null, blank CSV cell, and explicit not evaluated/not reported in Markdown",
        "inputs_read": inputs_read,
        "rows": _json_records(frame),
    }
    _atomic_csv(output / OUTPUT_NAMES[0], frame)
    _atomic_json(output / OUTPUT_NAMES[1], payload)
    _atomic_text(output / OUTPUT_NAMES[2], _markdown(frame, policy, timestamp))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_comparison(
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"rows: {len(payload['rows'])}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()

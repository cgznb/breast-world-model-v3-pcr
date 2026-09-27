"""Compare final independent and unified full978 models on fixed QC cohorts.

This script only reads saved patient-level predictions. It never trains a model
and never uses an exclusion result to select a checkpoint or hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_dual_strategy_qc_sensitivity import excluded_patient_ids
from src.metrics import METRIC_KEYS, compute_metrics


DEPTHS = (("T0", 1), ("T0-T1", 2), ("T0-T2", 3), ("T0-T3", 4))
SEEDS = tuple(range(42, 52))
SOURCES = ("real", "generated")
COHORTS = ("full_locked102", "exclude_all_9_qc_patients")
SCHEMA = "pillar_full978_final_strategy_comparison_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_independent_cv"
    / "analysis/all_strategy_comparison"
)

STRATEGIES = {
    "independent_final": {
        "display_name": "four_independent_models_final",
        "weight_sharing": "none_across_temporal_depths",
        "status": "current_independent_result",
        "prefix_policy": "one_target_depth_per_parameter_set",
    },
    "shared_contiguous": {
        "display_name": "one_shared_model_contiguous_equal_depth_loss",
        "weight_sharing": "one_parameter_set_across_all_temporal_depths",
        "status": "current_unified_result",
        "prefix_policy": "each_unique_contiguous_prefix_once_equal_depth_loss",
    },
    "shared_full_random_5fold": {
        "display_name": "one_shared_model_full_plus_random_prefix_5fold",
        "weight_sharing": "one_parameter_set_across_all_temporal_depths",
        "status": "historical_high_score_reference",
        "prefix_policy": "full_sequence_plus_one_random_prefix_per_batch",
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
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate patient IDs: {path}")
    return values


def _read_predictions(path: Path, depths: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame = frame[frame["temporal_depth"].isin(depths)].copy()
    return frame


def _load_strategy_predictions(strategy: str, source: str) -> pd.DataFrame:
    if strategy == "independent_final":
        early = _read_predictions(
            ROOT
            / "results/mewm_ispy2_full978_locked102_dual_strategy"
            / "independent/evaluation"
            / source
            / "all_test_predictions.csv",
            ("T0", "T0-T1"),
        )
        late = _read_predictions(
            ROOT
            / "results/mewm_ispy2_full978_locked102_independent_cv/evaluation"
            / source
            / "all_test_predictions.csv",
            ("T0-T2", "T0-T3"),
        )
        frame = pd.concat([early, late], ignore_index=True)
    elif strategy == "shared_contiguous":
        frame = _read_predictions(
            ROOT
            / "results/mewm_ispy2_full978_locked102_dual_strategy"
            / "shared_contiguous/evaluation"
            / source
            / "all_test_predictions.csv",
            tuple(depth for depth, _ in DEPTHS),
        )
    elif strategy == "shared_full_random_5fold":
        frame = _read_predictions(
            ROOT
            / "results/mewm_ispy2_full978_locked102_anti_overfit/evaluation"
            / source
            / "all_test_predictions.csv",
            tuple(depth for depth, _ in DEPTHS),
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    frame["strategy"] = strategy
    return frame


def _development_protocol(strategy: str, depth: str) -> tuple[str, int]:
    if strategy == "independent_final" and depth in {"T0", "T0-T1"}:
        return "fixed_778_train_98_validation", 1
    if strategy == "shared_contiguous":
        return "fixed_778_train_98_validation", 1
    return "development_876_five_fold", 5


def _validate_predictions(
    frame: pd.DataFrame,
    strategy: str,
    source: str,
    test_ids: list[str],
) -> None:
    required = {
        "patient_id", "label", "probability", "seed", "split",
        "temporal_depth", "max_tp", "source", "strategy",
    }
    if required - set(frame):
        raise ValueError(f"{strategy}/{source} is missing required columns")
    if len(frame) != len(DEPTHS) * len(SEEDS) * len(test_ids):
        raise ValueError(f"{strategy}/{source} prediction count mismatch")
    if set(frame["source"]) != {source} or set(frame["strategy"]) != {strategy}:
        raise ValueError(f"{strategy}/{source} source or strategy mismatch")
    if not np.isfinite(frame["probability"]).all():
        raise ValueError(f"{strategy}/{source} has non-finite probabilities")
    if not frame["probability"].between(0, 1).all():
        raise ValueError(f"{strategy}/{source} probabilities outside [0,1]")
    for depth, max_tp in DEPTHS:
        _, models_per_seed = _development_protocol(strategy, depth)
        expected_split = "test_fold_ensemble" if models_per_seed == 5 else "test"
        depth_frame = frame[frame["temporal_depth"] == depth]
        if set(depth_frame["split"]) != {expected_split}:
            raise ValueError(f"{strategy}/{source}/{depth} test split mismatch")
        for seed in SEEDS:
            selected = frame[
                (frame["temporal_depth"] == depth)
                & (frame["seed"].astype(int) == seed)
            ]
            if (
                selected["patient_id"].tolist() != test_ids
                or selected["patient_id"].duplicated().any()
                or set(selected["max_tp"].astype(int)) != {max_tp}
            ):
                raise ValueError(f"{strategy}/{source}/{depth}/seed{seed} ID mismatch")


def _metric_rows(predictions: pd.DataFrame, qc_ids: set[str]) -> pd.DataFrame:
    rows = []
    for (strategy, source, depth, max_tp, seed), frame in predictions.groupby(
        ["strategy", "source", "temporal_depth", "max_tp", "seed"],
        sort=False,
    ):
        protocol, models_per_seed = _development_protocol(strategy, depth)
        for cohort in COHORTS:
            selected = (
                frame
                if cohort == "full_locked102"
                else frame[~frame["patient_id"].isin(qc_ids)]
            )
            expected_n = 102 if cohort == "full_locked102" else 93
            expected_positive = 32 if cohort == "full_locked102" else 27
            if len(selected) != expected_n or int(selected["label"].sum()) != expected_positive:
                raise ValueError(f"{strategy}/{source}/{depth}/{seed}/{cohort} mismatch")
            rows.append(
                {
                    "analysis_status": (
                        "primary_intact_locked_test"
                        if cohort == "full_locked102"
                        else "post_hoc_sensitivity_not_primary"
                    ),
                    "strategy": strategy,
                    "display_name": STRATEGIES[strategy]["display_name"],
                    "strategy_status": STRATEGIES[strategy]["status"],
                    "weight_sharing": STRATEGIES[strategy]["weight_sharing"],
                    "prefix_policy": STRATEGIES[strategy]["prefix_policy"],
                    "development_protocol": protocol,
                    "models_per_seed": models_per_seed,
                    "source": source,
                    "temporal_depth": depth,
                    "max_tp": int(max_tp),
                    "seed": int(seed),
                    "cohort": cohort,
                    "n_patients": len(selected),
                    "n_positive": int(selected["label"].sum()),
                    "n_negative": int((selected["label"] == 0).sum()),
                    "n_excluded": len(frame) - len(selected),
                    **compute_metrics(
                        selected["label"], selected["probability"], threshold=0.5
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "analysis_status", "strategy", "display_name", "strategy_status",
        "weight_sharing", "prefix_policy", "development_protocol",
        "models_per_seed", "source", "temporal_depth", "max_tp", "cohort",
        "n_patients", "n_positive", "n_negative", "n_excluded",
    ]
    rows = []
    for values, frame in metrics.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = len(frame)
        for metric in METRIC_KEYS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    reference = summary[summary["cohort"] == "full_locked102"][
        ["strategy", "source", "temporal_depth"]
        + [f"{metric}_mean" for metric in METRIC_KEYS]
    ].rename(
        columns={f"{metric}_mean": f"full_{metric}_mean" for metric in METRIC_KEYS}
    )
    summary = summary.merge(
        reference,
        on=["strategy", "source", "temporal_depth"],
        validate="many_to_one",
    )
    for metric in METRIC_KEYS:
        summary[f"{metric}_delta_vs_full"] = (
            summary[f"{metric}_mean"] - summary[f"full_{metric}_mean"]
        )
    return summary


def _format_value(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _report_table(summary: pd.DataFrame, source: str) -> str:
    selected = summary[summary["source"] == source]
    lines = [
        "| Strategy | Depth | Full AUROC | Excluded AUROC | AUROC delta | Full PR-AUC | Excluded PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        for depth, _ in DEPTHS:
            full = selected[
                (selected["strategy"] == strategy)
                & (selected["temporal_depth"] == depth)
                & (selected["cohort"] == "full_locked102")
            ].iloc[0]
            excluded = selected[
                (selected["strategy"] == strategy)
                & (selected["temporal_depth"] == depth)
                & (selected["cohort"] == "exclude_all_9_qc_patients")
            ].iloc[0]
            lines.append(
                "| "
                + " | ".join(
                    [
                        strategy,
                        depth,
                        _format_value(full.auroc_mean, full.auroc_std),
                        _format_value(excluded.auroc_mean, excluded.auroc_std),
                        f"{excluded.auroc_delta_vs_full:+.4f}",
                        _format_value(full.prauc_mean, full.prauc_std),
                        _format_value(excluded.prauc_mean, excluded.prauc_std),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


METRIC_DISPLAY_NAMES = {
    "acc": "Accuracy",
    "auroc": "AUROC",
    "sens": "Sensitivity",
    "spec": "Specificity",
    "prec": "Precision",
    "npv": "NPV",
    "bacc": "Balanced accuracy",
    "prauc": "PR-AUC (AP)",
}


def _complete_metric_table(
    summary: pd.DataFrame,
    strategy: str,
    source: str,
    cohort: str,
) -> str:
    selected = summary[
        (summary["strategy"] == strategy)
        & (summary["source"] == source)
        & (summary["cohort"] == cohort)
    ].set_index("temporal_depth")
    headers = ["Depth", "N (+/-)"] + [METRIC_DISPLAY_NAMES[key] for key in METRIC_KEYS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|" + "---:|" * len(METRIC_KEYS),
    ]
    for depth, _ in DEPTHS:
        row = selected.loc[depth]
        values = [
            depth,
            f"{int(row.n_patients)} ({int(row.n_positive)}/{int(row.n_negative)})",
        ]
        values.extend(
            _format_value(row[f"{metric}_mean"], row[f"{metric}_std"])
            for metric in METRIC_KEYS
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    complete_sections = []
    for strategy, metadata in STRATEGIES.items():
        complete_sections.append(
            f"## {strategy}\n\n"
            f"Status: `{metadata['status']}`. Weight sharing: "
            f"`{metadata['weight_sharing']}`.\n"
        )
        for source in SOURCES:
            for cohort in COHORTS:
                cohort_label = (
                    "intact locked102"
                    if cohort == "full_locked102"
                    else "post-hoc exclusion of all nine QC patients"
                )
                complete_sections.append(
                    f"### {source}: {cohort_label}\n\n"
                    f"{_complete_metric_table(summary, strategy, source, cohort)}\n"
                )
    complete_tables = "\n".join(complete_sections)
    report = f"""# Full978 final strategy comparison

## Contract

- Every row uses the same locked102 IDs and seeds 42-51. Full rows contain 102 patients (32 pCR); exclusion rows contain 93 patients (27 pCR) after removing the same nine model-independent QC-flagged patients.
- Exclusion is post-hoc sensitivity analysis and cannot replace the intact locked102 result.
- `independent_final` combines fixed-split T0/T0-T1 with five-fold T0-T2/T0-T3, so its four-depth curve is not a pure temporal-depth comparison.
- `shared_contiguous` is the current no-duplicate unified strategy requested after the sampling review. `shared_full_random_5fold` is retained only as a historical high-score reference because it gives the full prefix extra expected training weight.
- AUROC and PR-AUC are threshold-free. Accuracy, sensitivity, specificity, precision, NPV, and balanced accuracy in the CSV use the fixed 0.5 threshold.
- Every value is the mean +/- sample SD of ten seed-level metrics. It is not a metric computed after pooling 1,020 predictions and is not a patient-bootstrap confidence interval.
- PR-AUC is computed with `average_precision_score` (average precision).

## Real test

{_report_table(summary, "real")}

## Generated-future test

{_report_table(summary, "generated")}

## Interpretation

Removing the same nine QC-flagged patients lowers mean AUROC for every strategy, source, and temporal depth in this comparison. Patient deletion therefore does not provide a valid performance improvement. The full 102-patient results remain primary.

# Complete eight-metric tables

{complete_tables}
"""
    _atomic_text(output_dir / "comparison_report.md", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    test_ids = _read_ids(
        ROOT / "data/mewm_ispy2_full978_locked102/splits/test_ids.txt"
    )
    if len(test_ids) != 102:
        raise ValueError("locked test must contain 102 patients")
    qc_ids = set(excluded_patient_ids("exclude_all_9_qc_patients", "T0-T3"))
    if len(qc_ids) != 9 or not qc_ids.issubset(test_ids):
        raise ValueError("QC exclusion contract mismatch")

    frames = []
    label_reference = None
    for strategy in STRATEGIES:
        for source in SOURCES:
            frame = _load_strategy_predictions(strategy, source)
            _validate_predictions(frame, strategy, source, test_ids)
            for depth, _ in DEPTHS:
                labels = frame[
                    (frame["temporal_depth"] == depth)
                    & (frame["seed"].astype(int) == SEEDS[0])
                ][["patient_id", "label"]].reset_index(drop=True)
                if label_reference is None:
                    label_reference = labels
                elif not labels.equals(label_reference):
                    raise ValueError(f"label mismatch: {strategy}/{source}/{depth}")
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    metrics = _metric_rows(predictions, qc_ids)
    summary = _summarize(metrics)

    expected_rows = len(STRATEGIES) * len(SOURCES) * len(DEPTHS) * len(SEEDS) * len(COHORTS)
    if len(metrics) != expected_rows or len(summary) != expected_rows // len(SEEDS):
        raise ValueError("comparison output row count mismatch")
    excluded = summary[summary["cohort"] == "exclude_all_9_qc_patients"]
    if not (excluded["auroc_delta_vs_full"] < 0).all():
        raise ValueError("expected every all-nine AUROC sensitivity delta to be negative")

    _atomic_csv(output_dir / "metrics_per_seed.csv", metrics)
    _atomic_csv(output_dir / "summary.csv", summary)
    _write_report(output_dir, summary)
    _atomic_json(
        output_dir / "ANALYSIS_COMPLETE.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "strategies": list(STRATEGIES),
            "sources": list(SOURCES),
            "temporal_depths": [depth for depth, _ in DEPTHS],
            "seeds": list(SEEDS),
            "locked_test_patients": len(test_ids),
            "excluded_qc_patients": len(qc_ids),
            "metrics_recomputed_from_patient_probabilities": True,
            "exclusion_used_for_model_selection": False,
        },
    )
    print(f"comparison complete: {output_dir}")


if __name__ == "__main__":
    main()

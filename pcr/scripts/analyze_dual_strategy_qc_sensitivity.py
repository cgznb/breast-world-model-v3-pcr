"""Post-hoc, model-independent QC exclusion sensitivity for locked102 predictions.

This script never trains or re-runs a model. It recomputes metrics from the
saved dual-strategy real/generated predictions after applying QC cohorts that
were defined by the image/registration audit, not by labels or prediction
errors. The intact locked102 cohort remains the primary result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import METRIC_KEYS, compute_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "results/mewm_ispy2_full978_locked102_dual_strategy"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis/qc_exclusion_sensitivity_v1"
DEFAULT_TEST_IDS = ROOT / "data/mewm_ispy2_full978_locked102/splits/test_ids.txt"
DEFAULT_AUDIT_REPORT = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_table2/diagnostics"
    / "temporal_depth_regression_audit.md"
)

SCHEMA = "pillar_locked102_qc_exclusion_sensitivity_v1"
ANALYSIS_STATUS = "post_hoc_sensitivity_not_primary"
DEPTHS = ("T0", "T0-T1", "T0-T2", "T0-T3")
DEPTH_TO_MAX_TP = {depth: index + 1 for index, depth in enumerate(DEPTHS)}
VISIT_TO_INDEX = {f"T{index}": index for index in range(4)}
STRATEGIES = ("independent", "shared_contiguous")
SOURCES = ("real", "generated")
SEEDS = tuple(range(42, 52))
COHORTS = (
    "full_locked102",
    "exclude_depth_visible_qc_patients",
    "exclude_all_9_qc_patients",
)

# Source: temporal_depth_regression_audit.md, section "Image And Registration
# Findings" (2026-09-03). These ten visit flags identify nine patients using
# image support/correlation and decoded transform properties. Prediction error
# and pCR label were not used. Thresholds were selected post hoc, so every
# exclusion result is sensitivity analysis only.
QC_VISITS = (
    {
        "patient_id": "ISPY2-294265",
        "visit": "T3",
        "evidence": "gross_local_transform",
        "detail": "selected late phase",
    },
    {
        "patient_id": "ISPY2-804233",
        "visit": "T3",
        "evidence": "gross_local_transform",
        "detail": "selected early phase",
    },
    {
        "patient_id": "ISPY2-355292",
        "visit": "T1",
        "evidence": "gross_local_transform",
        "detail": "selected late phase",
    },
    {
        "patient_id": "ISPY2-614434",
        "visit": "T3",
        "evidence": "gross_local_transform",
        "detail": "selected early phase",
    },
    {
        "patient_id": "ISPY2-797499",
        "visit": "T2",
        "evidence": "gross_local_transform",
        "detail": "selected early phase",
    },
    {
        "patient_id": "ISPY2-797499",
        "visit": "T3",
        "evidence": "gross_local_transform",
        "detail": "selected late phase",
    },
    {
        "patient_id": "ISPY2-493775",
        "visit": "T3",
        "evidence": "additional_selected_transform_outlier",
        "detail": "selected transform outlier",
    },
    {
        "patient_id": "ISPY2-703401",
        "visit": "T2",
        "evidence": "additional_selected_transform_outlier",
        "detail": "selected transform outlier",
    },
    {
        "patient_id": "ISPY2-341238",
        "visit": "T3",
        "evidence": "support_only_warning",
        "detail": "support/correlation screen",
    },
    {
        "patient_id": "ISPY2-900959",
        "visit": "T3",
        "evidence": "support_only_warning",
        "detail": "support/correlation screen",
    },
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_test_ids(path: Path) -> list[str]:
    patient_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError(f"duplicate patient IDs in {path}")
    return patient_ids


def _qc_patient_ids() -> frozenset[str]:
    return frozenset(row["patient_id"] for row in QC_VISITS)


def excluded_patient_ids(cohort: str, temporal_depth: str) -> frozenset[str]:
    """Return the predeclared patient exclusion set for one temporal depth."""
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort: {cohort}")
    if temporal_depth not in DEPTHS:
        raise ValueError(f"unknown temporal depth: {temporal_depth}")
    if cohort == "full_locked102":
        return frozenset()
    if cohort == "exclude_all_9_qc_patients":
        return _qc_patient_ids()

    visible_index = DEPTH_TO_MAX_TP[temporal_depth] - 1
    return frozenset(
        row["patient_id"]
        for row in QC_VISITS
        if VISIT_TO_INDEX[row["visit"]] <= visible_index
    )


def _validate_qc_source(path: Path) -> None:
    text = path.read_text()
    if "Locked-102 Temporal-Depth Regression Audit" not in text:
        raise ValueError(f"unexpected QC source report: {path}")
    if "post-hoc" not in text.lower():
        raise ValueError("QC source report does not identify the analysis as post-hoc")
    for row in QC_VISITS:
        marker = f'{row["patient_id"]} {row["visit"]}'
        if marker not in text:
            raise ValueError(f"QC source report is missing declared visit: {marker}")


def load_predictions(input_root: Path) -> pd.DataFrame:
    frames = []
    for strategy in STRATEGIES:
        for source in SOURCES:
            path = input_root / strategy / "evaluation" / source / "all_test_predictions.csv"
            frame = pd.read_csv(path, dtype={"patient_id": str})
            if "strategy" not in frame or set(frame["strategy"].astype(str)) != {strategy}:
                raise ValueError(f"strategy does not match prediction path: {path}")
            if "source" not in frame or set(frame["source"].astype(str)) != {source}:
                raise ValueError(f"source does not match prediction path: {path}")
            frame["input_file"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_prediction_contract(
    predictions: pd.DataFrame,
    expected_patient_ids: Sequence[str],
    *,
    strategies: Sequence[str] = STRATEGIES,
    sources: Sequence[str] = SOURCES,
    seeds: Sequence[int] = SEEDS,
    depths: Sequence[str] = DEPTHS,
) -> None:
    required = {
        "patient_id",
        "label",
        "probability",
        "seed",
        "strategy",
        "temporal_depth",
        "source",
        "split",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")

    expected_ids = set(expected_patient_ids)
    if len(expected_ids) != len(expected_patient_ids):
        raise ValueError("expected patient IDs are not unique")
    if predictions[list(required)].isna().any().any():
        raise ValueError("prediction contract columns contain missing values")
    if set(predictions["strategy"].astype(str)) != set(strategies):
        raise ValueError("prediction strategy set mismatch")
    if set(predictions["source"].astype(str)) != set(sources):
        raise ValueError("prediction source set mismatch")
    if set(predictions["seed"].astype(int)) != set(seeds):
        raise ValueError("prediction seed set mismatch")
    if set(predictions["temporal_depth"].astype(str)) != set(depths):
        raise ValueError("prediction temporal-depth set mismatch")
    if set(predictions["split"].astype(str)) != {"test"}:
        raise ValueError("non-test predictions were supplied")
    if not predictions["label"].astype(int).isin([0, 1]).all():
        raise ValueError("predictions contain non-binary labels")
    probabilities = predictions["probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not predictions["probability"].between(0, 1).all():
        raise ValueError("predictions contain invalid probabilities")
    if "max_tp" in predictions:
        expected_max_tp = predictions["temporal_depth"].map(DEPTH_TO_MAX_TP)
        if not np.array_equal(predictions["max_tp"].to_numpy(dtype=int), expected_max_tp):
            raise ValueError("max_tp does not match temporal depth")

    keys = ["strategy", "source", "seed", "temporal_depth", "patient_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("duplicate prediction rows detected")
    for key, frame in predictions.groupby(keys[:-1], sort=False):
        actual_ids = set(frame["patient_id"].astype(str))
        if actual_ids != expected_ids or len(frame) != len(expected_ids):
            raise ValueError(f"locked test coverage mismatch for {key}")

    labels_per_patient = predictions.groupby("patient_id")["label"].nunique()
    if not labels_per_patient.eq(1).all():
        raise ValueError("patient labels disagree across predictions")


def compute_sensitivity_metrics(
    predictions: pd.DataFrame,
    expected_patient_ids: Sequence[str],
    *,
    strategies: Sequence[str] = STRATEGIES,
    sources: Sequence[str] = SOURCES,
    seeds: Sequence[int] = SEEDS,
    depths: Sequence[str] = DEPTHS,
) -> pd.DataFrame:
    expected_ids = set(expected_patient_ids)
    if not _qc_patient_ids() <= expected_ids:
        missing = sorted(_qc_patient_ids() - expected_ids)
        raise ValueError(f"QC patients are absent from locked test: {missing}")

    rows = []
    for strategy in strategies:
        for source in sources:
            for seed in seeds:
                for depth in depths:
                    base = predictions[
                        (predictions["strategy"] == strategy)
                        & (predictions["source"] == source)
                        & (predictions["seed"].astype(int) == int(seed))
                        & (predictions["temporal_depth"] == depth)
                    ]
                    for cohort in COHORTS:
                        excluded = excluded_patient_ids(cohort, depth)
                        selected = base[~base["patient_id"].isin(excluded)]
                        labels = selected["label"].to_numpy(dtype=int)
                        if len(selected) != len(expected_ids) - len(excluded):
                            raise RuntimeError("unexpected sensitivity cohort size")
                        if len(np.unique(labels)) != 2:
                            raise RuntimeError("sensitivity cohort does not contain both classes")
                        rows.append(
                            {
                                "analysis_status": ANALYSIS_STATUS,
                                "strategy": strategy,
                                "source": source,
                                "seed": int(seed),
                                "temporal_depth": depth,
                                "max_tp": DEPTH_TO_MAX_TP[depth],
                                "cohort": cohort,
                                "cohort_alters_locked_test": bool(excluded),
                                "n_patients": int(len(selected)),
                                "n_positive": int(labels.sum()),
                                "n_negative": int(len(labels) - labels.sum()),
                                "n_excluded": int(len(excluded)),
                                "excluded_patient_ids": ";".join(sorted(excluded)),
                                "selection_uses_predictions": False,
                                "selection_uses_labels": False,
                                **compute_metrics(labels, selected["probability"], threshold=0.5),
                            }
                        )
    result = pd.DataFrame(rows)
    expected_rows = len(strategies) * len(sources) * len(seeds) * len(depths) * len(COHORTS)
    if len(result) != expected_rows:
        raise RuntimeError("sensitivity metric grid is incomplete")
    return result


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["strategy", "source", "temporal_depth", "max_tp", "cohort"]
    for key, frame in metrics.groupby(group_keys, sort=False):
        constant_columns = ("n_patients", "n_positive", "n_negative", "n_excluded")
        if any(frame[column].nunique() != 1 for column in constant_columns):
            raise RuntimeError("cohort counts vary across seeds")
        row = {
            "analysis_status": ANALYSIS_STATUS,
            **dict(zip(group_keys, key)),
            "n_seeds": int(len(frame)),
            **{column: int(frame[column].iloc[0]) for column in constant_columns},
            "excluded_patient_ids": frame["excluded_patient_ids"].iloc[0],
        }
        for metric in METRIC_KEYS:
            values = frame[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["strategy", "source", "seed", "temporal_depth", "max_tp"]
    full = metrics[metrics["cohort"] == "full_locked102"]
    rows = []
    for cohort in COHORTS[1:]:
        altered = metrics[metrics["cohort"] == cohort]
        paired = altered.merge(full, on=keys, suffixes=("_excluded", "_full"), validate="one_to_one")
        for record in paired.to_dict("records"):
            row = {
                "analysis_status": ANALYSIS_STATUS,
                **{key: record[key] for key in keys},
                "cohort": cohort,
                "n_patients": int(record["n_patients_excluded"]),
                "n_excluded": int(record["n_excluded_excluded"]),
                "excluded_patient_ids": record["excluded_patient_ids_excluded"],
            }
            for metric in METRIC_KEYS:
                row[f"{metric}_delta_vs_full"] = float(
                    record[f"{metric}_excluded"] - record[f"{metric}_full"]
                )
            rows.append(row)
    detail = pd.DataFrame(rows)

    summary_rows = []
    summary_keys = ["strategy", "source", "temporal_depth", "max_tp", "cohort"]
    for key, frame in detail.groupby(summary_keys, sort=False):
        row = {
            "analysis_status": ANALYSIS_STATUS,
            **dict(zip(summary_keys, key)),
            "n_seeds": int(len(frame)),
            "n_patients": int(frame["n_patients"].iloc[0]),
            "n_excluded": int(frame["n_excluded"].iloc[0]),
            "excluded_patient_ids": frame["excluded_patient_ids"].iloc[0],
        }
        for metric in METRIC_KEYS:
            values = frame[f"{metric}_delta_vs_full"].to_numpy(dtype=float)
            row[f"{metric}_delta_vs_full_mean"] = float(np.mean(values))
            row[f"{metric}_delta_vs_full_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def _qc_cases_frame(audit_report: Path) -> pd.DataFrame:
    first_visit = {}
    for row in QC_VISITS:
        patient_id = row["patient_id"]
        visit_index = VISIT_TO_INDEX[row["visit"]]
        first_visit[patient_id] = min(first_visit.get(patient_id, visit_index), visit_index)
    rows = []
    for row in QC_VISITS:
        rows.append(
            {
                **row,
                "first_affected_depth": DEPTHS[first_visit[row["patient_id"]]],
                "source_report": _safe_relative(audit_report),
                "analysis_status": ANALYSIS_STATUS,
                "selection_uses_predictions": False,
                "selection_uses_labels": False,
            }
        )
    return pd.DataFrame(rows)


def _cohort_definitions(expected_patient_ids: Sequence[str]) -> pd.DataFrame:
    expected = set(expected_patient_ids)
    rows = []
    for cohort in COHORTS:
        for depth in DEPTHS:
            excluded = excluded_patient_ids(cohort, depth)
            rows.append(
                {
                    "analysis_status": ANALYSIS_STATUS,
                    "cohort": cohort,
                    "temporal_depth": depth,
                    "n_patients": len(expected - excluded),
                    "n_excluded": len(excluded),
                    "excluded_patient_ids": ";".join(sorted(excluded)),
                    "cohort_alters_locked_test": bool(excluded),
                    "selection_uses_predictions": False,
                    "selection_uses_labels": False,
                }
            )
    return pd.DataFrame(rows)


def _format_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame[list(columns)].copy()
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "|" + "|".join(["---"] * len(selected.columns)) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()]
    return "\n".join([header, divider, *body])


def _write_report(output_dir: Path, summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    report_rows = summary.copy()
    report_rows["AUROC mean +/- SD"] = [
        _format_metric(mean, std)
        for mean, std in zip(report_rows["auroc_mean"], report_rows["auroc_std"])
    ]
    report_rows["PR-AUC mean +/- SD"] = [
        _format_metric(mean, std)
        for mean, std in zip(report_rows["prauc_mean"], report_rows["prauc_std"])
    ]
    deltas = delta_summary[
        ["strategy", "source", "temporal_depth", "cohort", "auroc_delta_vs_full_mean"]
    ]
    report_rows = report_rows.merge(
        deltas,
        on=["strategy", "source", "temporal_depth", "cohort"],
        how="left",
        validate="one_to_one",
    )
    report_rows["AUROC delta vs full"] = report_rows["auroc_delta_vs_full_mean"].map(
        lambda value: "reference" if pd.isna(value) else f"{value:+.4f}"
    )

    lines = [
        "# Locked102 QC Exclusion Sensitivity",
        "",
        "## Interpretation Contract",
        "",
        "- This entire document is a post-hoc sensitivity analysis. The intact 102-patient result remains primary.",
        "- Cohort membership comes only from the existing image/registration QC audit. Labels, prediction errors, and whether exclusion improves AUROC were not used.",
        "- `exclude_depth_visible_qc_patients` removes a patient only once a flagged visit is visible: 0 at T0, 1 at T0-T1, 3 at T0-T2, and all 9 at T0-T3.",
        "- `exclude_all_9_qc_patients` removes the same nine patients at every depth. Its T0 change therefore measures altered case mix, not an effect of a bad later image.",
        "- Patient exclusion changes the target population and cannot establish that a malformed visit caused an error. It must not replace locked102 reporting or guide further test-set filtering.",
        "",
    ]
    for strategy in STRATEGIES:
        for source in SOURCES:
            selected = report_rows[
                (report_rows["strategy"] == strategy) & (report_rows["source"] == source)
            ]
            lines.extend(
                [
                    f"## {strategy}: {source}",
                    "",
                    _markdown_table(
                        selected,
                        [
                            "temporal_depth",
                            "cohort",
                            "n_patients",
                            "n_positive",
                            "n_negative",
                            "AUROC mean +/- SD",
                            "AUROC delta vs full",
                            "PR-AUC mean +/- SD",
                        ],
                    ),
                    "",
                ]
            )
    _atomic_text(output_dir / "report.md", "\n".join(lines))


def _safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def run_analysis(
    input_root: Path,
    output_dir: Path,
    test_ids_path: Path,
    audit_report: Path,
) -> pd.DataFrame:
    if output_dir.resolve() == input_root.resolve():
        raise ValueError("output directory must be separate from the experiment root")
    _validate_qc_source(audit_report)
    patient_ids = _read_test_ids(test_ids_path)
    if len(patient_ids) != 102:
        raise ValueError(f"locked test must contain 102 patients, found {len(patient_ids)}")

    predictions = load_predictions(input_root)
    validate_prediction_contract(predictions, patient_ids)
    metrics = compute_sensitivity_metrics(predictions, patient_ids)
    summary = summarize_metrics(metrics)
    delta_detail, delta_summary = paired_deltas(metrics)

    _atomic_csv(output_dir / "metrics_per_seed.csv", metrics)
    _atomic_csv(output_dir / "summary.csv", summary)
    _atomic_csv(output_dir / "paired_deltas_per_seed.csv", delta_detail)
    _atomic_csv(output_dir / "paired_delta_summary.csv", delta_summary)
    _atomic_csv(output_dir / "qc_cases.csv", _qc_cases_frame(audit_report))
    _atomic_csv(output_dir / "cohort_definitions.csv", _cohort_definitions(patient_ids))
    _write_report(output_dir, summary, delta_summary)
    _atomic_json(
        output_dir / "ANALYSIS_COMPLETE.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "analysis_status": ANALYSIS_STATUS,
            "primary_locked_test_result_changed": False,
            "test_used_for_training_or_model_selection": False,
            "qc_selection_uses_predictions": False,
            "qc_selection_uses_labels": False,
            "qc_thresholds_declared_post_hoc": True,
            "input_root": _safe_relative(input_root),
            "test_ids_source": _safe_relative(test_ids_path),
            "qc_case_source": _safe_relative(audit_report),
            "output_dir": _safe_relative(output_dir),
            "strategies": list(STRATEGIES),
            "sources": list(SOURCES),
            "seeds": list(SEEDS),
            "temporal_depths": list(DEPTHS),
            "cohorts": list(COHORTS),
            "locked_test_patients": len(patient_ids),
            "qc_patients": len(_qc_patient_ids()),
            "qc_flagged_visits": len(QC_VISITS),
            "metric_rows": len(metrics),
            "summary_rows": len(summary),
        },
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_analysis(args.input_root, args.output_dir, args.test_ids, args.audit_report)
    print(
        summary[
            [
                "strategy",
                "source",
                "temporal_depth",
                "cohort",
                "n_patients",
                "auroc_mean",
                "auroc_std",
                "prauc_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

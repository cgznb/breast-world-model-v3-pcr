"""Freeze fixed-epoch versus validation-peak policies using development OOF only.

The selector reads exactly four ``formal_variant_summary.csv`` files. It never
walks a result tree and never reads an evaluation or test prediction artifact.
The locked-test cohort must remain unopened until this decision is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pillar_full978_fixed_epoch_policy_selection_v1"
EXPECTED_FORMAL_SEEDS = 10
AUROC_TOLERANCE = 0.005
FORMAL_SUMMARY_NAME = "formal_variant_summary.csv"

INDEPENDENT_DEPTHS = {"T0-T2": 3, "T0-T3": 4}
INDEPENDENT_VALIDATION_POLICY = "best_validation"
SHARED_VALIDATION_POLICY = "best_validation_macro"
FIXED_POLICY = "final_epoch"

DEFAULT_INDEPENDENT_VALIDATION_DIR = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_independent_cv_regularization_v2"
)
DEFAULT_INDEPENDENT_FIXED_DIR = (
    ROOT / "results/mewm_ispy2_full978_locked102_independent_cv_fixed_epoch"
)
DEFAULT_SHARED_VALIDATION_DIR = (
    ROOT / "results/mewm_ispy2_full978_locked102_shared_contiguous_cv"
)
DEFAULT_SHARED_FIXED_DIR = (
    ROOT / "results/mewm_ispy2_full978_locked102_shared_contiguous_cv_fixed_epoch12"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_fixed_epoch_policy_selection"
    / "analysis/fixed_epoch_policy_selection"
)

OUTPUT_NAMES = (
    "checkpoint_policy_selection.json",
    "checkpoint_policy_selection.csv",
    "checkpoint_policy_selection.md",
)
FORBIDDEN_INPUT_COMPONENTS = {"evaluation", "test", "tests"}


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


def _contains_forbidden_input_component(path: Path) -> bool:
    return any(part.lower() in FORBIDDEN_INPUT_COMPONENTS for part in path.parts)


def _formal_summary_path(result_dir: str | Path) -> tuple[Path, Path]:
    supplied = Path(result_dir).expanduser()
    if not supplied.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {supplied}")
    resolved_dir = supplied.resolve()
    if _contains_forbidden_input_component(resolved_dir):
        raise ValueError(
            f"refusing non-development input directory: {resolved_dir}"
        )
    summary = resolved_dir / FORMAL_SUMMARY_NAME
    if not summary.is_file():
        raise FileNotFoundError(f"missing formal summary: {summary}")
    resolved_summary = summary.resolve()
    if (
        resolved_summary.parent != resolved_dir
        or resolved_summary.name != FORMAL_SUMMARY_NAME
        or _contains_forbidden_input_component(resolved_summary)
    ):
        raise ValueError(f"formal summary escapes its result directory: {summary}")
    return resolved_dir, resolved_summary


def _require_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _validate_numeric_column(
    frame: pd.DataFrame,
    column: str,
    context: str,
    *,
    lower: float,
    upper: float,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise ValueError(f"{context} has non-finite {column}")
    if not values.between(lower, upper, inclusive="both").all():
        raise ValueError(
            f"{context} has {column} outside [{lower}, {upper}]"
        )
    frame[column] = values.astype(float)


def _validate_seed_counts(frame: pd.DataFrame, context: str) -> None:
    values = pd.to_numeric(frame["n_seeds"], errors="coerce")
    if values.isna().any() or not (values == EXPECTED_FORMAL_SEEDS).all():
        found = sorted(set(frame["n_seeds"].astype(str)))
        raise ValueError(
            f"{context} requires exactly {EXPECTED_FORMAL_SEEDS} formal seeds "
            f"per row; found {found}"
        )
    frame["n_seeds"] = values.astype(int)


def _validate_policy(
    frame: pd.DataFrame,
    expected_policy: str,
    context: str,
    *,
    allow_legacy_missing_policy: bool,
) -> str:
    if "checkpoint_policy" not in frame.columns:
        if allow_legacy_missing_policy:
            frame["checkpoint_policy"] = expected_policy
            return "legacy_default_result_role"
        raise ValueError(f"{context} is missing required checkpoint_policy")
    if frame["checkpoint_policy"].isna().any():
        raise ValueError(f"{context} has missing checkpoint_policy values")
    policies = set(frame["checkpoint_policy"].astype(str))
    if policies != {expected_policy}:
        raise ValueError(
            f"{context} checkpoint_policy must be {expected_policy!r}; "
            f"found {sorted(policies)}"
        )
    frame["checkpoint_policy"] = expected_policy
    return "formal_variant_summary"


def _load_independent_summary(
    result_dir: str | Path,
    expected_policy: str,
    role: str,
    *,
    allow_legacy_missing_policy: bool = False,
) -> tuple[Path, Path, pd.DataFrame, str]:
    resolved_dir, summary_path = _formal_summary_path(result_dir)
    frame = pd.read_csv(summary_path)
    context = f"independent/{role}"
    _require_columns(
        frame,
        {
            "temporal_depth",
            "max_tp",
            "variant",
            "oof_auroc_mean",
            "positive_train_oof_gap",
            "n_seeds",
        },
        context,
    )
    if len(frame) != len(INDEPENDENT_DEPTHS):
        raise ValueError(
            f"{context} must contain exactly one row for each independent depth"
        )
    frame = frame.copy()
    frame["temporal_depth"] = frame["temporal_depth"].astype(str)
    if frame["temporal_depth"].duplicated().any():
        raise ValueError(f"{context} has duplicate temporal depths")
    if set(frame["temporal_depth"]) != set(INDEPENDENT_DEPTHS):
        raise ValueError(
            f"{context} temporal depths must be {list(INDEPENDENT_DEPTHS)}"
        )
    max_tp = pd.to_numeric(frame["max_tp"], errors="coerce")
    observed_depths = dict(zip(frame["temporal_depth"], max_tp))
    if any(
        not math.isfinite(float(observed_depths[name]))
        or int(observed_depths[name]) != expected_max_tp
        for name, expected_max_tp in INDEPENDENT_DEPTHS.items()
    ):
        raise ValueError(f"{context} max_tp does not match the required depths")
    if frame["variant"].isna().any() or not frame["variant"].astype(str).str.strip().all():
        raise ValueError(f"{context} has a missing variant strategy")
    _validate_seed_counts(frame, context)
    _validate_numeric_column(
        frame, "oof_auroc_mean", context, lower=0.0, upper=1.0
    )
    _validate_numeric_column(
        frame, "positive_train_oof_gap", context, lower=0.0, upper=1.0
    )
    policy_source = _validate_policy(
        frame,
        expected_policy,
        context,
        allow_legacy_missing_policy=allow_legacy_missing_policy,
    )
    frame["max_tp"] = max_tp.astype(int)
    return resolved_dir, summary_path, frame, policy_source


def _load_shared_summary(
    result_dir: str | Path,
    expected_policy: str,
    role: str,
    *,
    allow_legacy_missing_policy: bool = False,
) -> tuple[Path, Path, pd.DataFrame, str]:
    resolved_dir, summary_path = _formal_summary_path(result_dir)
    frame = pd.read_csv(summary_path)
    context = f"shared_contiguous/{role}"
    _require_columns(
        frame,
        {
            "variant",
            "macro_oof_auroc_mean",
            "positive_train_oof_gap",
            "n_seeds",
        },
        context,
    )
    if len(frame) != 1:
        raise ValueError(f"{context} must contain exactly one shared strategy row")
    frame = frame.copy()
    if frame["variant"].isna().any() or not frame["variant"].astype(str).str.strip().all():
        raise ValueError(f"{context} has a missing variant strategy")
    _validate_seed_counts(frame, context)
    _validate_numeric_column(
        frame, "macro_oof_auroc_mean", context, lower=0.0, upper=1.0
    )
    _validate_numeric_column(
        frame, "positive_train_oof_gap", context, lower=0.0, upper=1.0
    )
    policy_source = _validate_policy(
        frame,
        expected_policy,
        context,
        allow_legacy_missing_policy=allow_legacy_missing_policy,
    )
    return resolved_dir, summary_path, frame, policy_source


def _decision(
    *,
    strategy: str,
    temporal_depth: str,
    metric_scope: str,
    validation_row: pd.Series,
    fixed_row: pd.Series,
    validation_dir: Path,
    fixed_dir: Path,
    auroc_column: str,
) -> dict[str, object]:
    validation_auroc = float(validation_row[auroc_column])
    fixed_auroc = float(fixed_row[auroc_column])
    validation_gap = float(validation_row["positive_train_oof_gap"])
    fixed_gap = float(fixed_row["positive_train_oof_gap"])
    fixed_drop = validation_auroc - fixed_auroc
    eligible = fixed_drop <= AUROC_TOLERANCE + 1e-12
    fixed_has_smaller_gap = fixed_gap < validation_gap
    choose_fixed = eligible and fixed_has_smaller_gap

    if not eligible:
        reason = "fixed_ineligible_auroc_drop_exceeds_0.005"
    elif choose_fixed:
        reason = "fixed_selected_eligible_and_smaller_positive_train_oof_gap"
    else:
        reason = "best_validation_selected_fixed_gap_not_smaller"

    chosen_row = fixed_row if choose_fixed else validation_row
    chosen_dir = fixed_dir if choose_fixed else validation_dir
    return {
        "strategy": strategy,
        "temporal_depth": temporal_depth,
        "metric_scope": metric_scope,
        "validation_result_dir": str(validation_dir),
        "validation_policy": str(validation_row["checkpoint_policy"]),
        "validation_variant": str(validation_row["variant"]),
        "validation_oof_auroc": validation_auroc,
        "validation_positive_train_oof_gap": validation_gap,
        "fixed_result_dir": str(fixed_dir),
        "fixed_policy": str(fixed_row["checkpoint_policy"]),
        "fixed_variant": str(fixed_row["variant"]),
        "fixed_oof_auroc": fixed_auroc,
        "fixed_positive_train_oof_gap": fixed_gap,
        "fixed_minus_validation_oof_auroc": fixed_auroc - validation_auroc,
        "fixed_auroc_drop": fixed_drop,
        "auroc_tolerance": AUROC_TOLERANCE,
        "fixed_eligible": bool(eligible),
        "fixed_has_smaller_positive_gap": bool(fixed_has_smaller_gap),
        "chosen_result_dir": str(chosen_dir),
        "chosen_policy": str(chosen_row["checkpoint_policy"]),
        "chosen_variant": str(chosen_row["variant"]),
        "decision_reason": reason,
    }


def _normalize_frozen_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid frozen_at timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("frozen_at timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _markdown(decisions: pd.DataFrame, payload: dict[str, object]) -> str:
    lines = [
        "# Fixed-Epoch Checkpoint Policy Selection",
        "",
        f"Frozen at (UTC): `{payload['frozen_at_utc']}`",
        "",
        "This decision uses only ten-seed formal OOF summaries from the 876-patient "
        "development pool. No evaluation or test file was read.",
        "",
        "Fixed epoch is eligible when its OOF AUROC decrease is at most 0.005. "
        "An eligible fixed policy is selected only when its positive train-OOF gap "
        "is smaller; otherwise the validation-peak policy is retained.",
        "",
        "| Strategy | Scope | Validation AUROC | Fixed AUROC | Fixed drop | "
        "Validation gap | Fixed gap | Eligible | Chosen policy | Chosen result dir |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---|---|",
    ]
    for row in decisions.to_dict(orient="records"):
        lines.append(
            "| {strategy} | {temporal_depth} | {validation_oof_auroc:.6f} | "
            "{fixed_oof_auroc:.6f} | {fixed_auroc_drop:.6f} | "
            "{validation_positive_train_oof_gap:.6f} | "
            "{fixed_positive_train_oof_gap:.6f} | {eligible} | "
            "{chosen_policy} | `{chosen_result_dir}` |".format(
                **row, eligible="yes" if row["fixed_eligible"] else "no"
            )
        )
    lines.extend(
        [
            "",
            "Selection inputs: mean formal OOF AUROC and positive mean fold-train "
            "minus OOF AUROC gap.",
            "",
            "Excluded from selection: PR-AUC, temporal monotonicity, and every "
            "locked-test or evaluation result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_selection(
    *,
    independent_validation_dir: str | Path = DEFAULT_INDEPENDENT_VALIDATION_DIR,
    independent_fixed_dir: str | Path = DEFAULT_INDEPENDENT_FIXED_DIR,
    shared_validation_dir: str | Path = DEFAULT_SHARED_VALIDATION_DIR,
    shared_fixed_dir: str | Path = DEFAULT_SHARED_FIXED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_at: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    input_values = (
        independent_validation_dir,
        independent_fixed_dir,
        shared_validation_dir,
        shared_fixed_dir,
    )
    resolved_inputs = [Path(value).expanduser().resolve() for value in input_values]
    if len(set(resolved_inputs)) != 4:
        raise ValueError("all four policy-comparison result directories must be distinct")

    allow_legacy_independent = (
        resolved_inputs[0] == DEFAULT_INDEPENDENT_VALIDATION_DIR.resolve()
    )
    allow_legacy_shared = (
        resolved_inputs[2] == DEFAULT_SHARED_VALIDATION_DIR.resolve()
    )
    (
        independent_validation_resolved,
        independent_validation_summary,
        independent_validation,
        independent_validation_policy_source,
    ) = _load_independent_summary(
        independent_validation_dir,
        INDEPENDENT_VALIDATION_POLICY,
        "validation_peak",
        allow_legacy_missing_policy=allow_legacy_independent,
    )
    (
        independent_fixed_resolved,
        independent_fixed_summary,
        independent_fixed,
        independent_fixed_policy_source,
    ) = _load_independent_summary(
        independent_fixed_dir,
        FIXED_POLICY,
        "fixed_epoch",
    )
    (
        shared_validation_resolved,
        shared_validation_summary,
        shared_validation,
        shared_validation_policy_source,
    ) = _load_shared_summary(
        shared_validation_dir,
        SHARED_VALIDATION_POLICY,
        "validation_peak",
        allow_legacy_missing_policy=allow_legacy_shared,
    )
    (
        shared_fixed_resolved,
        shared_fixed_summary,
        shared_fixed,
        shared_fixed_policy_source,
    ) = _load_shared_summary(
        shared_fixed_dir,
        FIXED_POLICY,
        "fixed_epoch",
    )

    independent_validation_by_depth = independent_validation.set_index(
        "temporal_depth"
    )
    independent_fixed_by_depth = independent_fixed.set_index("temporal_depth")
    rows = []
    for depth in INDEPENDENT_DEPTHS:
        rows.append(
            _decision(
                strategy="independent",
                temporal_depth=depth,
                metric_scope="depth_specific_formal_oof_auroc",
                validation_row=independent_validation_by_depth.loc[depth],
                fixed_row=independent_fixed_by_depth.loc[depth],
                validation_dir=independent_validation_resolved,
                fixed_dir=independent_fixed_resolved,
                auroc_column="oof_auroc_mean",
            )
        )
    rows.append(
        _decision(
            strategy="shared_contiguous",
            temporal_depth="macro_T0_to_T0-T3",
            metric_scope="four_depth_macro_formal_oof_auroc",
            validation_row=shared_validation.iloc[0],
            fixed_row=shared_fixed.iloc[0],
            validation_dir=shared_validation_resolved,
            fixed_dir=shared_fixed_resolved,
            auroc_column="macro_oof_auroc_mean",
        )
    )
    decisions = pd.DataFrame(rows)

    output = Path(output_dir).expanduser()
    if output.name != "fixed_epoch_policy_selection" or output.parent.name != "analysis":
        raise ValueError(
            "output_dir must end with analysis/fixed_epoch_policy_selection"
        )
    output = output.resolve()
    if output in set(resolved_inputs):
        raise ValueError("output_dir must be isolated from all input result directories")
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "frozen selection artifacts already exist; pass overwrite=True only "
            "for an explicit replacement"
        )

    frozen_at_utc = _normalize_frozen_at(frozen_at)
    decisions["frozen_at_utc"] = frozen_at_utc
    input_records = [
        {
            "strategy": "independent",
            "role": "validation_peak",
            "result_dir": str(independent_validation_resolved),
            "formal_summary": str(independent_validation_summary),
            "expected_policy": INDEPENDENT_VALIDATION_POLICY,
            "policy_source": independent_validation_policy_source,
        },
        {
            "strategy": "independent",
            "role": "fixed_epoch",
            "result_dir": str(independent_fixed_resolved),
            "formal_summary": str(independent_fixed_summary),
            "expected_policy": FIXED_POLICY,
            "policy_source": independent_fixed_policy_source,
        },
        {
            "strategy": "shared_contiguous",
            "role": "validation_peak",
            "result_dir": str(shared_validation_resolved),
            "formal_summary": str(shared_validation_summary),
            "expected_policy": SHARED_VALIDATION_POLICY,
            "policy_source": shared_validation_policy_source,
        },
        {
            "strategy": "shared_contiguous",
            "role": "fixed_epoch",
            "result_dir": str(shared_fixed_resolved),
            "formal_summary": str(shared_fixed_summary),
            "expected_policy": FIXED_POLICY,
            "policy_source": shared_fixed_policy_source,
        },
    ]
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "frozen_at_utc": frozen_at_utc,
        "development_scope": "formal_five_fold_oof_876_patients",
        "expected_formal_seeds": EXPECTED_FORMAL_SEEDS,
        "auroc_tolerance": AUROC_TOLERANCE,
        "selection_rule": (
            "fixed_eligible_when_auroc_drop_lte_0.005_then_choose_only_if_"
            "positive_train_oof_gap_is_smaller_else_best_validation"
        ),
        "selection_inputs_used": [
            "mean_formal_oof_auroc",
            "positive_train_oof_gap",
        ],
        "selection_inputs_not_used": [
            "prauc",
            "temporal_monotonicity",
            "evaluation_metrics",
            "test_metrics",
        ],
        "test_data_used": False,
        "test_embeddings_or_labels_loaded": False,
        "evaluation_files_read": False,
        "input_files": input_records,
        "decisions": decisions.to_dict(orient="records"),
        "chosen": {
            f"{row['strategy']}/{row['temporal_depth']}": {
                "result_dir": row["chosen_result_dir"],
                "policy": row["chosen_policy"],
                "variant": row["chosen_variant"],
            }
            for row in rows
        },
    }

    _atomic_csv(output / OUTPUT_NAMES[1], decisions)
    _atomic_json(output / OUTPUT_NAMES[0], payload)
    _atomic_text(output / OUTPUT_NAMES[2], _markdown(decisions, payload))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--independent-validation-dir",
        type=Path,
        default=DEFAULT_INDEPENDENT_VALIDATION_DIR,
    )
    parser.add_argument(
        "--independent-fixed-dir",
        type=Path,
        default=DEFAULT_INDEPENDENT_FIXED_DIR,
    )
    parser.add_argument(
        "--shared-validation-dir",
        type=Path,
        default=DEFAULT_SHARED_VALIDATION_DIR,
    )
    parser.add_argument(
        "--shared-fixed-dir", type=Path, default=DEFAULT_SHARED_FIXED_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing frozen selection",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_selection(
        independent_validation_dir=args.independent_validation_dir,
        independent_fixed_dir=args.independent_fixed_dir,
        shared_validation_dir=args.shared_validation_dir,
        shared_fixed_dir=args.shared_fixed_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    for scope, choice in payload["chosen"].items():
        print(f"{scope}: {choice['policy']} -> {choice['result_dir']}")
    print(f"frozen_at_utc: {payload['frozen_at_utc']}")


if __name__ == "__main__":
    main()

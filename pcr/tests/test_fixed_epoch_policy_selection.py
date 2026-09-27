import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.select_full978_fixed_epoch_policy as selector


def _write_independent(path: Path, policy: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "temporal_depth": "T0-T2",
                "max_tp": 3,
                "variant": "independent_variant",
                "oof_auroc_mean": 0.800,
                "oof_prauc_mean": 0.99,
                "positive_train_oof_gap": 0.090,
                "n_seeds": 10,
                "checkpoint_policy": policy,
            },
            {
                "temporal_depth": "T0-T3",
                "max_tp": 4,
                "variant": "independent_variant",
                "oof_auroc_mean": 0.810,
                "oof_prauc_mean": 0.99,
                "positive_train_oof_gap": 0.050,
                "n_seeds": 10,
                "checkpoint_policy": policy,
            },
        ]
    ).to_csv(path / selector.FORMAL_SUMMARY_NAME, index=False)


def _write_independent_fixed(path: Path) -> None:
    _write_independent(path, selector.FIXED_POLICY)
    summary = pd.read_csv(path / selector.FORMAL_SUMMARY_NAME)
    summary.loc[summary["temporal_depth"] == "T0-T2", "oof_auroc_mean"] = 0.795
    summary.loc[
        summary["temporal_depth"] == "T0-T2", "positive_train_oof_gap"
    ] = 0.040
    summary.loc[summary["temporal_depth"] == "T0-T3", "oof_auroc_mean"] = 0.804
    summary.loc[
        summary["temporal_depth"] == "T0-T3", "positive_train_oof_gap"
    ] = 0.010
    summary["oof_prauc_mean"] = 0.01
    summary.to_csv(path / selector.FORMAL_SUMMARY_NAME, index=False)


def _write_shared(
    path: Path,
    policy: str,
    *,
    auroc: float,
    gap: float,
    prauc: float,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "variant": "moderate",
                "macro_oof_auroc_mean": auroc,
                "macro_oof_prauc_mean": prauc,
                "positive_train_oof_gap": gap,
                "n_seeds": 10,
                "checkpoint_policy": policy,
            }
        ]
    ).to_csv(path / selector.FORMAL_SUMMARY_NAME, index=False)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "independent_validation_dir": tmp_path / "independent_validation",
        "independent_fixed_dir": tmp_path / "independent_fixed",
        "shared_validation_dir": tmp_path / "shared_validation",
        "shared_fixed_dir": tmp_path / "shared_fixed",
        "output_dir": tmp_path / "decision" / "analysis/fixed_epoch_policy_selection",
    }
    _write_independent(
        paths["independent_validation_dir"],
        selector.INDEPENDENT_VALIDATION_POLICY,
    )
    _write_independent_fixed(paths["independent_fixed_dir"])
    _write_shared(
        paths["shared_validation_dir"],
        selector.SHARED_VALIDATION_POLICY,
        auroc=0.750,
        gap=0.070,
        prauc=0.99,
    )
    _write_shared(
        paths["shared_fixed_dir"],
        selector.FIXED_POLICY,
        auroc=0.746,
        gap=0.030,
        prauc=0.01,
    )
    return paths


def test_selector_uses_only_four_formal_oof_summaries_and_writes_frozen_outputs(
    monkeypatch, tmp_path
):
    paths = _inputs(tmp_path)
    actual_read_csv = selector.pd.read_csv
    reads = []

    def tracked_read_csv(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return actual_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(selector.pd, "read_csv", tracked_read_csv)
    payload = selector.run_selection(
        **paths,
        frozen_at="2026-09-04T01:02:03+00:00",
    )

    expected_reads = {
        (paths[key] / selector.FORMAL_SUMMARY_NAME).resolve()
        for key in (
            "independent_validation_dir",
            "independent_fixed_dir",
            "shared_validation_dir",
            "shared_fixed_dir",
        )
    }
    assert len(reads) == 4
    assert set(reads) == expected_reads
    assert all("evaluation" not in path.parts and "test" not in path.parts for path in reads)
    assert payload["frozen_at_utc"] == "2026-09-04T01:02:03Z"
    assert payload["test_data_used"] is False
    assert payload["test_embeddings_or_labels_loaded"] is False
    assert payload["evaluation_files_read"] is False
    assert set(payload["selection_inputs_used"]) == {
        "mean_formal_oof_auroc",
        "positive_train_oof_gap",
    }
    assert "prauc" in payload["selection_inputs_not_used"]

    decisions = pd.read_csv(
        paths["output_dir"] / "checkpoint_policy_selection.csv"
    ).set_index(["strategy", "temporal_depth"])
    t2 = decisions.loc[("independent", "T0-T2")]
    t3 = decisions.loc[("independent", "T0-T3")]
    shared = decisions.loc[("shared_contiguous", "macro_T0_to_T0-T3")]
    assert bool(t2["fixed_eligible"])
    assert t2["chosen_policy"] == selector.FIXED_POLICY
    assert t2["fixed_auroc_drop"] == pytest.approx(0.005)
    assert not bool(t3["fixed_eligible"])
    assert t3["chosen_policy"] == selector.INDEPENDENT_VALIDATION_POLICY
    assert shared["chosen_policy"] == selector.FIXED_POLICY
    assert Path(shared["chosen_result_dir"]) == paths["shared_fixed_dir"].resolve()

    assert {path.name for path in paths["output_dir"].iterdir()} == set(
        selector.OUTPUT_NAMES
    )
    saved = json.loads(
        (paths["output_dir"] / "checkpoint_policy_selection.json").read_text()
    )
    assert saved["chosen"]["independent/T0-T2"]["policy"] == "final_epoch"
    report = (
        paths["output_dir"] / "checkpoint_policy_selection.md"
    ).read_text()
    assert "No evaluation or test file was read" in report
    assert "PR-AUC" in report and "Excluded from selection" in report


def test_eligible_fixed_policy_requires_strictly_smaller_positive_gap(tmp_path):
    paths = _inputs(tmp_path)
    fixed_path = paths["independent_fixed_dir"] / selector.FORMAL_SUMMARY_NAME
    fixed = pd.read_csv(fixed_path)
    fixed.loc[
        fixed["temporal_depth"] == "T0-T2", "positive_train_oof_gap"
    ] = 0.090
    fixed.to_csv(fixed_path, index=False)

    payload = selector.run_selection(
        **paths,
        frozen_at="2026-09-04T00:00:00Z",
    )
    decision = next(
        row
        for row in payload["decisions"]
        if row["strategy"] == "independent" and row["temporal_depth"] == "T0-T2"
    )
    assert decision["fixed_eligible"] is True
    assert decision["fixed_has_smaller_positive_gap"] is False
    assert decision["chosen_policy"] == selector.INDEPENDENT_VALIDATION_POLICY
    assert decision["decision_reason"] == "best_validation_selected_fixed_gap_not_smaller"


@pytest.mark.parametrize(
    ("target_key", "mutation", "message"),
    [
        (
            "independent_fixed_dir",
            lambda frame: frame.assign(n_seeds=9),
            "exactly 10 formal seeds",
        ),
        (
            "shared_fixed_dir",
            lambda frame: frame.assign(checkpoint_policy="best_validation_macro"),
            "checkpoint_policy must be 'final_epoch'",
        ),
        (
            "independent_validation_dir",
            lambda frame: frame.assign(temporal_depth=["T0-T1", "T0-T3"]),
            "temporal depths",
        ),
        (
            "independent_validation_dir",
            lambda frame: frame.drop(columns=["checkpoint_policy"]),
            "missing required checkpoint_policy",
        ),
        (
            "shared_validation_dir",
            lambda frame: pd.concat([frame, frame], ignore_index=True),
            "exactly one shared strategy row",
        ),
    ],
)
def test_selector_rejects_incomplete_or_wrong_formal_contract_before_writing(
    tmp_path, target_key, mutation, message
):
    paths = _inputs(tmp_path)
    summary_path = paths[target_key] / selector.FORMAL_SUMMARY_NAME
    mutation(pd.read_csv(summary_path)).to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match=message):
        selector.run_selection(
            **paths,
            frozen_at="2026-09-04T00:00:00Z",
        )
    assert not paths["output_dir"].exists()


def test_selector_rejects_missing_summary_before_writing(tmp_path):
    paths = _inputs(tmp_path)
    (paths["shared_fixed_dir"] / selector.FORMAL_SUMMARY_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="missing formal summary"):
        selector.run_selection(**paths)
    assert not paths["output_dir"].exists()


def test_selector_refuses_evaluation_path_without_reading_it(monkeypatch, tmp_path):
    paths = _inputs(tmp_path)
    unsafe = tmp_path / "evaluation" / "independent_validation"
    _write_independent(unsafe, selector.INDEPENDENT_VALIDATION_POLICY)
    paths["independent_validation_dir"] = unsafe

    def fail_read(*args, **kwargs):
        raise AssertionError("read_csv must not be called for an evaluation path")

    monkeypatch.setattr(selector.pd, "read_csv", fail_read)
    with pytest.raises(ValueError, match="refusing non-development input"):
        selector.run_selection(**paths)
    assert not paths["output_dir"].exists()


def test_custom_output_must_use_isolated_analysis_directory(tmp_path):
    paths = _inputs(tmp_path)
    paths["output_dir"] = tmp_path / "not_a_frozen_analysis"

    with pytest.raises(ValueError, match="analysis/fixed_epoch_policy_selection"):
        selector.run_selection(**paths)
    assert not paths["output_dir"].exists()

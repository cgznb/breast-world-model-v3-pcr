import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.compare_full978_final_optimization as comparison


def _result_dirs(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for strategy in comparison.STRATEGY_SPECS:
        path = tmp_path / "inputs" / strategy
        path.mkdir(parents=True)
        result[strategy] = path
    return result


def _policy(result_dirs: dict[str, Path]) -> dict:
    decisions = [
        {
            "strategy": "independent",
            "temporal_depth": "T0-T2",
            "fixed_auroc_drop": 0.01747,
            "auroc_tolerance": 0.005,
            "validation_positive_train_oof_gap": 0.08869,
            "fixed_positive_train_oof_gap": 0.12357,
        },
        {
            "strategy": "independent",
            "temporal_depth": "T0-T3",
            "fixed_auroc_drop": 0.00842,
            "auroc_tolerance": 0.005,
            "validation_positive_train_oof_gap": 0.10452,
            "fixed_positive_train_oof_gap": 0.08714,
        },
        {
            "strategy": "shared_contiguous",
            "temporal_depth": "macro_T0_to_T0-T3",
            "fixed_auroc_drop": 0.00499,
            "auroc_tolerance": 0.005,
            "validation_positive_train_oof_gap": 0.05436,
            "fixed_positive_train_oof_gap": 0.04751,
        },
    ]
    return {
        "complete": True,
        "frozen_at_utc": "2026-09-03T16:37:21Z",
        "test_data_used": False,
        "test_embeddings_or_labels_loaded": False,
        "evaluation_files_read": False,
        "selection_inputs_not_used": ["prauc", "test_metrics"],
        "chosen": {
            "independent/T0-T2": {
                "policy": "best_validation",
                "variant": "layerwise_adamw",
                "result_dir": str(result_dirs["independent_regularization_v2"]),
            },
            "independent/T0-T3": {
                "policy": "best_validation",
                "variant": "residual_1e3",
                "result_dir": str(result_dirs["independent_regularization_v2"]),
            },
            "shared_contiguous/macro_T0_to_T0-T3": {
                "policy": "final_epoch",
                "variant": "moderate",
                "result_dir": str(result_dirs["shared_contiguous_fixed12"]),
            },
        },
        "decisions": decisions,
    }


def _fake_rows(strategy: str, spec: dict) -> list[dict]:
    rows = []
    for depth in spec["depths"]:
        row = comparison._base_row(
            strategy,
            spec,
            depth,
            "fixture_variant",
            spec["expected_parameters"][depth],
        )
        row.update(
            {
                "development_auroc_mean": 0.70,
                "development_auroc_std": 0.01,
                "development_prauc_mean": (
                    None
                    if spec["development_estimate"] == "heldout_validation_not_oof"
                    else 0.55
                ),
                "development_prauc_std": (
                    None
                    if spec["development_estimate"] == "heldout_validation_not_oof"
                    else 0.02
                ),
            }
        )
        rows.append(row)
    return rows


def _fake_test_values(spec: dict) -> dict:
    return {
        source: {
            depth: {
                f"{metric}_{suffix}": 0.75 if suffix == "mean" else 0.01
                for metric in comparison.METRICS
                for suffix in ("mean", "std")
            }
            for depth in spec["depths"]
        }
        for source in ("real", "generated")
    }


def test_run_comparison_writes_22_rows_and_preserves_real_missing_values(
    monkeypatch, tmp_path
):
    result_dirs = _result_dirs(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy(result_dirs)))
    calls = []

    def fake_no_cv(result_dir, spec, inputs_read):
        calls.append("development:independent_no_cv")
        return _fake_rows("independent_no_cv", spec)

    def fake_cv(strategy, result_dir, spec, inputs_read):
        calls.append(f"development:{strategy}")
        return _fake_rows(strategy, spec)

    def fake_test(strategy, result_dir, spec, inputs_read, reference_labels):
        calls.append(f"test:{strategy}")
        if not spec["test_evaluated"]:
            return {}, reference_labels
        if reference_labels is None:
            reference_labels = {
                f"P{index:03d}": int(index < 32) for index in range(102)
            }
        return _fake_test_values(spec), reference_labels

    monkeypatch.setattr(comparison, "_load_no_cv_development", fake_no_cv)
    monkeypatch.setattr(comparison, "_load_cv_development", fake_cv)
    monkeypatch.setattr(comparison, "_load_test_metrics", fake_test)
    output = tmp_path / "report" / "analysis/final_optimization_comparison"
    payload = comparison.run_comparison(
        result_dirs=result_dirs,
        policy_path=policy_path,
        output_dir=output,
        generated_at="2026-09-04T01:02:03Z",
    )

    assert calls[0] == "development:independent_no_cv"
    assert payload["inputs_read"] == [str(policy_path.resolve())]
    assert payload["test_metrics_used_to_change_frozen_decision"] is False
    assert len(payload["rows"]) == 22
    assert {path.name for path in output.iterdir()} == set(comparison.OUTPUT_NAMES)

    frame = pd.read_csv(output / comparison.OUTPUT_NAMES[0])
    fixed = frame[frame["strategy"] == "independent_fixed_epoch"]
    shared_peak = frame[
        frame["strategy"] == "shared_contiguous_validation_peak"
    ]
    no_cv = frame[frame["strategy"] == "independent_no_cv"]
    assert fixed["real_auroc_mean"].isna().all()
    assert fixed["generated_auroc_mean"].isna().all()
    assert shared_peak["real_auroc_mean"].isna().all()
    assert set(fixed["test_evaluation_status"]) == {"not_evaluated"}
    assert set(no_cv["development_estimate"]) == {"heldout_validation_not_oof"}

    saved = json.loads((output / comparison.OUTPUT_NAMES[1]).read_text())
    fixed_json = [
        row for row in saved["rows"] if row["strategy"] == "independent_fixed_epoch"
    ]
    assert all(row["real_auroc_mean"] is None for row in fixed_json)
    assert all(row["generated_auroc_mean"] is None for row in fixed_json)
    report = (output / comparison.OUTPUT_NAMES[2]).read_text()
    assert "98-val; not OOF" in report
    assert "not evaluated" in report
    assert "never re-ranks them using real or generated locked102" in report


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(test_data_used=True), "development-only"),
        (
            lambda value: value["chosen"]["independent/T0-T2"].update(
                policy="final_epoch"
            ),
            "winner mismatch",
        ),
        (
            lambda value: value.update(selection_inputs_not_used=["prauc"]),
            "exclude test metrics",
        ),
    ],
)
def test_frozen_policy_rejects_test_contamination_or_changed_winner(
    tmp_path, mutation, message
):
    result_dirs = _result_dirs(tmp_path)
    payload = _policy(result_dirs)
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        comparison._validate_frozen_policy(payload, result_dirs)


def _locked_predictions(depths=("T0",), split="test_fold_ensemble"):
    rows = []
    for depth in depths:
        for seed in comparison.SEEDS:
            for index in range(102):
                label = int(index < 32)
                rows.append(
                    {
                        "patient_id": f"P{index:03d}",
                        "label": label,
                        "probability": 0.9 if label else 0.1,
                        "seed": seed,
                        "split": split,
                        "temporal_depth": depth,
                        "max_tp": comparison.DEPTH_TO_MAX_TP[depth],
                        "source": "real",
                    }
                )
    return pd.DataFrame(rows)


def test_prediction_contract_recomputes_metrics_on_exact_locked102():
    predictions = _locked_predictions()

    labels, metrics = comparison._validate_test_predictions(
        predictions,
        strategy="fixture",
        source="real",
        depths=("T0",),
        folds=5,
        reference_labels=None,
    )

    assert len(labels) == 102
    assert sum(labels.values()) == 32
    assert metrics["T0"]["auroc_mean"] == pytest.approx(1.0)
    assert metrics["T0"]["prauc_mean"] == pytest.approx(1.0)
    assert metrics["T0"]["auroc_std"] == pytest.approx(0.0)


@pytest.mark.parametrize("failure", ["missing_patient", "wrong_membership", "wrong_split"])
def test_prediction_contract_rejects_non_locked102_or_wrong_protocol(failure):
    predictions = _locked_predictions()
    if failure == "missing_patient":
        predictions = predictions.drop(index=0)
        message = "not exactly locked102"
    elif failure == "wrong_membership":
        selected = (predictions["seed"] == 43) & (
            predictions["patient_id"] == "P101"
        )
        predictions.loc[selected, "patient_id"] = "P999"
        message = "membership mismatch"
    else:
        predictions["split"] = "test"
        message = "split mismatch"

    with pytest.raises(ValueError, match=message):
        comparison._validate_test_predictions(
            predictions,
            strategy="fixture",
            source="real",
            depths=("T0",),
            folds=5,
            reference_labels=None,
        )


def test_cv_loader_requires_explicit_final_epoch_policy(tmp_path):
    result_dir = tmp_path / "fixed"
    result_dir.mkdir()
    rows = []
    for seed in comparison.SEEDS:
        for depth in ("T0-T2", "T0-T3"):
            rows.append(
                {
                    "seed": seed,
                    "temporal_depth": depth,
                    "max_tp": comparison.DEPTH_TO_MAX_TP[depth],
                    "variant": "fixed_epoch_v2_selected",
                    "parameters": comparison.STRATEGY_SPECS[
                        "independent_fixed_epoch"
                    ]["expected_parameters"][depth],
                    "auroc": 0.7,
                    "prauc": 0.55,
                    "train_oof_gap": 0.05,
                }
            )
    pd.DataFrame(rows).to_csv(result_dir / "formal_oof_metrics_all.csv", index=False)

    with pytest.raises(ValueError, match="must record final_epoch explicitly"):
        comparison._load_cv_development(
            "independent_fixed_epoch",
            result_dir,
            comparison.STRATEGY_SPECS["independent_fixed_epoch"],
            [],
        )


def test_not_evaluated_strategy_rejects_hidden_evaluation_directory(tmp_path):
    result_dir = tmp_path / "fixed"
    (result_dir / "evaluation").mkdir(parents=True)
    spec = comparison.STRATEGY_SPECS["independent_fixed_epoch"]

    with pytest.raises(ValueError, match="declared not evaluated"):
        comparison._load_test_metrics(
            "independent_fixed_epoch", result_dir, spec, [], None
        )

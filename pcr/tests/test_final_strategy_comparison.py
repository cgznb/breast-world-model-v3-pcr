import numpy as np
import pandas as pd
import pytest

import scripts.compare_full978_final_strategies as comparison


def _predictions(strategy, source="real"):
    patient_ids = [f"P{index:03d}" for index in range(102)]
    rows = []
    for depth, max_tp in comparison.DEPTHS:
        _, models_per_seed = comparison._development_protocol(strategy, depth)
        split = "test_fold_ensemble" if models_per_seed == 5 else "test"
        for seed in comparison.SEEDS:
            for index, patient_id in enumerate(patient_ids):
                label = int(index < 32)
                rows.append(
                    {
                        "patient_id": patient_id,
                        "label": label,
                        "probability": 0.8 if label else 0.2,
                        "seed": seed,
                        "split": split,
                        "temporal_depth": depth,
                        "max_tp": max_tp,
                        "source": source,
                        "strategy": strategy,
                    }
                )
    return patient_ids, pd.DataFrame(rows)


@pytest.mark.parametrize("source", comparison.SOURCES)
@pytest.mark.parametrize("strategy", comparison.STRATEGIES)
def test_prediction_contract_accepts_protocol_specific_test_splits(strategy, source):
    patient_ids, predictions = _predictions(strategy, source)

    comparison._validate_predictions(predictions, strategy, source, patient_ids)


@pytest.mark.parametrize(
    ("strategy", "depth"),
    [
        ("independent_final", "T0"),
        ("independent_final", "T0-T2"),
        ("shared_contiguous", "T0-T3"),
        ("shared_full_random_5fold", "T0-T1"),
    ],
)
def test_prediction_contract_rejects_wrong_split_at_one_depth(strategy, depth):
    patient_ids, predictions = _predictions(strategy)
    selected = predictions["temporal_depth"] == depth
    current = predictions.loc[selected, "split"].iloc[0]
    wrong = "test" if current == "test_fold_ensemble" else "test_fold_ensemble"
    predictions.loc[selected, "split"] = wrong

    with pytest.raises(ValueError, match=rf"{strategy}/real/{depth} test split mismatch"):
        comparison._validate_predictions(predictions, strategy, "real", patient_ids)


def test_metric_summary_preserves_locked_and_fixed_exclusion_cohorts():
    qc_ids = {f"P{index:03d}" for index in range(5)} | {
        f"P{index:03d}" for index in range(32, 36)
    }
    frames = []
    for strategy in comparison.STRATEGIES:
        for source in comparison.SOURCES:
            _, frame = _predictions(strategy, source)
            frames.append(frame)
    metrics = comparison._metric_rows(pd.concat(frames, ignore_index=True), qc_ids)
    summary = comparison._summarize(metrics)

    assert len(metrics) == 480
    assert len(summary) == 48
    assert set(summary.loc[summary["cohort"] == "full_locked102", "n_patients"]) == {102}
    assert set(summary.loc[summary["cohort"] == "full_locked102", "n_positive"]) == {32}
    excluded = summary[summary["cohort"] == "exclude_all_9_qc_patients"]
    assert set(excluded["n_patients"]) == {93}
    assert set(excluded["n_positive"]) == {27}
    assert np.isfinite(summary[[f"{key}_mean" for key in comparison.METRIC_KEYS]]).all().all()

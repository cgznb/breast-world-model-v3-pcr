import pandas as pd
import pytest

import scripts.analyze_full978_independent_cv as analysis


def _prediction_rows(patient_ids):
    rows = []
    for source in analysis.SOURCES:
        for depth, max_tp in analysis.DEPTHS:
            for seed in analysis.SEEDS:
                for index, patient_id in enumerate(patient_ids):
                    label = index % 2
                    probability = 0.2 + 0.55 * label + 0.001 * (seed - 42)
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "label": label,
                            "probability": probability,
                            "seed": seed,
                            "temporal_depth": depth,
                            "max_tp": max_tp,
                            "source": source,
                        }
                    )
    return pd.DataFrame(rows)


def test_qc_sensitivity_uses_predeclared_depth_visible_patient_sets():
    flagged = sorted(analysis.excluded_patient_ids("exclude_all_9_qc_patients", "T0-T3"))
    patient_ids = flagged + [f"CONTROL-{index:02d}" for index in range(12)]

    metrics, summary = analysis._qc_sensitivity(_prediction_rows(patient_ids))

    assert len(metrics) == len(analysis.SOURCES) * len(analysis.DEPTHS) * len(analysis.SEEDS) * 3
    selected = summary.set_index(["source", "temporal_depth", "cohort"])
    assert selected.loc[("real", "T0-T2", "exclude_depth_visible_qc_patients"), "n_excluded"] == 3
    assert selected.loc[("real", "T0-T3", "exclude_depth_visible_qc_patients"), "n_excluded"] == 9
    assert selected.loc[("generated", "T0-T2", "exclude_all_9_qc_patients"), "n_excluded"] == 9
    assert set(metrics["analysis_status"]) == {analysis.ANALYSIS_STATUS}


def test_oof_thresholds_are_applied_unchanged_to_both_test_sources():
    rows = []
    probabilities = {
        "real": [0.1, 0.3, 0.65, 0.9],
        "generated": [0.2, 0.5, 0.55, 0.8],
    }
    labels = [0, 0, 1, 1]
    thresholds = []
    for depth, max_tp in analysis.DEPTHS:
        threshold = 0.4 if depth == "T0-T2" else 0.6
        for seed in analysis.SEEDS:
            thresholds.append(
                {
                    "temporal_depth": depth,
                    "max_tp": max_tp,
                    "seed": seed,
                    "threshold": threshold,
                }
            )
            for source in analysis.SOURCES:
                for index, (label, probability) in enumerate(
                    zip(labels, probabilities[source])
                ):
                    rows.append(
                        {
                            "patient_id": f"P{index}",
                            "label": label,
                            "probability": probability,
                            "seed": seed,
                            "temporal_depth": depth,
                            "max_tp": max_tp,
                            "source": source,
                        }
                    )

    metrics, summary = analysis._threshold_test_metrics(
        pd.DataFrame(rows), pd.DataFrame(thresholds)
    )

    assert len(metrics) == len(analysis.SOURCES) * len(analysis.DEPTHS) * len(analysis.SEEDS)
    assert set(metrics["threshold_selected_on"]) == {"formal_oof_876"}
    selected = summary.set_index(["source", "temporal_depth"])
    assert selected.loc[("real", "T0-T2"), "threshold_mean"] == pytest.approx(0.4)
    assert selected.loc[("generated", "T0-T2"), "threshold_mean"] == pytest.approx(0.4)
    assert selected.loc[("real", "T0-T3"), "balanced_accuracy_mean"] == pytest.approx(1.0)
    assert selected.loc[("generated", "T0-T3"), "balanced_accuracy_mean"] == pytest.approx(0.75)

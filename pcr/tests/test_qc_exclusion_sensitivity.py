import numpy as np
import pandas as pd
import pytest

import scripts.analyze_dual_strategy_qc_sensitivity as qc


QC_IDS = {
    "ISPY2-294265",
    "ISPY2-804233",
    "ISPY2-355292",
    "ISPY2-614434",
    "ISPY2-797499",
    "ISPY2-493775",
    "ISPY2-703401",
    "ISPY2-341238",
    "ISPY2-900959",
}


def test_qc_patient_sets_follow_first_visible_flagged_visit():
    assert qc._qc_patient_ids() == QC_IDS
    assert qc.excluded_patient_ids("full_locked102", "T0-T3") == set()
    assert qc.excluded_patient_ids("exclude_depth_visible_qc_patients", "T0") == set()
    assert qc.excluded_patient_ids(
        "exclude_depth_visible_qc_patients", "T0-T1"
    ) == {"ISPY2-355292"}
    assert qc.excluded_patient_ids(
        "exclude_depth_visible_qc_patients", "T0-T2"
    ) == {"ISPY2-355292", "ISPY2-797499", "ISPY2-703401"}
    assert qc.excluded_patient_ids("exclude_depth_visible_qc_patients", "T0-T3") == QC_IDS
    assert qc.excluded_patient_ids("exclude_all_9_qc_patients", "T0") == QC_IDS


def _synthetic_predictions(patient_ids, *, duplicate=False, disagree=False):
    rows = []
    labels = {patient_id: index % 2 for index, patient_id in enumerate(patient_ids)}
    for strategy in qc.STRATEGIES:
        for source in qc.SOURCES:
            for seed in (42, 43):
                for depth_index, depth in enumerate(qc.DEPTHS):
                    for patient_index, patient_id in enumerate(patient_ids):
                        label = labels[patient_id]
                        if disagree and strategy == "shared_contiguous" and patient_index == 0:
                            label = 1 - label
                        rows.append(
                            {
                                "patient_id": patient_id,
                                "label": label,
                                "probability": float(
                                    np.clip(0.15 + 0.65 * labels[patient_id] + patient_index / 1000, 0, 1)
                                ),
                                "seed": seed,
                                "strategy": strategy,
                                "temporal_depth": depth,
                                "max_tp": depth_index + 1,
                                "source": source,
                                "split": "test",
                            }
                        )
    if duplicate:
        rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


def test_prediction_contract_rejects_duplicates_and_label_disagreement():
    patient_ids = sorted(QC_IDS | {"SAFE-0", "SAFE-1"})
    good = _synthetic_predictions(patient_ids)
    qc.validate_prediction_contract(good, patient_ids, seeds=(42, 43))

    with pytest.raises(ValueError, match="duplicate prediction"):
        qc.validate_prediction_contract(
            _synthetic_predictions(patient_ids, duplicate=True), patient_ids, seeds=(42, 43)
        )
    with pytest.raises(ValueError, match="labels disagree"):
        qc.validate_prediction_contract(
            _synthetic_predictions(patient_ids, disagree=True), patient_ids, seeds=(42, 43)
        )


def test_metrics_preserve_full_and_apply_only_predeclared_exclusions():
    patient_ids = sorted(QC_IDS | {"SAFE-0", "SAFE-1"})
    predictions = _synthetic_predictions(patient_ids)
    metrics = qc.compute_sensitivity_metrics(
        predictions,
        patient_ids,
        strategies=("independent",),
        sources=("real",),
        seeds=(42, 43),
    )

    assert len(metrics) == 2 * 4 * 3
    one_seed = metrics[metrics["seed"] == 42].set_index(["temporal_depth", "cohort"])
    assert one_seed.loc[("T0", "full_locked102"), "n_patients"] == len(patient_ids)
    assert one_seed.loc[("T0", "exclude_depth_visible_qc_patients"), "n_excluded"] == 0
    assert not one_seed.loc[
        ("T0", "exclude_depth_visible_qc_patients"), "cohort_alters_locked_test"
    ]
    assert one_seed.loc[("T0-T1", "exclude_depth_visible_qc_patients"), "n_excluded"] == 1
    assert one_seed.loc[("T0-T2", "exclude_depth_visible_qc_patients"), "n_excluded"] == 3
    assert one_seed.loc[("T0-T3", "exclude_depth_visible_qc_patients"), "n_excluded"] == 9
    assert one_seed.loc[("T0", "exclude_all_9_qc_patients"), "n_patients"] == 2
    assert not metrics["selection_uses_predictions"].any()
    assert not metrics["selection_uses_labels"].any()
    assert set(metrics["analysis_status"]) == {qc.ANALYSIS_STATUS}

    summary = qc.summarize_metrics(metrics)
    detail, delta_summary = qc.paired_deltas(metrics)
    assert len(summary) == 4 * 3
    assert len(detail) == 2 * 4 * 2
    assert len(delta_summary) == 4 * 2
    unchanged = detail[
        (detail["cohort"] == "exclude_depth_visible_qc_patients")
        & (detail["temporal_depth"] == "T0")
    ]
    np.testing.assert_array_equal(unchanged["auroc_delta_vs_full"], 0.0)


def test_qc_source_must_contain_every_declared_visit_and_post_hoc_warning(tmp_path):
    report = tmp_path / "audit.md"
    markers = "\n".join(f'{row["patient_id"]} {row["visit"]}' for row in qc.QC_VISITS)
    report.write_text(
        "# Locked-102 Temporal-Depth Regression Audit\n\nThis is post-hoc.\n" + markers
    )
    qc._validate_qc_source(report)

    report.write_text("# Locked-102 Temporal-Depth Regression Audit\n\nThis is post-hoc.\n")
    with pytest.raises(ValueError, match="missing declared visit"):
        qc._validate_qc_source(report)

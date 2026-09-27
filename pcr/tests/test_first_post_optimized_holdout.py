import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from scripts.evaluate_first_post_optimized_holdout import (
    SOURCES, aggregate, attach_resized_t0, paired_auc_difference, validate_patient_boundary,
)


def test_holdout_checkpoint_boundary_and_inner_model_rejection():
    task = {"train_ids": ["A", "B"], "val_ids": ["C"], "stage": "outer"}
    validate_patient_boundary(task, ["A", "B", "C"], ["H"])
    with pytest.raises(ValueError, match="boundary"):
        validate_patient_boundary(task, ["A", "B", "C"], ["C"])
    with pytest.raises(ValueError, match="outer"):
        validate_patient_boundary({**task, "stage": "inner"}, ["A", "B", "C"], ["H"])


def test_paired_auc_matches_sklearn_mean_seeds_including_ties():
    labels = np.array([0, 1, 0, 1, 0, 1])
    left = np.array([[.1, .7, .4, .7, .8, .8], [.2, .3, .3, .6, .6, .5]])
    right = np.array([[.6, .4, .4, .4, .4, .6], [.6, .5, .2, .6, .2, .2]])
    result = paired_auc_difference(labels, left, right, samples=100)
    expected = np.mean([roc_auc_score(labels, a) - roc_auc_score(labels, b) for a, b in zip(left, right)])
    assert result["difference"] == pytest.approx(expected, abs=1e-12)
    equal = paired_auc_difference(labels, left, left, samples=100)
    assert equal["difference"] == equal["ci95_low"] == equal["ci95_high"] == 0


def test_resize_only_replaces_t0_and_preserves_evaluation_metadata(tmp_path):
    real = {"pids": ["P0", "P1"], "embs": np.zeros((2, 4, 1152), np.float32),
            "masks": np.ones((2, 4), np.float32), "days": np.zeros((2, 4), np.float32),
            "clinical": np.zeros((2, 17)), "labels": np.array([0, 1])}
    for pid in real["pids"]:
        (tmp_path / pid).mkdir()
        torch.save(torch.ones(1152), tmp_path / pid / f"{pid}_T0.pt")
    result = attach_resized_t0(real, tmp_path)
    assert np.allclose(np.linalg.norm(result["embs"][:, 0], axis=1), 1)
    assert not result["embs"][:, 1:].any()
    for key in ("pids", "masks", "days", "clinical", "labels"):
        np.testing.assert_array_equal(result[key], real[key])
    torch.save(torch.ones(1152), tmp_path / "P0/P0_T1.pt")
    with pytest.raises(ValueError, match="inventory"):
        attach_resized_t0(real, tmp_path)


def test_report_uses_same_96_patients_and_averages_folds_before_auc(tmp_path):
    baseline, output = tmp_path / "baseline", tmp_path / "evaluation"
    (baseline / "evaluation").mkdir(parents=True)
    ids = [f"P{i:03d}" for i in range(96)]
    labels = (np.arange(96) % 3 == 0).astype(int)
    rng = np.random.default_rng(3)
    old_rows, old_scores, records, references, expected = [], [], [], [], []
    for seed in (42, 43):
        old_probability = rng.uniform(.1, .9, len(ids))
        for source in SOURCES:
            old_rows.append(pd.DataFrame({"patient_id": ids, "label": labels, "source": source,
                                         "seed": seed, "temporal_depth": "T0", "probability": old_probability}))
        old_scores.append(roc_auc_score(labels, old_probability))
        for fold in range(5):
            model_id = len(records)
            path = f"model_path_{model_id}"
            records.append({"path": path, "model_id": model_id})
            references.append({"path": path, "arm": "physical_roi_selected"})
            probability = rng.uniform(.1, .9, len(ids))
            if seed == 42:
                expected.append(probability)
            rows = [pd.DataFrame({"patient_id": ids, "label": labels, "source": source, "depth": 1,
                                   "seed": seed, "outer": fold, "probability": probability}) for source in SOURCES]
            directory = output / "models" / f"model_{model_id:04d}"
            directory.mkdir(parents=True)
            pd.concat(rows).to_csv(directory / "predictions.csv", index=False)
    pd.concat(old_rows).to_csv(baseline / "evaluation/predictions.csv", index=False)
    pd.DataFrame([{"source": s, "temporal_depth": "T0", "population": "all_prefixes",
                   "threshold_policy": "development_oof_bacc", "auroc_mean": np.mean(old_scores)}
                  for s in SOURCES]).to_csv(baseline / "evaluation/summary.csv", index=False)
    cfg = {"baseline_run": str(baseline), "formal_seeds": [42, 43], "depths": [1]}
    contract = {"records": records, "references": references, "holdout_ids": ids,
                "bootstrap_samples": 32, "generator_snapshots": {}, "evaluation_role": "internal_holdout"}
    summary = aggregate(cfg, output, contract, ["physical_roi_selected"], "unit")
    assert len(summary) == 6
    predictions = pd.read_csv(output / "reports/unit/predictions.csv")
    selected = predictions[(predictions.version == "physical_roi_selected") & (predictions.source == "real") & (predictions.seed == 42)]
    np.testing.assert_allclose(selected.sort_values("patient_id").probability, np.mean(expected, axis=0), atol=1e-15)
    metrics = pd.read_csv(output / "reports/unit/metrics_per_seed.csv")
    row = metrics[(metrics.version == "physical_roi_selected") & (metrics.source == "real") & (metrics.seed == 42)].iloc[0]
    assert row.auroc == pytest.approx(roc_auc_score(labels, np.mean(expected, axis=0)), abs=1e-12)

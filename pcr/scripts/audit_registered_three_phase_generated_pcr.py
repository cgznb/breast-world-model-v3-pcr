"""Independently replay saved generated-pCR aggregates and cohort accounting."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.first_post_pcr_data import identity, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/registered_three_phase_roi32_pcr_v1_20260917"
OUTPUT = ROOT / "results/registered_three_phase_roi32_generated_pcr_v1_20260918"
WORLD = ROOT.parent / "MeWM-ISPY2/runs/registered_dce0_roi32_firstpostmask_v1"


def cohort_accounting():
    old = ROOT / "data/mewm_ispy2_full978_locked102"
    development = set(sum([(old / "splits" / f"{name}_ids.txt").read_text().split()
                           for name in ("train", "val")], []))
    holdout = set((old / "splits/test_ids.txt").read_text().split())
    cohort = read_json(SOURCE / "cohort.json")
    retained = set(cohort["split"]["train"])
    inventory = read_json(WORLD / "data/inventory.json")
    bundle = Path(inventory["data_configuration"]["bundle_dir"])
    bundle_development = set(read_json(bundle / "split.json")["train"])
    outside = development - bundle_development
    empty_t0 = bundle_development - retained
    assert retained < bundle_development < development
    assert holdout == set(cohort["split"]["val"])
    assert empty_t0 == {r["patient_id"] for r in inventory["exclusions"]
                        if r["reason"] == "originally_empty_model_T0"}
    old_manifest = pd.read_csv(old / "patient_manifest.csv")
    omitted = old_manifest[old_manifest.patient_id.isin(outside)]
    excluded = pd.read_csv(bundle / "exclusions.csv")
    excluded = excluded[excluded.patient_id.isin(outside)]
    reason_patients = {}
    for row in excluded.itertuples():
        for reason in row.reason.split("+"):
            reason_patients.setdefault(reason, set()).add(row.patient_id)
    report = {"old_development": len(development), "bundle_development": len(bundle_development),
              "roi32_development": len(retained), "omitted_before_roi32": len(outside),
              "empty_T0_mask_excluded": len(empty_t0), "unchanged_holdout": len(holdout),
              "outside_bundle_registered_visit_patterns": dict(Counter(omitted.registered_timepoints)),
              "outside_bundle_with_transition_exclusion_records": int(excluded.patient_id.nunique()),
              "outside_bundle_exclusion_reason_patient_counts_overlapping": {k: len(v) for k, v in reason_patients.items()},
              "old_development_positives": int(old_manifest[old_manifest.patient_id.isin(development)].label.sum()),
              "roi32_development_positives": int(old_manifest[old_manifest.patient_id.isin(retained)].label.sum()),
              "interpretation": "Inherited generator cohort restriction, not a tensor-size requirement or proof of absent MRI",
              "checked_utc": now()}
    write_json(OUTPUT / "cohort_comparison.json", report)
    return report


def audit():
    cohort = read_json(SOURCE / "cohort.json")
    protocol = read_json(OUTPUT / "protocol.json")
    routes = read_json(OUTPUT / "routes.json")["routes"]
    cases, vectors, maximum_replay = 0, 0, 0.
    for route in routes:
        receipt = read_json(OUTPUT / "cases" / (route["key"] + ".json"))
        assert receipt["route"] == route and receipt["protocol"] == identity(OUTPUT / "protocol.json")
        for artifact in receipt["artifacts"]:
            assert identity(artifact["path"]) == artifact["identity"]
        with np.load(OUTPUT / "volumes" / (route["key"] + ".npz"), allow_pickle=False) as stored:
            samples = stored["samples"]
            assert samples.shape == (4, 3, 32, 128, 128) and np.isfinite(samples).all()
            assert stored["source_foreground"].shape == samples.shape[1:]
        for draw in range(4):
            path = OUTPUT / "embeddings" / f"draw_{draw}" / route["patient_id"] / (route["key"] + ".pt")
            vector = torch.load(path, map_location="cpu", weights_only=True)
            assert vector.shape == (1152,) and torch.isfinite(vector).all() and vector.norm() > 0
            vectors += 1
        guard = receipt["source_only_audit"]
        assert guard["passed"] and guard["future_image_reads"] == guard["future_latent_reads"] == 0
        assert not guard["future_masks_used"]
        maximum_replay = max(maximum_replay, receipt["T0_pillar_replay_max_error"])
        cases += 1
    for model in protocol["classifiers"]:
        assert identity(model["path"]) == model["identity"]
    contract = read_json(SOURCE / "input_contract.json")
    for section in ("runtime_sources", "input_sources", "crop_reports"):
        for path, expected in contract[section].items():
            assert identity(path) == expected

    keys = ["source", "temporal_depth", "seed", "patient_id", "label"]
    folds = pd.read_csv(OUTPUT / "evaluation/fold_predictions.csv", dtype={"probability": np.float32})
    predictions = pd.read_csv(OUTPUT / "evaluation/predictions.csv", dtype={"probability": np.float32})
    assert len(folds) == 122400 and len(predictions) == 28560
    assert not folds.duplicated(keys + ["fold"]).any() and not predictions.duplicated(keys).any()
    counts = folds.groupby(keys).fold.nunique()
    assert (counts == 5).all()
    replay = folds.groupby(keys).probability.mean()
    predicted = predictions.set_index(keys).probability
    fold_error = float((replay - predicted.loc[replay.index]).abs().max())
    assert fold_error == 0
    draws = predictions[predictions.source.str.startswith("draw_")]
    monte_carlo = draws.groupby(keys[1:]).probability.mean()
    mc_saved = predictions[predictions.source == "generated_mc4"].set_index(keys[1:]).probability
    assert float((monte_carlo - mc_saved).abs().max()) == 0
    old = pd.read_csv(SOURCE / "evaluation/real/predictions.csv", dtype={"probability": np.float32}).set_index(keys[1:])
    real = predictions[predictions.source == "real"].set_index(keys[1:])
    real_error = float((old.probability - real.probability).abs().max())
    assert len(real) == len(old) == 4080 and real_error <= 2e-7
    t0 = predictions[predictions.temporal_depth == "T0"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    assert float(t0.sub(t0.real, axis=0).abs().max().max()) <= 1e-7
    heldout = set(cohort["split"]["val"])
    values = []
    for (arm, depth, seed), group in predictions.groupby(["source", "temporal_depth", "seed"]):
        assert set(group.patient_id) == heldout and len(group) == 102 and group.label.sum() == 32
        values.append({"source": arm, "depth": depth, "seed": seed,
                       "auroc": roc_auc_score(group.label, group.probability)})
    metrics = pd.DataFrame(values)
    summary = metrics.groupby(["source", "depth"]).auroc.agg(["mean", "std"])
    saved = pd.read_csv(OUTPUT / "evaluation/summary.csv")
    saved = saved[saved.threshold_policy == "fixed_0.5"].set_index(["source", "depth"])
    error = float(max((summary["mean"] - saved.auroc_mean).abs().max(),
                      (summary["std"] - saved.auroc_std).abs().max()))
    assert error < 1e-12
    differences = pd.read_csv(OUTPUT / "evaluation/paired_auroc_differences.csv")
    for row in differences.itertuples():
        left, right = row.comparison.split("_minus_")
        value = summary.loc[(left, row.depth), "mean"] - summary.loc[(right, row.depth), "mean"]
        assert abs(value - row.auroc_difference) < 1e-12
    report = {"passed": True, "cases": cases, "generated_vectors": vectors, "frozen_classifiers": len(protocol["classifiers"]),
              "fold_rows": len(folds), "patient_seed_window_arm_rows": len(predictions),
              "fold_probability_replay_max_error": fold_error, "real_probability_replay_max_error": real_error,
              "T0_feature_replay_max_error": maximum_replay, "auroc_summary_replay_max_error": error,
              "checked_utc": now()}
    write_json(OUTPUT / "evaluation/independent_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cohort_accounting(), indent=2))
    if not args.cohort_only:
        print(json.dumps(audit(), indent=2))

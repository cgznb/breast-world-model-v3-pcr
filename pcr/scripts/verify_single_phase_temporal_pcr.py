"""Independently audit completed temporal pCR v3 artifacts on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.first_post_pcr_data import identity, now, read_json, write_json


def metrics(frame):
    assert np.isfinite(frame.probability).all() and frame.probability.between(0, 1).all()
    return {"auroc": roc_auc_score(frame.label, frame.probability),
            "logloss": log_loss(frame.label, frame.probability, labels=[0, 1]),
            "brier": brier_score_loss(frame.label, frame.probability)}


def verify(root):
    root = Path(root).resolve()
    final = read_json(root / "COMPLETE.json")
    assert final["complete"] and final["formal_seeds"] == list(range(42, 72))
    assert final["holdout_evaluated"] and not final["generated_evaluated"]
    contract = read_json(root / "temporal_contract.json")
    cfg = contract["config"]
    for path, expected in contract["runtime"].items():
        assert identity(ROOT / path) == expected
    generic = read_json(root / "contract.json")
    for path, expected in generic["sources"].items():
        assert identity(ROOT / path) == expected
    source = Path(cfg["baseline_run"])
    for path, expected in generic["inputs"].items():
        assert identity(source / path) == expected
    inventory = read_json(root / "physical_roi_feature_inventory.json")
    for path, expected in inventory.items():
        assert identity(source / "embeddings/real" / path) == expected
    cohort = read_json(source / "cohort.json")
    development, holdout = set(cohort["split"]["train"]), set(cohort["split"]["val"])
    assert not development & holdout
    assert (len(development), len(holdout)) == (cfg["expected_development"], cfg["expected_holdout"])
    labels = {}
    for filename in ("metadata_enriched.csv", "holdout_metadata.csv"):
        metadata = pd.read_csv(source / filename, dtype={"pid": str})
        labels.update(dict(zip(metadata.pid, metadata.pCR.astype(int))))
    t0 = {v["canonical_patient_id"] for v in cohort["visits"] if v["timepoint"] == 0}
    folds = {r["fold"]: r for r in read_json(root / "nested_folds.json")}
    assert set(folds) == set(range(5))
    maximum_error = 0.0

    def equal(actual, expected):
        nonlocal maximum_error
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        assert np.isfinite(error) and error < 1e-10
        maximum_error = max(maximum_error, error)

    # Reconstruct each candidate score and duration from its six inner fits.
    ranking = pd.read_csv(root / "selection/temporal.csv")
    assert len(ranking) == 140
    inner_count = 0
    for row in ranking.itertuples():
        directory = root / "fits/inner" / row.source / ("T0" if row.depth == 1 else f"T0-T{row.depth - 1}") / row.candidate / f"outer_{row.outer}"
        results = [read_json(p) for p in directory.glob("inner_*/seed_*/COMPLETE.json")]
        assert len(results) == 6
        assert {(r["task"]["inner"], r["task"]["seed"]) for r in results} == {(i, s) for i in range(3) for s in (42, 43)}
        for result in results:
            task = result["task"]
            assert task["stage"] == "inner" and task.get("fixed_epochs") is None
            assert set(task["train_ids"]) | set(task["val_ids"]) == set(folds[row.outer]["train_ids"])
            assert not set(task["train_ids"]) & set(task["val_ids"])
        for metric in ("auroc", "logloss", "brier"):
            equal(np.mean([r["validation"][metric] for r in results]), getattr(row, metric))
        assert int(np.ceil(np.median([r["selected_epochs"] for r in results]))) == row.fixed_epochs
        inner_count += len(results)
    selections = read_json(root / "selection/temporal.json")["selections"]
    assert len(selections) == 20
    for row in selections:
        candidates = ranking[(ranking.depth == row["depth"]) & (ranking.outer == row["outer"])]
        allowed = candidates[candidates.auroc >= candidates.auroc.max() - cfg["auroc_tolerance"]]
        chosen = allowed.sort_values(["logloss", "parameters", "source", "candidate"]).iloc[0]
        assert row["selected"]["candidate"] == chosen.candidate
        assert row["selected"]["fixed_epochs"] == chosen.fixed_epochs

    references = read_json(root / "selection/temporal_outer_models.json")
    assert len(references) == 1200
    paths = {r["path"] for r in references}
    selected = read_json(root / "frozen_models.json")
    assert not selected["holdout_used_for_selection"]
    assert len(selected["models"]) == final["models"] == 600
    assert {(r["max_tp"], r["seed"], r["fold"]) for r in selected["models"]} == {
        (d, s, f) for d in range(1, 5) for s in range(42, 72) for f in range(5)}
    for ref in selected["models"]:
        assert identity(ref["path"]) == ref["identity"]
        assert str(Path(ref["path"]).parent) in paths
    frozen_by_path = {str(Path(r["path"]).parent): r for r in selected["models"]}
    fit_values = {}
    for name in paths:
        directory = Path(name)
        result = read_json(directory / "COMPLETE.json")
        task = result["task"]
        assert task["stage"] == "outer" and task["inner"] == -1
        assert set(task["train_ids"]) == set(folds[task["outer"]]["train_ids"])
        assert set(task["val_ids"]) == set(folds[task["outer"]]["val_ids"])
        assert set(task["train_ids"]) | set(task["val_ids"]) == development
        assert not set(task["train_ids"]) & set(task["val_ids"])
        expected = next(r for r in selections if r["depth"] == task["depth"] and r["outer"] == task["outer"])
        arm = "baseline" if task["candidate"] == "baseline" else "selected"
        assert task["candidate"] == expected[arm]["candidate"]
        assert result["selected_epochs"] == result["epochs_trained"] == task["fixed_epochs"] == expected[arm]["fixed_epochs"]
        assert result["clinical_prior_train_count"] == len(task["train_ids"])
        assert result["neural_train_count"] == len(set(task["train_ids"]) & t0)
        history = pd.read_csv(directory / "history.csv")
        assert history.epoch.tolist() == list(range(1, task["fixed_epochs"] + 1))
        assert not any(c.startswith("validation_") for c in history)
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["task"] == task and checkpoint["selected_epochs"] == task["fixed_epochs"]
        assert not checkpoint["outer_validation_used_for_selection"]
        assert not checkpoint["holdout_labels_or_embeddings_loaded"]
        assert all(torch.isfinite(v).all() for v in checkpoint["model_state"].values())
        effective = task["effective"]
        if name in frozen_by_path:
            reference = frozen_by_path[name]
            assert (reference["max_tp"], reference["seed"], reference["fold"]) == (task["depth"], task["seed"], task["outer"])
            assert reference["input_max_tp"] == min(task["depth"], effective.get("input_depth", task["depth"]))
        predictions = pd.read_csv(directory / "predictions.csv", dtype={"patient_id": str})
        assert (predictions.patient_id.map(labels) == predictions.label).all()
        values = {"epochs": task["fixed_epochs"]}
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            group = predictions[predictions.role == role]
            assert len(group) == len(ids) and set(group.patient_id) == set(ids)
            values.update({f"{role}_{k}": v for k, v in metrics(group).items()})
            for metric in ("auroc", "logloss", "brier"):
                equal(values[f"{role}_{metric}"], result[role][metric])
        values["gap"] = values["train_auroc"] - values["validation_auroc"]
        equal(values["gap"], result["auroc_gap"])
        fit_values[name] = values
        mode = effective["feature_transform"]
        if mode != "identity":
            limit = min(task["depth"], effective.get("input_depth", task["depth"]))
            width = effective["pca_dim"] if mode == "pca" else effective["input_dim"]
            cache = root / "feature_statistics" / f"{mode}_{width}_{effective['visit_policy']}_depth{limit}" / f"outer_{task['outer']}_inner_-1.pt"
            transform = torch.load(cache, map_location="cpu", weights_only=False)
            assert set(transform["spec"]["train_ids"]) == set(task["train_ids"])
            assert transform["spec"]["training_visits"] == result["feature_statistics_training_visits"]
            for key in ("mean", "scale", "components"):
                if key in transform["statistics"]:
                    assert torch.equal(checkpoint["model_state"][f"features.{key}"], transform["statistics"][key])

    fits = pd.DataFrame([{**{k: r[k] for k in ("arm", "depth", "seed", "outer")},
                          **fit_values[r["path"]]} for r in references])
    reported_fits = pd.read_csv(root / "reports/fit_metrics.csv").set_index(["arm", "depth", "seed", "outer"]).sort_index()
    rebuilt_fits = fits.set_index(["arm", "depth", "seed", "outer"]).sort_index()
    assert reported_fits.index.equals(rebuilt_fits.index)
    for column in rebuilt_fits:
        equal(rebuilt_fits[column], reported_fits[column])
    oof = pd.read_csv(root / "reports/outer_oof_predictions.csv", dtype={"patient_id": str})
    held = pd.read_csv(root / "evaluation/real/predictions.csv", dtype={"patient_id": str})
    fold_predictions = pd.read_csv(root / "evaluation/real/fold_predictions.csv", dtype={"patient_id": str})
    keys = ["depth", "seed", "patient_id", "label"]
    assert len(fold_predictions) == 600 * len(holdout)
    assert not fold_predictions.duplicated(keys + ["fold"]).any()
    assert (fold_predictions.groupby(keys).fold.nunique() == 5).all()
    reconstructed = fold_predictions.groupby(keys).probability.mean().sort_index()
    stored = held.set_index(keys).probability.sort_index()
    assert reconstructed.index.equals(stored.index)
    equal(reconstructed, stored)
    groups_checked = 0
    for subdir, frame, patients, group_keys in (
        ("reports", oof, development, ["depth", "arm"]),
        ("evaluation/real", held, holdout, ["depth"]),
    ):
        saved = pd.read_csv(root / subdir / "seed_metrics.csv").set_index(group_keys + ["seed"])
        assert len(saved) == (240 if subdir == "reports" else 120)
        assert (frame.patient_id.map(labels) == frame.label).all()
        for key, group in frame.groupby(group_keys + ["seed"]):
            assert len(group) == len(patients) and set(group.patient_id) == patients
            assert group.patient_id.is_unique and group.label.isin([0, 1]).all()
            for metric, value in metrics(group).items():
                equal(value, saved.loc[key, metric])
            if subdir == "reports":
                equal(roc_auc_score(group.label, group.clinical_prior_probability), saved.loc[key, "prior_auroc"])
                selected_fits = fits[(fits.depth == key[0]) & (fits.arm == key[1]) & (fits.seed == key[2])]
                assert len(selected_fits) == 5
                for reported, recomputed in (("train_auroc", "train_auroc"), ("fold_validation_auroc", "validation_auroc"),
                                             ("gap", "gap"), ("epochs", "epochs")):
                    equal(saved.loc[key, reported], selected_fits[recomputed].mean())
            groups_checked += 1
        saved = saved.reset_index()
        for filename in ("summary.csv", "seed_group_summary.csv"):
            summary = pd.read_csv(root / subdir / filename)
            for row in summary.to_dict("records"):
                group = saved
                for key in group_keys:
                    group = group[group[key] == row[key]]
                seed_group = row.get("seed_group", "all_30")
                seeds = set(range(42, 52)) if seed_group == "original_10" else set(range(52, 72)) if seed_group == "additional_20" else set(range(42, 72))
                group = group[group.seed.isin(seeds)]
                assert set(group.seed) == seeds and len(group) == len(seeds)
                for column in summary.columns:
                    if column.endswith("_mean"):
                        metric = column[:-5]
                        equal(group[metric].mean(), row[column])
                        equal(group[metric].std(ddof=1), row[metric + "_std"])
    matched = pd.read_csv(root / "evaluation/real/matched_v2_comparison.csv").set_index(["depth", "seed"]).sort_index()
    old = pd.read_csv(Path(cfg["previous_run"]) / "evaluation/real/seed_metrics.csv").set_index(["depth", "seed"]).sort_index()
    current = pd.read_csv(root / "evaluation/real/seed_metrics.csv").set_index(["depth", "seed"])
    assert matched.index.equals(old.index) and len(matched) == 40
    for metric in ("auroc", "logloss", "brier"):
        equal(matched[metric + "_v2"], old[metric])
        equal(matched[metric + "_v3"], current.loc[matched.index, metric])
        equal(matched["delta_" + metric], matched[metric + "_v3"] - matched[metric + "_v2"])
    payload = {"complete": True, "verified_utc": now(), "branch": cfg["branch"],
        "development_patients": len(development), "historical_holdout_patients": len(holdout),
        "frozen_development_feature_identities": len(inventory), "inner_fits": inner_count,
        "distinct_fixed_epoch_outer_fits": len(paths), "selected_models": len(selected["models"]),
        "metric_groups": groups_checked, "fold_prediction_rows": len(fold_predictions),
        "maximum_numeric_error": maximum_error,
        "selected_candidate_counts": dict(Counter(r["selected"]["candidate"] for r in selections)),
        "t0_only_long_window_models": sum(r["max_tp"] > 1 and r["input_max_tp"] == 1 for r in selected["models"]),
        "checks": ["cohort_and_partition_isolation", "inner_candidate_selection_and_median_duration",
                   "frozen_runtime_and_model_identities", "outer_fixed_epochs_without_validation_history",
                   "neural_loss_cohort", "training_only_transform_cache_matches_model_buffers",
                   "five_fold_probability_averaging", "oof_and_holdout_metric_replay",
                   "all_original_additional_seed_groups", "matched_v2_seed_comparison"],
        "limitations": "CPU artifact audit; no new fitting or whole-model inference. Historical holdouts are reused internal cohorts; seed SD is not a patient confidence interval."}
    write_json(root / "independent_verification.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/single_phase_temporal_pcr_v3_20260915")
    parser.add_argument("--branch", choices=["registered_dce0", "first_post"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    for branch in [args.branch] if args.branch else ["registered_dce0", "first_post"]:
        print(json.dumps(verify(ROOT / args.root / branch)), flush=True)


if __name__ == "__main__":
    main()

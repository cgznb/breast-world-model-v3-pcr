"""Independently replay fixed recipes and per-seed five-model mean AUROCs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from scripts.run_full978_anti_overfit import _load_split
from scripts.verify_registered_three_phase_optimization import (
    independent_predict, score, weighted_auc_samples,
)
from src import first_post_optimization as opt
from src import registered_three_phase_fixed_pcr as fixed
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import now, read_json, write_json
from src.registered_three_phase_optimization_report import holdout_splits, require_frozen
from src.single_phase_temporal_models import window_split


def read_csv(path):
    return pd.read_csv(path, dtype={"patient_id": str}, float_precision="round_trip")


def assert_close(actual, expected, tolerance=1e-12):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert np.isfinite(error) and error <= tolerance, (error, tolerance)
    return error


def audit_models(cfg, refs):
    root = Path(cfg["output_dir"])
    contract = read_json(root / "contract.json")
    shared.verify_inventory(contract["runtime"])
    shared.verify_inventory({cfg["config_path"]: contract["config_identity"]})
    folds = read_json(root / "nested_folds.json")
    original = read_json(Path(cfg["baseline_run"]) / "folds.json")
    if isinstance(original, dict):
        original = list(original.values())
    assert folds == original and {f["fold"] for f in folds} == set(range(5))
    cohort = read_json(root / "cohort.json")
    dev_ids, held_ids = cohort["split"]["train"], cohort["split"]["val"]
    assert len(dev_ids) == 764 and len(held_ids) == 102
    opt.validate_partitions(dev_ids, held_ids, folds)
    expected = {(a, d, f, s) for a in fixed.ARMS for d in range(1, 5)
                for f in range(5) for s in range(42, 52)}
    assert len(refs) == 400
    assert {(r["arm"], r["depth"], r["outer"], r["seed"]) for r in refs} == expected
    by_path = {}
    for ref in refs:
        by_path.setdefault(ref["path"], []).append(ref)
    assert len(by_path) == 350
    raw = _load_split(opt.source_directory(cfg, "physical_roi"),
                      root / "development_metadata.csv", sorted(dev_ids))
    priors = {}
    for fold in folds:
        train = opt.select_patients(raw, fold["train_ids"], 1)
        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        model.fit(train["clinical"], train["labels"])
        priors[fold["fold"]] = model
    maximum, prior_error, metric_error, shapes = 0.0, 0.0, 0.0, {}
    for path, usages in by_path.items():
        saved = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        task, effective = saved["task"], saved["task"]["effective"]
        fold = next(f for f in folds if f["fold"] == task["outer"])
        assert task["train_ids"] == fold["train_ids"] and task["val_ids"] == fold["val_ids"]
        assert not set(held_ids) & (set(task["train_ids"]) | set(task["val_ids"]))
        assert task["stage"] == "outer" and task["inner"] == -1
        assert saved["selected_epochs"] == saved["epochs_trained"] == task["fixed_epochs"] == 40
        assert effective["epochs"] == effective["scheduler_t_max"] == 40
        assert effective["epoch_selection"] == "fixed" and effective["feature_transform"] == "identity"
        assert not saved["outer_validation_used_for_selection"]
        assert not saved["holdout_labels_or_embeddings_loaded"]
        state_shapes = {k: tuple(v.shape) for k, v in saved["model_state"].items()}
        assert all(torch.isfinite(v).all() for v in saved["model_state"].values())
        for ref in usages:
            assert (task["depth"], task["seed"], task["outer"]) == (ref["depth"], ref["seed"], ref["outer"])
            assert effective == contract["resolved_recipes"][ref["arm"]][str(ref["depth"])]
            key = (ref["arm"], ref["depth"])
            assert shapes.setdefault(key, state_shapes) == state_shapes
        history = read_csv(Path(path) / "history.csv")
        assert history.epoch.tolist() == list(range(1, 41))
        assert not any(c.startswith("validation_") for c in history)
        assert np.isfinite(history.select_dtypes(include="number").to_numpy()).all()
        expected_lr = effective["lr"] * (1 + np.cos(np.pi * np.arange(1, 41) / 40)) / 2
        assert_close(history.lr.to_numpy(), expected_lr)
        prior = priors[task["outer"]]
        assert saved["clinical_prior"]["fitted_on"] == "fold_train_only"
        prior_error = max(prior_error, assert_close(saved["clinical_prior"]["coef"], prior.coef_, 1e-10),
                          assert_close(saved["clinical_prior"]["intercept"], prior.intercept_, 1e-10))
        frame = read_csv(Path(path) / "predictions.csv").set_index("patient_id")
        completed = read_json(Path(path) / "COMPLETE.json")
        assert frame.index.is_unique and set(frame.index) == set(dev_ids)
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            part = window_split(raw, ids, task["depth"], effective)
            expected_frame = frame.loc[part["pids"]]
            assert (expected_frame.role == role).all()
            assert np.array_equal(expected_frame.label, part["labels"])
            probability = independent_predict(saved, part)
            maximum = max(maximum, assert_close(probability, expected_frame.probability, 1e-6))
            metrics = score(expected_frame.label, expected_frame.probability)
            metric_error = max(metric_error, *[assert_close(metrics[k], completed[role][k]) for k in metrics])
    return {"unique_models": len(by_path), "references": len(refs), "uniform_window_arm_configs": len(shapes),
            "epochs_every_model": 40, "independent_training_only_prior_refits": len(priors),
            "maximum_prior_error": prior_error, "maximum_development_prediction_error": maximum,
            "maximum_fitted_metric_error": metric_error}


def audit_holdout(cfg, refs):
    root = Path(cfg["output_dir"]) / "evaluation/holdout"
    raw = read_csv(root / "draw_fold_predictions.csv")
    assert len(raw) == 530400
    key = ["model", "depth", "seed", "outer", "source"]
    groups = {k: p.set_index("patient_id") for k, p in raw.groupby(key)}
    splits = {k: v for k, v in holdout_splits(cfg).items() if k != "copy_T0"}
    assert len(splits) == 13 and len(groups) == 5200
    by_path = {}
    for ref in refs:
        by_path.setdefault(ref["path"], []).append(ref)
    maximum, count = 0.0, 0
    for index, (path, usages) in enumerate(by_path.items(), 1):
        saved = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        for source, split in splits.items():
            probability = independent_predict(saved, split)
            for ref in usages:
                frame = groups[(ref["arm"], ref["depth"], ref["seed"], ref["outer"], source)]
                assert frame.index.is_unique and set(frame.index) == set(split["pids"])
                frame = frame.loc[split["pids"]]
                assert np.array_equal(frame.label, split["labels"])
                maximum = max(maximum, assert_close(probability, frame.probability, 1e-6))
                count += len(frame)
        if index % 100 == 0:
            print(f"Independently replayed {index}/{len(by_path)} models on all 13 sources.", flush=True)
    mc4 = read_csv(root / "mc4_fold_predictions.csv")
    assert len(mc4) == 163200
    matrices = raw.pivot(index=key[:4] + ["patient_id", "label"], columns="source", values="probability")
    assert np.isfinite(matrices.to_numpy()).all()
    mc4_error = 0.0
    for source in fixed.SOURCES:
        replay = (matrices["real"] if source == "real" else
                  matrices[[source.replace("_mc4", f"_draw_{d}") for d in range(4)]].mean(axis=1))
        stored = mc4[mc4.source == source].set_index(list(matrices.index.names))
        assert stored.index.is_unique and set(stored.index) == set(matrices.index)
        mc4_error = max(mc4_error, assert_close(replay, stored.loc[matrices.index, "probability"]))
    return {"prediction_rows": count, "maximum_prediction_error": maximum,
            "mc4_rows": len(mc4), "maximum_mc4_error": mc4_error}


def audit_tables(cfg):
    root = Path(cfg["output_dir"]) / "evaluation"
    maximum, fold_count, summary_count = 0.0, 0, 0
    for population in ("development", "holdout"):
        directory = root / population
        metrics = read_csv(directory / "fold_metrics.csv")
        summaries = read_csv(directory / "seed_fivefold_summary.csv")
        sources = ["real"] if population == "development" else list(fixed.SOURCES)
        expected = {(a, d, s, source) for a in fixed.ARMS for d in range(1, 5)
                    for s in range(42, 52) for source in sources}
        keys = ["model", "depth", "seed", "source"]
        assert set(summaries[keys].itertuples(index=False, name=None)) == expected
        assert len(summaries) == len(expected) and len(metrics) == len(expected) * 5
        if population == "holdout":
            predictions = read_csv(directory / "mc4_fold_predictions.csv")
        else:
            predictions = read_csv(directory / "oof_predictions.csv")
            predictions["source"] = "real"
        stored_metrics = metrics.set_index(keys + ["outer"])
        for key, part in predictions.groupby(keys + ["outer"]):
            assert part.patient_id.is_unique
            actual = score(part.label, part.probability)
            stored = stored_metrics.loc[key]
            maximum = max(maximum, *[assert_close(actual[k], stored[k]) for k in actual])
            fold_count += 1
        for row in summaries.itertuples():
            part = metrics[(metrics.model == row.model) & (metrics.depth == row.depth)
                           & (metrics.seed == row.seed) & (metrics.source == row.source)]
            assert len(part) == 5 and set(part.outer) == set(range(5)) and row.folds == 5
            for metric in ("auroc", "logloss", "brier", "train_auroc", "train_logloss", "train_brier", "auroc_gap"):
                if metric not in part:
                    continue
                maximum = max(maximum, assert_close(part[metric].mean(), getattr(row, metric + "_mean")),
                              assert_close(part[metric].std(ddof=1), getattr(row, metric + "_fold_sd")))
            for value in part.itertuples():
                maximum = max(maximum, assert_close(value.auroc, getattr(row, f"fold_{value.outer + 1}_auroc")))
            summary_count += 1
    return {"fold_metric_rows_recomputed": fold_count, "seed_summaries_recomputed": summary_count,
            "maximum_metric_summary_error": maximum, "classifier_seeds_never_averaged": True}


def audit_intervals(cfg):
    root = Path(cfg["output_dir"]) / "evaluation/holdout"
    predictions = read_csv(root / "mc4_fold_predictions.csv")
    table = read_csv(root / "paired_seed_intervals.csv")
    assert len(table) == 280
    ids = sorted(predictions.patient_id.unique())
    y = predictions.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    assert (len(positive), len(negative)) == (32, 70)
    weights = np.zeros((cfg["bootstrap_samples"], len(ids)))
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    weights[:, positive] = rng.multinomial(len(positive), np.full(len(positive), 1 / len(positive)), size=len(weights))
    weights[:, negative] = rng.multinomial(len(negative), np.full(len(negative), 1 / len(negative)), size=len(weights))
    maximum, count = 0.0, 0
    for (depth, seed), part in predictions.groupby(["depth", "seed"]):
        samples, points = {}, {}
        for key, frame in part.groupby(["model", "source"]):
            matrix = frame.pivot(index="outer", columns="patient_id", values="probability").loc[list(range(5)), ids].to_numpy()
            samples[key] = weighted_auc_samples(y, matrix, weights)
            points[key] = np.mean([score(y, p)["auroc"] for p in matrix])
        rows = table[(table.depth == depth) & (table.seed == seed)]
        assert len(rows) == 7
        for row in rows.itertuples():
            left, right = (row.left_model, row.left_source), (row.right_model, row.right_source)
            low, high = np.quantile(samples[left] - samples[right], [.025, .975])
            maximum = max(maximum, assert_close(low, row.ci95_low), assert_close(high, row.ci95_high),
                          assert_close(points[left] - points[right], row.mean_fold_auc_difference))
            count += 1
    return {"separate_seed_contrasts": count, "bootstrap_replicates": len(weights), "maximum_interval_error": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_fixed_pcr_v3.yaml")
    parser.add_argument("--check-complete", action="store_true",
                        help="Check completed-run identities without refitting or replaying inference")
    args = parser.parse_args()
    cfg = fixed.load_config(args.config)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    root = Path(cfg["output_dir"])
    assert read_json(root / "COMPLETE.json")["passed"]
    contract = read_json(root / "contract.json")
    # YAML integer mapping keys become strings when the contract is stored as JSON.
    assert json.loads(json.dumps(cfg)) == contract["config"]
    shared.verify_inventory(contract["runtime"])
    shared.verify_inventory({cfg["config_path"]: contract["config_identity"]})
    for name in ("prior_inventory.json", "evaluation_input_inventory.json"):
        shared.verify_inventory(read_json(root / name))
    require_frozen(cfg)
    if args.check_complete:
        assert read_json(root / "evaluation/independent_audit.json")["passed"]
        print("Completed study and frozen identities verified; no fitting or inference repeated.", flush=True)
        return
    refs = read_json(root / "model_references.json")
    result = {"models": audit_models(cfg, refs)}
    print("Identical saved recipes, fixed histories, training-only priors and development predictions verified.", flush=True)
    result["holdout"] = audit_holdout(cfg, refs)
    result["tables"] = audit_tables(cfg)
    result["intervals"] = audit_intervals(cfg)
    for name in ("prior_inventory.json", "evaluation_input_inventory.json"):
        shared.verify_inventory(read_json(root / name))
    require_frozen(cfg)
    result.update(passed=True, checked_utc=now(), neural_optimizer_updates=0)
    write_json(root / "evaluation/independent_audit.json", result)
    print(__import__("json").dumps(result), flush=True)


if __name__ == "__main__":
    main()

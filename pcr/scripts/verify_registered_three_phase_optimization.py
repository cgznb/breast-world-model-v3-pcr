"""Read-only independent audit of selection, transforms, predictions and intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full978_anti_overfit import _load_split
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as study
from src.first_post_pcr_data import now, read_json, write_json
from src.registered_three_phase_optimization_report import holdout_splits, require_frozen
from src.single_phase_temporal_models import FeatureTDN, fit_feature_statistics, window_split
from src.tdn import TDN


def score(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p))}


def audit_selection(cfg):
    output = Path(cfg["output_dir"])
    folds = read_json(output / "nested_folds.json")
    heldout = set(read_json(Path(cfg["baseline_run"]) / "cohort.json")["split"]["val"])
    grouped, maximum = {}, 0.0
    tasks = opt.inner_tasks(cfg, folds, cfg["feature_sources"])
    for task in tasks:
        path = opt.task_path(output, task)
        assert opt.task_complete(path, task)
        outer = next(f for f in folds if f["fold"] == task["outer"])
        assert not (set(task["train_ids"]) | set(task["val_ids"])) & (heldout | set(outer["val_ids"]))
        summary = read_json(path / "COMPLETE.json")
        predictions = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        for role in ("train", "validation"):
            part = predictions[predictions.role == role]
            metrics = score(part.label.to_numpy(), part.probability.to_numpy())
            maximum = max(maximum, max(abs(metrics[k] - summary[role][k]) for k in metrics))
        if task["effective"]["kind"] != "clinical":
            history = pd.read_csv(path / "history.csv")
            values = (history.validation_auroc if task["effective"]["epoch_selection"] == "auroc"
                      else -history.validation_logloss)
            best, chosen = -np.inf, 0
            for epoch, value in enumerate(values, 1):
                if value > best + task["effective"].get("min_delta", 0):
                    best, chosen = value, epoch
            assert chosen == summary["selected_epochs"]
        key = (task["depth"], task["outer"], task["candidate"])
        grouped.setdefault(key, []).append(summary)
    assert maximum < 1e-10
    selected = read_json(output / "selection" / f"{study.STUDY}.json")["selections"]
    for selection in selected:
        ranking = []
        for candidate in cfg["candidates"]:
            records = grouped[(selection["depth"], selection["outer"], candidate)]
            ranking.append({"candidate": candidate,
                            "auroc": np.mean([r["validation"]["auroc"] for r in records]),
                            "logloss": np.mean([r["validation"]["logloss"] for r in records]),
                            "parameters": records[0]["parameters"],
                            "fixed_epochs": int(np.ceil(np.median([r["selected_epochs"] for r in records])))})
        best = max(r["auroc"] for r in ranking)
        eligible = [r for r in ranking if r["auroc"] >= best - cfg["auroc_tolerance"]]
        chosen = sorted(eligible, key=lambda r: (r["logloss"], r["parameters"], r["candidate"]))[0]
        baseline = next(r for r in ranking if r["candidate"] == "baseline")
        for role, expected in (("selected", chosen), ("baseline", baseline)):
            assert selection[role]["candidate"] == expected["candidate"]
            assert selection[role]["fixed_epochs"] == expected["fixed_epochs"]
    refs = read_json(output / "selection" / f"{study.STUDY}_outer_models.json")
    expected_keys = {(arm, d, f["fold"], seed) for arm in ("selected", "baseline")
                     for d in cfg["depths"] for f in folds for seed in cfg["formal_seeds"]}
    actual_keys = {(r["arm"], r["depth"], r["outer"], r["seed"]) for r in refs}
    assert len(refs) == len(expected_keys) and actual_keys == expected_keys
    for ref in refs:
        saved = torch.load(Path(ref["path"]) / "model.pt", map_location="cpu", weights_only=False)
        task = saved["task"]
        outer = next(f for f in folds if f["fold"] == ref["outer"])
        chosen = next(s for s in selected if s["depth"] == ref["depth"] and s["outer"] == ref["outer"])[ref["arm"]]
        assert task["train_ids"] == outer["train_ids"] and task["val_ids"] == outer["val_ids"]
        assert task["candidate"] == chosen["candidate"] and task["fixed_epochs"] == chosen["fixed_epochs"]
        assert task["seed"] == ref["seed"] and task["depth"] == ref["depth"]
        assert opt.task_complete(Path(ref["path"]), task)
    return {"inner_fits": len(tasks), "outer_references": len(refs), "choices": len(selected),
            "maximum_inner_metric_error": maximum}


def audit_transforms(cfg):
    output = Path(cfg["output_dir"])
    folds = read_json(output / "nested_folds.json")
    ids = sorted(folds[0]["train_ids"] + folds[0]["val_ids"])
    raw = _load_split(opt.source_directory(cfg, "physical_roi"), output / "development_metadata.csv", ids)
    count, maximum = 0, 0.0
    for path in sorted((output / "feature_statistics").glob("**/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec, original = payload["spec"], payload["statistics"]
        effective = {"feature_transform": spec["mode"], "pca_dim": spec["width"],
                     "input_dim": spec["input_dim"], "visit_policy": spec["policy"]}
        part = window_split(raw, spec["train_ids"], spec["depth"], effective)
        assert part["pids"] == spec["train_ids"]
        assert int(part["masks"].sum()) == spec["training_visits"]
        replay = fit_feature_statistics(part, effective)
        assert original["fitted_visits"] == replay["fitted_visits"]
        for key in ("mean", "components", "scale"):
            maximum = max(maximum, float((original[key] - replay[key]).abs().max()))
        count += 1
    assert maximum < 1e-6
    return {"training_only_transforms_refitted": count, "maximum_transform_error": maximum}


@torch.no_grad()
def independent_predict(checkpoint, raw):
    task, effective = checkpoint["task"], checkpoint["task"]["effective"]
    split = window_split(raw, raw["pids"], task["depth"], effective)
    prior_state = checkpoint["clinical_prior"]
    prior = split["clinical"] @ np.asarray(prior_state["coef"], np.float32)[0]
    prior = np.clip(prior + float(prior_state["intercept"][0]), -30, 30)
    if effective["kind"] == "clinical":
        return 1 / (1 + np.exp(-prior))
    cls = FeatureTDN(effective, task["depth"]) if effective.get("feature_transform") == "pca" else TDN({"downstream": effective})
    cls.load_state_dict(checkpoint["model_state"], strict=True)
    cls.eval()
    inputs = [torch.tensor(split[k], dtype=torch.float32) for k in ("embs", "masks", "clinical")]
    logits = cls(*inputs, days=torch.tensor(split["days"]), prior_logit=torch.tensor(prior))
    return torch.sigmoid(logits).numpy()


def audit_holdout_replay(cfg):
    output = Path(cfg["output_dir"])
    splits = holdout_splits(cfg)
    refs = read_json(output / "selection" / f"{study.STUDY}_outer_models.json")
    stored = pd.read_csv(output / "evaluation/holdout/fold_predictions.csv", dtype={"patient_id": str})
    groups = {(m, s, d, seed, f): g.set_index("patient_id") for (m, s, d, seed, f), g in
              stored.groupby(["model", "source", "depth", "seed", "outer"])}
    by_path = {}
    for ref in refs:
        by_path.setdefault(ref["path"], []).append(ref)
    maximum, predictions = 0.0, 0
    for path, usages in by_path.items():
        checkpoint = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        for source, raw in splits.items():
            probability = independent_predict(checkpoint, raw)
            for ref in usages:
                frame = groups[(ref["arm"], source, ref["depth"], ref["seed"], ref["outer"])].loc[raw["pids"]]
                assert np.array_equal(frame.label.to_numpy(), raw["labels"])
                maximum = max(maximum, float(np.abs(probability - frame.probability.to_numpy()).max()))
                predictions += len(probability)
    assert maximum < 1e-6
    return {"holdout_models_replayed": len(by_path), "fold_prediction_rows_replayed": predictions,
            "maximum_holdout_prediction_error": maximum}


def weighted_auc_samples(y, matrix, weights):
    """Independent weighted-rank implementation, including exact probability ties."""
    values = []
    for p in matrix:
        order = np.argsort(p, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
        negative = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
        before = np.cumsum(negative, axis=1) - negative
        numerator = (positive * (before + .5 * negative)).sum(axis=1)
        values.append(numerator / (positive.sum(axis=1) * negative.sum(axis=1)))
    return np.mean(values, axis=0)


def audit_bootstrap(cfg):
    output = Path(cfg["output_dir"])
    maximum, count = 0.0, 0
    for population in ("development", "holdout"):
        root = output / "evaluation" / population
        predictions = pd.read_csv(root / "predictions.csv", dtype={"patient_id": str})
        table = pd.read_csv(root / "paired_differences.csv")
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        for (source, depth), frame in predictions.groupby(["source", "depth"]):
            ids = sorted(frame.patient_id.unique())
            y = frame.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
            positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            weights = np.zeros((cfg["bootstrap_samples"], len(ids)), dtype=np.float64)
            weights[:, positive] = rng.multinomial(len(positive), np.full(len(positive), 1 / len(positive)), size=len(weights))
            weights[:, negative] = rng.multinomial(len(negative), np.full(len(negative), 1 / len(negative)), size=len(weights))
            rows = table[(table.source == source) & (table.depth == depth)]
            names = {name for text in rows.comparison for name in text.split("_minus_")}
            points, samples = {}, {}
            for name in names:
                matrix = frame[frame.model == name].pivot(index="seed", columns="patient_id", values="probability").loc[cfg["formal_seeds"], ids].to_numpy()
                samples[name] = weighted_auc_samples(y, matrix, weights)
                points[name] = float(np.mean([roc_auc_score(y, p) for p in matrix]))
            for row in rows.itertuples():
                left, right = row.comparison.split("_minus_")
                lo, hi = np.quantile(samples[left] - samples[right], [.025, .975])
                maximum = max(maximum, abs(lo - row.ci95_low), abs(hi - row.ci95_high),
                              abs(points[left] - points[right] - row.auroc_difference))
                count += 1
    assert maximum < 1e-12
    return {"paired_contrasts_recomputed": count, "maximum_bootstrap_error": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_pcr_optimization_v2.yaml")
    args = parser.parse_args()
    cfg = study.load_config(args.config)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(cfg["output_dir"])
    require_frozen(cfg)
    assert read_json(output / "COMPLETE.json")["passed"]
    result = {"selection": audit_selection(cfg)}
    print("Selection and duration audit passed.", flush=True)
    result["transforms"] = audit_transforms(cfg)
    print("Training-only transform refits passed.", flush=True)
    result["holdout"] = audit_holdout_replay(cfg)
    print("All frozen holdout predictions replayed.", flush=True)
    result["bootstrap"] = audit_bootstrap(cfg)
    for name in ("original_inventory.json", "evaluation_input_inventory.json"):
        study.verify_inventory(read_json(output / name))
    require_frozen(cfg)
    result.update(passed=True, checked_utc=now(), optimizer_updates=0)
    write_json(output / "evaluation/independent_audit.json", result)
    print(__import__("json").dumps(result), flush=True)


if __name__ == "__main__":
    main()

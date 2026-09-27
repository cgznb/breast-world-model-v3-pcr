"""Read-only replay of the completed anchored pCR study; no MRI or GPU access."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(cfg):
    import numpy as np
    import pandas as pd
    import torch
    from src.first_post_pcr_data import identity, now, read_json, write_json
    from src.single_phase_anchor_training import (
        STATISTICAL, as_tensors, load_data, model_from_bundle, predict_bundle, task_complete, task_path,
    )
    from src.single_phase_anchor_workflow import add_selected, choose_candidate, formal_tasks, inner_tasks
    from src.single_phase_fmbcmri_training import METRICS
    from src.single_phase_response_data import load_data as source_data, subset
    from src.single_phase_response_reporting import metric_rows, summarize, validate_prediction_frame
    from src.single_phase_response_training import fit_clinical
    root = Path(cfg["output_dir"])
    if not (root / "COMPLETE.json").exists():
        raise ValueError("Independent replay requires a completed branch")
    for file, stamp in read_json(Path(cfg["study_dir"]) / "runtime_sources.json").items():
        if identity(ROOT / file) != stamp:
            raise ValueError("The runtime source differs from the frozen experiment")
    for file, stamp in read_json(root / "source_features.json").items():
        if identity(file) != stamp:
            raise ValueError("A reused source feature changed")
    choices = read_json(root / "selection.json")
    scores = pd.read_csv(root / "inner_candidate_scores.csv")
    for selected in choices:
        part = scores[(scores.fold == selected["fold"]) & (scores.prefix == selected["prefix"])]
        calculated = choose_candidate(part.set_index("arm").validation_logloss.to_dict(),
                                      cfg["selection"]["candidates"], cfg["selection"]["min_improvement"])
        if calculated != selected["selected"]:
            raise ValueError("Saved inner policy differs from its declared rule")
    formal, inner = formal_tasks(cfg, write_selection=False), inner_tasks(cfg)
    data = {roi: load_data(cfg, roi) for roi in ("context", "tumor")}
    independent_clinical = {}
    for fold in read_json(root / "folds.json"):
        independent_clinical[fold["fold"]] = fit_clinical(subset(data["context"], fold["train_ids"]), 42)
    fold_prediction_path = root / "evaluation/holdout_fold_predictions.csv"
    saved_holdout = pd.read_csv(fold_prediction_path, dtype={"patient_id": str}, low_memory=False)
    holdout_groups = {k: part for k, part in saved_holdout.groupby(["arm", "fold", "seed", "source"])}
    study = read_json(Path(cfg["source_dir"]) / "cohort.json")
    cached_holdout, max_replay, baseline_checks, neural_count, zero_refits = {}, 0.0, 0, 0, 0
    anchor_probability_roundoff = 0.0
    for number, task in enumerate(formal):
        if not task_complete(cfg, task):
            raise ValueError("Incomplete or modified formal artifact")
        path = task_path(cfg, task)
        bundle = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        if bundle["task"] != task or set(bundle["clinical"]["fitted_ids"]) != set(task["train_ids"]):
            raise ValueError("Clinical fitting partition is incorrect")
        reference = independent_clinical[task["fold"]]
        for key in ("coefficient", "intercept"):
            np.testing.assert_allclose(bundle["clinical"][key], reference[key], rtol=0, atol=1e-7)
        for key in ("median", "mean", "scale"):
            np.testing.assert_array_equal(bundle["clinical"]["transform"][key], reference["transform"][key])
        arm = cfg["arms"][task["arm"]]
        train = subset(data[arm["roi"]], task["train_ids"])
        mask = train["masks"].copy()
        if arm["kind"] == "t0":
            mask[:, 1:] = 0
        rows = train["embs"][mask > 0].astype(np.float64)
        np.testing.assert_allclose(bundle["statistics"]["mean"], rows.mean(0), rtol=0, atol=1e-7)
        if bundle["statistics"]["fitted_visits"] != len(rows):
            raise ValueError("PCA fitted-visit count is incorrect")
        validation = subset(data[arm["roi"]], task["val_ids"])
        predicted = predict_bundle(bundle, validation)
        saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        expected = saved.pivot(index="patient_id", columns="prefix", values="probability").reindex(validation["pids"]).to_numpy()
        max_replay = max(max_replay, float(np.max(np.abs(predicted - expected))))
        if arm["kind"] not in STATISTICAL:
            neural_count += 1
            if bundle["binding"]["validation_ids"] or bundle["selected_epochs"] != task["fixed_epochs"]:
                raise ValueError("Outer data influenced neural stopping")
            if any(row["validation_logloss"] is not None for row in bundle["history"]):
                raise ValueError("Outer validation loss was monitored during fitting")
            zero_refits += int(bundle["zero_epoch_fallback"])
            model = model_from_bundle(bundle)
            np.testing.assert_array_equal(model.base.clinical.coefficient, bundle["clinical"]["coefficient"])
            if "base" in arm:
                parent = root / "training/formal" / arm["base"] / f"fold_{task['fold']}" / f"seed_{task['seed']}" / "model.pt"
                parent_bundle = torch.load(parent, map_location="cpu", weights_only=False)
                parent_model = model_from_bundle(parent_bundle)
                for key, value in model.base.state_dict().items():
                    if not torch.equal(value, parent_model.base.state_dict()[key]):
                        raise ValueError("Frozen T0 weights differ from the matching parent")
                tensors = as_tensors(validation)
                args = tuple(tensors[key] for key in ("embs", "masks", "clinical", "days"))
                with torch.no_grad():
                    child_logits, parent_logits = model(*args)[0], parent_model(*args)[0]
                if not torch.equal(child_logits[:, 0], parent_logits[:, 0]):
                    raise ValueError("Frozen T0 logits changed")
                parent_probability = predict_bundle(parent_bundle, validation)[:, 0]
                anchor_probability_roundoff = max(anchor_probability_roundoff,
                                                  float(np.max(np.abs(predicted[:, 0] - parent_probability))))
                # Expanded versus contiguous tensors can use different FP32 sigmoid kernels.
                np.testing.assert_allclose(predicted[:, 0], parent_probability, rtol=0, atol=1e-6)
                baseline_checks += 1
            altered = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in validation.items()}
            altered["embs"][:, 2:] *= 100
            altered["days"][:, 2:] += 1000
            np.testing.assert_array_equal(predicted[:, :2], predict_bundle(bundle, altered)[:, :2])
        for source in ("real", "symm", "bifm", "copy"):
            key = (arm["roi"], source)
            if key not in cached_holdout:
                cached_holdout[key] = source_data(cfg["source_cfg"], study, arm["roi"], "val", source)
            current = cached_holdout[key]
            actual = predict_bundle(bundle, current)
            group = holdout_groups[(task["arm"], task["fold"], task["seed"], source)]
            expected = group.pivot(index="patient_id", columns="prefix", values="probability").reindex(current["pids"]).to_numpy()
            max_replay = max(max_replay, float(np.max(np.abs(actual - expected))))
        if number % 50 == 0:
            print(f"{cfg['branch']}: replayed {number + 1}/{len(formal)} formal bundles", flush=True)
    if max_replay > 1e-6:
        raise ValueError("Saved-model predictions exceed replay tolerance")
    for task in inner:
        if not task_complete(cfg, task):
            raise ValueError("Incomplete inner fit")
        bundle = torch.load(task_path(cfg, task) / "model.pt", map_location="cpu", weights_only=False)
        if set(bundle["clinical"]["fitted_ids"]) != set(task["train_ids"]):
            raise ValueError("Inner clinical fitting crossed its partition")
        if cfg["arms"][task["arm"]]["kind"] not in STATISTICAL:
            best = min(bundle["history"], key=lambda r: r["validation_logloss"])["epoch"]
            if best != bundle["selected_epochs"]:
                raise ValueError("Inner epoch selection differs from minimum validation log loss")
    # Rebuild the selected fold predictions, including the missing-prefix policy.
    raw = saved_holdout[saved_holdout.arm != "inner_selected"].copy()
    selected = add_selected(raw, choices, cfg["selection"]["candidates"])
    keys = ["arm", "fold", "seed", "source", "patient_id", "prefix"]
    a = saved_holdout.set_index(keys).sort_index().probability
    b = selected.set_index(keys).sort_index().probability
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)
    frame = pd.read_csv(root / "evaluation/predictions.csv", dtype={"patient_id": str}, low_memory=False)
    validate_prediction_frame(frame, {"oof": study["split"]["train"], "holdout": study["split"]["val"]}, cfg["seeds"])
    for _, group in frame.groupby(["arm", "population", "source", "seed"]):
        p = group.pivot(index="patient_id", columns="prefix", values="probability")
        available = group.drop_duplicates("patient_id").set_index("patient_id").available_prefix.reindex(p.index)
        for t in range(2, 5):
            absent = available < t
            np.testing.assert_allclose(p.loc[absent, t], p.loc[absent, t - 1], rtol=0, atol=1e-6)
    groups = ["arm", "source", "population", "seed", "prefix", "patient_id", "label", "complete", "available_prefix"]
    averaged = saved_holdout.groupby(groups, as_index=False).probability.mean()
    compare_keys = ["arm", "source", "seed", "prefix", "patient_id"]
    a = averaged.set_index(compare_keys).sort_index().probability
    b = frame[frame.population == "holdout"].set_index(compare_keys).sort_index().probability
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    recalculated = metric_rows(frame)
    stored = pd.read_csv(root / "evaluation/metrics_per_seed.csv")
    metric_keys = ["arm", "population", "source", "prefix", "seed", "subset"]
    a = recalculated.set_index(metric_keys).sort_index()[list(METRICS)].to_numpy()
    b = stored.set_index(metric_keys).sort_index()[list(METRICS)].to_numpy()
    metric_error = float(np.nanmax(np.abs(a - b)))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
    summary = summarize(recalculated)
    stored_summary = pd.read_csv(root / "evaluation/summary.csv")
    summary_keys = ["arm", "population", "source", "prefix", "subset"]
    columns = [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    np.testing.assert_allclose(summary.set_index(summary_keys).sort_index()[columns],
                               stored_summary.set_index(summary_keys).sort_index()[columns], rtol=0, atol=1e-12)
    result = dict(passed=True, verified_utc=now(),
                  formal_bundles=len(formal), inner_bundles=len(inner), neural_refits=neural_count,
                  zero_epoch_neural_refits=zero_refits, frozen_t0_checks=baseline_checks,
                  maximum_prediction_replay_error=max_replay, maximum_metric_replay_error=metric_error,
                  maximum_anchor_probability_roundoff=anchor_probability_roundoff, t0_logits_exact=True,
                  audit_only_clinical_refits=5, neural_optimizer_updates=0, gpu_used=False, mri_pixels_read=False,
                  bootstrap_samples=0, verified_metric_rows=len(stored))
    write_json(root / "evaluation/independent_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_anchor_pcr_v3.yaml")
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    from src.first_post_pcr_data import write_json
    from src.single_phase_anchor_workflow import load_config
    from src.single_phase_fmbcmri_data import BRANCHES
    from src.single_phase_fmbcmri_training import deterministic_runtime
    deterministic_runtime()
    result, configs = {}, [load_config(args.config, b) for b in BRANCHES]
    for cfg in configs:
        result[cfg["branch"]] = verify(cfg)
    write_json(Path(configs[0]["study_dir"]) / "independent_verification.json", result)
    print("All branch replay, partition, frozen-anchor and metric checks passed", flush=True)


if __name__ == "__main__":
    main()

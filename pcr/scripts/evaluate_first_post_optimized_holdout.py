"""Evaluate frozen v2 classifiers on the original 96-person three-source holdout."""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_first_post_pcr import overlay
from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward, staged_chunks
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state
from src.data import EmbStore
from src.first_post_optimization import (
    DEPTH_NAMES, load_development, load_study, metric_values, predict,
    select_patients, source_directory, task_complete, tensor_split,
)
from src.first_post_pcr_data import (
    embedding_path, identity, now, read_json, repo_path, save_tensor,
    validate_embedding, write_json,
)
from src.first_post_roi_optimization import build_resized_volume, extraction_config, pin_pillar
from src.tdn import TDN

SCHEMA = "first_post_frozen_optimized_holdout96_v1"
SOURCES = ("real", "symm", "bifm")


def status(output, stage, **fields):
    row = {"updated_utc": now(), "stage": stage, "pid": os.getpid(), **fields}
    write_json(output / "progress.json", row)
    print(__import__("json").dumps(row), flush=True)


def validate_patient_boundary(task, development, holdout):
    train, val, dev, held = map(set, (task["train_ids"], task["val_ids"], development, holdout))
    if train & val or train | val != dev or (train | val) & held or dev & held:
        raise ValueError("Checkpoint does not respect the development/holdout boundary")
    if task["stage"] != "outer":
        raise ValueError("Only frozen outer refits can enter holdout evaluation")


def freeze_inputs(cfg, output):
    study, baseline = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    if read_json(study / "COMPLETE.json").get("holdout_evaluated") is not False:
        raise ValueError("Expected completed development-only study")
    cohort = read_json(baseline / "cohort.json")
    heldout, development = cohort["split"]["val"], cohort["split"]["train"]
    if len(heldout) != 96 or len(development) != 881:
        raise ValueError("Unexpected fixed cohort")
    references, records, selection_files = [], {}, {}
    for name in ("physical_roi", "joint_geometry"):
        path = study / "selection" / f"{name}_outer_models.json"
        selection_files[str(path)] = identity(path)
        selection = study / "selection" / f"{name}.json"
        selection_files[str(selection)] = identity(selection)
        for ref in read_json(path):
            if name == "joint_geometry" and ref["arm"] == "baseline":
                continue
            arm = "nested_baseline" if ref["arm"] == "baseline" else f"{name}_selected"
            references.append({**ref, "arm": arm})
            if ref["path"] in records:
                continue
            run = Path(ref["path"])
            checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
            task = checkpoint["task"]
            validate_patient_boundary(task, development, heldout)
            if any(task[k] != ref[k] for k in ("depth", "seed", "outer")):
                raise ValueError("Selected reference and checkpoint differ")
            if not task_complete(run, task):
                raise ValueError("Selected training artifacts do not validate")
            if task["source"] == "resized_roi" and task["depth"] != 1:
                raise ValueError("This frozen selection requires only resized T0; changed selection rejected")
            records[ref["path"]] = {"path": ref["path"], "geometry": task["source"],
                                   "depth": task["depth"], "seed": task["seed"], "outer": task["outer"],
                                   "checkpoint_identity": identity(run / "model.pt")}
    ordered = sorted(records.values(), key=lambda r: (r["geometry"] != "physical_roi", r["depth"], r["seed"], r["outer"], r["path"]))
    for index, record in enumerate(ordered):
        record["model_id"] = index
    snapshots = read_json(baseline / "world_checkpoints/snapshots.json")
    for value in snapshots.values():
        if identity(value["path"]) != value["identity"]:
            raise ValueError("Archived generator snapshot changed")
    files = [baseline / name for name in ("cohort.json", "metadata_enriched.csv", "evaluation/predictions.csv",
             "evaluation/summary.csv", "HYBRID_FEATURES_COMPLETE.json", "EVALUATION_COMPLETE.json")]
    sources = [Path(__file__), repo_path("src/tdn.py"), repo_path("src/data.py"),
               repo_path("src/first_post_optimization.py"), repo_path("src/first_post_roi_optimization.py"),
               repo_path("src/first_post_pcr_data.py"), repo_path("src/temporal.py"),
               repo_path("scripts/evaluate_first_post_pcr.py"), repo_path("scripts/run_first_post_pcr.py")]
    inventory = {}
    for visit in cohort["visits"]:
        if visit["split"] != "val":
            continue
        pid, tp = visit["canonical_patient_id"], visit["timepoint"]
        for source in SOURCES if tp > 0 else ("real",):
            path = baseline / "embeddings" / source / pid / f"{pid}_T{tp}.pt"
            inventory[str(path)] = identity(path)
    contract = {"schema": SCHEMA, "development_study": str(study), "baseline_run": str(baseline),
                "records": ordered, "references": references, "holdout_ids": heldout,
                "input_files": {str(p): identity(p) for p in files}, "feature_inventory": inventory,
                "runtime_sources": {str(p): identity(p) for p in sources}, "selection_files": selection_files,
                "generator_snapshots": snapshots, "fit_or_selection_on_holdout": False,
                "aggregation": "mean_five_fold_probabilities_per_seed_then_mean_ten_seed_metrics",
                "evaluation_role": cohort["test_role"], "bootstrap_samples": 2000}
    path = output / "contract.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("Frozen evaluation inputs changed")
    write_json(path, contract)
    return contract, cohort


def resized_t0(cfg, output, cohort):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    extraction = extraction_config(cfg)
    extraction["output_dir"] = str(output / "resized_t0")
    directory = Path(extraction["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    pin_pillar(directory)
    if read_json(directory / "pillar_files.json") != read_json(Path(cfg["output_dir"]) / "resized_roi/pillar_files.json"):
        raise ValueError("Holdout encoder differs from development encoder")
    visits = [v for v in cohort["visits"] if v["split"] == "val" and v["timepoint"] == 0]
    if len(visits) != 96 or {v["canonical_patient_id"] for v in visits} != set(cohort["split"]["val"]):
        raise ValueError("T0 extraction does not cover exactly the fixed 96 patients")
    pending = []
    for visit in visits:
        path = embedding_path(extraction, "real", visit)
        audit = directory / "geometry_audit" / f"{visit['canonical_patient_id']}_T0.json"
        if path.exists() and audit.exists():
            validate_embedding(path)
        else:
            pending.append(visit)
    completed, started = len(visits) - len(pending), time.perf_counter()
    if pending:
        status(output, "extracting_holdout_resized_t0", completed=completed, total=96)
        model = load_frozen_pillar()
        for chunk in staged_chunks(extraction, pending):
            with ThreadPoolExecutor(max_workers=2) as pool:
                queue, submitted = {}, 0
                for index, visit in enumerate(chunk):
                    while submitted < min(len(chunk), index + 2):
                        queue[submitted] = pool.submit(build_resized_volume, extraction, chunk[submitted])
                        submitted += 1
                    volume, audit = queue.pop(index).result()
                    vector = pillar_forward(model, volume[None].to("cuda:0"))[0]
                    save_tensor(embedding_path(extraction, "real", visit), vector.clone())
                    write_json(directory / "geometry_audit" / f"{visit['canonical_patient_id']}_T0.json", audit)
                    completed += 1
                    status(output, "extracting_holdout_resized_t0", completed=completed, total=96,
                           elapsed_seconds=time.perf_counter() - started)
                    del volume, vector
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(directory / "COMPLETE.json", {"patients": 96, "timepoint": 0, "inference_only": True, "completed_utc": now()})
    return directory / "embeddings/real"


def attach_resized_t0(real, directory):
    result = copy.deepcopy(real)
    expected = {str(pid) for pid in real["pids"]}
    files = list(Path(directory).glob("*/*.pt"))
    if len(files) != len(expected) or {p.parent.name for p in files} != expected or any(not p.stem.endswith("_T0") for p in files):
        raise ValueError("Resized T0 inventory differs from the fixed cohort")
    store = EmbStore(str(directory))
    for i, pid in enumerate(real["pids"]):
        result["embs"][i, 0] = store.load(pid, 0, require_finite=True)
    return result


def frozen_probability(checkpoint, split, device):
    effective = checkpoint["task"]["effective"]
    prior = _prior_from_state(split["clinical"], checkpoint["clinical_prior"])
    if effective["kind"] == "clinical":
        return (1 / (1 + np.exp(-prior))).astype(np.float64)
    model = TDN({"downstream": effective}).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    probability, _ = predict(model, tensor_split(split, prior, device))
    return probability.astype(np.float64)


def evaluate_record(cfg, output, record, splits, device="cuda:0"):
    path = Path(record["path"])
    if identity(path / "model.pt") != record["checkpoint_identity"]:
        raise ValueError("Frozen checkpoint changed during evaluation")
    destination = output / "models" / f"model_{record['model_id']:04d}"
    done, predictions = destination / "COMPLETE.json", destination / "predictions.csv"
    if done.exists() and predictions.exists() and read_json(done)["record"] == record:
        frame = pd.read_csv(predictions, dtype={"patient_id": str})
        if len(frame) == 96 * 3 and np.isfinite(frame.probability).all():
            return
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    task = checkpoint["task"]
    development = load_development(cfg["output_dir"], str(source_directory(cfg, record["geometry"])))
    val = select_patients(development, task["val_ids"], record["depth"])
    fresh = frozen_probability(checkpoint, val, device)
    saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
    saved = saved[saved.role == "validation"].set_index("patient_id").loc[val["pids"]]
    difference = float(np.max(np.abs(fresh - saved.probability.to_numpy())))
    if difference > 1e-5 or not np.array_equal(val["labels"], saved.label.to_numpy()):
        raise ValueError("Frozen checkpoint failed development prediction replay")
    rows = []
    for source in SOURCES:
        split = select_patients(splits[source], splits[source]["pids"], record["depth"])
        probability = frozen_probability(checkpoint, split, device)
        rows.append(pd.DataFrame({"patient_id": split["pids"], "label": split["labels"].astype(int),
                                  "source": source, "probability": probability,
                                  "depth": record["depth"], "seed": record["seed"], "outer": record["outer"]}))
    frame = pd.concat(rows, ignore_index=True)
    if record["depth"] == 1:
        pivot = frame.pivot(index="patient_id", columns="source", values="probability")
        if not (np.array_equal(pivot.real, pivot.symm) and np.array_equal(pivot.real, pivot.bifm)):
            raise ValueError("T0 must be identical across real and generated branches")
    _atomic_csv(predictions, frame)
    write_json(done, {"record": record, "development_replay_max_error": difference,
                      "completed_utc": now(), "inference_only": True})


def paired_auc_difference(labels, left, right, samples=2000, seed=2026):
    """Stratified patient bootstrap of mean seed AUROC using pairwise ranks."""
    labels, left, right = np.asarray(labels), np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != len(labels):
        raise ValueError("Paired probability matrices must have shape seeds x patients")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Paired AUROC requires both classes")

    def comparisons(values):
        a, b = values[:, positive, None], values[:, None, negative]
        return (a > b).mean(axis=0) + .5 * (a == b).mean(axis=0)

    delta = comparisons(left) - comparisons(right)
    rng = np.random.default_rng(seed)
    count_p = rng.multinomial(len(positive), np.full(len(positive), 1 / len(positive)), size=samples)
    count_n = rng.multinomial(len(negative), np.full(len(negative), 1 / len(negative)), size=samples)
    draws = np.einsum("bi,ij,bj->b", count_p, delta, count_n, optimize=True) / (len(positive) * len(negative))
    return {"difference": float(delta.mean()), "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "bootstrap_samples": samples}


def aggregate(cfg, output, contract, versions, name):
    paths = {r["path"]: r for r in contract["records"]}
    rows = []
    for ref in contract["references"]:
        if ref["arm"] not in versions:
            continue
        record = paths[ref["path"]]
        frame = pd.read_csv(output / "models" / f"model_{record['model_id']:04d}/predictions.csv", dtype={"patient_id": str})
        frame["version"] = ref["arm"]
        rows.append(frame)
    folds = pd.concat(rows, ignore_index=True)
    keys = ["version", "source", "depth", "seed", "patient_id"]
    if not (folds.groupby(keys).outer.nunique() == 5).all() or not (folds.groupby(keys).size() == 5).all():
        raise ValueError("Each patient requires exactly five distinct fold predictions")
    averaged = folds.groupby(keys, as_index=False).agg(label=("label", "first"), probability=("probability", "mean"))
    old = pd.read_csv(Path(cfg["baseline_run"]) / "evaluation/predictions.csv", dtype={"patient_id": str})
    old = old[old.source.isin(SOURCES)].copy()
    old["depth"] = old.temporal_depth.map({v: k for k, v in DEPTH_NAMES.items()})
    old["version"] = "original_v1"
    all_predictions = pd.concat([old[averaged.columns], averaged], ignore_index=True)
    metric_rows, label_reference = [], old[old.source == "real"].drop_duplicates("patient_id").set_index("patient_id").label
    for key, frame in all_predictions.groupby(["version", "source", "depth", "seed"]):
        if len(frame) != 96 or not frame.patient_id.is_unique or set(frame.patient_id) != set(contract["holdout_ids"]):
            raise ValueError("Patient membership differs between reported sources")
        if not np.array_equal(frame.label.to_numpy(), label_reference.loc[frame.patient_id].to_numpy()):
            raise ValueError("Paired evaluation labels differ")
        metric_rows.append({**dict(zip(("version", "source", "depth", "seed"), key)),
                            **metric_values(frame.label.to_numpy(), frame.probability.to_numpy()),
                            "prauc": float(average_precision_score(frame.label, frame.probability))})
    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby(["version", "source", "depth"])[["auroc", "logloss", "brier", "prauc"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    old_summary = pd.read_csv(Path(cfg["baseline_run"]) / "evaluation/summary.csv")
    old_summary = old_summary[(old_summary.population == "all_prefixes") & (old_summary.threshold_policy == "development_oof_bacc")]
    for row in summary[summary.version == "original_v1"].itertuples():
        expected = old_summary[(old_summary.source == row.source) & (old_summary.temporal_depth == DEPTH_NAMES[row.depth])].iloc[0]
        if abs(row.auroc_mean - expected.auroc_mean) > 1e-12:
            raise ValueError("Original 96-patient results did not replay")
    paired = []
    for source in SOURCES:
        for depth in cfg["depths"]:
            group = all_predictions[(all_predictions.source == source) & (all_predictions.depth == depth)]
            ids = sorted(contract["holdout_ids"])
            matrices = {version: frame.pivot(index="seed", columns="patient_id", values="probability").loc[:, ids].to_numpy()
                        for version, frame in group.groupby("version")}
            labels = label_reference.loc[ids].to_numpy()
            for version in versions:
                paired.append({"version": version, "source": source, "depth": depth,
                               **paired_auc_difference(labels, matrices[version], matrices["original_v1"], contract["bootstrap_samples"])})
    destination = output / "reports" / name
    _atomic_csv(destination / "predictions.csv", all_predictions)
    _atomic_csv(destination / "metrics_per_seed.csv", metrics)
    _atomic_csv(destination / "summary.csv", summary)
    _atomic_csv(destination / "paired_auroc_differences.csv", pd.DataFrame(paired))
    write_json(destination / "summary.json", {"schema": SCHEMA, "patients": 96,
               "positives": int(label_reference.sum()), "seeds": cfg["formal_seeds"],
               "summary": summary.to_dict("records"), "paired_auroc_differences": paired,
               "generator_snapshots": contract["generator_snapshots"], "fit_or_selection_on_holdout": False,
               "role": contract["evaluation_role"], "created_utc": now()})
    lines = ["# Nonregistered first-post pCR: fixed 96-patient holdout", "",
             "Every entry uses the same 96 patients (33 pCR positive), including incomplete T0-starting prefixes.",
             "For each seed, average five fold probabilities first; report the mean and sample SD of ten seed metrics.",
             "The classifiers were frozen before this evaluation. No fitting, epoch selection, threshold tuning, or model ranking uses these labels.",
             "The generators are the archived Symm80000 EMA and BiFM10000 ordinary snapshots, not newer ongoing generator weights.",
             "Generated branches retain real T0, real future pre/late channels and target ROI geometry; only future first-post is generated from the previous real visit.",
             "This is an internal pCR holdout previously used for upstream model selection, not an independent end-to-end test.", "",
             "| Version | Source | Window | AUROC mean | Seed SD | Log loss mean |", "|---|---|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.version} | {row.source} | {DEPTH_NAMES[row.depth]} | {row.auroc_mean:.5f} | {row.auroc_std:.5f} | {row.logloss_mean:.5f} |")
    lines.extend(["", "Paired AUROC intervals resample patients while retaining their paired seed predictions; they target the mean seed AUROC difference, not the AUROC of a seed-probability ensemble.",
                  "Intervals condition on fixed fitted models, omit refitting uncertainty, and are not adjusted for multiple comparisons.", ""])
    _atomic_text(destination / "README.md", "\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="configs/first_post_pcr_optimization_v2.yaml")
    parser.add_argument("--output", default="results/first_post_pcr_optimization_v2_holdout96_20260915")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    cfg, output = load_study(args.study), repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        status(output, "freezing_inputs")
        contract, cohort = freeze_inputs(cfg, output)
        baseline = Path(cfg["baseline_run"])
        real = _load_split(baseline / "embeddings/real", baseline / "metadata_enriched.csv", contract["holdout_ids"])
        splits = {"real": real, **{s: overlay(real, baseline / "embeddings" / s) for s in ("symm", "bifm")}}
        physical = [r for r in contract["records"] if r["geometry"] == "physical_roi"]
        for index, record in enumerate(physical):
            evaluate_record(cfg, output, record, splits)
            if (index + 1) % 20 == 0 or index + 1 == len(physical):
                status(output, "evaluating_physical_models", completed=index + 1, total=len(physical))
        aggregate(cfg, output, contract, ["nested_baseline", "physical_roi_selected"], "physical_roi")
        scaled = [r for r in contract["records"] if r["geometry"] == "resized_roi"]
        if scaled:
            directory = resized_t0(cfg, output, cohort)
            scaled_real = attach_resized_t0(real, directory)
            scaled_splits = {s: scaled_real for s in SOURCES}
            for index, record in enumerate(scaled):
                evaluate_record(cfg, output, record, scaled_splits)
                status(output, "evaluating_resized_t0_models", completed=index + 1, total=len(scaled))
        aggregate(cfg, output, contract, ["nested_baseline", "physical_roi_selected", "joint_geometry_selected"], "final")
        freeze_inputs(cfg, output)
        write_json(output / "COMPLETE.json", {"schema": SCHEMA, "completed_utc": now(), "patients": 96,
                   "sources": list(SOURCES), "unique_classifiers": len(contract["records"]),
                   "holdout_evaluated": True, "optimizer_updates": 0, "selection_changed": False})
        status(output, "complete", patients=96, unique_classifiers=len(contract["records"]), optimizer_updates=0)
    except BaseException as error:
        status(output, "failed", error_type=type(error).__name__, error=str(error)[-1200:])
        raise


if __name__ == "__main__":
    main()

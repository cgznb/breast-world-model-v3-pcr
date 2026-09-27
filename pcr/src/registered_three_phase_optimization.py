"""Nested improvement study using the frozen three-phase ROI32 Pillar features."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _prior_from_state, _read_metadata
from src import first_post_optimization as opt
from src import single_phase_temporal_training as temporal
from src.first_post_pcr_data import identity, now, read_json, repo_path, write_json
from src.single_phase_repeat_pcr import pillar_identity
from src.single_phase_temporal_models import build_model, window_split

SCHEMA = "registered_three_phase_roi32_pcr_optimization_v2"
STUDY = "four_windows"
RUNTIME = (
    "src/registered_three_phase_optimization.py",
    "scripts/run_registered_three_phase_optimization.py",
    "src/first_post_optimization.py", "src/single_phase_temporal_training.py",
    "src/single_phase_temporal_models.py", "src/single_phase_repeat_retraining.py",
    "src/tdn.py", "src/temporal.py", "src/data.py", "src/first_post_pcr_data.py",
    "scripts/run_full978_anti_overfit.py", "scripts/run_full978_independent_cv.py",
)


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("Wrong three-phase optimization schema")
    cfg["config_path"] = str(path)
    for key in ("output_dir", "baseline_run", "direct_run", "sequential_run"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if cfg["depths"] != [1, 2, 3, 4] or cfg["formal_seeds"] != list(range(42, 52)):
        raise ValueError("Four windows and the ten original classifier seeds are required")
    if cfg["feature_sources"] != ["physical_roi"]:
        raise ValueError("Only the original physical ROI32 features are allowed")
    if cfg["output_dir"] in [cfg[k] for k in ("baseline_run", "direct_run", "sequential_run")]:
        raise ValueError("The new study must use a separate output directory")
    return cfg


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        write_json(path, value)


def verify_inventory(inventory):
    for path, expected in inventory.items():
        if identity(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def prepare(cfg):
    output, baseline = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    output.mkdir(parents=True, exist_ok=True)
    for name in ("COMPLETE.json", "DEVELOPMENT_FEATURES_COMPLETE.json", "TRAINING_COMPLETE.json"):
        if not (baseline / name).is_file():
            raise ValueError("The original ROI32 training is incomplete")
    cohort, outer = read_json(baseline / "cohort.json"), read_json(baseline / "folds.json")
    if isinstance(outer, dict):
        outer = list(outer.values())
    development, holdout = cohort["split"]["train"], cohort["split"]["val"]
    if (len(development), len(holdout)) != (cfg["expected_development"], cfg["expected_holdout"]):
        raise ValueError("The original patient population changed")
    if cohort["image_shape_czyx"] != [3, 32, 128, 128] or cohort["embedding_dim"] != 1152:
        raise ValueError("Wrong image or feature representation")
    original = read_json(baseline / "input_contract.json")
    if pillar_identity() != original["pillar_files"]:
        raise ValueError("The original frozen encoder changed")
    metadata = _read_metadata(baseline / "metadata_enriched.csv", development)
    labels = dict(zip(metadata.pid, metadata.pCR.astype(int)))
    folds = opt.nested_folds(development, holdout, outer, labels,
                             cfg["inner_folds"], cfg["inner_fold_seed"])
    contract = {
        "schema": SCHEMA, "config": cfg,
        "runtime": {str(repo_path(p)): identity(repo_path(p)) for p in RUNTIME},
        "config_identity": identity(cfg["config_path"]),
        "pillar_files": original["pillar_files"],
        "development_patients": len(development), "holdout_patients": len(holdout),
        "effective_candidates": {str(d): {c: opt.effective_config(cfg, c, d) for c in cfg["candidates"]}
                                 for d in cfg["depths"]},
        "selection": "per outer fold/window: inner mean AUROC within 0.005 of best, then minimum logloss",
        "duration": "ceil(median selected inner epochs); fixed-epoch outer refit",
        "pca": "training patients and window-allowed contiguous visits only; 32 whitened components",
        "outer_validation_used_for_selection": False, "holdout_used_for_selection": False,
        "holdout_role": cohort["test_role"], "reuse_existing_generated_features": True,
    }
    frozen_json(output / "contract.json", contract)
    frozen_json(output / "nested_folds.json", folds)
    metadata_path = output / "development_metadata.csv"
    if metadata_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(metadata_path, dtype={"pid": str}),
                                      metadata.reset_index(drop=True), check_dtype=False)
    else:
        _atomic_csv(metadata_path, metadata)
    inventory_path = output / "original_inventory.json"
    if inventory_path.exists():
        verify_inventory(read_json(inventory_path))
    else:
        paths = {p.resolve() for p in baseline.rglob("*") if p.is_file()}
        for section in ("runtime_sources", "input_sources", "crop_reports"):
            entries = original.get(section, {})
            if isinstance(entries, dict):
                for name, expected in entries.items():
                    if isinstance(expected, dict) and "size_bytes" in expected:
                        p = Path(name)
                        if not p.is_absolute():
                            p = repo_path(p)
                        if identity(p) != expected:
                            raise ValueError(f"Original experiment source identity changed: {p}")
                        paths.add(p.resolve())
        frozen_json(inventory_path, {str(p): identity(p) for p in sorted(paths)})
    return folds


def task_uses_transform(task):
    return task["effective"].get("feature_transform", "identity") != "identity"


def run_task(task, output, embedding_dir, device="cpu"):
    if task["stage"] not in ("inner", "outer"):
        raise ValueError("Invalid fitting stage")
    if (task["stage"] == "outer") != (task.get("fixed_epochs") is not None):
        raise ValueError("Outer fitting requires a fixed duration; inner fitting cannot use one")
    if task_uses_transform(task):
        return temporal.run_task(task, output, embedding_dir, device)
    return opt.run_task(task, output, embedding_dir, device)


def load_model(checkpoint, device="cpu"):
    task = checkpoint["task"]
    if task["effective"]["kind"] == "clinical":
        return None
    if task_uses_transform(task):
        model = build_model(task["effective"], task["depth"])
    else:
        model = opt.TDN({"downstream": task["effective"]})
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def predict_checkpoint(checkpoint, raw, device="cpu", model=None):
    task = checkpoint["task"]
    split = window_split(raw, raw["pids"], task["depth"], task["effective"])
    prior = _prior_from_state(split["clinical"], checkpoint["clinical_prior"])
    if task["effective"]["kind"] == "clinical":
        return 1 / (1 + np.exp(-prior)), np.zeros(len(prior), dtype=np.float32), prior
    if model is None:
        model = load_model(checkpoint, device)
    probability, residual = opt.predict(model, opt.tensor_split(split, prior, device))
    return probability, residual, prior


def freeze_models(cfg, folds, selection):
    output = Path(cfg["output_dir"])
    tasks = opt.formal_tasks(cfg, folds, selection)
    inventory = {}
    for task in tasks:
        path = opt.task_path(output, task)
        if not opt.task_complete(path, task):
            raise ValueError("All formal fits must complete before evaluation")
        for name in ("model.pt", "COMPLETE.json", "predictions.csv", "history.csv"):
            inventory[str(path / name)] = identity(path / name)
    for name in ("contract.json", "nested_folds.json", f"selection/{STUDY}.json",
                 f"selection/{STUDY}.csv", f"selection/{STUDY}_outer_models.json"):
        inventory[str(output / name)] = identity(output / name)
    frozen_json(output / "FROZEN_MODELS.json", {
        "schema": SCHEMA, "models": len(tasks), "artifacts": inventory,
        "holdout_used_for_selection": False,
    })
    verify_inventory(read_json(output / "original_inventory.json"))
    return tasks


def verify_fitted_predictions(cfg, tasks, device="cpu"):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(cfg["output_dir"])
    raw = opt.load_development(str(output), str(opt.source_directory(cfg, "physical_roi")))
    max_error = 0.0
    for task in tasks:
        path = opt.task_path(output, task)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        if not opt.task_complete(path, task):
            raise ValueError("Invalid saved fit")
        frame = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str}).set_index("patient_id")
        model = load_model(checkpoint, device)
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            part = window_split(raw, ids, task["depth"], task["effective"])
            probability, _, _ = predict_checkpoint(checkpoint, part, device, model)
            saved = frame.loc[part["pids"]]
            if not (saved.role == role).all():
                raise ValueError("Saved prediction roles changed")
            error = float(np.max(np.abs(probability - saved.probability.to_numpy())))
            max_error = max(max_error, error)
            if error > 1e-6:
                raise ValueError("Saved-weight prediction replay failed")
    return {"fits": len(tasks), "max_prediction_error": max_error}


def smoke(cfg, device):
    local = copy.deepcopy(cfg)
    local["output_dir"] = str(Path(cfg["output_dir"]) / "smoke" / device.replace(":", "_"))
    folds = prepare(local)
    tasks = opt.inner_tasks(local, folds, ["physical_roi"])
    elapsed, checked = [], []
    for candidate in cfg["candidates"]:
        task = copy.deepcopy(next(t for t in tasks if t["candidate"] == candidate and t["depth"] == 4))
        if candidate != "clinical_only":
            task["effective"].update(epochs=3, patience=3)
        start = time.perf_counter()
        result = run_task(task, local["output_dir"], opt.source_directory(cfg, "physical_roi"), device)
        elapsed.append({"candidate": candidate, "seconds": time.perf_counter() - start,
                        "skipped": result["skipped"]})
        checked.append(task)
        fixed = copy.deepcopy(task)
        fixed.update(stage="outer", inner=-1, fixed_epochs=0 if candidate == "clinical_only" else 3)
        run_task(fixed, local["output_dir"], opt.source_directory(cfg, "physical_roi"), device)
        checked.append(fixed)
        if not run_task(fixed, local["output_dir"], opt.source_directory(cfg, "physical_roi"), device)["skipped"]:
            raise ValueError("Completed smoke fit did not resume as a no-op")
    verification = verify_fitted_predictions(local, checked, device)
    write_json(Path(local["output_dir"]) / "SMOKE_COMPLETE.json", {
        "passed": True, "created_utc": now(), "device": device,
        "timings": elapsed, "verification": verification,
    })
    return elapsed


def progress(cfg, stage, **fields):
    payload = {"updated_utc": now(), "pid": os.getpid(), "stage": stage, **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", payload)
    print(__import__("json").dumps(payload), flush=True)

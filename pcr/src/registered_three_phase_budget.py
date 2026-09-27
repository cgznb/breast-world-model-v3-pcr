"""Matched V1/V4 refits with validation-AUC stopping at 300 epochs/patience 50."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _prior_from_state, _read_metadata
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import identity, now, public, read_json, repo_path, save_tensor, write_json

SCHEMA = "registered_three_phase_roi32_pcr_300_50"
RUNTIME = (*shared.RUNTIME, "src/registered_three_phase_budget.py",
           "scripts/run_registered_three_phase_budget.py")


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema"] != SCHEMA or cfg["formal_seeds"] != list(range(42, 52)):
        raise ValueError("Wrong study or classifier seeds")
    if cfg["windows"] != {"v1": [1, 2, 3, 4], "v4": [4]}:
        raise ValueError("V1 has four windows; the existing V4 recipe has T3 only")
    if cfg["training"] != {"epochs": 300, "patience": 50, "scheduler_t_max": 300,
                           "epoch_selection": "auroc", "min_delta": 0.0}:
        raise ValueError("Expected 300/50 with strict validation-AUC improvement")
    cfg["config_path"] = str(path)
    for key in ("output_dir", "baseline_run", "fixed_run", "v4_run", "direct_run",
                "sequential_run", "comparison_run"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if any(Path(cfg["output_dir"]).is_relative_to(cfg[k])
           or Path(cfg[k]).is_relative_to(cfg["output_dir"])
           for k in cfg if k.endswith("_run")):
        raise ValueError("New results must be isolated from all earlier experiments")
    return cfg


def effective_config(cfg, arm, depth):
    if depth not in cfg["windows"][arm]:
        raise ValueError("Undefined version/window")
    name = opt.DEPTH_NAMES[depth].lower().replace("-", "_")
    effective = yaml.safe_load((Path(cfg["baseline_run"]) / "configs" / f"{name}.yaml").read_text())["downstream"]
    effective.pop("checkpoint_policy", None)
    effective.setdefault("use_clinical_token", True)
    effective.setdefault("positive_class_weight", "balanced")
    effective.setdefault("residual_logit_limit", None)
    effective.update(cfg["training"])
    effective.update(kind="tdn", architecture="tdn", feature_transform="identity",
                     visit_policy="contiguous", input_dim=1152, clinical_dim=17,
                     residual_l2_weight=0.1 if arm == "v4" else 0.0)
    return effective


def task_path(output, task):
    return (Path(output) / "fits" / task["candidate"] / opt.DEPTH_NAMES[task["depth"]]
            / f"fold_{task['outer']}" / f"seed_{task['seed']}")


def build_tasks(cfg, folds):
    tasks = []
    for arm, depths in cfg["windows"].items():
        for depth in depths:
            for fold in folds:
                for seed in cfg["formal_seeds"]:
                    tasks.append({"stage": "cv_checkpoint_selection", "source": "physical_roi",
                                  "depth": depth, "candidate": arm, "outer": fold["fold"],
                                  "inner": -1, "seed": seed,
                                  "train_ids": fold["train_ids"], "val_ids": fold["val_ids"],
                                  "effective": effective_config(cfg, arm, depth)})
    return tasks


def prepare(cfg):
    output, baseline = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    cohort, folds = read_json(baseline / "cohort.json"), read_json(baseline / "folds.json")
    if isinstance(folds, dict):
        folds = list(folds.values())
    dev, held = cohort["split"]["train"], cohort["split"]["val"]
    opt.validate_partitions(dev, held, folds)
    if (len(dev), len(held), {f["fold"] for f in folds}) != (764, 102, set(range(5))):
        raise ValueError("Original cohort or five folds changed")
    if cohort["embedding_dim"] != 1152 or cohort["image_shape_czyx"] != [3, 32, 128, 128]:
        raise ValueError("Wrong cached representation")
    tasks = build_tasks(cfg, folds)
    if len(tasks) != 250:
        raise ValueError("Expected 250 fresh classifier fits")
    refs = [{"arm": t["candidate"], "depth": t["depth"], "seed": t["seed"],
             "outer": t["outer"], "path": str(task_path(output, t))} for t in tasks]
    contract = public({
        "schema": SCHEMA, "config": cfg, "config_identity": identity(cfg["config_path"]),
        "runtime": {str(repo_path(p)): identity(repo_path(p)) for p in RUNTIME},
        "resolved_recipes": {a: {str(d): effective_config(cfg, a, d) for d in ds}
                             for a, ds in cfg["windows"].items()},
        "new_fits": len(tasks), "development_patients": len(dev), "historical_patients": len(held),
        "validation_used_for_checkpoint_selection": True, "holdout_used_for_selection": False,
        "selection": "First epoch attaining the highest validation AUC; stop after50 consecutive non-improving epochs or at300; restore selected weights.",
        "min_delta_change": "All windows use0.0; original V1 T2 used0.001. Cosine T_max is300; other recipe settings are retained.",
        "rng_seed": "classifier_seed*10000 + fold*100 + depth; identical for V1/V4 T3",
        "v4_change": "T3 residual-logit L2 penalty0.1; no other T3 architecture/loss differences",
        "generated_inputs": "Reuse frozen Heun25/MC4 features; train classifiers on real development input only.",
        "main_estimands": ["Mean of five model AUCs on the same102patients, with sample fold SD",
                           "AUC of the mean of five patient probabilities"],
        "across_seed_averaging": False,
        "limitations": "Development folds select checkpoints and are selection validation. Historical102patients also served as generator validation and were repeatedly evaluated; exploratory, not untouched end-to-end testing.",
    })
    for name, value in (("contract.json", contract), ("cohort.json", public(cohort)),
                        ("nested_folds.json", folds), ("model_references.json", refs)):
        shared.frozen_json(output / name, value)
    metadata = _read_metadata(baseline / "metadata_enriched.csv", dev).reset_index(drop=True)
    metadata_path = output / "development_metadata.csv"
    if metadata_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(metadata_path, dtype={"pid": str}), metadata, check_dtype=False)
    else:
        _atomic_csv(metadata_path, metadata)
    inventory_path = output / "input_inventory.json"
    if not inventory_path.exists():
        inventory = {}
        fixed, v4 = Path(cfg["fixed_run"]), Path(cfg["v4_run"])
        for path, key in ((fixed / "prior_inventory.json", None),
                          (fixed / "evaluation_input_inventory.json", None),
                          (fixed / "FROZEN_MODELS.json", "artifacts"),
                          (v4 / "FROZEN_MODELS.json", "artifacts"),
                          (v4 / "candidate_followup/input_inventory.json", None),
                          (Path(cfg["comparison_run"]) / "verification.json", "input_identities")):
            entries = read_json(path)
            entries = entries[key] if key else entries
            shared.verify_inventory(entries)
            inventory.update(entries)
            inventory[str(path)] = identity(path)
        for key in ("baseline_run", "comparison_run"):
            inventory.update({str(p): identity(p) for p in Path(cfg[key]).rglob("*") if p.is_file()})
        inventory.update(contract["runtime"])
        for path in (Path(cfg["config_path"]), metadata_path, output / "contract.json",
                     output / "cohort.json", output / "nested_folds.json", output / "model_references.json"):
            inventory[str(path)] = identity(path)
        shared.frozen_json(inventory_path, inventory)
    shared.verify_inventory(read_json(inventory_path))
    return folds, tasks, refs


def validate_history(history, effective, selected):
    if history.epoch.tolist() != list(range(1, len(history) + 1)):
        raise ValueError("Noncontiguous training history")
    if not np.isfinite(history.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite training history")
    best, best_epoch, stale = -np.inf, 0, 0
    for row in history.itertuples():
        if row.validation_auroc > best + effective.get("min_delta", 0):
            best, best_epoch, stale = row.validation_auroc, row.epoch, 0
        else:
            stale += 1
        if row.epoch < len(history) and stale >= effective["patience"]:
            raise ValueError("Training continued beyond patience")
    if (selected != best_epoch or not 1 <= len(history) <= effective["epochs"]
            or (len(history) < effective["epochs"] and stale != effective["patience"])):
        raise ValueError("Selected epoch or stopping time violates the declared rule")
    epochs = history.epoch.to_numpy(dtype=float)
    eta = effective.get("scheduler_eta_min", 0.0)
    expected_lr = eta + (effective["lr"] - eta) * (1 + np.cos(np.pi * epochs / effective["scheduler_t_max"])) / 2
    np.testing.assert_allclose(history.lr, expected_lr, rtol=1e-9, atol=1e-15)
    return "patience" if stale >= effective["patience"] else "epoch_cap"


def task_complete(path, task):
    if not opt.task_complete(path, task):
        return False
    try:
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        if not state["validation_used_for_checkpoint_selection"] or state["holdout_labels_or_embeddings_loaded"]:
            return False
        history = pd.read_csv(path / "history.csv", float_precision="round_trip")
        validate_history(history, task["effective"], state["selected_epochs"])
        return True
    except (OSError, ValueError, KeyError, AssertionError):
        return False


def run_task(task, output, embedding_dir, device="cpu"):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    path = task_path(output, task)
    if task_complete(path, task):
        return {"path": str(path), "skipped": True}
    if set(task["train_ids"]) & set(task["val_ids"]):
        raise ValueError("Training/validation overlap")
    started = time.perf_counter()
    raw = opt.load_development(str(output), str(embedding_dir))
    train = opt.select_patients(raw, task["train_ids"], task["depth"])
    val = opt.select_patients(raw, task["val_ids"], task["depth"])
    seed = task["seed"] * 10000 + task["outer"] * 100 + task["depth"]
    model, state = opt.fit_tdn(train, val, task["effective"], seed, task["depth"], device)
    history = pd.DataFrame(state.pop("history"))
    reason = validate_history(history, task["effective"], state["selected_epochs"])
    state.update(schema=SCHEMA, task=task, created_utc=now(), initialization_seed=seed,
                 validation_used_for_checkpoint_selection=True,
                 outer_validation_used_for_selection=True, holdout_labels_or_embeddings_loaded=False)
    save_tensor(path / "model.pt", state)
    restored = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(restored["model_state"], strict=True)
    predictions, metrics = [], {}
    for role, split in (("train", train), ("validation", val)):
        prior = _prior_from_state(split["clinical"], restored["clinical_prior"])
        probability, residual = opt.predict(model, opt.tensor_split(split, prior, device))
        metrics[role] = opt.metric_values(split["labels"], probability)
        predictions.append(pd.DataFrame({"patient_id": split["pids"], "role": role,
            "label": split["labels"].astype(int), "probability": probability.astype(float),
            "residual_logit": residual.astype(float),
            "clinical_prior_probability": (1 / (1 + np.exp(-prior))).astype(float)}))
    _atomic_csv(path / "predictions.csv", pd.concat(predictions, ignore_index=True))
    _atomic_csv(path / "history.csv", history)
    write_json(path / "COMPLETE.json", {
        "schema": SCHEMA, "task": task, **metrics, "selected_epochs": state["selected_epochs"],
        "epochs_trained": state["epochs_trained"], "parameters": state["parameters"],
        "stop_reason": reason, "seconds": time.perf_counter() - started, "completed_utc": now(),
        "auroc_gap": metrics["train"]["auroc"] - metrics["validation"]["auroc"],
        "checkpoint_reloaded_before_prediction": True,
    })
    if not task_complete(path, task):
        raise RuntimeError("Saved fit failed validation")
    return {"path": str(path), "skipped": False}


def verify_training(cfg, tasks):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(cfg["output_dir"])
    raw = opt.load_development(str(output), str(opt.source_directory(cfg, "physical_roi")))
    max_error, epoch_rows, priors = 0.0, [], {}
    for index, task in enumerate(tasks, 1):
        path = task_path(output, task)
        if not task_complete(path, task):
            raise ValueError(f"Invalid completed fit: {path}")
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        train = opt.select_patients(raw, task["train_ids"], task["depth"])
        if task["outer"] not in priors:
            _, _, priors[task["outer"]] = opt._fit_prior(train, train, state["initialization_seed"])
        if state["clinical_prior"] != priors[task["outer"]]:
            raise ValueError("Clinical prior differs from training-patients-only refit")
        model = shared.load_model(state)
        saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str},
                            float_precision="round_trip").set_index("patient_id")
        history = pd.read_csv(path / "history.csv", float_precision="round_trip")
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            split = opt.select_patients(raw, ids, task["depth"])
            probability, _, _ = shared.predict_checkpoint(state, split, model=model)
            rows = saved.loc[split["pids"]]
            if not (rows.role == role).all():
                raise ValueError("Prediction partition changed")
            np.testing.assert_array_equal(rows.label, split["labels"])
            error = float(np.max(np.abs(probability - rows.probability.to_numpy())))
            max_error = max(max_error, error)
            if error > 1e-7:
                raise ValueError("Saved-weight replay changed predictions")
            actual = opt.metric_values(split["labels"], probability)
            selected = history.iloc[state["selected_epochs"] - 1]
            for metric in actual:
                if abs(actual[metric] - selected[f"{role}_{metric}"]) > 1e-6:
                    raise ValueError("Restored weights do not match selected-epoch metrics")
        record = read_json(path / "COMPLETE.json")
        epoch_rows.append({"model": task["candidate"], "depth": task["depth"],
                           "window": opt.DEPTH_NAMES[task["depth"]], "seed": task["seed"],
                           "outer": task["outer"], **{k: record[k] for k in
                           ("selected_epochs", "epochs_trained", "stop_reason", "auroc_gap", "parameters")},
                           "train_auroc": record["train"]["auroc"],
                           "validation_auroc": record["validation"]["auroc"]})
        if index % 50 == 0:
            shared.progress(cfg, "training_verification", completed=index, total=len(tasks))
    _atomic_csv(output / "training_epochs.csv", pd.DataFrame(epoch_rows))
    return {"passed": True, "fits": len(tasks), "max_prediction_error": max_error,
            "train_only_clinical_priors_refitted": len(priors),
            "stopping_and_cosine_schedule_verified": True, "selected_weights_verified": True}


def freeze_models(cfg, tasks):
    output = Path(cfg["output_dir"])
    artifacts = {}
    for task in tasks:
        path = task_path(output, task)
        if not task_complete(path, task):
            raise ValueError("Every fit must finish before historical evaluation")
        for name in ("model.pt", "history.csv", "predictions.csv", "COMPLETE.json"):
            artifacts[str(path / name)] = identity(path / name)
    shared.frozen_json(output / "FROZEN_MODELS.json", {
        "schema": SCHEMA, "models": len(tasks), "artifacts": artifacts,
        "holdout_used_for_selection": False, "validation_used_for_checkpoint_selection": True})


def smoke(cfg):
    local = copy.deepcopy(cfg)
    local["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    local["training"].update(epochs=3, patience=2, scheduler_t_max=3)
    _, all_tasks, _ = prepare(local)
    tasks = [t for t in all_tasks if t["seed"] == 42 and t["outer"] == 0]
    for task in tasks:
        run_task(task, local["output_dir"], opt.source_directory(cfg, "physical_roi"))
        if not run_task(task, local["output_dir"], opt.source_directory(cfg, "physical_roi"))["skipped"]:
            raise ValueError("Completed smoke fit was not reused")
    audit = verify_training(local, tasks)
    write_json(Path(local["output_dir"]) / "SMOKE_COMPLETE.json", audit)
    return audit

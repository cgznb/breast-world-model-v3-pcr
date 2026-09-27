"""Uniform five-fold recipes, fixed duration, and separate classifier seeds."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _read_metadata
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as previous
from src.first_post_pcr_data import identity, now, read_json, repo_path, write_json

SCHEMA = "registered_three_phase_roi32_fixed_pcr_v3"
ARMS = ("primary", "reference")
SOURCES = ("real", "direct_mc4", "rollout_mc4", "previous_real_mc4")
RUNTIME = tuple(dict.fromkeys((*previous.RUNTIME, "src/registered_three_phase_fixed_pcr.py",
                              "scripts/run_registered_three_phase_fixed_pcr.py")))


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("Wrong uniform-fold study schema")
    cfg["config_path"] = str(path)
    for key in ("output_dir", "baseline_run", "previous_run", "direct_run", "sequential_run"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if cfg["depths"] != [1, 2, 3, 4] or cfg["formal_seeds"] != list(range(42, 52)):
        raise ValueError("This study requires all four windows and seeds42-51")
    if cfg["common_training"]["epochs"] != 40 or cfg["common_training"]["scheduler_t_max"] != 40:
        raise ValueError("The declared experiment uses the same40-epoch budget throughout")
    if cfg["output_dir"] in [cfg[k] for k in ("baseline_run", "previous_run", "direct_run", "sequential_run")]:
        raise ValueError("A separate result directory is required")
    return cfg


def effective_config(cfg, arm, depth):
    if arm not in ARMS or depth not in cfg["depths"]:
        raise ValueError("Unknown model arm or input window")
    name = opt.DEPTH_NAMES[depth].lower().replace("-", "_")
    effective = yaml.safe_load((Path(cfg["baseline_run"]) / "configs" / f"{name}.yaml").read_text())["downstream"]
    effective.pop("checkpoint_policy", None)
    effective.setdefault("use_clinical_token", True)
    effective.setdefault("positive_class_weight", "balanced")
    effective.setdefault("residual_logit_limit", None)
    effective.update(cfg["common_training"])
    if arm == "primary":
        effective.update(cfg["primary_overrides"][depth])
    effective.update(kind="tdn", architecture="tdn", feature_transform="identity",
                     visit_policy="contiguous", input_dim=1152, clinical_dim=17)
    if effective["epochs"] != cfg["common_training"]["epochs"] or effective["epoch_selection"] != "fixed":
        raise ValueError("Per-window overrides cannot change the fixed fitting budget")
    return effective


def build_tasks(cfg, folds):
    tasks, refs, seen = [], [], set()
    for depth in cfg["depths"]:
        reference = effective_config(cfg, "reference", depth)
        for arm in ARMS:
            effective = effective_config(cfg, arm, depth)
            candidate = "reference" if effective == reference else "fixed_regularized"
            for fold in folds:
                for seed in cfg["formal_seeds"]:
                    task = {"stage": "outer", "source": "physical_roi", "depth": depth,
                            "candidate": candidate, "outer": fold["fold"], "inner": -1, "seed": seed,
                            "train_ids": fold["train_ids"], "val_ids": fold["val_ids"],
                            "effective": copy.deepcopy(effective), "fixed_epochs": effective["epochs"]}
                    path = str(opt.task_path(cfg["output_dir"], task))
                    refs.append({"arm": arm, "depth": depth, "outer": fold["fold"], "seed": seed, "path": path})
                    if path not in seen:
                        tasks.append(task)
                        seen.add(path)
    validate_uniformity(cfg, tasks, refs, folds)
    return tasks, refs


def validate_uniformity(cfg, tasks, refs, folds):
    lookup = {str(opt.task_path(cfg["output_dir"], t)): t for t in tasks}
    expected = {(arm, depth, f["fold"], seed) for arm in ARMS for depth in cfg["depths"]
                for f in folds for seed in cfg["formal_seeds"]}
    actual = {(r["arm"], r["depth"], r["outer"], r["seed"]) for r in refs}
    if len(refs) != len(expected) or actual != expected:
        raise ValueError("Missing or duplicate fold/seed references")
    by_fold = {f["fold"]: f for f in folds}
    for ref in refs:
        task = lookup[ref["path"]]
        expected_config = effective_config(cfg, ref["arm"], ref["depth"])
        if task["effective"] != expected_config or task["fixed_epochs"] != expected_config["epochs"]:
            raise ValueError("Five folds and ten seeds must use identical complete configurations")
        if (task["train_ids"] != by_fold[ref["outer"]]["train_ids"]
                or task["val_ids"] != by_fold[ref["outer"]]["val_ids"]
                or task["seed"] != ref["seed"] or task["depth"] != ref["depth"]
                or task["outer"] != ref["outer"]):
            raise ValueError("Model reference points to another training partition")


def prepare(cfg):
    output, baseline = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    output.mkdir(parents=True, exist_ok=True)
    if not (baseline / "COMPLETE.json").is_file():
        raise ValueError("Original ROI32 feature/classifier study is incomplete")
    cohort = read_json(baseline / "cohort.json")
    folds = read_json(baseline / "folds.json")
    if isinstance(folds, dict):
        folds = list(folds.values())
    development, holdout = cohort["split"]["train"], cohort["split"]["val"]
    opt.validate_partitions(development, holdout, folds)
    if (len(development), len(holdout)) != (cfg["expected_development"], cfg["expected_holdout"]):
        raise ValueError("Patient population changed")
    if cohort["image_shape_czyx"] != [3, 32, 128, 128] or cohort["embedding_dim"] != 1152:
        raise ValueError("Wrong three-phase ROI32 representation")
    metadata = _read_metadata(baseline / "metadata_enriched.csv", development)
    previous.frozen_json(output / "nested_folds.json", folds)
    previous.frozen_json(output / "cohort.json", cohort)
    metadata_path = output / "development_metadata.csv"
    if metadata_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(metadata_path, dtype={"pid": str}),
                                      metadata.reset_index(drop=True), check_dtype=False)
    else:
        _atomic_csv(metadata_path, metadata)
    tasks, refs = build_tasks(cfg, folds)
    previous.frozen_json(output / "model_references.json", refs)
    contract = {
        "schema": SCHEMA, "config": cfg, "config_identity": identity(cfg["config_path"]),
        "runtime": {str(repo_path(p)): identity(repo_path(p)) for p in RUNTIME},
        "resolved_recipes": {arm: {str(d): effective_config(cfg, arm, d) for d in cfg["depths"]} for arm in ARMS},
        "unique_fits": len(tasks), "model_references": len(refs),
        "same_config_and_duration_across_folds_and_seeds": True,
        "outer_validation_selects_weights_or_recipe": False, "holdout_used_for_selection": False,
        "current_run_selection": "none; fixed primary and fixed reference declared before training",
        "prior_evidence": "Recipes informed by earlier development studies; these reused patients do not constitute fresh validation.",
        "holdout_role": cohort["test_role"],
        "main_estimand": "For each seed/window/source: mean of five fold-model AUROCs on the same102patients; MC4 probabilities averaged within each model first.",
        "development_estimand": "For each seed/window: mean of five outer-validation AUROCs, disjoint held-out patients.",
        "across_seed_averaging": False,
    }
    previous.frozen_json(output / "contract.json", contract)
    inventory_path = output / "prior_inventory.json"
    if inventory_path.exists():
        previous.verify_inventory(read_json(inventory_path))
    else:
        old = Path(cfg["previous_run"])
        inventory = read_json(old / "original_inventory.json")
        previous.verify_inventory(inventory)
        previous.verify_inventory(read_json(old / "FROZEN_MODELS.json")["artifacts"])
        previous.verify_inventory(read_json(old / "evaluation_input_inventory.json"))
        for root in (baseline, old):
            inventory.update({str(p): identity(p) for p in root.rglob("*") if p.is_file()})
        previous.frozen_json(inventory_path, inventory)
    return folds, tasks, refs


def freeze_models(cfg, folds, tasks, refs):
    output = Path(cfg["output_dir"])
    restored_tasks, artifacts = [], {}
    for task in tasks:
        path = opt.task_path(output, task)
        if not opt.task_complete(path, task):
            raise ValueError("Every declared fit must complete before model freeze")
        saved = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        restored_tasks.append(saved["task"])
        if saved["selected_epochs"] != task["fixed_epochs"] or saved["epochs_trained"] != task["fixed_epochs"]:
            raise ValueError("A fit changed its declared training duration")
        for name in ("model.pt", "history.csv", "predictions.csv", "COMPLETE.json"):
            artifacts[str(path / name)] = identity(path / name)
    validate_uniformity(cfg, restored_tasks, refs, folds)
    for name in ("contract.json", "cohort.json", "nested_folds.json", "model_references.json"):
        artifacts[str(output / name)] = identity(output / name)
    previous.frozen_json(output / "FROZEN_MODELS.json", {
        "schema": SCHEMA, "models": len(tasks), "artifacts": artifacts,
        "same_configuration_verified": True, "holdout_used_for_selection": False,
    })


def smoke(cfg, device):
    local = copy.deepcopy(cfg)
    local["output_dir"] = str(Path(cfg["output_dir"]) / "smoke" / device.replace(":", "_"))
    local["common_training"].update(epochs=2, scheduler_t_max=2, patience=3)
    _, tasks, _ = prepare(local)
    tasks = [t for t in tasks if t["outer"] == 0 and t["seed"] == cfg["formal_seeds"][0]]
    for task in tasks:
        opt.run_task(task, local["output_dir"], opt.source_directory(local, "physical_roi"), device)
        if not opt.run_task(task, local["output_dir"], opt.source_directory(local, "physical_roi"), device)["skipped"]:
            raise ValueError("Completed smoke fit must be reused")
    audit = previous.verify_fitted_predictions(local, tasks, device)
    write_json(Path(local["output_dir"]) / "SMOKE_COMPLETE.json", {
        "passed": True, "created_utc": now(), "device": device, "audit": audit,
    })
    return audit

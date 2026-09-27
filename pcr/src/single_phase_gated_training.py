"""Nested fitting and dependency validation for the five-seed gated pCR pilot."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _prior_from_state
from scripts.run_full978_independent_cv import _make_optimizer, _positive_class_weight
from src import first_post_optimization as opt
from src.first_post_pcr_data import identity, now, read_json, repo_path, save_tensor, write_json
from src.single_phase_gated_models import build_model, drop_followups, predict
from src.single_phase_repeat_retraining import unchanged
from src.single_phase_temporal_models import window_split
from src.single_phase_temporal_training import fit_model as fit_plain

SCHEMA = "single_phase_gated_pcr_v4"
CANDIDATES = ("baseline", "wide64", "gated64", "shared64")


def load_config(path, branch):
    path = repo_path(path).resolve()
    master = yaml.safe_load(path.read_text())
    if master.get("schema") != SCHEMA or branch not in master["branches"]:
        raise ValueError("Unsupported gated pCR configuration")
    cfg = {k: copy.deepcopy(v) for k, v in master.items() if k != "branches"}
    cfg.update(master["branches"][branch], branch=branch, config_path=str(path))
    cfg["output_dir"] = str(repo_path(master["output_dir"]) / branch)
    for key in ("baseline_run", "previous_run", "input_config"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if (cfg["formal_seeds"] != list(range(42, 47)) or cfg["depths"] != [1, 2, 3, 4]
            or tuple(cfg["candidates"]) != CANDIDATES or cfg["evaluate_holdout"]):
        raise ValueError("This is a prespecified five-seed development-only pilot")
    return cfg


def prepare(cfg):
    output, source = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    cohort = read_json(source / "cohort.json")
    marker = read_json(source / "REAL_FEATURES_COMPLETE.json")
    if not marker.get("complete") or marker["input_mode"] != cfg["input_mode"]:
        raise ValueError("Frozen single-phase features are incomplete or incompatible")
    if (len(cohort["split"]["train"]), len(cohort["split"]["val"])) != (
            cfg["expected_development"], cfg["expected_holdout"]):
        raise ValueError("Original patient cohort changed")
    runtime = ["src/single_phase_gated_models.py", "src/single_phase_gated_training.py",
               "src/single_phase_gated_reporting.py", "scripts/run_single_phase_gated_pcr.py",
               "src/single_phase_temporal_training.py", "src/single_phase_temporal_models.py",
               "src/single_phase_repeat_retraining.py"]
    unchanged(output / "gated_contract.json", {"schema": SCHEMA, "config": cfg,
        "runtime": {p: identity(repo_path(p)) for p in runtime},
        "t0_reference": "baseline T0 fit on the exact same current training partition and seed",
        "holdout_evaluated": False, "automatic_thirty_seed_extension": False,
        "shared_model": "one checkpoint per fold and seed for all four causal prefixes",
        "input_policy": "frozen historical features; available visits require T0; no corrected crops"})
    return opt.prepare_study(cfg)


def effective_config(cfg, candidate, depth):
    effective = opt.effective_config(cfg, candidate, depth)
    if effective.get("architecture") == "gated":
        effective["t0_effective"] = opt.effective_config(cfg, "baseline", 1)
    return effective


def windows(task):
    return list(range(1, task["depth"] + 1)) if task["effective"].get("shared_prefix") else [task["depth"]]


def attach_t0(task, output):
    if task["effective"].get("architecture") != "gated":
        return task
    base = {**task, "candidate": "baseline", "depth": 1}
    path = opt.task_path(output, base)
    result = read_json(path / "COMPLETE.json")
    original = result["task"]
    for key in ("stage", "outer", "inner", "seed", "train_ids", "val_ids"):
        if original[key] != task[key]:
            raise ValueError("T0 reference does not match the current training partition")
    if (original["candidate"] != "baseline" or original["depth"] != 1
            or original["effective"] != task["effective"]["t0_effective"]
            or not task_complete(path, original)):
        raise ValueError("T0 reference is incomplete or has a different configuration")
    task["t0_reference"] = {"path": str(path / "model.pt"), "identity": identity(path / "model.pt")}
    return task


def tasks(cfg, folds, stage, gated=False, ranking=None):
    result = []
    for candidate in CANDIDATES:
        is_gated = cfg["candidates"][candidate].get("architecture") == "gated"
        if is_gated != gated:
            continue
        depths = [4] if candidate == "shared64" else [2, 3, 4] if is_gated else cfg["depths"]
        for depth in depths:
            for outer in folds:
                partitions = outer["inner_folds"] if stage == "inner" else [{**outer, "inner": -1}]
                for part in partitions:
                    for seed in cfg["tuning_seeds"] if stage == "inner" else cfg["formal_seeds"]:
                        task = {"stage": stage, "source": "physical_roi", "candidate": candidate,
                            "depth": depth, "outer": outer["fold"], "inner": part["inner"], "seed": seed,
                            "train_ids": part["train_ids"], "val_ids": part["val_ids"],
                            "effective": effective_config(cfg, candidate, depth)}
                        if stage == "outer":
                            chosen = next(r for r in ranking if r["candidate"] == candidate
                                          and r["depth"] == depth and r["outer"] == outer["fold"])
                            task["fixed_epochs"] = chosen["fixed_epochs"]
                        result.append(attach_t0(task, cfg["output_dir"]))
    return result


def canonical_spec(candidate, depth, outer, inner, seed, stage):
    actual = "baseline" if candidate == "gated64" and depth == 1 else candidate
    return {"candidate": actual, "depth": 4 if actual == "shared64" else depth,
            "outer": outer, "inner": inner, "seed": seed, "stage": stage, "source": "physical_roi"}


def select_models(cfg, folds):
    ranking, selections = [], []
    for depth in cfg["depths"]:
        for outer in folds:
            candidates = []
            for candidate in CANDIDATES:
                results = []
                for inner in outer["inner_folds"]:
                    for seed in cfg["tuning_seeds"]:
                        spec = canonical_spec(candidate, depth, outer["fold"], inner["inner"], seed, "inner")
                        path = opt.task_path(cfg["output_dir"], spec)
                        result = read_json(path / "COMPLETE.json")
                        if not task_complete(path, result["task"]):
                            raise ValueError("Selection requires every prespecified inner fit")
                        results.append(result)
                row = {"candidate": candidate, "depth": depth, "outer": outer["fold"], "source": "physical_roi",
                    **{k: float(np.mean([r["windows"][str(depth)]["validation"][k] for r in results]))
                       for k in ("auroc", "logloss", "brier")},
                    "parameters": results[0]["parameters"], "inner_fits": len(results),
                    "fixed_epochs": int(np.ceil(np.median([r["selected_epochs"] for r in results])))}
                candidates.append(row)
            selections.append({"depth": depth, "outer": outer["fold"],
                               "selected": opt.choose_candidate(candidates, cfg["auroc_tolerance"])})
            ranking.extend(candidates)
    payload = {"schema": SCHEMA, "ranking": ranking, "selections": selections,
               "selection_only_uses_inner_validation": True, "holdout_loaded": False}
    unchanged(Path(cfg["output_dir"]) / "selection/models.json", payload)
    _atomic_csv(Path(cfg["output_dir"]) / "selection/ranking.csv", pd.DataFrame(ranking))
    return payload


def score(model, data, depths):
    return {str(d): opt.metric_values(data["labels"].cpu().numpy(), predict(model, data, d)["probability"])
            for d in depths}


def fit_gated(train, val, task, device):
    effective, depth = task["effective"], task["depth"]
    fixed_epochs = task.get("fixed_epochs")
    if (fixed_epochs is not None and val is not None) or (fixed_epochs is None and val is None):
        raise ValueError("Fixed outer fitting must not receive validation; inner fitting requires it")
    reference = task["t0_reference"]
    if identity(reference["path"]) != reference["identity"]:
        raise ValueError("Frozen T0 reference changed")
    base = torch.load(reference["path"], map_location="cpu", weights_only=False)
    if any(base["task"][k] != task[k] for k in ("stage", "outer", "inner", "seed", "train_ids", "val_ids")):
        raise ValueError("T0 reference partition mismatch")
    if base["task"]["effective"] != effective["t0_effective"]:
        raise ValueError("T0 reference configuration mismatch")
    seed = task["seed"] * 10000 + task["outer"] * 100 + (task["inner"] + 1) * 10 + depth
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(effective, depth).to(device)
    model.t0.load_state_dict(base["model_state"])
    prior = base["clinical_prior"]
    data = opt.tensor_split(train, _prior_from_state(train["clinical"], prior), device)
    validation = opt.tensor_split(val, _prior_from_state(val["clinical"], prior), device) if val is not None else None
    eligible = torch.as_tensor(np.flatnonzero(train["masks"][:, 0] > 0), device=device)
    if not len(eligible):
        raise ValueError("No T0-anchored patients for neural fitting")
    pos_weight = torch.tensor(_positive_class_weight(effective, train["labels"][train["masks"][:, 0] > 0]), device=device)
    optimizer = _make_optimizer(model, effective)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(effective["scheduler_t_max"]), eta_min=float(effective["scheduler_eta_min"]))
    generator = torch.Generator(device=device).manual_seed(seed)
    epochs = int(fixed_epochs if fixed_epochs is not None else effective["epochs"])
    if epochs < 1:
        raise ValueError("Invalid gated training duration")
    history, best, selected, stale, state = [], -np.inf, 0, 0, None
    depths = windows(task)
    for epoch in range(1, epochs + 1):
        model.train()
        order = eligible[torch.randperm(len(eligible), device=device, generator=generator)]
        total, gate_total = 0.0, 0.0
        for indices in order.split(int(effective["batch_size"])):
            batch = {k: v[indices] for k, v in data.items()}
            batch = drop_followups(batch, float(effective["followup_dropout_probability"]), generator)
            optimizer.zero_grad(set_to_none=True)
            losses, gates = [], []
            for current in depths:
                logits, residual, gate, _ = model(batch["embs"], batch["masks"], batch["clinical"],
                    batch["days"], batch["prior"], max_depth=current, return_components=True)
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
                losses.append(loss + float(effective["residual_l2_weight"]) * residual.square().mean()
                              + float(effective["gate_l1_weight"]) * gate.mean())
                gates.append(gate.mean().detach())
            # A shared patient contributes one averaged objective, not four independent samples.
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite gated training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(effective["grad_clip"]), error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            gate_total += float(torch.stack(gates).mean()) * len(indices)
        scheduler.step()
        training = score(model, data, depths)
        row = {"epoch": epoch, "train_objective": total / len(eligible), "train_gate": gate_total / len(eligible),
               "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": float(np.mean([r[k] for r in training.values()])) for k in ("auroc", "logloss", "brier")}}
        if validation is not None:
            values = score(model, validation, depths)
            row.update({f"validation_{k}": float(np.mean([r[k] for r in values.values()]))
                        for k in ("auroc", "logloss", "brier")})
            value = -row["validation_logloss"]
            if value > best + float(effective["min_delta"]):
                best, selected, stale = value, epoch, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= int(effective["patience"]):
            break
    if validation is not None:
        if state is None:
            raise ValueError("No valid gated inner checkpoint")
        model.load_state_dict(state)
    else:
        selected = epochs
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(not torch.equal(state["t0." + k], v) for k, v in base["model_state"].items()):
        raise ValueError("The frozen T0 reference was updated")
    return model, {"model_state": state, "clinical_prior": prior, "history": history,
        "selected_epochs": selected, "epochs_trained": len(history),
        "clinical_prior_train_count": len(train["labels"]), "neural_train_count": len(eligible),
        "parameters": sum(p.numel() for p in model.parameters()),
        "t0_reference_frozen": True, "prefixes_per_patient": len(depths)}


def task_complete(path, task):
    path = Path(path)
    try:
        result = read_json(path / "COMPLETE.json")
        if result["schema"] != SCHEMA or result["task"] != task:
            return False
        if any(identity(path / key) != value for key, value in result["artifacts"].items()):
            return False
        if "t0_reference" in task and identity(task["t0_reference"]["path"]) != task["t0_reference"]["identity"]:
            return False
        history = pd.read_csv(path / "history.csv")
        if history.epoch.tolist() != list(range(1, result["epochs_trained"] + 1)):
            return False
        if not 1 <= result["selected_epochs"] <= len(history) or not np.isfinite(history.to_numpy()).all():
            return False
        if task["stage"] == "outer" and (len(history) != task["fixed_epochs"]
                or result["selected_epochs"] != task["fixed_epochs"]
                or any(k.startswith("validation_") for k in history)):
            return False
        frame = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        if set(frame.depth) != set(windows(task)):
            return False
        for depth in windows(task):
            for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
                part = frame[(frame.depth == depth) & (frame.role == role)]
                if len(part) != len(ids) or not part.patient_id.is_unique or set(part.patient_id) != set(ids):
                    return False
                values = opt.metric_values(part.label, part.probability)
                if any(abs(v - result["windows"][str(depth)][role][k]) > 1e-7 for k, v in values.items()):
                    return False
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        return (checkpoint["task"] == task and checkpoint["selected_epochs"] == result["selected_epochs"]
                and all(torch.isfinite(v).all() for v in checkpoint["model_state"].values()))
    except (OSError, ValueError, KeyError, RuntimeError, EOFError):
        return False


def run_task(task, output, embedding_dir, device="cuda:0"):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    path = opt.task_path(output, task)
    if task_complete(path, task):
        return {"path": str(path), "skipped": True}
    if set(task["train_ids"]) & set(task["val_ids"]):
        raise ValueError("Training/validation overlap")
    if task["stage"] not in ("inner", "outer") or (task["stage"] == "outer") != (task.get("fixed_epochs") is not None):
        raise ValueError("Only outer tasks require fixed epochs")
    started = time.perf_counter()
    raw = opt.load_development(str(output), str(embedding_dir))
    train = window_split(raw, task["train_ids"], task["depth"], task["effective"])
    validation = window_split(raw, task["val_ids"], task["depth"], task["effective"]) if task["stage"] == "inner" else None
    if task["effective"].get("architecture") == "gated":
        model, checkpoint = fit_gated(train, validation, task, device)
    else:
        seed = task["seed"] * 10000 + task["outer"] * 100 + (task["inner"] + 1) * 10 + task["depth"]
        model, checkpoint = fit_plain(train, validation, task["effective"], seed, task["depth"], device,
                                      fixed_epochs=task.get("fixed_epochs"))
    history = checkpoint.pop("history")
    checkpoint.update(schema=SCHEMA, task=task, created_utc=now(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        outer_validation_used_for_selection=False, holdout_loaded=False)
    save_tensor(path / "model.pt", checkpoint)
    restored = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(restored["model_state"])
    validation = window_split(raw, task["val_ids"], task["depth"], task["effective"])
    frames, metrics = [], {}
    for depth in windows(task):
        metrics[str(depth)] = {}
        for role, split in (("train", train), ("validation", validation)):
            prior = _prior_from_state(split["clinical"], restored["clinical_prior"])
            prediction = predict(model, opt.tensor_split(split, prior, device), depth)
            values = opt.metric_values(split["labels"], prediction["probability"])
            metrics[str(depth)][role] = values
            frames.append(pd.DataFrame({"patient_id": split["pids"], "label": split["labels"].astype(int),
                "role": role, "depth": depth, **prediction,
                "clinical_prior_probability": (1 / (1 + np.exp(-prior))).astype(np.float64)}))
        metrics[str(depth)]["gap"] = metrics[str(depth)]["train"]["auroc"] - metrics[str(depth)]["validation"]["auroc"]
    _atomic_csv(path / "predictions.csv", pd.concat(frames, ignore_index=True))
    _atomic_csv(path / "history.csv", pd.DataFrame(history))
    result = {"schema": SCHEMA, "task": task, "windows": metrics, "completed_utc": now(),
        "seconds": time.perf_counter() - started,
        **{k: checkpoint[k] for k in ("selected_epochs", "epochs_trained", "parameters", "trainable_parameters",
                                     "clinical_prior_train_count", "neural_train_count")},
        "artifacts": {p: identity(path / p) for p in ("model.pt", "history.csv", "predictions.csv")}}
    write_json(path / "COMPLETE.json", result)
    if not task_complete(path, task):
        raise ValueError("Written gated-pilot artifacts did not validate")
    return {"path": str(path), "seconds": result["seconds"], "skipped": False}

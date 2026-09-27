"""Isolated nested training for temporal feature transforms and thirty seeds."""

from __future__ import annotations

import copy
import fcntl
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _fit_prior, _prior_from_state
from scripts.run_full978_independent_cv import _make_optimizer, _positive_class_weight
from src import first_post_optimization as opt
from src.first_post_pcr_data import identity, now, read_json, repo_path, save_tensor, write_json
from src.single_phase_repeat_retraining import unchanged
from src.single_phase_temporal_models import build_model, fit_feature_statistics, window_split

SCHEMA = "single_phase_temporal_pcr_v3"


def load_config(path, branch):
    path = repo_path(path).resolve()
    master = yaml.safe_load(path.read_text())
    if master.get("schema") != SCHEMA or branch not in master["branches"]:
        raise ValueError("Unsupported temporal optimization configuration")
    cfg = {k: copy.deepcopy(v) for k, v in master.items() if k != "branches"}
    cfg.update(master["branches"][branch])
    cfg.update(branch=branch, config_path=str(path))
    cfg["output_dir"] = str(repo_path(master["output_dir"]) / branch)
    for key in ("baseline_run", "previous_run", "input_config"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if cfg["formal_seeds"] != list(range(42, 72)) or cfg["depths"] != [1, 2, 3, 4]:
        raise ValueError("This study requires thirty prespecified seeds and four windows")
    return cfg


def prepare(cfg):
    source, output = Path(cfg["baseline_run"]), Path(cfg["output_dir"])
    cohort = read_json(source / "cohort.json")
    marker = read_json(source / "REAL_FEATURES_COMPLETE.json")
    if not marker.get("complete") or marker["input_mode"] != cfg["input_mode"]:
        raise ValueError("Wrong or incomplete frozen single-phase features")
    if (len(cohort["split"]["train"]) != cfg["expected_development"]
            or len(cohort["split"]["val"]) != cfg["expected_holdout"]):
        raise ValueError("Original cohort changed")
    unchanged(output / "temporal_contract.json", {
        "schema": SCHEMA, "config": cfg,
        "runtime": {p: identity(repo_path(p)) for p in (
            "src/single_phase_temporal_models.py", "src/single_phase_temporal_training.py",
            "src/single_phase_temporal_reporting.py", "scripts/run_single_phase_temporal_pcr.py",
            "src/single_phase_repeat_retraining.py")},
        "transform_fitting": "current training partition; allowed observed visits only; no validation or holdout",
        "missing_t0_policy": "all images masked; excluded from neural loss; retained in clinical prior and predictions",
        "holdout_role": cohort["test_role"], "new_seeds": list(range(52, 72)),
        "prior_runs_preserved": True,
    })
    return opt.prepare_study(cfg)


def cached_statistics(task, train, output):
    effective = task["effective"]
    mode = effective.get("feature_transform", "identity")
    if mode == "identity":
        return fit_feature_statistics(train, effective)
    limit = min(task["depth"], effective.get("input_depth", task["depth"]))
    width = effective.get("pca_dim", 32) if mode == "pca" else effective["input_dim"]
    name = f"{mode}_{width}_{effective['visit_policy']}_depth{limit}"
    path = Path(output) / "feature_statistics" / name / f"outer_{task['outer']}_inner_{task['inner']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {"train_ids": train["pids"], "input_dim": effective["input_dim"], "mode": mode,
            "width": width, "depth": limit, "policy": effective["visit_policy"],
            "training_visits": int((train["masks"] > 0).sum())}
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["spec"] != spec:
                raise ValueError("Fold transform cache identity changed")
            state = payload["statistics"]
            if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)):
                raise ValueError("Nonfinite fold transform cache")
            return state
        state = fit_feature_statistics(train, effective)
        save_tensor(path, {"spec": spec, "statistics": state})
        return state


def fit_model(train, val, effective, seed, depth, device, statistics=None, fixed_epochs=None):
    if fixed_epochs is not None and val is not None:
        raise ValueError("Outer fixed-epoch fitting must not receive validation data")
    if fixed_epochs is None and val is None:
        raise ValueError("Inner fitting requires validation data")
    if effective.get("prefix_dropout_probability", 0):
        raise ValueError("Random prefix dropout is not part of this protocol")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior_train, prior_val, prior_state = _fit_prior(train, val if val is not None else train, seed)
    has_t0 = train["masks"][:, 0] > 0
    if not has_t0.any():
        raise ValueError("Training fold has no valid T0")
    if statistics is None:
        statistics = fit_feature_statistics(train, effective)
    data = opt.tensor_split(train, prior_train, device)
    validation = opt.tensor_split(val, prior_val, device) if val is not None else None
    eligible = torch.as_tensor(np.flatnonzero(has_t0), device=device)
    model = build_model(effective, depth, statistics).to(device)
    optimizer = _make_optimizer(model, effective)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0)))
    pos_weight = torch.tensor(_positive_class_weight(effective, train["labels"][has_t0]), device=device)
    epochs = int(fixed_epochs if fixed_epochs is not None else effective["epochs"])
    if epochs < 1:
        raise ValueError("Invalid training duration")
    best, selected, stale, state, history = -np.inf, 0, 0, None, []
    order_generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = eligible[torch.randperm(len(eligible), device=device, generator=order_generator)]
        total_loss = 0.0
        for indices in order.split(int(effective["batch_size"])):
            batch = {k: v[indices] for k, v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(batch["embs"], batch["masks"], batch["clinical"],
                                     days=batch["days"], prior_logit=batch["prior"], return_residual=True)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
            loss = loss + float(effective.get("residual_l2_weight", 0)) * residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(effective["grad_clip"]), error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        probability, _ = opt.predict(model, data)
        row = {"epoch": epoch, "train_objective": total_loss / len(eligible),
               "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in opt.metric_values(train["labels"], probability).items()}}
        if validation is not None:
            probability, _ = opt.predict(model, validation)
            metrics = opt.metric_values(val["labels"], probability)
            row.update({f"validation_{k}": v for k, v in metrics.items()})
            score = -metrics["logloss"]
            if score > best + float(effective.get("min_delta", 0)):
                best, selected, stale = score, epoch, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= int(effective["patience"]):
            break
    if validation is not None:
        if state is None:
            raise RuntimeError("No finite inner checkpoint")
        model.load_state_dict(state)
    else:
        selected = epochs
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, {"model_state": state, "clinical_prior": prior_state,
        "selected_epochs": selected, "epochs_trained": len(history), "history": history,
        "clinical_prior_train_count": len(train["labels"]), "neural_train_count": int(has_t0.sum()),
        "feature_statistics_training_visits": statistics["fitted_visits"],
        "parameters": sum(p.numel() for p in model.parameters())}


def run_task(task, output, embedding_dir, device="cuda:0"):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    path = opt.task_path(output, task)
    if opt.task_complete(path, task):
        return {"path": str(path), "skipped": True}
    if set(task["train_ids"]) & set(task["val_ids"]):
        raise ValueError("Task train/validation overlap")
    if task["stage"] not in ("inner", "outer") or (task["stage"] == "outer") != (task.get("fixed_epochs") is not None):
        raise ValueError("Only outer fits require fixed epochs")
    if task["effective"]["epoch_selection"] != "logloss":
        raise ValueError("This study selects duration using inner log loss")
    started = time.perf_counter()
    raw = opt.load_development(str(output), str(embedding_dir))
    train = window_split(raw, task["train_ids"], task["depth"], task["effective"])
    val_for_selection = (window_split(raw, task["val_ids"], task["depth"], task["effective"])
                         if task["stage"] == "inner" else None)
    statistics = cached_statistics(task, train, output)
    seed = task["seed"] * 10000 + task["outer"] * 100 + (task["inner"] + 1) * 10 + task["depth"]
    model, checkpoint = fit_model(train, val_for_selection, task["effective"], seed, task["depth"], device,
                                  statistics=statistics, fixed_epochs=task.get("fixed_epochs"))
    checkpoint.update(schema=SCHEMA, task=task, created_utc=now(),
                      outer_validation_used_for_selection=False, holdout_labels_or_embeddings_loaded=False)
    history = checkpoint.pop("history")
    # Freeze both classifier and training-only transform before scoring outer patients.
    save_tensor(path / "model.pt", checkpoint)
    restored = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(restored["model_state"])
    val = window_split(raw, task["val_ids"], task["depth"], task["effective"])
    predictions, summaries = [], {}
    for role, split in (("train", train), ("validation", val)):
        logits = _prior_from_state(split["clinical"], restored["clinical_prior"])
        probability, residual = opt.predict(model, opt.tensor_split(split, logits, device))
        summaries[role] = opt.metric_values(split["labels"], probability)
        predictions.append(pd.DataFrame({"patient_id": split["pids"], "role": role,
            "label": split["labels"].astype(int), "probability": probability.astype(np.float64),
            "residual_logit": residual.astype(np.float64),
            "clinical_prior_probability": (1 / (1 + np.exp(-logits))).astype(np.float64)}))
    _atomic_csv(path / "predictions.csv", pd.concat(predictions, ignore_index=True))
    _atomic_csv(path / "history.csv", pd.DataFrame(history))
    summary = {"schema": SCHEMA, "task": task, **summaries,
        **{k: checkpoint[k] for k in ("selected_epochs", "epochs_trained", "parameters",
            "clinical_prior_train_count", "neural_train_count", "feature_statistics_training_visits")},
        "seconds": time.perf_counter() - started,
        "auroc_gap": summaries["train"]["auroc"] - summaries["validation"]["auroc"],
        "completed_utc": now(), "checkpoint_reloaded_before_prediction": True}
    write_json(path / "COMPLETE.json", summary)
    if not opt.task_complete(path, task):
        raise RuntimeError("Written task artifacts did not replay")
    return {"path": str(path), "seconds": summary["seconds"], "skipped": False}

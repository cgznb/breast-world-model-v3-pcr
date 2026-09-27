"""Nested, real-image-only TDN fitting with epoch-boundary recovery."""

from __future__ import annotations

import copy
import functools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from scripts.run_full978_anti_overfit import _atomic_csv, _fit_prior, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _make_optimizer, _positive_class_weight, _sample_prefix_depths
from src.first_post_optimization import predict, select_patients, tensor_split
from src.first_post_pcr_data import identity, now, read_json, save_tensor, write_json
from src.single_phase_fmbcmri_data import DEPTHS, SCHEMA, unchanged_json
from src.tdn import TDN
from src.temporal import mask_torch_temporal_prefix

METRICS = ("auroc", "auprc", "logloss", "brier")


def deterministic_runtime():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)


def metrics(labels, probability):
    labels = np.asarray(labels)
    probability = np.asarray(probability, dtype=np.float64)
    if (len(labels) != len(probability) or not len(labels) or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Invalid saved probabilities")
    return {"auroc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None,
            "auprc": float(average_precision_score(labels, probability)) if (labels == 1).any() else None,
            "logloss": float(log_loss(labels, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(labels, probability))}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state(order, device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
            "order": order.get_state()}


def restore_rng(state, order, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    order.set_state(state["order"])


def fit(train, val, effective, seed, depth, device, directory, *, fixed_epochs=None,
        contract=None, pause_after=None, pause_marker=None):
    if (fixed_epochs is None) != (val is not None):
        raise ValueError("Inner fitting needs validation; fixed-epoch refits cannot receive validation")
    if train["dim"] != effective["input_dim"]:
        raise ValueError("TDN input dimension differs from the features")
    deterministic_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    directory = Path(directory)
    recovery_path = directory / "recovery.pt"
    binding = {"effective": effective, "seed": seed, "depth": depth, "fixed_epochs": fixed_epochs,
               "train_ids": train["pids"], "validation_ids": val["pids"] if val is not None else [],
               "task": contract}
    recovered = torch.load(recovery_path, map_location="cpu", weights_only=False) if recovery_path.exists() else None
    if recovered is not None and recovered["binding"] != binding:
        raise ValueError("Recovery configuration or patient partition changed")
    prior_state = recovered["clinical_prior"] if recovered else _fit_prior(train, train, seed)[2]
    # Use the portable prior formula during fitting too, so reloading cannot change it.
    data = tensor_split(train, _prior_from_state(train["clinical"], prior_state), device)
    validation = (tensor_split(val, _prior_from_state(val["clinical"], prior_state), device)
                  if val is not None else None)
    eligible = torch.nonzero(data["masks"][:, 0] > 0, as_tuple=True)[0]
    if not len(eligible):
        raise ValueError("No training patient has T0")
    model = TDN({"downstream": effective}).to(device)
    initial = cpu_state(model)
    optimizer = _make_optimizer(model, effective)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0)))
    order_generator = torch.Generator(device=device).manual_seed(seed)
    pos_weight = torch.tensor(_positive_class_weight(effective, train["labels"][train["masks"][:, 0] > 0]),
                              dtype=torch.float32, device=device)
    history, best_state, selected, best_loss, patience_loss, stale = [], None, 0, float("inf"), float("inf"), 0
    gradient_audit = None
    if recovered:
        model.load_state_dict(recovered["model_state"], strict=True)
        optimizer.load_state_dict(recovered["optimizer"])
        scheduler.load_state_dict(recovered["scheduler"])
        history, best_state = recovered["history"], recovered["best_state"]
        selected, best_loss = recovered["selected_epochs"], recovered["best_loss"]
        patience_loss, stale = recovered["patience_loss"], recovered["stale"]
        gradient_audit = recovered["gradient_audit"]
        restore_rng(recovered["rng"], order_generator, device)
    epochs = int(fixed_epochs if fixed_epochs is not None else effective["epochs"])
    stopped = validation is not None and stale >= int(effective["patience"])
    for epoch in range(len(history) + 1, epochs + 1) if not stopped else ():
        model.train()
        order = eligible[torch.randperm(len(eligible), generator=order_generator, device=device)]
        total_loss = 0.0
        for indices in order.split(int(effective["batch_size"])):
            batch = {k: v[indices] for k, v in data.items()}
            if effective.get("prefix_dropout_probability", 0) > 0:
                depths = _sample_prefix_depths(len(indices), depth, effective["prefix_dropout_probability"], device)
                batch["embs"], batch["masks"], batch["days"] = mask_torch_temporal_prefix(
                    batch["embs"], batch["masks"], batch["days"], depths)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(batch["embs"], batch["masks"], batch["clinical"],
                                     days=batch["days"], prior_logit=batch["prior"], return_residual=True)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
            loss = loss + float(effective.get("residual_l2_weight", 0)) * residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TDN objective")
            loss.backward()
            if gradient_audit is None:
                gradient_audit = {name: bool(p.grad is not None and torch.isfinite(p.grad).all())
                                  for name, p in model.named_parameters() if p.requires_grad}
                if not all(gradient_audit.values()) or model.proj.net[0].weight.grad.abs().sum() == 0:
                    raise FloatingPointError("A trainable TDN parameter lacks a finite gradient")
            if float(effective.get("grad_clip", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(effective["grad_clip"]), error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        probability, _ = predict(model, data)
        row = {"epoch": epoch, "train_objective": total_loss / len(eligible),
               "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in metrics(train["labels"], probability).items()}}
        if validation is not None:
            probability, _ = predict(model, validation)
            values = metrics(val["labels"], probability)
            row.update({f"validation_{k}": v for k, v in values.items()})
            loss_value = values["logloss"]
            if loss_value < best_loss:
                best_loss, selected, best_state = loss_value, epoch, cpu_state(model)
            if loss_value < patience_loss - float(effective["min_delta"]):
                patience_loss, stale = loss_value, 0
            else:
                stale += 1
        history.append(row)
        recovery = {"schema": SCHEMA, "binding": binding, "model_state": cpu_state(model),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "clinical_prior": prior_state, "history": history, "best_state": best_state,
                    "best_loss": best_loss, "patience_loss": patience_loss, "selected_epochs": selected,
                    "stale": stale, "gradient_audit": gradient_audit, "rng": rng_state(order_generator, device)}
        save_tensor(recovery_path, recovery)
        write_json(directory / "progress.json", {"epoch": epoch, "maximum_epochs": epochs,
                   "selected_epochs": selected if validation is not None else epoch, "stale": stale,
                   "updated_utc": now(), "inner_selection": validation is not None})
        if (pause_after is not None and epoch >= pause_after) or (pause_marker and Path(pause_marker).exists()):
            raise InterruptedError("TDN paused at a recoverable epoch boundary")
        if validation is not None and stale >= int(effective["patience"]):
            break
    if validation is not None:
        if best_state is None:
            raise RuntimeError("Inner training did not produce a finite selected checkpoint")
        model.load_state_dict(best_state, strict=True)
    else:
        selected, best_state = epochs, cpu_state(model)
    return model, {"model_state": best_state, "clinical_prior": prior_state, "selected_epochs": selected,
                   "epochs_trained": len(history), "history": history, "effective": effective,
                   "parameters": sum(p.numel() for p in model.parameters()), "gradient_audit": gradient_audit,
                   "updated_parameter_tensors": sum(not torch.equal(initial[k], v) for k, v in best_state.items()),
                   "neural_loss_patients": len(eligible), "clinical_prior_patients": len(train["pids"]),
                   "initialized_from_pretrained_tdn": False, "resumed": recovered is not None}


@functools.lru_cache(maxsize=2)
def load_development(output):
    root = Path(output)
    folds = read_json(root / "nested_folds.json")
    ids = sorted(folds[0]["train_ids"] + folds[0]["val_ids"])
    split = _load_split(root / "embeddings/real", root / "development_metadata.csv", ids)
    if split["dim"] != 768:
        raise ValueError("Development features are not FM-BCMRI CLS768")
    return split


def task_path(output, task):
    root = Path(output) / "fits" / task["stage"] / task["window"] / f"outer_{task['outer']}"
    if task["stage"] == "inner":
        root = root / f"inner_{task['inner']}"
    return root / f"seed_{task['seed']}"


def task_complete(path, task):
    if not (Path(path) / "COMPLETE.json").exists():
        return False
    summary = read_json(Path(path) / "COMPLETE.json")
    if summary["task"] != task:
        raise ValueError("Completed task configuration changed")
    for name, previous in summary["artifacts"].items():
        if identity(Path(path) / name) != previous:
            raise ValueError("A completed TDN artifact changed")
    history = pd.read_csv(Path(path) / "history.csv")
    if history.epoch.tolist() != list(range(1, summary["epochs_trained"] + 1)):
        raise ValueError("Incomplete learning curve")
    if task["stage"] == "outer" and (len(history) != task["fixed_epochs"]
            or any(c.startswith("validation_") for c in history.columns)):
        raise ValueError("Outer refit used validation selection or wrong epochs")
    if task["stage"] == "inner" and int(history.loc[history.validation_logloss.idxmin(), "epoch"]) != summary["selected_epochs"]:
        raise ValueError("Inner selected checkpoint is not minimum log loss")
    frame = pd.read_csv(Path(path) / "predictions.csv", dtype={"patient_id": str})
    for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
        rows = frame[frame.role == role]
        if len(rows) != len(ids) or set(rows.patient_id) != set(ids) or not rows.patient_id.is_unique:
            raise ValueError("Saved prediction patient set differs")
        actual = metrics(rows.label, rows.probability)
        for key, value in actual.items():
            if value is not None and abs(value - summary[role][key]) > 1e-7:
                raise ValueError("Metrics cannot be reproduced from predictions")
    return True


def run_task(task, output, device="cuda:0"):
    deterministic_runtime()
    directory = task_path(output, task)
    if task_complete(directory, task):
        return {"path": str(directory), "skipped": True}
    started = time.monotonic()
    split = load_development(str(output))
    depth = DEPTHS[task["window"]]
    if set(task["train_ids"]) & set(task["val_ids"]):
        raise ValueError("Training and validation patients overlap")
    train = select_patients(split, task["train_ids"], depth)
    val = select_patients(split, task["val_ids"], depth) if task["stage"] == "inner" else None
    seed = task["seed"] * 10000 + task["outer"] * 100 + (task["inner"] + 1) * 10 + depth
    model, checkpoint = fit(train, val, task["effective"], seed, depth, device, directory,
                            fixed_epochs=task.get("fixed_epochs"), contract=task,
                            pause_marker=Path(output) / "PAUSE_REQUESTED")
    history = checkpoint.pop("history")
    checkpoint.update(schema=SCHEMA, task=task, created_utc=now(),
                      outer_validation_used_for_selection=False, holdout_used_for_fitting=False)
    # Freeze the model before reading outer-validation labels or scoring them.
    save_tensor(directory / "model.pt", checkpoint)
    restored = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    replay = TDN({"downstream": task["effective"]}).to(device)
    replay.load_state_dict(restored["model_state"], strict=True)
    val = select_patients(split, task["val_ids"], depth)
    frames, summaries, replay_error = [], {}, 0.0
    for role, data in (("train", train), ("validation", val)):
        prior = _prior_from_state(data["clinical"], restored["clinical_prior"])
        tensors = tensor_split(data, prior, device)
        original_probability, _ = predict(model, tensors)
        probability, residual = predict(replay, tensors)
        replay_error = max(replay_error, float(np.max(np.abs(original_probability - probability))))
        summaries[role] = metrics(data["labels"], probability)
        frames.append(pd.DataFrame({"patient_id": data["pids"], "role": role,
                      "label": data["labels"].astype(int), "probability": probability.astype(np.float64),
                      "residual_logit": residual.astype(np.float64), "complete_window": data["masks"].sum(1) == depth,
                      "available_prefix": data["masks"].sum(1).astype(int)}))
    if replay_error > 1e-6:
        raise ValueError("Checkpoint prediction replay exceeds 1e-6")
    _atomic_csv(directory / "predictions.csv", pd.concat(frames, ignore_index=True))
    _atomic_csv(directory / "history.csv", pd.DataFrame(history))
    summary = {"schema": SCHEMA, "task": task, **summaries,
               **{k: checkpoint[k] for k in ("selected_epochs", "epochs_trained", "parameters", "gradient_audit",
                                             "neural_loss_patients", "clinical_prior_patients", "updated_parameter_tensors")},
               "seconds": time.monotonic() - started, "checkpoint_replay_max_error": replay_error,
               "artifacts": {p: identity(directory / p) for p in ("model.pt", "history.csv", "predictions.csv")},
               "completed_utc": now()}
    write_json(directory / "COMPLETE.json", summary)
    task_complete(directory, task)
    return {"path": str(directory), "skipped": False, "seconds": summary["seconds"]}


def inner_tasks(cfg, folds, recipes):
    return [{"stage": "inner", "window": window, "outer": outer["fold"], "inner": inner["inner"],
             "seed": seed, "train_ids": inner["train_ids"], "val_ids": inner["val_ids"], "effective": recipes[window]}
            for window in DEPTHS for outer in folds for inner in outer["inner_folds"] for seed in cfg["tuning_seeds"]]


def select_epochs(cfg, folds, recipes):
    groups = {}
    for task in inner_tasks(cfg, folds, recipes):
        directory = task_path(cfg["output_dir"], task)
        if not task_complete(directory, task):
            raise RuntimeError("All six inner fits are required before epoch selection")
        groups.setdefault((task["window"], task["outer"]), []).append(read_json(directory / "COMPLETE.json")["selected_epochs"])
    selection = [{"window": window, "outer": fold, "inner_selected_epochs": epochs,
                  "fixed_epochs": int(np.ceil(np.median(epochs)))} for (window, fold), epochs in groups.items()]
    if len(selection) != 20 or any(len(row["inner_selected_epochs"]) != 6 for row in selection):
        raise ValueError("Wrong nested epoch selection count")
    unchanged_json(Path(cfg["output_dir"]) / "selected_epochs.json", selection)
    return selection


def formal_tasks(cfg, folds, recipes, selection):
    lookup = {(r["window"], r["outer"]): r["fixed_epochs"] for r in selection}
    return [{"stage": "outer", "window": window, "outer": outer["fold"], "inner": -1, "seed": seed,
             "train_ids": outer["train_ids"], "val_ids": outer["val_ids"], "effective": recipes[window],
             "fixed_epochs": lookup[window, outer["fold"]]}
            for window in DEPTHS for outer in folds for seed in cfg["seeds"]]


def freeze_models(cfg, tasks):
    refs = []
    for task in tasks:
        directory = task_path(cfg["output_dir"], task)
        if task["stage"] != "outer" or not task_complete(directory, task):
            raise ValueError("Formal refit is incomplete")
        refs.append({"window": task["window"], "outer": task["outer"], "seed": task["seed"],
                     "path": str(directory / "model.pt"), "identity": identity(directory / "model.pt")})
    if len(refs) != 200 or len({(r["window"], r["outer"], r["seed"]) for r in refs}) != 200:
        raise ValueError("Exactly 200 distinct formal classifiers are required per branch")
    unchanged_json(Path(cfg["output_dir"]) / "frozen_models.json", refs)
    return refs

"""Nested selection, fixed refits, and exact epoch-boundary recovery."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from scripts.run_full978_anti_overfit import _atomic_csv
from src.first_post_pcr_data import identity, read_json, save_tensor, write_json
from src.single_phase_fmbcmri_data import load_encoder
from src.single_phase_fmbcmri_training import deterministic_runtime, metrics, restore_rng, rng_state
from src.single_phase_response_data import check_pause, load_data, subset
from src.single_phase_response_models import (
    ResponseModel, apply_clinical_transform, fit_clinical_transform, patient_prefix_loss,
)


def fit_clinical(data, seed):
    transform = fit_clinical_transform(data["clinical"])
    values = apply_clinical_transform(data["clinical"], transform)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    model.fit(values, data["labels"])
    if int(model.n_iter_[0]) >= 2000:
        raise ValueError("Clinical baseline failed to converge")
    return dict(transform=transform, coefficient=model.coef_[0].astype(np.float32),
                intercept=float(model.intercept_[0]), fitted_ids=data["pids"],
                objective="ordinary_unweighted_logistic_BCE_L2_C1")


def clinical_predict(data, state):
    c = apply_clinical_transform(data["clinical"], state["transform"])
    logits = c @ state["coefficient"] + state["intercept"]
    return np.repeat((1 / (1 + np.exp(-logits)))[:, None], 4, axis=1)


def tensors(data, clinical_state, device):
    return {k: torch.from_numpy(v).to(device) for k, v in dict(
        features=data["embs"], mask=data["masks"], days=data["days"], labels=data["labels"],
        clinical=apply_clinical_transform(data["clinical"], clinical_state["transform"])).items()}


def load_tokens(data, indices, device):
    values = []
    for i in indices:
        active = True
        for t in range(4):
            active = active and bool(data["masks"][i, t])
            if active:
                path = data["token_paths"][i][t]
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                token = payload["tokens"]
                if token.shape != (217, 768) or not torch.isfinite(token).all():
                    raise ValueError("Invalid prefix tokens")
                values.append(token)
    return torch.stack(values).to(device) if values else torch.empty((0, 217, 768), device=device)


def forward_batch(model, data, tensor_data, indices, device):
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    batch = {k: v[index] for k, v in tensor_data.items()}
    tokens = load_tokens(data, indices, device) if model.adapter is not None else None
    return model(batch["features"], batch["mask"], batch["clinical"], batch["days"], tokens), batch


@torch.no_grad()
def predict(model, data, clinical_state, device, batch_size=64):
    model.eval()
    tensor_data = tensors(data, clinical_state, device)
    results = []
    for start in range(0, len(data["pids"]), batch_size):
        indices = list(range(start, min(start + batch_size, len(data["pids"]))))
        logits, _ = forward_batch(model, data, tensor_data, indices, device)
        results.append(logits.float().cpu())
    logits = torch.cat(results)
    loss = patient_prefix_loss(logits, torch.from_numpy(data["labels"]), torch.from_numpy(data["masks"]))
    return torch.sigmoid(logits).numpy(), float(loss)


def fit(train, validation, cfg, arm, seed, directory, *, fixed_epochs=None, encoder=None,
        device="cpu", pause_after=None, binding_extra=None):
    if (validation is None) != (fixed_epochs is not None):
        raise ValueError("Inner fitting requires validation; outer fixed refits must have none")
    if arm.get("exclude_conflicts_from_fit"):
        train = subset(train, [p for p, bad in zip(train["pids"], train["conflicted"]) if not bad])
    if validation is not None and set(train["pids"]) & set(validation["pids"]):
        raise ValueError("Training/validation patients overlap")
    deterministic_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = dict(cfg["training"])
    if arm.get("lora"):
        settings.update(epochs=cfg["adaptation"]["epochs"], batch_size=cfg["adaptation"]["batch_size"],
                        lr=cfg["adaptation"]["head_lr"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = dict(arm=arm, settings=settings, adaptation=cfg["adaptation"] if arm.get("lora") else None,
                   seed=seed, train_ids=train["pids"], validation_ids=validation["pids"] if validation else [],
                   fixed_epochs=fixed_epochs, source=binding_extra)
    recovery_path = directory / "recovery.pt"
    recovered = torch.load(recovery_path, map_location="cpu", weights_only=False) if recovery_path.exists() else None
    if recovered and recovered["binding"] != binding:
        raise ValueError("Resume config, data or patient partition differs")
    clinical = recovered["clinical"] if recovered else fit_clinical(train, seed)
    prevalence = float(np.clip(train["labels"].mean(), 1e-5, 1 - 1e-5))
    model = ResponseModel(arm, settings, prevalence, encoder, cfg["adaptation"]).to(device)
    with torch.no_grad():
        model.head.fallback_clinical_coef.copy_(torch.from_numpy(clinical["coefficient"]).to(device))
        model.head.fallback_clinical_intercept.fill_(clinical["intercept"])
    groups = [dict(params=model.head.parameters(), lr=settings["lr"])]
    if model.adapter is not None:
        groups.append(dict(params=[p for p in model.adapter.parameters() if p.requires_grad], lr=cfg["adaptation"]["lr"]))
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    order = torch.Generator().manual_seed(seed)
    train_tensors = tensors(train, clinical, device)
    eligible = np.flatnonzero(train["masks"][:, 0] > 0)
    if not len(eligible):
        raise ValueError("No training patient has T0")
    history, best, best_loss, patience_loss, stale, selected, gradient_audit = [], None, float("inf"), float("inf"), 0, 0, None
    if recovered:
        model.load_portable_state(recovered["state"])
        optimizer.load_state_dict(recovered["optimizer"])
        scheduler.load_state_dict(recovered["scheduler"])
        history, best = recovered["history"], recovered["best"]
        best_loss, patience_loss, stale, selected = (recovered[k] for k in ("best_loss", "patience_loss", "stale", "selected"))
        gradient_audit = recovered["gradient_audit"]
        restore_rng(recovered["rng"], order, device)
    count = int(fixed_epochs if fixed_epochs is not None else settings["epochs"])
    stopped = validation is not None and stale >= settings["patience"]
    for epoch in range(len(history) + 1, count + 1) if not stopped else ():
        model.train()
        shuffled = eligible[torch.randperm(len(eligible), generator=order).numpy()]
        sum_loss, examples = 0.0, 0
        for start in range(0, len(shuffled), settings["batch_size"]):
            indices = shuffled[start:start + settings["batch_size"]].tolist()
            optimizer.zero_grad(set_to_none=True)
            logits, batch = forward_batch(model, train, train_tensors, indices, device)
            loss = patient_prefix_loss(logits, batch["labels"], batch["mask"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ordinary BCE")
            loss.backward()
            params = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Nonfinite response/adapter gradient")
            if gradient_audit is None:
                gradient_audit = dict(trainable_parameters=sum(p.numel() for p in params),
                                     trainable_tensors=sum(p.requires_grad for p in model.parameters()),
                                     finite_gradients=True, head_gradient_norm=sum(float(p.grad.norm()) for p in model.head.parameters() if p.grad is not None),
                                     adapter_parameters=sum(p.numel() for p in model.adapter.parameters() if p.requires_grad) if model.adapter else 0,
                                     frozen_base_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad))
                if gradient_audit["head_gradient_norm"] == 0 or not gradient_audit["frozen_base_gradients_absent"]:
                    raise ValueError("Gradient/freeze audit failed")
            torch.nn.utils.clip_grad_norm_(params, settings["grad_clip"])
            optimizer.step()
            sum_loss += float(loss.detach()) * len(indices)
            examples += len(indices)
        scheduler.step()
        _, train_loss = predict(model, train, clinical, device, settings["batch_size"])
        val_loss = predict(model, validation, clinical, device, settings["batch_size"])[1] if validation else None
        if validation is not None:
            if val_loss < best_loss:
                best_loss, selected, best = val_loss, epoch, model.portable_state()
            if val_loss < patience_loss - settings["min_delta"]:
                patience_loss, stale = val_loss, 0
            else:
                stale += 1
        else:
            selected, best = epoch, model.portable_state()
        history.append(dict(epoch=epoch, train_objective=sum_loss / examples, train_logloss=train_loss,
                            validation_logloss=val_loss, head_lr=optimizer.param_groups[0]["lr"]))
        recovered = dict(binding=binding, state=model.portable_state(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), clinical=clinical, history=history, best=best,
                         best_loss=best_loss, patience_loss=patience_loss, stale=stale, selected=selected,
                         gradient_audit=gradient_audit, rng=rng_state(order, device))
        save_tensor(recovery_path, recovered)
        _atomic_csv(directory / "learning_curve.csv", pd.DataFrame(history))
        write_json(directory / "progress.json", dict(epoch=epoch, maximum_epochs=count, selected_epoch=selected,
                   train_logloss=train_loss, validation_logloss=val_loss, stale=stale))
        if pause_after == epoch:
            raise InterruptedError("Requested deterministic recovery test")
        check_pause(cfg)
        if validation is not None and stale >= settings["patience"]:
            break
    model.load_portable_state(best)
    model.eval()
    result = dict(binding=binding, state=best, clinical=clinical, prevalence=prevalence, history=history,
                  selected_epochs=selected, gradient_audit=gradient_audit,
                  parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  holdout_used_for_selection=False)
    if model.adapter is not None:
        state = model.adapter.adapter_state()
        result["adapter_nonzero_update"] = any(float(v.abs().sum()) > 0 for k, v in state.items() if k.endswith(".b"))
        if not result["adapter_nonzero_update"]:
            raise ValueError("The last-block adapter did not learn")
    return model, result


def prediction_frame(data, probability, role):
    return pd.DataFrame([dict(patient_id=p, label=int(data["labels"][i]), prefix=t + 1,
                              probability=float(probability[i, t]), complete=bool(data["masks"][i, t]),
                              available_prefix=int(data["masks"][i].sum()), role=role)
                         for i, p in enumerate(data["pids"]) for t in range(4)])


def task_path(cfg, task):
    root = Path(cfg["output_dir"]) / "training" / task["stage"] / task["arm"] / f"fold_{task['fold']}"
    if task["stage"] == "inner":
        root /= f"inner_{task['inner_fold']}"
    return root / f"seed_{task['seed']}"


def task_complete(cfg, task):
    root = task_path(cfg, task)
    if not (root / "COMPLETE.json").exists():
        return False
    done = read_json(root / "COMPLETE.json")
    if done["task"] != task or done["checkpoint"] != identity(root / "model.pt") or done["predictions"] != identity(root / "predictions.csv"):
        raise ValueError("A completed task changed")
    return True


def inner_tasks(cfg):
    nested = read_json(Path(cfg["output_dir"]) / "nested_folds.json")
    tasks = []
    for outer in nested:
        for name, arm in cfg["arms"].items():
            if arm["model"] == "clinical":
                continue
            for inner in outer["inner_folds"]:
                for seed in cfg["tuning_seeds"]:
                    tasks.append(dict(stage="inner", arm=name, fold=outer["fold"], inner_fold=inner["inner"], seed=seed,
                                      train_ids=inner["train_ids"], val_ids=inner["val_ids"], fixed_epochs=None))
    return tasks


def formal_tasks(cfg, arm_names=None):
    folds = read_json(Path(cfg["output_dir"]) / "folds.json")
    inner = inner_tasks(cfg)
    tasks, selection = [], []
    for outer in folds:
        for name, arm in cfg["arms"].items():
            if arm_names is not None and name not in arm_names:
                continue
            selected = []
            if arm["model"] != "clinical":
                group = [t for t in inner if t["fold"] == outer["fold"] and t["arm"] == name]
                for t in group:
                    if not task_complete(cfg, t):
                        raise ValueError("Epoch selection requires all six inner fits")
                    selected.append(read_json(task_path(cfg, t) / "summary.json")["selected_epochs"])
                epochs = math.ceil(float(np.median(selected)))
            else:
                epochs = 0
            selection.append(dict(arm=name, fold=outer["fold"], inner_epochs=selected, selected_epochs=epochs))
            for seed in cfg["seeds"] if epochs else [cfg["seeds"][0]]:
                tasks.append(dict(stage="formal", arm=name, fold=outer["fold"], seed=seed,
                                  train_ids=outer["train_ids"], val_ids=outer["val_ids"], fixed_epochs=epochs))
    path = Path(cfg["output_dir"]) / "epoch_selection.json"
    existing = read_json(path) if path.exists() else []
    updated = {(r["arm"], r["fold"]): r for r in existing + selection}
    write_json(path, list(updated.values()))
    return tasks


def run_task(cfg, task):
    deterministic_runtime()
    if task_complete(cfg, task):
        return
    check_pause(cfg)
    study = read_json(Path(cfg["output_dir"]) / "cohort.json")
    arm = cfg["arms"][task["arm"]]
    data = load_data(cfg, study, arm["roi"], "train")
    train, val = subset(data, task["train_ids"]), subset(data, task["val_ids"])
    if set(task["train_ids"] + task["val_ids"]) & set(study["split"]["val"]):
        raise ValueError("The fixed holdout entered model fitting")
    root = task_path(cfg, task)
    started = time.monotonic()
    if arm["model"] == "clinical":
        state = fit_clinical(train, task["seed"])
        result = dict(clinical=state, selected_epochs=0, task=task, arm=arm)
        probabilities = clinical_predict(val, state)
    else:
        device = "cuda:0" if arm.get("lora") else "cpu"
        if arm.get("lora"):
            torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"], 0)
        encoder = load_encoder(cfg, "cpu") if arm.get("lora") else None
        model, result = fit(train, val if task["stage"] == "inner" else None, cfg, arm, task["seed"], root,
                            fixed_epochs=task["fixed_epochs"], encoder=encoder, device=device,
                            binding_extra=dict(task=task, contract=identity(Path(cfg["output_dir"]) / "contract.json"),
                                               runtime=read_json(Path(cfg["study_dir"]) / "runtime_sources.json")))
        batch = cfg["adaptation"]["batch_size"] if arm.get("lora") else cfg["training"]["batch_size"]
        probabilities = predict(model, val, result["clinical"], device, batch)[0]
        reloaded = ResponseModel(arm, cfg["training"], result["prevalence"], encoder, cfg["adaptation"]).to(device)
        reloaded.load_portable_state(result["state"])
        replay = predict(reloaded, val, result["clinical"], device, batch)[0]
        result["reload_max_error"] = float(np.max(np.abs(probabilities - replay)))
        if result["reload_max_error"] > 1e-6:
            raise ValueError("Checkpoint reconstruction exceeds the prediction tolerance")
    result.update(task=task, arm=arm)
    save_tensor(root / "model.pt", result)
    _atomic_csv(root / "predictions.csv", prediction_frame(val, probabilities, "validation" if task["stage"] == "inner" else "oof"))
    write_json(root / "summary.json", {k: result[k] for k in ("selected_epochs", "gradient_audit", "parameters", "reload_max_error", "adapter_nonzero_update") if k in result})
    write_json(root / "COMPLETE.json", dict(task=task, checkpoint=identity(root / "model.pt"),
                                           predictions=identity(root / "predictions.csv"), seconds=time.monotonic() - started))

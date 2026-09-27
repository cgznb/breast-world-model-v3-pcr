"""Training-partition-only preprocessing, staged refits and recoverable optimization."""

from __future__ import annotations

import copy
import fcntl
import functools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from scripts.run_full978_anti_overfit import _atomic_csv
from src.first_post_pcr_data import identity, read_json, save_tensor, write_json
from src.single_phase_anchor_models import (
    AnchoredResponse, ClinicalAnchor, learning_mask, objective, probe_inputs,
)
from src.single_phase_fmbcmri_training import cpu_state, deterministic_runtime, restore_rng, rng_state
from src.single_phase_response_data import subset
from src.single_phase_response_training import fit_clinical, prediction_frame
from src.single_phase_temporal_models import FrozenFeatureTransform, fit_feature_statistics

STATISTICAL = ("clinical", "current", "pair")


def check_pause(cfg):
    if (Path(cfg["study_dir"]) / "PAUSE_REQUESTED").exists():
        raise InterruptedError("Requested pause at a recoverable boundary")


@functools.lru_cache(maxsize=6)
def read_data(path, size, modified):
    if identity(path) != dict(size_bytes=size, mtime_ns=modified):
        raise ValueError("Prepared features changed")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_data(cfg, roi):
    path = Path(cfg["output_dir"]) / "data" / f"train_{roi}.pt"
    stamp = identity(path)
    return read_data(str(path), stamp["size_bytes"], stamp["mtime_ns"])


def task_path(cfg, task):
    path = Path(cfg["output_dir"]) / "training" / task["stage"] / task["arm"] / f"fold_{task['fold']}"
    if task["stage"] == "inner":
        path /= f"inner_{task['inner_fold']}"
    return path / f"seed_{task['seed']}"


def task_complete(cfg, task):
    root = task_path(cfg, task)
    if not (root / "COMPLETE.json").exists():
        return False
    done = read_json(root / "COMPLETE.json")
    if (done["task"] != task or done["model"] != identity(root / "model.pt")
            or done["predictions"] != identity(root / "predictions.csv")):
        raise ValueError("A completed anchored task changed")
    return True


def preprocessing(cfg, task, data):
    arm = cfg["arms"][task["arm"]]
    basis = "t0" if arm["kind"] == "t0" else "all"
    root = Path(cfg["output_dir"]) / "preprocessing" / task["stage"] / f"fold_{task['fold']}"
    root /= f"inner_{task['inner_fold']}" if task["stage"] == "inner" else "outer"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{arm['roi']}_{basis}.pt"
    source = Path(cfg["output_dir"]) / "data" / f"train_{arm['roi']}.pt"
    binding = dict(train_ids=data["pids"], source=identity(source), basis=basis, settings=cfg["training"])
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            result = torch.load(path, map_location="cpu", weights_only=False)
            if result["binding"] != binding:
                raise ValueError("Preprocessing partition or source changed")
            return result
        inputs = dict(data, masks=data["masks"].copy())
        if basis == "t0":
            inputs["masks"][:, 1:] = 0
        result = dict(binding=binding, clinical=fit_clinical(data, 42),
                      statistics=fit_feature_statistics(inputs, cfg["training"]))
        save_tensor(path, result)
        return result


def as_tensors(data):
    return {k: torch.from_numpy(np.asarray(data[k], np.float32))
            for k in ("embs", "masks", "clinical", "days", "labels") if k in data}


@torch.no_grad()
def predict_model(model, data):
    model.eval()
    values = as_tensors(data)
    logits, residual = model(values["embs"], values["masks"], values["clinical"], values["days"])
    loss = objective(logits, residual, values["labels"], values["masks"], model.kind)
    return torch.sigmoid(logits).numpy(), float(loss)


def model_from_bundle(bundle):
    model = AnchoredResponse(bundle["settings"], bundle["kind"], bundle["statistics"],
                             bundle["clinical"], bundle.get("base_spec"))
    model.load_state_dict(bundle["state"], strict=True)
    return model.eval()


@torch.no_grad()
def predict_bundle(bundle, data):
    if bundle["kind"] not in STATISTICAL:
        # Prediction does not require labels or any training-directory paths.
        model = model_from_bundle(bundle)
        values = as_tensors(data)
        logits = model(values["embs"], values["masks"], values["clinical"], values["days"])[0]
        return torch.sigmoid(logits).numpy()
    if bundle["kind"] == "clinical":
        logits = ClinicalAnchor(bundle["clinical"])(torch.from_numpy(data["clinical"]))
        return torch.sigmoid(logits).numpy()[:, None].repeat(4, axis=1)
    transform = FrozenFeatureTransform(bundle["settings"], bundle["statistics"])
    features = transform(torch.from_numpy(data["embs"]), torch.from_numpy(data["masks"])).numpy()
    last = np.full(len(features), bundle["prevalence"], np.float64)
    output = []
    for t, head in enumerate(bundle["heads"]):
        values = probe_inputs(features, data["days"], bundle["kind"], t)
        logits = values @ head["coefficient"] + head["intercept"]
        probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        last = np.where(data["masks"][:, t] > 0, probability, last)
        output.append(last.copy())
    return np.stack(output, axis=1)


def fit_probe(train, kind, settings, prep, probe_settings):
    bundle = dict(kind=kind, settings=settings, clinical=prep["clinical"], statistics=prep["statistics"],
                  selected_epochs=0, prevalence=float(train["labels"].mean()))
    if kind == "clinical":
        return bundle
    transform = FrozenFeatureTransform(settings, prep["statistics"])
    features = transform(torch.from_numpy(train["embs"]), torch.from_numpy(train["masks"])).numpy()
    bundle["heads"] = []
    for t in range(4):
        keep = train["masks"][:, t] > 0
        x = probe_inputs(features, train["days"], kind, t)
        labels = train["labels"][keep]
        if len(np.unique(labels)) < 2:
            probability = float((labels.sum() + 0.5) / (len(labels) + 1))
            state = dict(coefficient=np.zeros(x.shape[1]), intercept=float(np.log(probability / (1 - probability))))
        else:
            estimator = LogisticRegression(C=probe_settings["inverse_regularization"],
                                           max_iter=probe_settings["max_iter"], solver="lbfgs", random_state=42)
            estimator.fit(x[keep], labels)
            if int(estimator.n_iter_[0]) >= probe_settings["max_iter"]:
                raise ValueError("An image probe did not converge")
            state = dict(coefficient=estimator.coef_[0], intercept=float(estimator.intercept_[0]))
        bundle["heads"].append(state)
    return bundle


def fit_neural(train, validation, settings, kind, prep, seed, directory, *,
               fixed_epochs=None, base_spec=None, pause_after=None, pause_check=lambda: None, contract=None):
    if (validation is None) != (fixed_epochs is not None):
        raise ValueError("Inner fitting requires validation; fixed refits must not receive it")
    if validation is not None and set(train["pids"]) & set(validation["pids"]):
        raise ValueError("Training and validation patients overlap")
    deterministic_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = dict(settings=settings, kind=kind, seed=seed, train_ids=train["pids"],
                   validation_ids=validation["pids"] if validation is not None else [],
                   fixed_epochs=fixed_epochs, contract=contract)
    recovery = directory / "recovery.pt"
    resumed = torch.load(recovery, map_location="cpu", weights_only=False) if recovery.exists() else None
    if resumed and resumed["binding"] != binding:
        raise ValueError("Recovery config or partition differs")
    model = AnchoredResponse(settings, kind, prep["statistics"], prep["clinical"], base_spec)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    order = torch.Generator().manual_seed(seed)
    tensors = as_tensors(train)
    eligible = torch.nonzero(learning_mask(tensors["masks"], kind).any(1), as_tuple=True)[0]
    if not len(eligible):
        raise ValueError("No eligible patients in the neural fitting partition")
    _, initial_train = predict_model(model, train)
    initial_val = predict_model(model, validation)[1] if validation is not None else None
    history = [dict(epoch=0, train_objective=None, train_logloss=initial_train, validation_logloss=initial_val)]
    best, selected, epoch_done, stale = cpu_state(model), 0, 0, 0
    best_loss = float(initial_val) if initial_val is not None else float("inf")
    patience_loss, gradient = best_loss, None
    if resumed:
        model.load_state_dict(resumed["state"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        history, best, selected, epoch_done = (resumed[k] for k in ("history", "best", "selected", "epoch_done"))
        best_loss, patience_loss, stale, gradient = (resumed[k] for k in ("best_loss", "patience_loss", "stale", "gradient"))
        restore_rng(resumed["rng"], order, "cpu")
    maximum = settings["epochs"] if fixed_epochs is None else int(fixed_epochs)
    stopped = validation is not None and stale >= settings["patience"]
    for epoch in range(epoch_done + 1, maximum + 1) if not stopped else ():
        model.train()
        shuffled = eligible[torch.randperm(len(eligible), generator=order)]
        total = 0.0
        for index in shuffled.split(settings["batch_size"]):
            values = {k: v[index] for k, v in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(values["embs"], values["masks"], values["clinical"], values["days"])
            loss = objective(logits, residual, values["labels"], values["masks"], kind, settings["residual_l2"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite anchored loss")
            loss.backward()
            params = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Missing or nonfinite trainable gradient")
            if gradient is None:
                gradient = dict(finite=True, nonzero=any(bool(p.grad.abs().sum()) for p in params),
                                frozen_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),
                                trainable_parameters=sum(p.numel() for p in params))
                if not gradient["nonzero"] or not gradient["frozen_gradients_absent"]:
                    raise ValueError("Frozen-anchor or gradient audit failed")
            torch.nn.utils.clip_grad_norm_(params, settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        scheduler.step()
        _, train_loss = predict_model(model, train)
        val_loss = predict_model(model, validation)[1] if validation is not None else None
        if validation is None:
            selected, best = epoch, cpu_state(model)
        else:
            if val_loss < best_loss:
                selected, best, best_loss = epoch, cpu_state(model), val_loss
            if val_loss < patience_loss - settings["min_delta"]:
                patience_loss, stale = val_loss, 0
            else:
                stale += 1
        history.append(dict(epoch=epoch, train_objective=total / len(eligible), train_logloss=train_loss,
                            validation_logloss=val_loss))
        state = dict(binding=binding, state=cpu_state(model), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), rng=rng_state(order, "cpu"), history=history,
                     best=best, selected=selected, epoch_done=epoch, best_loss=best_loss,
                     patience_loss=patience_loss, stale=stale, gradient=gradient)
        save_tensor(recovery, state)
        _atomic_csv(directory / "learning_curve.csv", pd.DataFrame(history))
        write_json(directory / "progress.json", dict(epoch=epoch, maximum_epochs=maximum,
                   selected_epoch=selected, validation_logloss=val_loss))
        if pause_after == epoch:
            raise InterruptedError("Requested exact-recovery check")
        pause_check()
        if validation is not None and stale >= settings["patience"]:
            break
    model.load_state_dict(best, strict=True)
    if kind != "t0" and any(not torch.equal(v, base_spec["state"][k]) for k, v in model.base.state_dict().items()):
        raise ValueError("The frozen T0 reference changed during follow-up fitting")
    _atomic_csv(directory / "learning_curve.csv", pd.DataFrame(history))
    return dict(kind=kind, settings=settings, statistics=prep["statistics"], clinical=prep["clinical"],
                base_spec=base_spec, state=best, selected_epochs=selected, gradient_audit=gradient,
                history=history, binding=binding, zero_epoch_fallback=selected == 0)


def run_task(cfg, task):
    deterministic_runtime()
    if task_complete(cfg, task):
        return
    check_pause(cfg)
    arm = cfg["arms"][task["arm"]]
    all_data = load_data(cfg, arm["roi"])
    if not set(task["train_ids"] + task["val_ids"]) <= set(all_data["pids"]):
        raise ValueError("Fitting task crosses the development boundary")
    if set(task["train_ids"]) & set(task["val_ids"]):
        raise ValueError("Fitting and evaluation patients overlap")
    train, validation = subset(all_data, task["train_ids"]), subset(all_data, task["val_ids"])
    prep = preprocessing(cfg, task, train)
    root = task_path(cfg, task)
    base_spec, reference = None, None
    if "base" in arm:
        reference = copy.deepcopy(task)
        reference["arm"] = arm["base"]
        if task["stage"] == "formal":
            selection = read_json(Path(cfg["output_dir"]) / "epoch_selection.json")
            reference["fixed_epochs"] = next(x["selected_epochs"] for x in selection
                                             if x["arm"] == reference["arm"] and x["fold"] == task["fold"])
        if not task_complete(cfg, reference):
            raise ValueError("The matching training-partition T0 fit is incomplete")
        parent = torch.load(task_path(cfg, reference) / "model.pt", map_location="cpu", weights_only=False)
        base_spec = dict(statistics=parent["statistics"], state=model_from_bundle(parent).base.state_dict())
        for key in ("coefficient",):
            np.testing.assert_array_equal(prep["clinical"][key], parent["clinical"][key])
    contract = dict(task=task, runtime=read_json(Path(cfg["study_dir"]) / "runtime_sources.json"),
                    base=identity(task_path(cfg, reference) / "model.pt") if reference else None)
    if arm["kind"] in STATISTICAL:
        bundle = fit_probe(train, arm["kind"], cfg["training"], prep, cfg["probe"])
    else:
        bundle = fit_neural(train, validation if task["stage"] == "inner" else None, cfg["training"], arm["kind"],
                            prep, task["seed"], root, fixed_epochs=task["fixed_epochs"], base_spec=base_spec,
                            pause_check=lambda: check_pause(cfg), contract=contract)
    bundle.update(task=task, arm=arm, contract=contract)
    probability = predict_bundle(bundle, validation)
    save_tensor(root / "model.pt", bundle)
    reloaded = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    replay = predict_bundle(reloaded, validation)
    error = float(np.max(np.abs(probability - replay)))
    if error > 1e-6:
        raise ValueError("Saved checkpoint replay exceeds tolerance")
    _atomic_csv(root / "predictions.csv", prediction_frame(validation, probability, "inner" if task["stage"] == "inner" else "oof"))
    write_json(root / "summary.json", dict(selected_epochs=bundle["selected_epochs"], reload_error=error,
               zero_epoch_fallback=bundle.get("zero_epoch_fallback", False), gradient_audit=bundle.get("gradient_audit")))
    write_json(root / "COMPLETE.json", dict(task=task, model=identity(root / "model.pt"),
               predictions=identity(root / "predictions.csv")))

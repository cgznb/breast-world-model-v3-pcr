"""Nested regularized TDN refits using completed repeated-single-phase features."""

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

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _fit_prior, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _canonical_split, _make_optimizer, _positive_class_weight
from src import first_post_optimization as opt
from src.first_post_pcr_data import identity, now, read_json, repo_path, save_tensor, write_json
from src.tdn import TDN

SCHEMA = "single_phase_repeat_pcr_nested_v2"


def unchanged(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen retraining contract changed: {path}")
    else:
        write_json(path, value)


def load_config(path, branch):
    path = repo_path(path).resolve()
    master = yaml.safe_load(path.read_text())
    if master.get("schema") != SCHEMA or branch not in master["branches"]:
        raise ValueError("Unsupported retraining configuration")
    cfg = {k: copy.deepcopy(v) for k, v in master.items() if k != "branches"}
    cfg.update(master["branches"][branch])
    cfg.update(branch=branch, config_path=str(path))
    cfg["output_dir"] = str(repo_path(master["output_dir"]) / branch)
    for key in ("input_config", "baseline_run"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if any(opt.effective_config(cfg, c, d).get("prefix_dropout_probability", 0) != 0
           for c in cfg["candidates"] for d in cfg["depths"]):
        raise ValueError("This refit protocol does not use temporal prefix dropout")
    return cfg


def prepare(cfg):
    baseline, output = Path(cfg["baseline_run"]), Path(cfg["output_dir"])
    marker = read_json(baseline / "REAL_FEATURES_COMPLETE.json")
    if not marker.get("complete") or marker.get("input_mode") != cfg["input_mode"]:
        raise ValueError("Completed single-phase development features are required")
    cohort = read_json(baseline / "cohort.json")
    if (len(cohort["split"]["train"]) != cfg["expected_development"]
            or len(cohort["split"]["val"]) != cfg["expected_holdout"]):
        raise ValueError("Original cohort counts changed")
    unchanged(output / "retraining_contract.json", {
        "schema": SCHEMA, "config": cfg,
        "runtime": {p: identity(repo_path(p)) for p in (
            "src/single_phase_repeat_retraining.py", "scripts/run_single_phase_repeat_retraining.py")},
        "missing_t0_policy": "retain clinical prior and predictions; exclude from neural loss",
        "baseline_definition": "v1 compact32 unbounded_l2 candidate, inner logloss stopping",
        "historical_holdout_role": cohort["test_role"],
        "seed_selection": "all ten prespecified seeds; no holdout selection",
    })
    folds = opt.prepare_study(cfg)
    split = opt.load_development(str(output), str(opt.source_directory(cfg, "physical_roi")))
    if int((split["masks"][:, 0] <= 0).sum()) != cfg["expected_missing_t0"]:
        raise ValueError("Missing-T0 cohort membership changed")
    return folds


def fit_tdn(train, val, effective, seed, depth, device, fixed_epochs=None):
    """Keep the clinical cohort intact while excluding no-T0 rows from NN loss."""
    if fixed_epochs is not None and val is not None:
        raise ValueError("Outer fixed-epoch fitting must not receive validation data")
    if fixed_epochs is None and val is None:
        raise ValueError("Inner fitting requires validation data")
    if effective.get("prefix_dropout_probability", 0):
        raise ValueError("Temporal prefix dropout is not part of this protocol")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior_train, prior_val, prior_state = _fit_prior(train, val if val is not None else train, seed)
    has_t0 = train["masks"][:, 0] > 0
    if not has_t0.any():
        raise ValueError("Training fold has no valid T0")
    data = opt.tensor_split(train, prior_train, device)
    eligible = torch.as_tensor(np.flatnonzero(has_t0), device=device)
    validation = opt.tensor_split(val, prior_val, device) if val is not None else None
    model = TDN({"downstream": effective}).to(device)
    optimizer = _make_optimizer(model, effective)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0)))
    pos_weight = torch.tensor(_positive_class_weight(effective, train["labels"][has_t0]), device=device)
    epochs = int(fixed_epochs if fixed_epochs is not None else effective["epochs"])
    if epochs < 1:
        raise ValueError("A refit needs at least one epoch")
    selected, best, stale, state, history = 0, -np.inf, 0, None, []
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
            score = metrics["auroc"] if effective["epoch_selection"] == "auroc" else -metrics["logloss"]
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
    if task["stage"] not in ("inner", "outer"):
        raise ValueError("Unknown fitting stage")
    if (task["stage"] == "outer") != (task.get("fixed_epochs") is not None):
        raise ValueError("Only outer fits require fixed epochs")
    started = time.perf_counter()
    all_data = opt.load_development(str(output), str(embedding_dir))
    train = opt.select_patients(all_data, task["train_ids"], task["depth"])
    val_for_selection = (opt.select_patients(all_data, task["val_ids"], task["depth"])
                         if task["stage"] == "inner" else None)
    seed = task["seed"] * 10000 + task["outer"] * 100 + (task["inner"] + 1) * 10 + task["depth"]
    model, checkpoint = fit_tdn(train, val_for_selection, task["effective"], seed, task["depth"], device,
                                fixed_epochs=task.get("fixed_epochs"))
    checkpoint.update(schema=SCHEMA, task=task, created_utc=now(),
                      outer_validation_used_for_selection=False, holdout_labels_or_embeddings_loaded=False)
    history = checkpoint.pop("history")
    # Outer outcomes are scored only after the fixed weights have been persisted.
    save_tensor(path / "model.pt", checkpoint)
    restored = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(restored["model_state"])
    val = opt.select_patients(all_data, task["val_ids"], task["depth"])
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
                                     "clinical_prior_train_count", "neural_train_count")},
        "seconds": time.perf_counter() - started,
        "auroc_gap": summaries["train"]["auroc"] - summaries["validation"]["auroc"],
        "completed_utc": now(), "checkpoint_reloaded_before_prediction": True}
    write_json(path / "COMPLETE.json", summary)
    if not opt.task_complete(path, task):
        raise RuntimeError("Written task artifacts did not replay")
    return {"path": str(path), "seconds": summary["seconds"], "skipped": False}


def report(cfg):
    output = Path(cfg["output_dir"])
    refs = read_json(output / "selection/regularized_outer_models.json")
    folds = read_json(output / "nested_folds.json")
    expected = set(folds[0]["train_ids"] + folds[0]["val_ids"])
    predictions, fits = [], []
    for ref in refs:
        path = Path(ref["path"])
        summary = read_json(path / "COMPLETE.json")
        if not opt.task_complete(path, summary["task"]):
            raise ValueError("Incomplete outer fit in report")
        frame = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        frame = frame.loc[frame.role == "validation"].copy()
        fields = {k: ref[k] for k in ("arm", "depth", "seed", "outer")}
        predictions.append(frame.assign(**fields))
        fits.append({**fields, "candidate": summary["task"]["candidate"],
            "epochs": summary["selected_epochs"], "gap": summary["auroc_gap"],
            **{f"train_{k}": v for k, v in summary["train"].items()},
            **{f"validation_{k}": v for k, v in summary["validation"].items()}})
    predictions, fits = pd.concat(predictions, ignore_index=True), pd.DataFrame(fits)
    metrics = []
    for (arm, depth, seed), frame in predictions.groupby(["arm", "depth", "seed"]):
        if set(frame.patient_id) != expected or len(frame) != len(expected) or not frame.patient_id.is_unique:
            raise ValueError("Outer OOF coverage changed")
        fit = fits[(fits.arm == arm) & (fits.depth == depth) & (fits.seed == seed)]
        metrics.append({"arm": arm, "depth": depth, "seed": int(seed),
            **opt.metric_values(frame.label.to_numpy(), frame.probability.to_numpy()),
            "train_auroc": fit.train_auroc.mean(), "fold_validation_auroc": fit.validation_auroc.mean(),
            "gap": fit.gap.mean(), "epochs": fit.epochs.mean()})
    metrics = pd.DataFrame(metrics)
    summary = metrics.groupby(["depth", "arm"])[
        ["auroc", "logloss", "brier", "train_auroc", "fold_validation_auroc", "gap", "epochs"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    for name, frame in (("outer_oof_predictions", predictions), ("fit_metrics", fits),
                        ("seed_metrics", metrics), ("summary", summary)):
        _atomic_csv(output / "reports" / f"{name}.csv", frame)
    selected = metrics[metrics.arm == "selected"]
    rankings = selected.groupby("seed").auroc.mean().sort_values(ascending=False)
    payload = {"schema": SCHEMA, "branch": cfg["branch"], "created_utc": now(),
               "development_patients": len(expected), "holdout_evaluated": False,
               "summary": summary.to_dict("records"),
               "descriptive_development_seed_ranking": {str(k): float(v) for k, v in rankings.items()},
               "selection_nested_inside_outer_training": True,
               "baseline_definition": "v1 compact32 unbounded_l2, inner logloss stopping"}
    write_json(output / "reports/summary.json", payload)
    lines = [f"# {cfg['branch']}: regularized repeated-single-phase pCR", "",
        "Frozen Pillar features; original five outer folds; three inner folds; ten formal seeds.",
        "Every candidate uses inner log-loss early stopping. Inner mean AUROC within 0.005 of best gates minimum log loss.",
        "Outer refits use the ceiling median inner-selected duration without outer early stopping.",
        "The baseline arm is the v1 compact32 unbounded_l2 candidate, not the old mixed per-window policy.", "",
        "| Window | Arm | OOF AUC | Train AUC | Mean fold AUC | Gap | Log loss |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {opt.DEPTH_NAMES[row['depth']]} | {row['arm']} | {row['auroc_mean']:.5f} | "
                     f"{row['train_auroc_mean']:.5f} | {row['fold_validation_auroc_mean']:.5f} | "
                     f"{row['gap_mean']:.5f} | {row['logloss_mean']:.5f} |")
    lines += ["", "A smaller gap alone does not establish better prediction. Seed SD is not a patient confidence interval.",
              "The reused historical holdout is an internal evaluation cohort, not an independent end-to-end test.",
              "See evaluation/real for post-freeze real holdout inference. Generated images are not evaluated here.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return payload


def freeze_models(cfg):
    output = Path(cfg["output_dir"])
    rows = read_json(output / "selection/regularized_outer_models.json")
    refs = []
    for row in rows:
        if row["arm"] == "selected":
            path = Path(row["path"]) / "model.pt"
            refs.append({"depth": opt.DEPTH_NAMES[row["depth"]], "max_tp": row["depth"],
                         "seed": row["seed"], "fold": row["outer"], "path": str(path), "identity": identity(path)})
    expected = {(d, s, f) for d in cfg["depths"] for s in cfg["formal_seeds"] for f in range(5)}
    if {(r["max_tp"], r["seed"], r["fold"]) for r in refs} != expected or len(refs) != len(expected):
        raise ValueError("Incomplete selected model set")
    unchanged(output / "frozen_models.json", {"models": refs, "holdout_used_for_selection": False})
    return refs


def evaluate_real(cfg, refs):
    output, baseline = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    heldout = read_json(baseline / "cohort.json")["split"]["val"]
    raw = _load_split(baseline / "embeddings/real", baseline / "holdout_metadata.csv", heldout)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    rows = []
    for ref in refs:
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Selected weights changed")
        saved = torch.load(ref["path"], map_location="cpu", weights_only=False)
        task = saved["task"]
        if (set(task["train_ids"]) | set(task["val_ids"])) & set(heldout):
            raise ValueError("Holdout entered classifier fitting")
        split = _canonical_split(raw, ref["max_tp"])
        logits = _prior_from_state(split["clinical"], saved["clinical_prior"])
        model = TDN({"downstream": task["effective"]}).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["model_state"])
        probability, _ = opt.predict(model, opt.tensor_split(split, logits, "cuda:0"))
        rows.append(pd.DataFrame({"patient_id": split["pids"], "label": split["labels"].astype(int),
            "probability": probability.astype(np.float64), "depth": ref["depth"],
            "seed": ref["seed"], "fold": ref["fold"]}))
    folds = pd.concat(rows, ignore_index=True)
    if not (folds.groupby(["depth", "seed", "patient_id"]).size() == 5).all():
        raise ValueError("Holdout ensemble requires five folds")
    predictions = folds.groupby(["depth", "seed", "patient_id", "label"], as_index=False).probability.mean()
    metrics = []
    for (depth, seed), frame in predictions.groupby(["depth", "seed"]):
        if set(frame.patient_id) != set(heldout) or len(frame) != len(heldout):
            raise ValueError("Holdout prediction coverage changed")
        metrics.append({"depth": depth, "seed": int(seed), "patients": len(heldout),
                        **opt.metric_values(frame.label.to_numpy(), frame.probability.to_numpy())})
    metrics = pd.DataFrame(metrics)
    summary = metrics.groupby("depth")[["auroc", "logloss", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    for name, frame in (("fold_predictions", folds), ("predictions", predictions),
                        ("seed_metrics", metrics), ("summary", summary)):
        _atomic_csv(output / "evaluation/real" / f"{name}.csv", frame)
    write_json(output / "evaluation/real/COMPLETE.json", {"complete": True, "patients": len(heldout),
        "models": len(refs), "completed_utc": now(), "optimizer_updates": 0, "generated_evaluated": False,
        "aggregation": "five_fold_probability_mean_then_mean_seed_metric"})
    return summary.to_dict("records")

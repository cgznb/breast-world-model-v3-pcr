"""Development-only, nested model selection for native first-post pCR."""

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
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.run_full978_anti_overfit import (
    _atomic_csv, _fit_prior, _load_split, _prior_from_state, _read_metadata,
)
from scripts.run_full978_independent_cv import (
    _canonical_split, _make_optimizer, _positive_class_weight, _sample_prefix_depths,
    _subset_split,
)
from src.first_post_pcr_data import identity, now, read_json, repo_path, save_tensor, write_json
from src.tdn import TDN
from src.temporal import mask_torch_temporal_prefix

SCHEMA = "first_post_pcr_nested_optimization_v2"
DEPTH_NAMES = {1: "T0", 2: "T0-T1", 3: "T0-T2", 4: "T0-T3"}


def load_study(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["schema"] != SCHEMA:
        raise ValueError("Unsupported optimization schema")
    for key in ("output_dir", "baseline_run", "input_config"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    return cfg


def validate_partitions(development, holdout, folds):
    dev, held = set(development), set(holdout)
    if len(dev) != len(development) or len(held) != len(holdout) or dev & held:
        raise ValueError("Development/holdout identities overlap or repeat")
    seen = []
    for fold in folds:
        train, val = fold["train_ids"], fold["val_ids"]
        if (len(set(train)) != len(train) or len(set(val)) != len(val)
                or set(train) & set(val) or set(train) | set(val) != dev):
            raise ValueError("Invalid outer patient partition")
        seen.extend(val)
    if len(seen) != len(dev) or set(seen) != dev:
        raise ValueError("Outer validation must cover development exactly once")


def nested_folds(development, holdout, outer, labels, n_inner, seed):
    validate_partitions(development, holdout, outer)
    result = []
    for spec in outer:
        ids = np.asarray(spec["train_ids"])
        y = np.asarray([labels[pid] for pid in ids], dtype=int)
        splitter = StratifiedKFold(n_inner, shuffle=True, random_state=seed + spec["fold"])
        inner = []
        for index, (train, val) in enumerate(splitter.split(ids, y)):
            inner.append({"inner": index, "train_ids": ids[train].tolist(),
                          "val_ids": ids[val].tolist()})
        validate_partitions(ids.tolist(), spec["val_ids"] + list(holdout), inner)
        result.append({**spec, "inner_folds": inner})
    return result


def effective_config(cfg, candidate, depth):
    variant = cfg["candidates"][candidate]
    if variant["kind"] == "clinical":
        return {"kind": "clinical", "epochs": 0}
    if variant.get("original_recipe"):
        name = DEPTH_NAMES[depth].lower().replace("-", "_")
        path = Path(cfg["baseline_run"]) / "configs" / f"{name}.yaml"
        result = yaml.safe_load(path.read_text())["downstream"]
        result["epoch_selection"] = "auroc"
    else:
        result = copy.deepcopy(cfg["default_candidate"])
        result.update({k: v for k, v in variant.items() if k != "kind"})
    result.update(kind="tdn", input_dim=1152, clinical_dim=17)
    return result


def metric_values(labels, probability):
    probability = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probabilities")
    return {"auroc": float(roc_auc_score(labels, probability)),
            "logloss": float(log_loss(labels, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(labels, probability))}


def choose_candidate(rows, tolerance):
    """Use discrimination as a gate, then proper probability loss; never rank by gap."""
    if not rows or not 0 <= tolerance <= 0.01:
        raise ValueError("Invalid candidate selection inputs")
    if any(not np.isfinite([r["auroc"], r["logloss"]]).all() for r in rows):
        raise ValueError("Nonfinite selection metric")
    best_auc = max(r["auroc"] for r in rows)
    eligible = [r for r in rows if r["auroc"] >= best_auc - tolerance]
    return min(eligible, key=lambda r: (r["logloss"], r["parameters"], r["source"], r["candidate"]))


def source_directory(cfg, source):
    if source == "physical_roi":
        return Path(cfg["baseline_run"]) / "embeddings/real"
    if source == "resized_roi":
        return Path(cfg["output_dir"]) / "resized_roi/embeddings/real"
    raise ValueError("Unknown feature source")


def prepare_study(cfg):
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path(cfg["baseline_run"])
    cohort = read_json(baseline / "cohort.json")
    outer = read_json(baseline / "folds.json")
    if isinstance(outer, dict):
        outer = list(outer.values())
    development, holdout = cohort["split"]["train"], cohort["split"]["val"]
    validate_partitions(development, holdout, outer)
    metadata = _read_metadata(baseline / "metadata_enriched.csv", development)
    labels = dict(zip(metadata.pid, metadata.pCR.astype(int)))
    folds = nested_folds(development, holdout, outer, labels,
                         int(cfg["inner_folds"]), int(cfg["inner_fold_seed"]))
    source_files = ["src/first_post_optimization.py", "src/first_post_roi_optimization.py",
                    "scripts/run_first_post_optimization.py", "scripts/report_first_post_optimization.py",
                    "src/tdn.py", "src/temporal.py", "src/data.py", "src/pillar.py",
                    "src/first_post_pcr_data.py", "scripts/run_first_post_pcr.py",
                    "scripts/run_full978_anti_overfit.py", "scripts/run_full978_independent_cv.py"]
    contract = {"schema": SCHEMA, "config": cfg,
                "sources": {p: identity(repo_path(p)) for p in source_files},
                "inputs": {p: identity(baseline / p) for p in
                           ("cohort.json", "folds.json", "metadata_enriched.csv", "REAL_FEATURES_COMPLETE.json")},
                "recipes": {str(d): effective_config(cfg, "baseline", d) for d in cfg["depths"]},
                "development_count": len(development), "holdout_count": len(holdout),
                "holdout_labels_or_embeddings_loaded": False,
                "outer_validation_selects_epochs_or_candidates": False,
                "selection_rule": "inner_mean_AUROC_within_0.005_then_minimum_mean_logloss",
                "original_holdout_role": cohort["test_role"]}
    path = output / "contract.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("Optimization contract changed; use a new output directory")
    write_json(path, contract)
    fold_path = output / "nested_folds.json"
    if fold_path.exists() and read_json(fold_path) != folds:
        raise ValueError("Nested folds changed")
    write_json(fold_path, folds)
    metadata_path = output / "development_metadata.csv"
    if metadata_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(metadata_path, dtype={"pid": str}),
                                      metadata.reset_index(drop=True), check_dtype=False)
    else:
        _atomic_csv(metadata_path, metadata)
    for source in cfg["feature_sources"]:
        if source == "resized_roi" and not (output / "resized_roi/REAL_FEATURES_COMPLETE.json").exists():
            continue
        manifest = {}
        for visit in cohort["visits"]:
            if visit["split"] != "train":
                continue
            pid = visit["canonical_patient_id"]
            relative = f"{pid}/{pid}_T{visit['timepoint']}.pt"
            manifest[relative] = identity(source_directory(cfg, source) / relative)
        path = output / f"{source}_feature_inventory.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError("Frozen development embeddings changed")
        write_json(path, manifest)
    return folds


@functools.lru_cache(maxsize=2)
def load_development(output_dir, embedding_dir):
    folds = read_json(Path(output_dir) / "nested_folds.json")
    ids = sorted(folds[0]["train_ids"] + folds[0]["val_ids"])
    return _load_split(Path(embedding_dir), Path(output_dir) / "development_metadata.csv", ids)


def select_patients(split, ids, depth):
    allowed = set(ids)
    result = _subset_split(split, [pid in allowed for pid in split["pids"]])
    if len(result["pids"]) != len(allowed):
        raise ValueError("Missing or duplicate task patient")
    return _canonical_split(result, depth)


def tensor_split(split, prior, device):
    result = {key: torch.as_tensor(split[key], dtype=torch.float32, device=device)
              for key in ("embs", "masks", "clinical", "labels", "days")}
    result["prior"] = torch.as_tensor(prior, dtype=torch.float32, device=device)
    return result


@torch.no_grad()
def predict(model, data):
    model.eval()
    probability, residuals = [], []
    for start in range(0, len(data["labels"]), 256):
        part = {k: v[start:start + 256] for k, v in data.items()}
        logits, residual = model(part["embs"], part["masks"], part["clinical"],
                                 days=part["days"], prior_logit=part["prior"], return_residual=True)
        probability.append(torch.sigmoid(logits).cpu().numpy())
        residuals.append(residual.cpu().numpy())
    return np.concatenate(probability), np.concatenate(residuals)


def fit_tdn(train, val, effective, seed, depth, device, fixed_epochs=None):
    """An outer refit cannot receive validation data while optimizing weights."""
    if fixed_epochs is not None and val is not None:
        raise ValueError("Outer fixed-epoch fitting must not receive validation data")
    if fixed_epochs is None and val is None:
        raise ValueError("Inner fitting requires validation data")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior_train, prior_val, prior_state = _fit_prior(train, val if val is not None else train, seed)
    data = tensor_split(train, prior_train, device)
    validation = tensor_split(val, prior_val, device) if val is not None else None
    model = TDN({"downstream": effective}).to(device)
    optimizer = _make_optimizer(model, effective)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0)))
    pos_weight = torch.tensor(_positive_class_weight(effective, train["labels"]), device=device)
    selected, best, stale, state, history = 0, -np.inf, 0, None, []
    epochs = int(fixed_epochs if fixed_epochs is not None else effective["epochs"])
    order_generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["labels"]), device=device, generator=order_generator)
        total_loss = 0.0
        for indices in order.split(int(effective["batch_size"])):
            batch = {k: v[indices] for k, v in data.items()}
            if effective.get("prefix_dropout_probability", 0) > 0:
                depths = _sample_prefix_depths(len(indices), depth,
                                               effective["prefix_dropout_probability"], device)
                batch["embs"], batch["masks"], batch["days"] = mask_torch_temporal_prefix(
                    batch["embs"], batch["masks"], batch["days"], depths)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(batch["embs"], batch["masks"], batch["clinical"],
                                     days=batch["days"], prior_logit=batch["prior"], return_residual=True)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
            loss = loss + float(effective.get("residual_l2_weight", 0)) * residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(effective["grad_clip"]))
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        train_probability, _ = predict(model, data)
        train_metrics = metric_values(train["labels"], train_probability)
        row = {"epoch": epoch, "train_objective": total_loss / len(train["labels"]),
               "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in train_metrics.items()}}
        if validation is not None:
            probability, _ = predict(model, validation)
            metrics = metric_values(val["labels"], probability)
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
    return model, {"model_state": state, "clinical_prior": prior_state, "selected_epochs": selected,
                   "epochs_trained": len(history), "history": history,
                   "parameters": sum(p.numel() for p in model.parameters())}


def task_path(output, task):
    return Path(output) / "fits" / task["stage"] / task["source"] / DEPTH_NAMES[task["depth"]] / task["candidate"] / f"outer_{task['outer']}" / f"inner_{task['inner']}" / f"seed_{task['seed']}"


def task_complete(path, task, verify_weights=True):
    try:
        payload = read_json(path / "COMPLETE.json")
        if payload["task"] != task or not (path / "model.pt").is_file() or not (path / "history.csv").is_file():
            return False
        if task["effective"]["kind"] == "tdn":
            history = pd.read_csv(path / "history.csv")
            if (history.epoch.tolist() != list(range(1, payload["epochs_trained"] + 1))
                    or not 1 <= payload["selected_epochs"] <= len(history)
                    or not np.isfinite(history.select_dtypes(include="number").to_numpy()).all()):
                return False
            if task["stage"] == "outer" and (
                    payload["selected_epochs"] != task["fixed_epochs"]
                    or len(history) != task["fixed_epochs"]
                    or any(c.startswith("validation_") for c in history.columns)):
                return False
        frame = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            rows = frame[frame.role == role]
            if len(rows) != len(ids) or set(rows.patient_id) != set(ids) or not rows.patient_id.is_unique:
                return False
            metrics = metric_values(rows.label.to_numpy(), rows.probability.to_numpy())
            if any(abs(metrics[k] - payload[role][k]) > 1e-7 for k in metrics):
                return False
        if verify_weights:
            checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
            if (checkpoint["task"] != task
                    or checkpoint["selected_epochs"] != payload["selected_epochs"]
                    or checkpoint["epochs_trained"] != payload["epochs_trained"]):
                return False
            if any(not torch.isfinite(v).all() for v in checkpoint.get("model_state", {}).values()):
                return False
        return True
    except (OSError, ValueError, KeyError, RuntimeError, EOFError):
        return False


def run_task(task, output, embedding_dir, device="cuda:0"):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    path = task_path(output, task)
    if task_complete(path, task):
        return {"path": str(path), "skipped": True}
    started = time.perf_counter()
    all_data = load_development(str(output), str(embedding_dir))
    train = select_patients(all_data, task["train_ids"], task["depth"])
    if set(task["train_ids"]) & set(task["val_ids"]):
        raise ValueError("Task train/validation overlap")
    if not (train["masks"][:, 0] > 0).all():
        raise ValueError("Study requires valid T0 for every development patient")
    seed = task["seed"] * 10000 + task["outer"] * 100 + (task["inner"] + 1) * 10 + task["depth"]
    effective = task["effective"]
    if effective["kind"] == "clinical":
        _, _, prior_state = _fit_prior(train, train, seed)
        checkpoint = {"clinical_prior": prior_state, "parameters": 18, "selected_epochs": 0,
                      "epochs_trained": 0, "history": []}
        model = None
    else:
        val_for_selection = (select_patients(all_data, task["val_ids"], task["depth"])
                             if task["stage"] == "inner" else None)
        model, checkpoint = fit_tdn(train, val_for_selection, effective, seed, task["depth"], device,
                                    fixed_epochs=task.get("fixed_epochs"))
    checkpoint.update(schema=SCHEMA, task=task, created_utc=now(),
                      outer_validation_used_for_selection=False,
                      holdout_labels_or_embeddings_loaded=False)
    history = checkpoint.pop("history")
    # Persist fixed weights before accessing outer validation outcomes.
    save_tensor(path / "model.pt", checkpoint)
    restored = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(restored["model_state"])
    prior_state = restored["clinical_prior"]
    val = select_patients(all_data, task["val_ids"], task["depth"])
    predictions, summaries = [], {}
    for role, split in (("train", train), ("validation", val)):
        logits = _prior_from_state(split["clinical"], prior_state)
        if model is None:
            probability, residual = 1 / (1 + np.exp(-logits)), np.zeros(len(logits))
        else:
            probability, residual = predict(model, tensor_split(split, logits, device))
        summaries[role] = metric_values(split["labels"], probability)
        predictions.append(pd.DataFrame({"patient_id": split["pids"], "role": role,
                                         "label": split["labels"].astype(int), "probability": probability.astype(np.float64),
                                         "residual_logit": residual.astype(np.float64),
                                         "clinical_prior_probability": (1 / (1 + np.exp(-logits))).astype(np.float64)}))
    _atomic_csv(path / "predictions.csv", pd.concat(predictions, ignore_index=True))
    _atomic_csv(path / "history.csv", pd.DataFrame(history))
    summary = {"schema": SCHEMA, "task": task, **summaries,
               "selected_epochs": checkpoint["selected_epochs"], "epochs_trained": checkpoint["epochs_trained"],
               "parameters": checkpoint["parameters"], "seconds": time.perf_counter() - started,
               "auroc_gap": summaries["train"]["auroc"] - summaries["validation"]["auroc"],
               "completed_utc": now(), "checkpoint_reloaded_before_prediction": True}
    write_json(path / "COMPLETE.json", summary)
    if not task_complete(path, task, verify_weights=True):
        raise RuntimeError("Written task artifacts did not replay")
    return {"path": str(path), "seconds": summary["seconds"], "skipped": False}


def inner_tasks(cfg, folds, sources):
    tasks = []
    for depth in cfg["depths"]:
        for outer in folds:
            for source in sources:
                for candidate in cfg["candidates"]:
                    if candidate == "clinical_only" and source != "physical_roi":
                        continue
                    effective = effective_config(cfg, candidate, depth)
                    seeds = cfg["tuning_seeds"][:1] if candidate == "clinical_only" else cfg["tuning_seeds"]
                    for inner in outer["inner_folds"]:
                        for seed in seeds:
                            tasks.append({"stage": "inner", "source": source, "depth": depth,
                                          "candidate": candidate, "outer": outer["fold"], "inner": inner["inner"],
                                          "seed": seed, "train_ids": inner["train_ids"],
                                          "val_ids": inner["val_ids"], "effective": effective})
    return tasks


def select_models(cfg, folds, sources, name):
    tasks = inner_tasks(cfg, folds, sources)
    grouped = {}
    for task in tasks:
        path = task_path(cfg["output_dir"], task)
        if not task_complete(path, task):
            raise RuntimeError("Selection requires all prespecified inner fits")
        row = read_json(path / "COMPLETE.json")
        key = task["depth"], task["outer"], task["source"], task["candidate"]
        grouped.setdefault(key, []).append(row)
    ranking, selections = [], []
    for depth in cfg["depths"]:
        for outer in folds:
            rows = []
            for (d, fold, source, candidate), results in grouped.items():
                if d != depth or fold != outer["fold"]:
                    continue
                rows.append({"depth": depth, "outer": fold, "source": source, "candidate": candidate,
                             **{k: float(np.mean([r["validation"][k] for r in results]))
                                for k in ("auroc", "logloss", "brier")},
                             "gap": float(np.mean([r["auroc_gap"] for r in results])),
                             "parameters": results[0]["parameters"], "inner_fits": len(results),
                             "fixed_epochs": int(np.ceil(np.median([r["selected_epochs"] for r in results])))})
            selected = choose_candidate(rows, cfg["auroc_tolerance"])
            baseline = next(r for r in rows if r["source"] == "physical_roi" and r["candidate"] == "baseline")
            selections.append({"depth": depth, "outer": outer["fold"], "selected": selected, "baseline": baseline})
            ranking.extend(rows)
    payload = {"schema": SCHEMA, "name": name, "sources": sources, "selections": selections,
               "selection_only_uses_inner_validation": True, "holdout_loaded": False}
    path = Path(cfg["output_dir"]) / "selection" / f"{name}.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("A frozen selection changed")
    write_json(path, payload)
    _atomic_csv(path.with_suffix(".csv"), pd.DataFrame(ranking))
    return payload


def formal_tasks(cfg, folds, selection):
    tasks, references, seen = [], [], set()
    for row in selection["selections"]:
        outer = next(f for f in folds if f["fold"] == row["outer"])
        for arm in ("selected", "baseline"):
            chosen = row[arm]
            for seed in cfg["formal_seeds"]:
                task = {"stage": "outer", "source": chosen["source"], "depth": row["depth"],
                        "candidate": chosen["candidate"], "outer": row["outer"], "inner": -1,
                        "seed": seed, "train_ids": outer["train_ids"], "val_ids": outer["val_ids"],
                        "effective": effective_config(cfg, chosen["candidate"], row["depth"]),
                        "fixed_epochs": chosen["fixed_epochs"]}
                path = str(task_path(cfg["output_dir"], task))
                references.append({"arm": arm, "depth": row["depth"], "outer": row["outer"],
                                   "seed": seed, "path": path})
                if path not in seen:
                    tasks.append(task)
                    seen.add(path)
    write_json(Path(cfg["output_dir"]) / "selection" / f"{selection['name']}_outer_models.json", references)
    return tasks

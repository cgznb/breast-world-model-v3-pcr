"""Refit the selected T0-T1 recipe on all development patients and evaluate it."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import (
    _atomic_csv,
    _atomic_json,
    _atomic_torch,
    _fit_prior,
    _load_split,
    _load_test_source,
    _loader,
    _prior_from_state,
    _repo_path,
    _resolve_device,
    _set_seed,
)
from scripts.run_full978_independent_cv import (
    _canonical_split,
    _fold_specs,
    _make_optimizer,
    _positive_class_weight,
    _prediction_frame,
)
from src.metrics import METRIC_KEYS, compute_metrics
from src.tdn import TDN


SCHEMA = "pillar_full978_t0_t1_optimized_single_refit_v1"


def _depth(config):
    value = config["refit"]["temporal_depth"]
    depth = {"name": str(value["name"]), "max_tp": int(value["max_tp"])}
    if depth != {"name": "T0-T1", "max_tp": 2}:
        raise ValueError("optimized refit is restricted to T0-T1")
    return depth


def _development_config(config):
    path = _repo_path(config["development_selection_json"])
    selection = json.loads(path.read_text())
    depth = _depth(config)
    selected = selection.get("depths", {}).get(depth["name"], {})
    effective = copy.deepcopy(selected.get("effective_config", {}))
    if (
        selection.get("schema") != "pillar_full978_independent_depth_cv_v1"
        or selection.get("stage") != "tune"
        or selection.get("test_data_used") is not False
        or selection.get("test_embeddings_or_labels_loaded") is not False
        or selected.get("variant") != config["refit"]["selected_variant"]
        or selected.get("checkpoint_policy") != "final_epoch"
        or int(selected.get("max_tp", -1)) != depth["max_tp"]
        or effective.get("checkpoint_policy") != "final_epoch"
        or int(effective.get("epochs", -1)) != int(config["refit"]["expected_epochs"])
        or float(effective.get("residual_l2_weight", -1)) != 0.0
    ):
        raise ValueError("development-selected T0-T1 policy contract changed")
    return effective, selection, path


def _paths(output_dir, seed):
    root = Path(output_dir) / "train" / f"seed_{int(seed)}"
    return {
        "root": root,
        "checkpoint": root / "final.pt",
        "history": root / "history.csv",
        "predictions": root / "development_predictions.csv",
        "summary": root / "summary.json",
        "sentinel": root / "TRAINING_COMPLETE.json",
    }


def _valid_predictions(frame, patient_ids, seed, split_name):
    required = {
        "patient_id", "label", "probability", "residual_logit",
        "clinical_prior_probability", "seed", "fold", "split",
        "temporal_depth", "max_tp",
    }
    numeric = frame[["probability", "residual_logit", "clinical_prior_probability"]]
    return (
        required <= set(frame.columns)
        and frame["patient_id"].astype(str).tolist() == list(patient_ids)
        and frame["patient_id"].astype(str).is_unique
        and frame["label"].isin([0, 1]).all()
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and set(frame["seed"]) == {int(seed)}
        and set(frame["fold"]) == {-1}
        and set(frame["split"]) == {split_name}
        and set(frame["temporal_depth"]) == {"T0-T1"}
        and set(frame["max_tp"]) == {2}
    )


def _training_complete(output_dir, seed, effective, pool_ids, expected_tdn):
    paths = _paths(output_dir, seed)
    if not all(path.is_file() for key, path in paths.items() if key != "root"):
        return False
    try:
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        history = pd.read_csv(paths["history"])
        predictions = pd.read_csv(paths["predictions"], dtype={"patient_id": str})
        summary = json.loads(paths["summary"].read_text())
        sentinel = json.loads(paths["sentinel"].read_text())
        epochs = int(effective["epochs"])
        return (
            checkpoint.get("schema") == SCHEMA
            and int(checkpoint.get("seed", -1)) == int(seed)
            and checkpoint.get("effective_config") == effective
            and checkpoint.get("development_ids") == list(pool_ids)
            and checkpoint.get("test_data_loaded_during_training") is False
            and checkpoint.get("test_embeddings_or_labels_loaded_during_training") is False
            and checkpoint.get("clinical_prior", {}).get("fitted_on")
            == "full_development"
            and len(history) == epochs
            and history["epoch"].astype(int).tolist() == list(range(epochs))
            and int(summary.get("tdn_training_patients", -1)) == int(expected_tdn)
            and summary.get("selection", {}).get("criterion") == "final_training_epoch"
            and sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("test_data_loaded") is False
            and sentinel.get("test_embeddings_or_labels_loaded") is False
            and _valid_predictions(predictions, pool_ids, seed, "full_development")
        )
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False


def _train_task(task):
    config_path, output_dir, seed, device_text, smoke, force = task
    config = yaml.safe_load(Path(config_path).read_text())
    effective, _, _ = _development_config(config)
    if smoke:
        effective["epochs"] = 2
    _, pool_ids, _, _ = _fold_specs(
        yaml.safe_load(_repo_path("configs/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch.yaml").read_text())
    )
    expected_tdn = int(config["refit"]["expected_tdn_training_patients"])
    if not force and _training_complete(output_dir, seed, effective, pool_ids, expected_tdn):
        return f"skip refit seed={seed}"

    task_seed = int(seed) * 1000 + 2
    _set_seed(task_seed)
    torch.set_num_threads(1)
    device = _resolve_device(device_text)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = _canonical_split(
        _load_split(
            _repo_path(config["data"]["train_embeddings_dir"]),
            _repo_path(config["data"]["metadata_csv"]),
            pool_ids,
        ),
        2,
    )
    train_prior, _, prior_state = _fit_prior(train, train, task_seed)
    prior_state["fitted_on"] = "full_development"
    has_t0 = train["masks"][:, 0] > 0
    train_for_loss = {
        **train,
        **{
            key: np.asarray(train[key])[has_t0]
            for key in ("embs", "masks", "clinical", "labels", "days")
        },
        "pids": [pid for pid, keep in zip(train["pids"], has_t0) if keep],
    }
    if len(train_for_loss["pids"]) != expected_tdn:
        raise ValueError("T0-T1 refit patient count changed")
    prior_for_loss = np.asarray(train_prior)[has_t0]
    loader = _loader(
        train_for_loss,
        prior_for_loss,
        effective["batch_size"],
        True,
        task_seed,
    )
    model = TDN({"downstream": effective}).to(device)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    if not smoke and parameters != int(config["refit"]["expected_parameters"]):
        raise ValueError("T0-T1 refit parameter count changed")
    optimizer = _make_optimizer(model, effective)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0.0)),
    )
    weight = _positive_class_weight(effective, train_for_loss["labels"])
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight], device=device)
    )
    history = []
    for epoch in range(int(effective["epochs"])):
        model.train()
        losses = []
        for embeddings, masks, clinical, target, prior_logit, days in loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prior_logit = prior_logit.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                embeddings, masks, clinical, days=days, prior_logit=prior_logit
            )
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite T0-T1 refit loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(effective["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "residual_scale": float(model.residual_scale().detach().cpu()),
            }
        )

    predictions = _prediction_frame(
        model,
        train,
        train_prior,
        _depth(config),
        seed,
        -1,
        "full_development",
        device,
        effective["batch_size"],
    )
    metrics = compute_metrics(
        predictions["label"], predictions["probability"], threshold=0.5
    )
    selection = {
        "criterion": "final_training_epoch",
        "best_epoch_zero_based": int(effective["epochs"]) - 1,
        "epochs_trained": int(effective["epochs"]),
        "validation_used_for_checkpoint_selection": False,
        "early_stopping_enabled": False,
    }
    paths = _paths(output_dir, seed)
    _atomic_torch(
        paths["checkpoint"],
        {
            "schema": SCHEMA,
            "seed": int(seed),
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "effective_config": effective,
            "clinical_prior": prior_state,
            "selection": selection,
            "development_ids": list(pool_ids),
            "test_data_loaded_during_training": False,
            "test_embeddings_or_labels_loaded_during_training": False,
        },
    )
    _atomic_csv(paths["history"], pd.DataFrame(history))
    _atomic_csv(paths["predictions"], predictions)
    _atomic_json(
        paths["summary"],
        {
            "schema": SCHEMA,
            "seed": int(seed),
            "parameters": parameters,
            "development_metrics": metrics,
            "development_patients": len(pool_ids),
            "tdn_training_patients": len(train_for_loss["pids"]),
            "selection": selection,
            "test_role": "not_loaded",
        },
    )
    _atomic_json(
        paths["sentinel"],
        {
            "schema": SCHEMA,
            "seed": int(seed),
            "test_data_loaded": False,
            "test_embeddings_or_labels_loaded": False,
            "complete": True,
        },
    )
    return f"done refit seed={seed} development_auroc={metrics['auroc']:.4f}"


def _evaluate_task(task):
    config_path, output_dir, source, seed, test_ids, device_text, smoke = task
    config = yaml.safe_load(Path(config_path).read_text())
    effective, _, _ = _development_config(config)
    if smoke:
        effective["epochs"] = 2
    pool_count = int(config["data"]["expected_pool_patients"])
    if not _training_complete(
        output_dir,
        seed,
        effective,
        [pid for pid in _fold_specs(yaml.safe_load(_repo_path("configs/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch.yaml").read_text()))[1]],
        int(config["refit"]["expected_tdn_training_patients"]),
    ):
        raise RuntimeError("refit checkpoint is incomplete")
    if pool_count != 876:
        raise ValueError("development cohort changed")
    checkpoint = torch.load(
        _paths(output_dir, seed)["checkpoint"], map_location="cpu", weights_only=False
    )
    test = _canonical_split(_load_test_source(config, source, test_ids), 2)
    prior = _prior_from_state(test["clinical"], checkpoint["clinical_prior"])
    device = _resolve_device(device_text)
    torch.set_num_threads(1)
    model = TDN({"downstream": effective}).to(device)
    model.load_state_dict(checkpoint["model_state"])
    predictions = _prediction_frame(
        model,
        test,
        prior,
        _depth(config),
        seed,
        -1,
        "test_single_refit",
        device,
        effective["batch_size"],
    )
    predictions["source"] = source
    run_dir = Path(output_dir) / "evaluation" / source / f"seed_{int(seed)}"
    _atomic_csv(run_dir / "test_predictions.csv", predictions)
    _atomic_json(
        run_dir / "EVALUATION_COMPLETE.json",
        {
            "schema": SCHEMA,
            "source": source,
            "seed": int(seed),
            "test_ids": list(test_ids),
            "models_ensembled": 1,
            "test_used_for_selection": False,
            "complete": True,
        },
    )
    return f"done evaluation {source} seed={seed}"


def _run_tasks(tasks, jobs, label):
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=context) as executor:
        futures = [executor.submit(task[0], task[1:]) for task in tasks]
        for future in as_completed(futures):
            print(f"[{label}] {future.result()}", flush=True)


def _aggregate_evaluation(config, output_dir, sources, seeds, test_ids):
    for source in sources:
        frames = []
        rows = []
        for seed in seeds:
            run_dir = Path(output_dir) / "evaluation" / source / f"seed_{int(seed)}"
            sentinel = json.loads((run_dir / "EVALUATION_COMPLETE.json").read_text())
            frame = pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
            if (
                sentinel.get("schema") != SCHEMA
                or sentinel.get("test_ids") != list(test_ids)
                or sentinel.get("models_ensembled") != 1
                or not _valid_predictions(frame, test_ids, seed, "test_single_refit")
                or set(frame["source"]) != {source}
            ):
                raise RuntimeError("invalid optimized T0-T1 evaluation artifact")
            frames.append(frame)
            rows.append(
                {
                    "source": source,
                    "seed": int(seed),
                    **compute_metrics(frame["label"], frame["probability"], threshold=0.5),
                }
            )
        metrics = pd.DataFrame(rows)
        summary = {"source": source, "n_seeds": len(seeds), "models_per_seed": 1}
        for metric in METRIC_KEYS:
            values = metrics[metric].to_numpy(dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        source_dir = Path(output_dir) / "evaluation" / source
        _atomic_csv(source_dir / "all_test_predictions.csv", pd.concat(frames, ignore_index=True))
        _atomic_csv(source_dir / "metrics_per_seed.csv", metrics)
        _atomic_csv(source_dir / "summary.csv", pd.DataFrame([summary]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/mewm_ispy2_full978_locked102_t0_t1_optimized_refit.yaml",
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    effective, selection, selection_path = _development_config(config)
    cv_config = yaml.safe_load(
        _repo_path("configs/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch.yaml").read_text()
    )
    _, pool_ids, test_ids, _ = _fold_specs(cv_config)
    section = config["refit"]
    seeds = [int(seed) for seed in section["seeds"]]
    if args.smoke:
        seeds = seeds[:1]
    jobs = int(args.jobs or section["jobs"])
    output_dir = _repo_path(section["output_dir"])
    if args.smoke:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema": SCHEMA,
            "config": str(config_path),
            "development_selection": str(selection_path),
            "development_selection_test_data_used": selection["test_data_used"],
            "development_patients": len(pool_ids),
            "locked_test_patients": len(test_ids),
            "test_membership_role_during_training": "contract_validation_only",
            "effective_config": ({**effective, "epochs": 2} if args.smoke else effective),
            "seeds": seeds,
            "models_per_seed": 1,
            "smoke": bool(args.smoke),
        },
    )

    if args.phase in ("train", "all"):
        tasks = [
            (
                _train_task,
                str(config_path),
                str(output_dir),
                seed,
                args.device,
                bool(args.smoke),
                bool(args.force),
            )
            for seed in seeds
        ]
        _run_tasks(tasks, jobs, "refit")
        training_effective = copy.deepcopy(effective)
        if args.smoke:
            training_effective["epochs"] = 2
        if not all(
            _training_complete(
                output_dir,
                seed,
                training_effective,
                pool_ids,
                section["expected_tdn_training_patients"],
            )
            for seed in seeds
        ):
            raise RuntimeError("optimized T0-T1 refit training audit failed")
        _atomic_json(
            output_dir / "TRAINING_PHASE_COMPLETE.json",
            {
                "schema": SCHEMA,
                "seeds": seeds,
                "checkpoints": len(seeds),
                "models_per_seed": 1,
                "test_data_loaded": False,
                "test_embeddings_or_labels_loaded": False,
                "complete": True,
            },
        )

    if args.phase in ("evaluate", "all"):
        sentinel = json.loads((output_dir / "TRAINING_PHASE_COMPLETE.json").read_text())
        if (
            sentinel.get("schema") != SCHEMA
            or sentinel.get("complete") is not True
            or sentinel.get("seeds") != seeds
            or sentinel.get("test_embeddings_or_labels_loaded") is not False
        ):
            raise RuntimeError("evaluation requires a complete test-blind training phase")
        sources = list(config["evaluation_sources"])
        tasks = [
            (
                _evaluate_task,
                str(config_path),
                str(output_dir),
                source,
                seed,
                test_ids,
                args.device,
                bool(args.smoke),
            )
            for source in sources
            for seed in seeds
        ]
        _run_tasks(tasks, jobs, "evaluate")
        _aggregate_evaluation(config, output_dir, sources, seeds, test_ids)
        _atomic_json(
            output_dir / "EXPERIMENT_COMPLETE.json",
            {
                "schema": SCHEMA,
                "seeds": seeds,
                "sources": sources,
                "models_per_seed": 1,
                "test_used_for_selection": False,
                "complete": True,
            },
        )


if __name__ == "__main__":
    main()

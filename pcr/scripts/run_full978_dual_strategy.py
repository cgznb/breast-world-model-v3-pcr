"""Optimize independent-depth and shared-contiguous TDN training strategies."""

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
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import (
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _atomic_torch,
    _fit_prior,
    _load_split,
    _loader,
    _predict_prefix,
    _prior_from_state,
    _repo_path,
    _resolve_device,
    _set_seed,
)
from src.data import EmbStore, TABULAR_FEATURE_NAMES, load_ids
from src.metrics import METRIC_KEYS, compute_metrics
from src.tdn import TDN
from src.temporal import (
    canonicalize_temporal_prefix,
    expand_contiguous_torch_temporal_prefixes,
)


SCHEMA = "pillar_full978_dual_temporal_strategy_v1"
SELECTION_CRITERION = "validation_auroc"
_DEVELOPMENT_CACHE = {}
_TEST_CACHE = {}


def _slug(value):
    return str(value).lower().replace("-", "_")


def _depths(config):
    return [
        {"name": str(row["name"]), "max_tp": int(row["max_tp"])}
        for row in config["experiment"]["temporal_depths"]
    ]


def _split_ids(config):
    cohort = _repo_path(config["data"]["cohort_dir"])
    ids = {
        split: load_ids(cohort / "splits" / f"{split}_ids.txt")
        for split in ("train", "val", "test")
    }
    expected = {
        "train": int(config["data"]["expected_train_patients"]),
        "val": int(config["data"]["expected_validation_patients"]),
        "test": int(config["data"]["expected_test_patients"]),
    }
    if any(len(ids[name]) != expected[name] for name in expected):
        raise ValueError("fixed split patient count mismatch")
    sets = {name: set(values) for name, values in ids.items()}
    if sets["train"] & sets["val"] or (sets["train"] | sets["val"]) & sets["test"]:
        raise ValueError("fixed train/validation/test splits overlap")
    reference = json.loads(
        _repo_path(config["data"]["locked_test_reference_json"]).read_text()
    )
    reference_ids = {
        str(value)
        for value in reference[config["data"].get("locked_test_reference_split", "val")]
    }
    if reference_ids != sets["test"]:
        raise ValueError("fixed test does not exactly match the BiFlow reference")
    return ids


def _load_development(config):
    ids = _split_ids(config)
    key = (
        str(_repo_path(config["data"]["real_embeddings_dir"])),
        str(_repo_path(config["data"]["metadata_csv"])),
        tuple(ids["train"]),
        tuple(ids["val"]),
    )
    if key not in _DEVELOPMENT_CACHE:
        _DEVELOPMENT_CACHE[key] = {
            split: _load_split(
                _repo_path(config["data"]["real_embeddings_dir"]),
                _repo_path(config["data"]["metadata_csv"]),
                ids[split],
            )
            for split in ("train", "val")
        }
    return _DEVELOPMENT_CACHE[key], ids


def _copy_split(split):
    result = dict(split)
    for key in ("embs", "masks", "clinical", "labels", "days"):
        result[key] = np.asarray(split[key]).copy()
    result["pids"] = list(split["pids"])
    return result


def _canonical_split(split, max_tp):
    result = _copy_split(split)
    result["embs"], result["masks"], result["days"] = canonicalize_temporal_prefix(
        result["embs"], result["masks"], result["days"], int(max_tp)
    )
    return result


def _subset_split(split, selector):
    selector = np.asarray(selector, dtype=bool)
    result = dict(split)
    for key in ("embs", "masks", "clinical", "labels", "days"):
        result[key] = np.asarray(split[key])[selector]
    result["pids"] = [pid for pid, keep in zip(split["pids"], selector) if keep]
    return result


def _effective_config(config, variant, smoke=False):
    if variant not in config["variants"]:
        raise ValueError(f"unknown variant: {variant}")
    result = copy.deepcopy(config["downstream"])
    for key, value in config["variants"][variant].items():
        if key != "description":
            result[key] = value
    result["input_dim"] = int(config["data"]["embedding_dim"])
    result["clinical_dim"] = len(TABULAR_FEATURE_NAMES)
    result["model_type"] = "tdn"
    if smoke:
        result["epochs"] = 2
        result["patience"] = 2
    return result


def _training_dir(output_dir, strategy, stage, variant, seed, depth=None):
    path = Path(output_dir) / strategy / "train" / stage
    if depth is not None:
        path = path / _slug(depth)
    return path / f"variant_{variant}" / f"seed_{seed}"


def _prediction_frame(model, split, prior, depths, seed, strategy, device, batch_size):
    frames = []
    prior_probability = 1.0 / (1.0 + np.exp(-np.asarray(prior, dtype=np.float64)))
    for depth in depths:
        canonical = _canonical_split(split, depth["max_tp"])
        loader = _loader(canonical, prior, int(batch_size) * 4, False)
        labels, probabilities = _predict_prefix(
            model, loader, int(depth["max_tp"]), device
        )
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": canonical["pids"],
                    "label": labels.astype(int),
                    "probability": probabilities,
                    "clinical_prior_probability": prior_probability,
                    "seed": int(seed),
                    "strategy": strategy,
                    "temporal_depth": depth["name"],
                    "max_tp": int(depth["max_tp"]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _metrics_by_depth(predictions):
    rows = []
    for (depth, max_tp), frame in predictions.groupby(
        ["temporal_depth", "max_tp"], sort=False
    ):
        rows.append(
            {
                "temporal_depth": str(depth),
                "max_tp": int(max_tp),
                **compute_metrics(frame["label"], frame["probability"], threshold=0.5),
                "clinical_prior_auroc": float(
                    roc_auc_score(frame["label"], frame["clinical_prior_probability"])
                ),
            }
        )
    return rows


def _validate_training_artifact(run_dir, expected):
    run_dir = Path(run_dir)
    paths = {
        "checkpoint": run_dir / "best.pt",
        "summary": run_dir / "summary.json",
        "history": run_dir / "history.csv",
        "train": run_dir / "train_predictions.csv",
        "val": run_dir / "val_predictions.csv",
        "sentinel": run_dir / "TRAINING_COMPLETE.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        sentinel = json.loads(paths["sentinel"].read_text())
        train_predictions = pd.read_csv(paths["train"], dtype={"patient_id": str})
        val_predictions = pd.read_csv(paths["val"], dtype={"patient_id": str})
        prediction_columns = {
            "patient_id",
            "label",
            "probability",
            "seed",
            "strategy",
            "temporal_depth",
            "max_tp",
        }

        def valid_predictions(frame, patient_ids):
            expected_pairs = {
                (patient_id, depth["name"], int(depth["max_tp"]))
                for patient_id in patient_ids
                for depth in expected["depths"]
            }
            actual_pairs = set(
                zip(
                    frame["patient_id"].astype(str),
                    frame["temporal_depth"].astype(str),
                    frame["max_tp"].astype(int),
                )
            )
            return (
                prediction_columns <= set(frame.columns)
                and len(frame) == len(expected_pairs)
                and actual_pairs == expected_pairs
                and frame["seed"].eq(int(expected["seed"])).all()
                and frame["strategy"].eq(expected["strategy"]).all()
                and frame["label"].isin([0, 1]).all()
                and np.isfinite(frame["probability"].to_numpy(dtype=float)).all()
                and frame["probability"].between(0, 1).all()
            )

        return (
            sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("strategy") == expected["strategy"]
            and sentinel.get("stage") == expected["stage"]
            and sentinel.get("variant") == expected["variant"]
            and int(sentinel.get("seed", -1)) == int(expected["seed"])
            and sentinel.get("temporal_depth") == expected.get("temporal_depth")
            and sentinel.get("effective_config") == expected["effective_config"]
            and checkpoint.get("schema") == SCHEMA
            and checkpoint.get("strategy") == expected["strategy"]
            and checkpoint.get("stage") == expected["stage"]
            and checkpoint.get("variant") == expected["variant"]
            and int(checkpoint.get("seed", -1)) == int(expected["seed"])
            and checkpoint.get("temporal_depth") == expected.get("temporal_depth")
            and checkpoint.get("effective_config") == expected["effective_config"]
            and checkpoint.get("selection", {}).get("criterion") == SELECTION_CRITERION
            and checkpoint.get("test_data_loaded_during_training") is False
            and checkpoint.get("clinical_prior", {}).get("fitted_on")
            == "fold_train_only"
            and checkpoint.get("clinical_prior", {}).get("feature_names")
            == list(TABULAR_FEATURE_NAMES)
            and checkpoint.get("train_ids") == list(expected["train_ids"])
            and checkpoint.get("validation_ids") == list(expected["validation_ids"])
            and valid_predictions(train_predictions, expected["train_ids"])
            and valid_predictions(val_predictions, expected["validation_ids"])
        )
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        return False


def _equal_depth_loss(point_losses, counts):
    """Average examples within each depth, then give every represented depth equal weight."""
    if point_losses.ndim != 1:
        raise ValueError("point_losses must be one-dimensional")
    if any(int(count) < 0 for count in counts):
        raise ValueError("prefix counts must be non-negative")
    offset = 0
    depth_losses = []
    for count in counts:
        count = int(count)
        if count > 0:
            depth_losses.append(point_losses[offset : offset + count].mean())
            offset += count
    if offset != len(point_losses):
        raise ValueError("contiguous-prefix loss partition mismatch")
    if not depth_losses:
        raise ValueError("no temporal depth is represented in the batch")
    return torch.stack(depth_losses).mean()


def _train_task(task):
    (
        config_path,
        output_dir,
        strategy,
        stage,
        variant,
        seed,
        depth_name,
        device_text,
        smoke,
        force,
    ) = task
    config = yaml.safe_load(Path(config_path).read_text())
    ds_cfg = _effective_config(config, variant, smoke=smoke)
    all_depths = _depths(config)
    selected_depths = (
        [row for row in all_depths if row["name"] == depth_name]
        if strategy == "independent"
        else all_depths
    )
    if not selected_depths:
        raise ValueError(f"unknown temporal depth: {depth_name}")
    run_dir = _training_dir(
        output_dir, strategy, stage, variant, seed, depth=depth_name
    )
    split_ids = _split_ids(config)
    expected = {
        "strategy": strategy,
        "stage": stage,
        "variant": variant,
        "seed": int(seed),
        "temporal_depth": depth_name,
        "effective_config": ds_cfg,
        "depths": selected_depths,
        "train_ids": split_ids["train"],
        "validation_ids": split_ids["val"],
    }
    if not force and _validate_training_artifact(run_dir, expected):
        return f"skip {strategy} {depth_name or 'all'} {variant} seed={seed}"

    data, ids = _load_development(config)
    _set_seed(int(seed))
    torch.set_num_threads(1)
    device = _resolve_device(device_text)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_prior, val_prior, prior_state = _fit_prior(
        data["train"], data["val"], int(seed)
    )
    model = TDN({"downstream": ds_cfg}).to(device)
    optimizer = Adam(
        model.parameters(), lr=float(ds_cfg["lr"]), weight_decay=float(ds_cfg["weight_decay"])
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(ds_cfg["epochs"]))

    if strategy == "independent":
        depth = selected_depths[0]
        train_for_loss = _canonical_split(data["train"], depth["max_tp"])
        has_t0 = train_for_loss["masks"][:, 0] > 0
        train_for_loss = _subset_split(train_for_loss, has_t0)
        prior_for_loss = np.asarray(train_prior)[has_t0]
        train_loader = _loader(
            train_for_loss, prior_for_loss, ds_cfg["batch_size"], True, int(seed)
        )
        train_labels_for_weight = train_for_loss["labels"]
    elif strategy == "shared_contiguous":
        has_t0 = data["train"]["masks"][:, 0] > 0
        train_for_loss = _subset_split(data["train"], has_t0)
        prior_for_loss = np.asarray(train_prior)[has_t0]
        train_loader = _loader(
            train_for_loss, prior_for_loss, ds_cfg["batch_size"], True, int(seed)
        )
        train_labels_for_weight = train_for_loss["labels"]
    else:
        raise ValueError(f"unsupported strategy: {strategy}")

    positives = float((train_labels_for_weight == 1).sum())
    negatives = float((train_labels_for_weight == 0).sum())
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_score = -np.inf
    best_state = None
    best_epoch = -1
    best_validation = None
    patience = 0
    history = []

    for epoch in range(int(ds_cfg["epochs"])):
        model.train()
        epoch_losses = []
        for embeddings, masks, clinical, target, prior, days in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prior = prior.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if strategy == "independent":
                logits = model(
                    embeddings, masks, clinical, days=days, prior_logit=prior
                )
                loss = loss_fn(logits, target)
            else:
                (
                    prefix_embeddings,
                    prefix_masks,
                    prefix_days,
                    repeated,
                    _,
                    counts,
                ) = expand_contiguous_torch_temporal_prefixes(
                    embeddings, masks, days, clinical, target, prior
                )
                prefix_clinical, prefix_target, prefix_prior = repeated
                logits = model(
                    prefix_embeddings,
                    prefix_masks,
                    prefix_clinical,
                    days=prefix_days,
                    prior_logit=prefix_prior,
                )
                point_losses = F.binary_cross_entropy_with_logits(
                    logits, prefix_target, pos_weight=pos_weight, reduction="none"
                )
                loss = _equal_depth_loss(point_losses, counts)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            if float(ds_cfg.get("grad_clip", 0)) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(ds_cfg["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()

        val_predictions = _prediction_frame(
            model,
            data["val"],
            val_prior,
            selected_depths,
            seed,
            strategy,
            device,
            ds_cfg["batch_size"],
        )
        val_metrics = _metrics_by_depth(val_predictions)
        selection_score = float(np.mean([row["auroc"] for row in val_metrics]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "residual_scale": float(model.residual_scale().detach().cpu()),
                "selection_score": selection_score,
                **{
                    f"val_auroc_{row['temporal_depth']}": row["auroc"]
                    for row in val_metrics
                },
            }
        )
        if selection_score > best_score + float(ds_cfg.get("min_delta", 0.0)):
            best_score = selection_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_validation = val_metrics
            patience = 0
        else:
            patience += 1
        if patience >= int(ds_cfg["patience"]):
            break

    if best_state is None:
        raise RuntimeError("validation never produced a finite model selection score")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    train_predictions = _prediction_frame(
        model,
        data["train"],
        train_prior,
        selected_depths,
        seed,
        strategy,
        device,
        ds_cfg["batch_size"],
    )
    val_predictions = _prediction_frame(
        model,
        data["val"],
        val_prior,
        selected_depths,
        seed,
        strategy,
        device,
        ds_cfg["batch_size"],
    )
    selection = {
        "criterion": SELECTION_CRITERION,
        "best_epoch_zero_based": int(best_epoch),
        "best_score": float(best_score),
        "best_validation_metrics": best_validation,
        "epochs_trained": len(history),
    }
    checkpoint = {
        "schema": SCHEMA,
        "strategy": strategy,
        "stage": stage,
        "variant": variant,
        "seed": int(seed),
        "temporal_depth": depth_name,
        "model_state": best_state,
        "effective_config": ds_cfg,
        "clinical_prior": prior_state,
        "selection": selection,
        "train_ids": ids["train"],
        "validation_ids": ids["val"],
        "test_data_loaded_during_training": False,
        "test_embeddings_or_labels_loaded_during_training": False,
        "test_membership_validated_during_training": True,
        "contiguous_prefix_policy": "stop_at_first_missing_visit",
    }
    _atomic_torch(run_dir / "best.pt", checkpoint)
    _atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
    _atomic_csv(run_dir / "train_predictions.csv", train_predictions)
    _atomic_csv(run_dir / "val_predictions.csv", val_predictions)
    _atomic_json(
        run_dir / "summary.json",
        {
            "schema": SCHEMA,
            "strategy": strategy,
            "stage": stage,
            "variant": variant,
            "seed": int(seed),
            "temporal_depth": depth_name,
            "selection": selection,
            "train_metrics": _metrics_by_depth(train_predictions),
            "validation_metrics": _metrics_by_depth(val_predictions),
            "clinical_prior_patients": len(ids["train"]),
            "tdn_training_patients": int((data["train"]["masks"][:, 0] > 0).sum()),
            "test_role": "not_loaded",
            "test_membership_role": "contract_validation_only",
        },
    )
    _atomic_json(
        run_dir / "TRAINING_COMPLETE.json",
        {
            "schema": SCHEMA,
            "strategy": strategy,
            "stage": stage,
            "variant": variant,
            "seed": int(seed),
            "temporal_depth": depth_name,
            "effective_config": ds_cfg,
            "complete": True,
        },
    )
    return (
        f"done {stage} {strategy} {depth_name or 'all'} {variant} "
        f"seed={seed} val={best_score:.4f}"
    )


def _read_summary(output_dir, strategy, stage, variant, seed, depth=None):
    path = _training_dir(output_dir, strategy, stage, variant, seed, depth)
    return json.loads((path / "summary.json").read_text())


def _select_with_tolerance(rows, variant_order, tolerance):
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("variant", sort=False)
        .agg(
            validation_auroc_mean=("validation_auroc", "mean"),
            validation_auroc_std=("validation_auroc", "std"),
            train_auroc_mean=("train_auroc", "mean"),
            train_validation_gap_mean=("train_validation_gap", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    summary["variant_order"] = summary["variant"].map(
        {name: index for index, name in enumerate(variant_order)}
    )
    best_validation = float(summary["validation_auroc_mean"].max())
    eligible = summary[
        summary["validation_auroc_mean"] >= best_validation - float(tolerance)
    ].copy()
    eligible["positive_gap"] = eligible["train_validation_gap_mean"].clip(lower=0)
    selected = str(
        eligible.sort_values(
            ["positive_gap", "validation_auroc_mean", "variant_order"],
            ascending=[True, False, True],
        ).iloc[0]["variant"]
    )
    summary["within_validation_tolerance"] = summary["validation_auroc_mean"] >= (
        best_validation - float(tolerance)
    )
    summary["selected"] = summary["variant"] == selected
    return selected, table, summary.sort_values("variant_order")


def _build_selection(config, output_dir, tune_seeds, smoke):
    depths = _depths(config)
    selection = {
        "schema": SCHEMA,
        "tune_seeds": [int(seed) for seed in tune_seeds],
        "test_data_used": False,
        "contiguous_prefix_policy": "stop_at_first_missing_visit",
        "independent": {},
        "shared_contiguous": {},
    }
    independent_rows = []
    independent_cfg = config["strategies"]["independent"]
    independent_variants = list(independent_cfg["variants"])
    for depth in depths:
        if depth["name"] not in independent_cfg["tune_depths"]:
            continue
        rows = []
        for variant in independent_variants:
            for seed in tune_seeds:
                summary = _read_summary(
                    output_dir, "independent", "tune", variant, seed, depth["name"]
                )
                train_metric = summary["train_metrics"][0]
                val_metric = summary["validation_metrics"][0]
                rows.append(
                    {
                        "temporal_depth": depth["name"],
                        "variant": variant,
                        "seed": int(seed),
                        "train_auroc": float(train_metric["auroc"]),
                        "validation_auroc": float(val_metric["auroc"]),
                        "train_validation_gap": float(
                            train_metric["auroc"] - val_metric["auroc"]
                        ),
                    }
                )
        selected, detail, summary = _select_with_tolerance(
            rows,
            independent_variants,
            independent_cfg["selection_tolerance"],
        )
        independent_rows.append(detail)
        _atomic_csv(
            Path(output_dir) / "independent" / "selection" / f"{_slug(depth['name'])}.csv",
            summary,
        )
        selection["independent"][depth["name"]] = {
            "variant": selected,
            "effective_config": _effective_config(config, selected, smoke=smoke),
        }
    selection["independent"]["T0"] = {
        "variant": "reused_original",
        "source": independent_cfg["reuse_t0_from"],
    }

    shared_rows = []
    shared_cfg = config["strategies"]["shared_contiguous"]
    shared_variants = list(shared_cfg["variants"])
    for variant in shared_variants:
        for seed in tune_seeds:
            summary = _read_summary(
                output_dir, "shared_contiguous", "tune", variant, seed
            )
            train_by_depth = {
                row["temporal_depth"]: row for row in summary["train_metrics"]
            }
            val_by_depth = {
                row["temporal_depth"]: row for row in summary["validation_metrics"]
            }
            for depth in depths:
                train_value = float(train_by_depth[depth["name"]]["auroc"])
                val_value = float(val_by_depth[depth["name"]]["auroc"])
                shared_rows.append(
                    {
                        "temporal_depth": depth["name"],
                        "variant": variant,
                        "seed": int(seed),
                        "train_auroc": train_value,
                        "validation_auroc": val_value,
                        "train_validation_gap": train_value - val_value,
                    }
                )
    shared_selected, shared_detail, shared_summary = _select_with_tolerance(
        shared_rows, shared_variants, shared_cfg["selection_tolerance"]
    )
    _atomic_csv(
        Path(output_dir) / "shared_contiguous" / "selection" / "variants.csv",
        shared_summary,
    )
    _atomic_csv(
        Path(output_dir) / "shared_contiguous" / "selection" / "runs.csv",
        shared_detail,
    )
    selection["shared_contiguous"] = {
        "variant": shared_selected,
        "effective_config": _effective_config(config, shared_selected, smoke=smoke),
    }
    if independent_rows:
        _atomic_csv(
            Path(output_dir) / "independent" / "selection" / "runs.csv",
            pd.concat(independent_rows, ignore_index=True),
        )
    _atomic_json(Path(output_dir) / "selected_variants.json", selection)
    return selection


def _load_test_source(config, source):
    ids = _split_ids(config)["test"]
    key = (source, tuple(ids))
    if key in _TEST_CACHE:
        return _TEST_CACHE[key], ids
    real = _load_split(
        _repo_path(config["data"]["real_embeddings_dir"]),
        _repo_path(config["data"]["metadata_csv"]),
        ids,
    )
    if source == "real":
        result = real
    elif source == "generated":
        result = _copy_split(real)
        overlay = EmbStore(str(_repo_path(config["data"]["generated_embeddings_dir"])))
        availability = pd.read_csv(
            _repo_path(config["data"]["availability_csv"]), dtype={"patient_id": str}
        )
        availability = availability[availability["patient_id"].isin(set(ids))]
        availability_index = {
            (str(row.patient_id), int(str(row.visit).removeprefix("T"))): int(row.valid_mask)
            for row in availability.itertuples(index=False)
        }
        expected_grid = {(patient_id, timepoint) for patient_id in ids for timepoint in range(4)}
        if set(availability_index) != expected_grid:
            raise ValueError("availability does not exactly cover the fixed test grid")
        replaced = 0
        for patient_index, patient_id in enumerate(ids):
            for timepoint in range(1, 4):
                valid = availability_index[(patient_id, timepoint)] > 0
                if valid:
                    if not overlay.exists(patient_id, timepoint):
                        raise FileNotFoundError(
                            f"generated embedding missing for {patient_id} T{timepoint}"
                        )
                    result["embs"][patient_index, timepoint] = overlay.load(
                        patient_id, timepoint, require_finite=True
                    )
                    replaced += 1
                elif overlay.exists(patient_id, timepoint):
                    raise ValueError("generated embedding exists for an unavailable future visit")
        if replaced != int(config["data"]["expected_generated_future_embeddings"]):
            raise ValueError("generated future embedding count mismatch")
    else:
        raise ValueError(f"unknown evaluation source: {source}")
    _TEST_CACHE[key] = result
    return result, ids


def _retained_future_tokens(split):
    _, masks, _ = canonicalize_temporal_prefix(
        split["embs"], split["masks"], split["days"], split["embs"].shape[1]
    )
    return int((masks[:, 1:] > 0).sum())


def _reused_t0_checkpoint_path(config, seed):
    return (
        _repo_path(config["strategies"]["independent"]["reuse_t0_from"])
        / f"seed_{int(seed)}"
        / "global_tab_temporal"
        / "best.pt"
    )


def _validate_reused_t0_checkpoint(config, seed):
    path = _reused_t0_checkpoint_path(config, seed)
    if not path.is_file():
        raise FileNotFoundError(f"reused T0 checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    effective = checkpoint.get("effective_config", {})
    run_context = checkpoint.get("run_context", {})
    expected_input_dim = int(config["data"]["embedding_dim"])
    if (
        int(checkpoint.get("seed", -1)) != int(seed)
        or checkpoint.get("cell") != "global | +tab+temporal"
        or int(effective.get("max_tp", -1)) != 1
        or effective.get("model_type") != "tdn"
        or effective.get("use_prior") is not True
        or int(effective.get("input_dim", -1)) != expected_input_dim
        or int(effective.get("clinical_dim", -1)) != len(TABULAR_FEATURE_NAMES)
        or run_context.get("temporal_depth") != "T0"
        or int(run_context.get("max_tp", -1)) != 1
        or checkpoint.get("clinical_prior", {}).get("feature_names")
        != list(TABULAR_FEATURE_NAMES)
    ):
        raise ValueError(f"reused T0 checkpoint contract mismatch: {path}")
    return path


def _load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ds_cfg = checkpoint["effective_config"]
    model = TDN({"downstream": ds_cfg}).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint, ds_cfg


def _evaluation_dir(output_dir, strategy, source, seed):
    return Path(output_dir) / strategy / "evaluation" / source / f"seed_{seed}"


def _validate_evaluation_artifact(run_dir, expected):
    run_dir = Path(run_dir)
    paths = {
        "predictions": run_dir / "test_predictions.csv",
        "metrics": run_dir / "metrics.csv",
        "sentinel": run_dir / "EVALUATION_COMPLETE.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        sentinel = json.loads(paths["sentinel"].read_text())
        predictions = pd.read_csv(paths["predictions"], dtype={"patient_id": str})
        metrics = pd.read_csv(paths["metrics"])
        required_prediction_columns = {
            "patient_id",
            "label",
            "probability",
            "seed",
            "strategy",
            "source",
            "split",
            "temporal_depth",
            "max_tp",
        }
        expected_pairs = {
            (patient_id, depth["name"], int(depth["max_tp"]))
            for patient_id in expected["test_ids"]
            for depth in expected["depths"]
        }
        actual_pairs = set(
            zip(
                predictions["patient_id"].astype(str),
                predictions["temporal_depth"].astype(str),
                predictions["max_tp"].astype(int),
            )
        )
        if not (
            required_prediction_columns <= set(predictions.columns)
            and len(predictions) == len(expected_pairs)
            and actual_pairs == expected_pairs
            and predictions["seed"].eq(int(expected["seed"])).all()
            and predictions["strategy"].eq(expected["strategy"]).all()
            and predictions["source"].eq(expected["source"]).all()
            and predictions["split"].eq("test").all()
            and predictions["label"].isin([0, 1]).all()
            and np.isfinite(predictions["probability"].to_numpy(dtype=float)).all()
            and predictions["probability"].between(0, 1).all()
        ):
            return False
        if not (
            sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("strategy") == expected["strategy"]
            and sentinel.get("source") == expected["source"]
            and int(sentinel.get("seed", -1)) == int(expected["seed"])
            and sentinel.get("test_ids") == list(expected["test_ids"])
            and sentinel.get("temporal_depths") == expected["depths"]
            and int(sentinel.get("retained_contiguous_future_tokens", -1))
            == int(expected["retained_future_tokens"])
        ):
            return False
        if set(metrics["temporal_depth"].astype(str)) != {
            depth["name"] for depth in expected["depths"]
        } or len(metrics) != len(expected["depths"]):
            return False
        recomputed = {
            row["temporal_depth"]: row for row in _metrics_by_depth(predictions)
        }
        for metric_row in metrics.to_dict("records"):
            if (
                metric_row.get("strategy") != expected["strategy"]
                or metric_row.get("source") != expected["source"]
                or int(metric_row.get("seed", -1)) != int(expected["seed"])
            ):
                return False
            reference = recomputed[str(metric_row["temporal_depth"])]
            for metric in METRIC_KEYS:
                if not np.isclose(
                    float(metric_row[metric]),
                    float(reference[metric]),
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    return False
        return True
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        return False


def _evaluate_task(task):
    config_path, output_dir, strategy, source, seed, device_text, smoke = task
    config = yaml.safe_load(Path(config_path).read_text())
    selection = json.loads((Path(output_dir) / "selected_variants.json").read_text())
    test, test_ids = _load_test_source(config, source)
    retained_future_tokens = _retained_future_tokens(test)
    expected_retained = int(config["data"]["expected_contiguous_future_embeddings"])
    if retained_future_tokens != expected_retained:
        raise ValueError(
            "contiguous future-token count mismatch: "
            f"expected {expected_retained}, found {retained_future_tokens}"
        )
    device = _resolve_device(device_text)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames = []
    depths = _depths(config)

    if strategy == "shared_contiguous":
        variant = selection[strategy]["variant"]
        checkpoint_path = _training_dir(
            output_dir, strategy, "formal", variant, seed
        ) / "best.pt"
        model, checkpoint, ds_cfg = _load_model(checkpoint_path, device)
        prior = _prior_from_state(test["clinical"], checkpoint["clinical_prior"])
        frame = _prediction_frame(
            model, test, prior, depths, seed, strategy, device, ds_cfg["batch_size"]
        )
        frames.append(frame)
        del model
    elif strategy == "independent":
        for depth in depths:
            if depth["name"] == "T0":
                checkpoint_path = _validate_reused_t0_checkpoint(config, seed)
            else:
                variant = selection[strategy][depth["name"]]["variant"]
                checkpoint_path = _training_dir(
                    output_dir,
                    strategy,
                    "formal",
                    variant,
                    seed,
                    depth["name"],
                ) / "best.pt"
            model, checkpoint, ds_cfg = _load_model(checkpoint_path, device)
            prior = _prior_from_state(test["clinical"], checkpoint["clinical_prior"])
            frame = _prediction_frame(
                model,
                test,
                prior,
                [depth],
                seed,
                strategy,
                device,
                ds_cfg["batch_size"],
            )
            frames.append(frame)
            del model
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != len(test_ids) * len(depths):
        raise RuntimeError("evaluation predictions do not cover the fixed test exactly")
    predictions["source"] = source
    predictions["split"] = "test"
    metrics = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "source": source,
                "seed": int(seed),
                **row,
            }
            for row in _metrics_by_depth(predictions)
        ]
    )
    run_dir = _evaluation_dir(output_dir, strategy, source, seed)
    _atomic_csv(run_dir / "test_predictions.csv", predictions)
    _atomic_csv(run_dir / "metrics.csv", metrics)
    _atomic_json(
        run_dir / "EVALUATION_COMPLETE.json",
        {
            "schema": SCHEMA,
            "strategy": strategy,
            "source": source,
            "seed": int(seed),
            "test_patients": len(test_ids),
            "test_ids": list(test_ids),
            "temporal_depths": depths,
            "retained_contiguous_future_tokens": retained_future_tokens,
            "contiguous_prefix_policy": "stop_at_first_missing_visit",
            "complete": True,
        },
    )
    return f"done evaluate {strategy} {source} seed={seed}"


def _aggregate(config, output_dir, formal_seeds):
    test_ids = _split_ids(config)["test"]
    depths = _depths(config)
    expected_retained = int(config["data"]["expected_contiguous_future_embeddings"])
    all_summaries = []
    for strategy in ("independent", "shared_contiguous"):
        for source in ("real", "generated"):
            prediction_frames = []
            metric_frames = []
            for seed in formal_seeds:
                run_dir = _evaluation_dir(output_dir, strategy, source, seed)
                if not _validate_evaluation_artifact(
                    run_dir,
                    {
                        "strategy": strategy,
                        "source": source,
                        "seed": int(seed),
                        "test_ids": test_ids,
                        "depths": depths,
                        "retained_future_tokens": expected_retained,
                    },
                ):
                    raise RuntimeError(f"incomplete evaluation: {run_dir}")
                prediction_frames.append(
                    pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
                )
                metric_frames.append(pd.read_csv(run_dir / "metrics.csv"))
            predictions = pd.concat(prediction_frames, ignore_index=True)
            metrics = pd.concat(metric_frames, ignore_index=True)
            summary_rows = []
            for depth in depths:
                selected = metrics[metrics["temporal_depth"] == depth["name"]]
                row = {
                    "strategy": strategy,
                    "source": source,
                    "temporal_depth": depth["name"],
                    "max_tp": depth["max_tp"],
                    "n_seeds": len(formal_seeds),
                }
                for metric in METRIC_KEYS:
                    values = selected[metric].to_numpy(dtype=float)
                    row[f"{metric}_mean"] = float(np.nanmean(values))
                    row[f"{metric}_std"] = (
                        float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                    )
                summary_rows.append(row)
            summary = pd.DataFrame(summary_rows)
            target = Path(output_dir) / strategy / "evaluation" / source
            _atomic_csv(target / "all_test_predictions.csv", predictions)
            _atomic_csv(target / "metrics_per_seed.csv", metrics)
            _atomic_csv(target / "summary.csv", summary)
            all_summaries.append(summary)
    combined = pd.concat(all_summaries, ignore_index=True)
    _atomic_csv(Path(output_dir) / "strategy_summary.csv", combined)
    return combined


def _run_tasks(tasks, jobs, label):
    if not tasks:
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=context) as executor:
        futures = [executor.submit(task[0], task[1:]) for task in tasks]
        for future in as_completed(futures):
            print(f"[{label}] {future.result()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/mewm_ispy2_full978_locked102_dual_strategy.yaml"
    )
    parser.add_argument("--phase", choices=("tune", "formal", "evaluate", "all"), default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    experiment = config["experiment"]
    output_dir = _repo_path(args.output_dir or experiment["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    tune_seeds = [int(seed) for seed in experiment["tune_seeds"]]
    formal_seeds = [int(seed) for seed in experiment["formal_seeds"]]
    if args.smoke:
        tune_seeds = tune_seeds[:1]
        formal_seeds = formal_seeds[:1]
    jobs = int(args.jobs or experiment.get("jobs", 10))
    ids = _split_ids(config)
    if args.phase in ("formal", "evaluate", "all"):
        for seed in formal_seeds:
            _validate_reused_t0_checkpoint(config, seed)
    _atomic_json(
        output_dir / "EXPERIMENT_COMPLETE.json",
        {
            "schema": SCHEMA,
            "phase": args.phase,
            "complete": False,
        },
    )
    _atomic_text(
        output_dir / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=True)
    )
    _atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema": SCHEMA,
            "config": str(config_path),
            "output_dir": str(output_dir),
            "phase": args.phase,
            "tune_seeds": tune_seeds,
            "formal_seeds": formal_seeds,
            "jobs": jobs,
            "smoke": bool(args.smoke),
            "split_counts": {name: len(values) for name, values in ids.items()},
            "test_role": "not_loaded_until_evaluate",
            "contiguous_prefix_policy": "stop_at_first_missing_visit",
        },
    )

    selection = None
    if args.phase in ("tune", "all"):
        tasks = []
        independent = config["strategies"]["independent"]
        for depth in independent["tune_depths"]:
            for variant in independent["variants"]:
                for seed in tune_seeds:
                    tasks.append(
                        (
                            _train_task,
                            str(config_path),
                            str(output_dir),
                            "independent",
                            "tune",
                            variant,
                            seed,
                            depth,
                            args.device,
                            bool(args.smoke),
                            bool(args.force),
                        )
                    )
        shared = config["strategies"]["shared_contiguous"]
        for variant in shared["variants"]:
            for seed in tune_seeds:
                tasks.append(
                    (
                        _train_task,
                        str(config_path),
                        str(output_dir),
                        "shared_contiguous",
                        "tune",
                        variant,
                        seed,
                        None,
                        args.device,
                        bool(args.smoke),
                        bool(args.force),
                    )
                )
        _run_tasks(tasks, jobs, "tune")
        selection = _build_selection(config, output_dir, tune_seeds, bool(args.smoke))

    if args.phase in ("formal", "all"):
        if selection is None:
            selection = json.loads((output_dir / "selected_variants.json").read_text())
        tasks = []
        for depth in config["strategies"]["independent"]["tune_depths"]:
            variant = selection["independent"][depth]["variant"]
            for seed in formal_seeds:
                tasks.append(
                    (
                        _train_task,
                        str(config_path),
                        str(output_dir),
                        "independent",
                        "formal",
                        variant,
                        seed,
                        depth,
                        args.device,
                        bool(args.smoke),
                        bool(args.force),
                    )
                )
        variant = selection["shared_contiguous"]["variant"]
        for seed in formal_seeds:
            tasks.append(
                (
                    _train_task,
                    str(config_path),
                    str(output_dir),
                    "shared_contiguous",
                    "formal",
                    variant,
                    seed,
                    None,
                    args.device,
                    bool(args.smoke),
                    bool(args.force),
                )
            )
        _run_tasks(tasks, jobs, "formal")

    if args.phase in ("evaluate", "all"):
        if selection is None:
            selection = json.loads((output_dir / "selected_variants.json").read_text())
        tasks = []
        for strategy in ("independent", "shared_contiguous"):
            for source in ("real", "generated"):
                for seed in formal_seeds:
                    tasks.append(
                        (
                            _evaluate_task,
                            str(config_path),
                            str(output_dir),
                            strategy,
                            source,
                            seed,
                            args.device,
                            bool(args.smoke),
                        )
                    )
        _run_tasks(tasks, jobs, "evaluate")
        summary = _aggregate(config, output_dir, formal_seeds)
        _atomic_json(
            output_dir / "EXPERIMENT_COMPLETE.json",
            {
                "schema": SCHEMA,
                "strategies": ["independent", "shared_contiguous"],
                "formal_seeds": formal_seeds,
                "fixed_test_patients": len(ids["test"]),
                "test_used_for_selection": False,
                "selected_variants": selection,
                "complete": True,
            },
        )
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

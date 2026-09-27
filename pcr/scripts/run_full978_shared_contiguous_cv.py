"""Train a strict contiguous-prefix shared TDN with five-fold development CV.

The locked 102-patient test set is never loaded by the train phase. Variant
selection uses only pooled OOF predictions from the 876-patient development
pool. Evaluation averages five fold probabilities for each patient before
computing test metrics.
"""

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
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import (
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _atomic_torch,
    _evaluation_source_contract,
    _fit_prior,
    _fold_specs as _shared_fold_specs,
    _load_split,
    _load_test_source,
    _loader,
    _prior_from_state,
    _repo_path,
    _resolve_device,
    _set_seed,
    _write_fold_manifests,
)
from scripts.run_full978_independent_cv import _make_optimizer
from scripts.run_full978_dual_strategy import _equal_depth_loss
from src.data import TABULAR_FEATURE_NAMES
from src.metrics import METRIC_KEYS, compute_metrics
from src.tdn import TDN
from src.temporal import (
    canonicalize_temporal_prefix,
    contiguous_prefix_lengths,
    expand_contiguous_torch_temporal_prefixes,
)


SCHEMA = "pillar_full978_shared_contiguous_cv_v1"
SELECTION_CRITERION = (
    "macro_oof_auroc_within_0.005_then_positive_train_oof_gap_parameters_auroc"
)
PREFIX_POLICY = "each_unique_valid_t0_starting_contiguous_prefix_once_per_epoch"
LOSS_WEIGHTING = "equal_mean_weighted_bce_per_temporal_depth"
INFERENCE_POLICY = "stop_at_first_missing_visit"
BEST_VALIDATION_CHECKPOINT_POLICY = "best_validation_macro"
FINAL_EPOCH_CHECKPOINT_POLICY = "final_epoch"
CHECKPOINT_POLICIES = {
    BEST_VALIDATION_CHECKPOINT_POLICY,
    FINAL_EPOCH_CHECKPOINT_POLICY,
}
BEST_VALIDATION_CHECKPOINT_CRITERION = (
    "mean_validation_auroc_across_four_contiguous_prefixes"
)
FINAL_EPOCH_CHECKPOINT_CRITERION = (
    "final_configured_epoch_without_validation_checkpoint_selection"
)


def _section(config):
    return config["shared_contiguous_cv"]


def _checkpoint_policy(config):
    policy = str(
        _section(config).get(
            "checkpoint_policy", BEST_VALIDATION_CHECKPOINT_POLICY
        )
    )
    if policy not in CHECKPOINT_POLICIES:
        choices = ", ".join(sorted(CHECKPOINT_POLICIES))
        raise ValueError(f"checkpoint_policy must be one of: {choices}")
    return policy


def _checkpoint_criterion(policy):
    if policy == BEST_VALIDATION_CHECKPOINT_POLICY:
        return BEST_VALIDATION_CHECKPOINT_CRITERION
    if policy == FINAL_EPOCH_CHECKPOINT_POLICY:
        return FINAL_EPOCH_CHECKPOINT_CRITERION
    raise ValueError(f"unsupported checkpoint policy: {policy}")


def _allows_legacy_checkpoint_policy(config):
    return (
        "checkpoint_policy" not in _section(config)
        and _checkpoint_policy(config) == BEST_VALIDATION_CHECKPOINT_POLICY
    )


def _artifact_policies_match(artifact, policies, allow_legacy=False):
    for key, value in policies.items():
        if (
            key == "checkpoint_policy"
            and allow_legacy
            and key not in artifact
            and value == BEST_VALIDATION_CHECKPOINT_POLICY
        ):
            continue
        if artifact.get(key) != value:
            return False
    return True


def _depths(config):
    rows = [
        {"name": str(row["name"]), "max_tp": int(row["max_tp"])}
        for row in _section(config)["temporal_depths"]
    ]
    expected = [
        {"name": "T0", "max_tp": 1},
        {"name": "T0-T1", "max_tp": 2},
        {"name": "T0-T2", "max_tp": 3},
        {"name": "T0-T3", "max_tp": 4},
    ]
    if rows != expected:
        raise ValueError("shared contiguous CV requires exactly T0 through T0-T3")
    return rows


def _policies(config):
    section = _section(config)
    policies = {
        "training_view_policy": str(section["prefix_policy"]),
        "loss_weighting": str(section["loss_weighting"]),
        "inference_policy": INFERENCE_POLICY,
        "checkpoint_policy": _checkpoint_policy(config),
    }
    if policies["training_view_policy"] != PREFIX_POLICY:
        raise ValueError("unexpected contiguous-prefix training policy")
    if policies["loss_weighting"] != LOSS_WEIGHTING:
        raise ValueError("unexpected temporal loss weighting")
    return policies


def _fold_specs(config):
    compatibility = copy.deepcopy(config)
    section = _section(config)
    compatibility["anti_overfit"] = {
        "folds": int(section["folds"]),
        "fold_seed": int(section["fold_seed"]),
    }
    return _shared_fold_specs(compatibility)


def _effective_config(config, variant, smoke=False):
    if variant not in config["variants"]:
        raise ValueError(f"unknown variant: {variant}")
    result = copy.deepcopy(config["downstream"])
    for key, value in config["variants"][variant].items():
        if key != "description":
            result[key] = copy.deepcopy(value)
    result["input_dim"] = int(config["data"]["embedding_dim"])
    result["clinical_dim"] = len(TABULAR_FEATURE_NAMES)
    result["model_type"] = "tdn"
    if smoke:
        result["epochs"] = 2
        result["patience"] = 2
        if "scheduler_t_max" not in result:
            result["scheduler_t_max"] = 2
    return result


def _parameter_count(config, variant, smoke=False):
    effective = _effective_config(config, variant, smoke=smoke)
    model = TDN({"downstream": effective})
    return int(sum(parameter.numel() for parameter in model.parameters()))


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
    if selector.shape != (len(split["pids"]),):
        raise ValueError("patient selector has the wrong shape")
    result = dict(split)
    for key in ("embs", "masks", "clinical", "labels", "days"):
        result[key] = np.asarray(split[key])[selector]
    result["pids"] = [
        patient_id
        for patient_id, keep in zip(split["pids"], selector)
        if keep
    ]
    return result


def _training_dir(output_dir, stage, variant, seed, fold):
    return (
        Path(output_dir)
        / "train"
        / str(stage)
        / f"variant_{variant}"
        / f"seed_{int(seed)}"
        / f"fold_{int(fold)}"
    )


def _predict_arrays(model, split, prior, device, batch_size):
    loader = _loader(split, prior, int(batch_size) * 4, False)
    labels = []
    probabilities = []
    residuals = []
    model.eval()
    with torch.no_grad():
        for embeddings, masks, clinical, target, prior_logit, days in loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            prior_logit = prior_logit.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            logits, residual = model(
                embeddings,
                masks,
                clinical,
                days=days,
                prior_logit=prior_logit,
                return_residual=True,
            )
            labels.append(target.numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            residuals.append(residual.cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        np.concatenate(residuals),
    )


def _prediction_frame(
    model, split, prior, depths, seed, fold, split_name, device, batch_size
):
    frames = []
    prior_probability = 1.0 / (1.0 + np.exp(-np.asarray(prior, dtype=np.float64)))
    for depth in depths:
        canonical = _canonical_split(split, depth["max_tp"])
        labels, probabilities, residuals = _predict_arrays(
            model, canonical, prior, device, batch_size
        )
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": canonical["pids"],
                    "label": labels.astype(int),
                    "probability": probabilities,
                    "residual_logit": residuals,
                    "clinical_prior_probability": prior_probability,
                    "seed": int(seed),
                    "fold": int(fold),
                    "split": split_name,
                    "strategy": "shared_contiguous_cv",
                    "temporal_depth": depth["name"],
                    "max_tp": int(depth["max_tp"]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _metrics_by_depth(predictions, depths):
    rows = []
    for depth in depths:
        selected = predictions[
            predictions["temporal_depth"] == str(depth["name"])
        ]
        metrics = compute_metrics(
            selected["label"], selected["probability"], threshold=0.5
        )
        rows.append(
            {
                "temporal_depth": str(depth["name"]),
                "max_tp": int(depth["max_tp"]),
                **metrics,
                "clinical_prior_auroc": float(
                    roc_auc_score(
                        selected["label"], selected["clinical_prior_probability"]
                    )
                ),
            }
        )
    return rows


def _valid_prediction_frame(
    frame, patient_ids, seed, fold, split_name, depths, source=None
):
    required = {
        "patient_id",
        "label",
        "probability",
        "residual_logit",
        "clinical_prior_probability",
        "seed",
        "fold",
        "split",
        "strategy",
        "temporal_depth",
        "max_tp",
    }
    if required - set(frame.columns):
        return False
    numeric = frame[["probability", "residual_logit", "clinical_prior_probability"]]
    if (
        not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not frame["probability"].between(0, 1).all()
        or not frame["clinical_prior_probability"].between(0, 1).all()
        or not frame["label"].isin([0, 1]).all()
        or set(frame["seed"]) != {int(seed)}
        or set(frame["fold"]) != {int(fold)}
        or set(frame["split"]) != {split_name}
        or set(frame["strategy"]) != {"shared_contiguous_cv"}
    ):
        return False
    if source is not None and (
        "source" not in frame.columns or set(frame["source"]) != {source}
    ):
        return False
    for depth in depths:
        selected = frame[frame["temporal_depth"] == depth["name"]]
        if (
            selected["patient_id"].astype(str).tolist() != list(patient_ids)
            or not selected["patient_id"].is_unique
            or set(selected["max_tp"]) != {int(depth["max_tp"])}
        ):
            return False
    return len(frame) == len(patient_ids) * len(depths)


def _valid_oof_prediction_frame(frame, patient_ids, seed, depths, folds):
    required = {
        "patient_id",
        "label",
        "probability",
        "residual_logit",
        "clinical_prior_probability",
        "seed",
        "fold",
        "split",
        "strategy",
        "temporal_depth",
        "max_tp",
    }
    if required - set(frame.columns):
        return False
    numeric = frame[["probability", "residual_logit", "clinical_prior_probability"]]
    if (
        not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not frame["probability"].between(0, 1).all()
        or not frame["clinical_prior_probability"].between(0, 1).all()
        or not frame["label"].isin([0, 1]).all()
        or set(frame["seed"]) != {int(seed)}
        or set(frame["fold"]) != set(range(int(folds)))
        or set(frame["split"]) != {"oof_val"}
        or set(frame["strategy"]) != {"shared_contiguous_cv"}
    ):
        return False
    for depth in depths:
        selected = frame[frame["temporal_depth"] == depth["name"]]
        if (
            selected["patient_id"].astype(str).tolist() != list(patient_ids)
            or not selected["patient_id"].is_unique
            or set(selected["max_tp"]) != {int(depth["max_tp"])}
        ):
            return False
    return len(frame) == len(patient_ids) * len(depths)


def _training_complete(run_dir, expected):
    run_dir = Path(run_dir)
    paths = {
        "checkpoint": run_dir / "best.pt",
        "history": run_dir / "history.csv",
        "train": run_dir / "train_predictions.csv",
        "val": run_dir / "val_predictions.csv",
        "summary": run_dir / "summary.json",
        "sentinel": run_dir / "TRAINING_COMPLETE.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=False
        )
        sentinel = json.loads(paths["sentinel"].read_text())
        summary = json.loads(paths["summary"].read_text())
        history = pd.read_csv(paths["history"])
        train = pd.read_csv(paths["train"], dtype={"patient_id": str})
        val = pd.read_csv(paths["val"], dtype={"patient_id": str})
        policies = expected["policies"]
        policy = policies["checkpoint_policy"]
        allow_legacy = bool(expected.get("allow_legacy_checkpoint_policy", False))
        selection = checkpoint.get("selection", {})
        summary_selection = summary.get("selection", {})
        selection_policy_matches = (
            selection.get("checkpoint_policy") == policy
            or (
                allow_legacy
                and "checkpoint_policy" not in selection
                and policy == BEST_VALIDATION_CHECKPOINT_POLICY
            )
        )
        summary_selection_policy_matches = (
            summary_selection.get("checkpoint_policy") == policy
            or (
                allow_legacy
                and "checkpoint_policy" not in summary_selection
                and policy == BEST_VALIDATION_CHECKPOINT_POLICY
            )
        )
        checkpoint_epoch = int(
            selection.get(
                "checkpoint_epoch_zero_based",
                selection.get("best_epoch_zero_based", -1),
            )
        )
        expected_validation_use = policy == BEST_VALIDATION_CHECKPOINT_POLICY
        strategy_metadata_matches = allow_legacy or all(
            artifact.get("validation_used_for_checkpoint_selection")
            is expected_validation_use
            for artifact in (checkpoint, summary, sentinel)
        )
        final_epoch_contract = True
        if policy == FINAL_EPOCH_CHECKPOINT_POLICY:
            expected_epochs = int(expected["effective_config"]["epochs"])
            final_epoch_contract = (
                len(history) == expected_epochs
                and history["epoch"].astype(int).tolist()
                == list(range(expected_epochs))
                and int(selection.get("epochs_trained", -1)) == expected_epochs
                and checkpoint_epoch == expected_epochs - 1
                and int(
                    summary_selection.get(
                        "checkpoint_epoch_zero_based",
                        summary_selection.get("best_epoch_zero_based", -1),
                    )
                )
                == expected_epochs - 1
                and selection.get("early_stopping_enabled") is False
                and selection.get("validation_used_for_checkpoint_selection") is False
            )
        common = (
            checkpoint.get("schema") == SCHEMA
            and checkpoint.get("stage") == expected["stage"]
            and checkpoint.get("variant") == expected["variant"]
            and int(checkpoint.get("seed", -1)) == int(expected["seed"])
            and int(checkpoint.get("fold", -1)) == int(expected["fold"])
            and checkpoint.get("effective_config") == expected["effective_config"]
            and selection.get("criterion") == _checkpoint_criterion(policy)
            and selection_policy_matches
            and summary_selection.get("criterion") == _checkpoint_criterion(policy)
            and summary_selection_policy_matches
            and checkpoint.get("train_ids") == list(expected["train_ids"])
            and checkpoint.get("validation_ids") == list(expected["validation_ids"])
            and checkpoint.get("test_data_loaded_during_training") is False
            and checkpoint.get("test_embeddings_or_labels_loaded_during_training")
            is False
            and checkpoint.get("clinical_prior", {}).get("fitted_on")
            == "fold_train_only"
            and checkpoint.get("clinical_prior", {}).get("feature_names")
            == list(TABULAR_FEATURE_NAMES)
            and _artifact_policies_match(checkpoint, policies, allow_legacy)
            and _artifact_policies_match(summary, policies, allow_legacy)
            and sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("effective_config") == expected["effective_config"]
            and _artifact_policies_match(sentinel, policies, allow_legacy)
            and strategy_metadata_matches
        )
        return common and final_epoch_contract and _valid_prediction_frame(
            train,
            expected["train_ids"],
            expected["seed"],
            expected["fold"],
            "fold_train",
            expected["depths"],
        ) and _valid_prediction_frame(
            val,
            expected["validation_ids"],
            expected["seed"],
            expected["fold"],
            "oof_val",
            expected["depths"],
        )
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False


def _train_fold_task(task):
    (
        config_path,
        output_dir,
        stage,
        variant,
        seed,
        spec,
        device_text,
        smoke,
        force,
    ) = task
    config = yaml.safe_load(Path(config_path).read_text())
    depths = _depths(config)
    policies = _policies(config)
    checkpoint_policy = policies["checkpoint_policy"]
    fold = int(spec["fold"])
    effective = _effective_config(config, variant, smoke=smoke)
    run_dir = _training_dir(output_dir, stage, variant, seed, fold)
    expected = {
        "stage": stage,
        "variant": variant,
        "seed": int(seed),
        "fold": fold,
        "effective_config": effective,
        "train_ids": spec["train_ids"],
        "validation_ids": spec["val_ids"],
        "depths": depths,
        "policies": policies,
        "allow_legacy_checkpoint_policy": _allows_legacy_checkpoint_policy(config),
    }
    if not force and _training_complete(run_dir, expected):
        return f"skip {stage} {variant} seed={seed} fold={fold}"

    task_seed = int(seed) * 100 + fold
    _set_seed(task_seed)
    torch.set_num_threads(1)
    device = _resolve_device(device_text)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    embeddings_dir = _repo_path(config["data"]["train_embeddings_dir"])
    metadata_csv = _repo_path(config["data"]["metadata_csv"])
    train = _load_split(embeddings_dir, metadata_csv, spec["train_ids"])
    val = _load_split(embeddings_dir, metadata_csv, spec["val_ids"])
    if (
        train["dim"] != int(effective["input_dim"])
        or val["dim"] != int(effective["input_dim"])
    ):
        raise ValueError("embedding dimension mismatch")

    train_prior, val_prior, prior_state = _fit_prior(train, val, task_seed)
    has_t0 = train["masks"][:, 0] > 0
    if not has_t0.any():
        raise ValueError("shared temporal training fold has no valid T0")
    train_for_loss = _subset_split(train, has_t0)
    prior_for_loss = np.asarray(train_prior)[has_t0]
    train_loader = _loader(
        train_for_loss, prior_for_loss, effective["batch_size"], True, task_seed
    )
    lengths = contiguous_prefix_lengths(train_for_loss["masks"])
    expected_prefix_counts = tuple(
        int((lengths >= depth).sum()) for depth in range(1, 5)
    )
    if expected_prefix_counts[0] != len(train_for_loss["pids"]):
        raise RuntimeError("valid-T0 filter disagrees with contiguous prefix counts")

    model = TDN({"downstream": effective}).to(device)
    optimizer = _make_optimizer(model, effective)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0.0)),
    )
    positives = float((train_for_loss["labels"] == 1).sum())
    negatives = float((train_for_loss["labels"] == 0).sum())
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    best_validation = None
    patience = 0
    history = []
    for epoch in range(int(effective["epochs"])):
        model.train()
        losses = []
        epoch_prefix_counts = np.zeros(4, dtype=np.int64)
        for embeddings, masks, clinical, target, prior_logit, days in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prior_logit = prior_logit.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            (
                prefix_embeddings,
                prefix_masks,
                prefix_days,
                repeated,
                _,
                counts,
            ) = expand_contiguous_torch_temporal_prefixes(
                embeddings, masks, days, clinical, target, prior_logit
            )
            prefix_clinical, prefix_target, prefix_prior = repeated
            epoch_prefix_counts += np.asarray(counts, dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(
                prefix_embeddings,
                prefix_masks,
                prefix_clinical,
                days=prefix_days,
                prior_logit=prefix_prior,
                return_residual=True,
            )
            point_losses = F.binary_cross_entropy_with_logits(
                logits,
                prefix_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = _equal_depth_loss(point_losses, counts)
            residual_weight = float(effective.get("residual_l2_weight", 0.0))
            if residual_weight > 0:
                loss = loss + residual_weight * residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite shared contiguous training loss")
            loss.backward()
            if float(effective.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(effective["grad_clip"])
                )
            optimizer.step()
            losses.append(float(loss.item()))
        if tuple(epoch_prefix_counts.tolist()) != expected_prefix_counts:
            raise RuntimeError("an epoch did not consume each valid patient-depth pair once")
        scheduler.step()

        val_predictions = _prediction_frame(
            model,
            val,
            val_prior,
            depths,
            seed,
            fold,
            "oof_val",
            device,
            effective["batch_size"],
        )
        val_metrics = _metrics_by_depth(val_predictions, depths)
        selection_score = float(np.mean([row["auroc"] for row in val_metrics]))
        if not np.isfinite(selection_score):
            raise RuntimeError("validation produced a non-finite checkpoint score")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "residual_scale": float(model.residual_scale().detach().cpu()),
                "selection_score": selection_score,
                **{
                    f"val_auroc_{row['temporal_depth']}": row["auroc"]
                    for row in val_metrics
                },
                **{
                    f"train_pairs_{depth['name']}": int(epoch_prefix_counts[index])
                    for index, depth in enumerate(depths)
                },
            }
        )
        if checkpoint_policy == FINAL_EPOCH_CHECKPOINT_POLICY:
            best_score = selection_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_validation = val_metrics
            patience = 0
        elif selection_score > best_score + float(effective.get("min_delta", 0.0)):
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
        if (
            checkpoint_policy == BEST_VALIDATION_CHECKPOINT_POLICY
            and patience >= int(effective["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("validation never produced a finite checkpoint")

    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    train_predictions = _prediction_frame(
        model,
        train,
        train_prior,
        depths,
        seed,
        fold,
        "fold_train",
        device,
        effective["batch_size"],
    )
    val_predictions = _prediction_frame(
        model,
        val,
        val_prior,
        depths,
        seed,
        fold,
        "oof_val",
        device,
        effective["batch_size"],
    )
    train_metrics = _metrics_by_depth(train_predictions, depths)
    val_metrics = _metrics_by_depth(val_predictions, depths)
    validation_used_for_selection = (
        checkpoint_policy == BEST_VALIDATION_CHECKPOINT_POLICY
    )
    selection = {
        "criterion": _checkpoint_criterion(checkpoint_policy),
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_epoch_zero_based": int(best_epoch),
        "checkpoint_score": float(best_score),
        "checkpoint_validation_metrics": best_validation,
        "validation_used_for_checkpoint_selection": validation_used_for_selection,
        "early_stopping_enabled": validation_used_for_selection,
        "best_epoch_zero_based": int(best_epoch),
        "best_score": float(best_score),
        "best_validation_metrics": best_validation,
        "epochs_trained": len(history),
    }
    checkpoint = {
        "schema": SCHEMA,
        "stage": stage,
        "variant": variant,
        "seed": int(seed),
        "fold": fold,
        "model_state": best_state,
        "effective_config": effective,
        "clinical_prior": prior_state,
        "selection": selection,
        "train_ids": list(spec["train_ids"]),
        "validation_ids": list(spec["val_ids"]),
        "expected_prefix_counts": list(expected_prefix_counts),
        "test_data_loaded_during_training": False,
        "test_embeddings_or_labels_loaded_during_training": False,
        "validation_used_for_checkpoint_selection": validation_used_for_selection,
        "early_stopping_enabled": validation_used_for_selection,
        **policies,
    }
    _atomic_torch(run_dir / "best.pt", checkpoint)
    _atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
    _atomic_csv(run_dir / "train_predictions.csv", train_predictions)
    _atomic_csv(run_dir / "val_predictions.csv", val_predictions)
    _atomic_json(
        run_dir / "summary.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "variant": variant,
            "seed": int(seed),
            "fold": fold,
            "selection": selection,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "clinical_prior_training_patients": len(train["pids"]),
            "tdn_training_patients": len(train_for_loss["pids"]),
            "expected_prefix_counts": list(expected_prefix_counts),
            "test_role": "not_loaded",
            "validation_used_for_checkpoint_selection": validation_used_for_selection,
            "early_stopping_enabled": validation_used_for_selection,
            **policies,
        },
    )
    _atomic_json(
        run_dir / "TRAINING_COMPLETE.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "variant": variant,
            "seed": int(seed),
            "fold": fold,
            "effective_config": effective,
            "validation_used_for_checkpoint_selection": validation_used_for_selection,
            "early_stopping_enabled": validation_used_for_selection,
            "complete": True,
            **policies,
        },
    )
    return (
        f"done {stage} {variant} seed={seed} fold={fold} "
        f"val={best_score:.4f}"
    )


def _select_variant(summary, variant_order, tolerance):
    table = summary.copy()
    best = float(table["macro_oof_auroc_mean"].max())
    table["within_oof_tolerance"] = (
        table["macro_oof_auroc_mean"] >= best - float(tolerance) - 1e-12
    )
    table["variant_order"] = table["variant"].map(
        {name: index for index, name in enumerate(variant_order)}
    )
    eligible = table[table["within_oof_tolerance"]].copy()
    selected = str(
        eligible.sort_values(
            [
                "positive_train_oof_gap",
                "parameters",
                "macro_oof_auroc_mean",
                "variant_order",
            ],
            ascending=[True, True, False, True],
        ).iloc[0]["variant"]
    )
    table["selected"] = table["variant"] == selected
    return selected, table.sort_values("variant_order")


def _build_oof(config, output_dir, specs, stage, variants, seeds, select, smoke):
    depths = _depths(config)
    policies = _policies(config)
    pool_ids = [patient_id for spec in specs for patient_id in spec["val_ids"]]
    if len(pool_ids) != len(set(pool_ids)):
        raise RuntimeError("fold validation sets are not a disjoint development partition")
    metric_rows = []
    for variant in variants:
        effective = _effective_config(config, variant, smoke=smoke)
        for seed in seeds:
            frames = []
            fold_train = {depth["name"]: [] for depth in depths}
            checkpoint_epochs = []
            for spec in specs:
                fold = int(spec["fold"])
                run_dir = _training_dir(output_dir, stage, variant, seed, fold)
                expected = {
                    "stage": stage,
                    "variant": variant,
                    "seed": int(seed),
                    "fold": fold,
                    "effective_config": effective,
                    "train_ids": spec["train_ids"],
                    "validation_ids": spec["val_ids"],
                    "depths": depths,
                    "policies": policies,
                    "allow_legacy_checkpoint_policy": _allows_legacy_checkpoint_policy(config),
                }
                if not _training_complete(run_dir, expected):
                    raise RuntimeError(f"incomplete training artifact: {run_dir}")
                frames.append(
                    pd.read_csv(run_dir / "val_predictions.csv", dtype={"patient_id": str})
                )
                fold_summary = json.loads((run_dir / "summary.json").read_text())
                for row in fold_summary["train_metrics"]:
                    fold_train[row["temporal_depth"]].append(float(row["auroc"]))
                fold_selection = fold_summary["selection"]
                checkpoint_epochs.append(
                    int(
                        fold_selection.get(
                            "checkpoint_epoch_zero_based",
                            fold_selection["best_epoch_zero_based"],
                        )
                    )
                )
            oof = pd.concat(frames, ignore_index=True)
            if not _valid_oof_prediction_frame(
                oof, pool_ids, seed, depths, len(specs)
            ):
                raise RuntimeError("OOF prediction contract mismatch")
            for depth in depths:
                selected_rows = oof[oof["temporal_depth"] == depth["name"]]
                metrics = compute_metrics(
                    selected_rows["label"], selected_rows["probability"], threshold=0.5
                )
                mean_train = float(np.mean(fold_train[depth["name"]]))
                metric_rows.append(
                    {
                        "stage": stage,
                        "variant": variant,
                        "seed": int(seed),
                        "temporal_depth": depth["name"],
                        "max_tp": int(depth["max_tp"]),
                        "mean_fold_train_auroc": mean_train,
                        "train_oof_gap": mean_train - float(metrics["auroc"]),
                        "checkpoint_policy": policies["checkpoint_policy"],
                        "mean_checkpoint_epoch": float(np.mean(checkpoint_epochs)),
                        "mean_best_epoch": float(np.mean(checkpoint_epochs)),
                        "parameters": _parameter_count(config, variant, smoke=smoke),
                        **metrics,
                    }
                )
            oof_dir = (
                Path(output_dir)
                / "oof"
                / stage
                / f"variant_{variant}"
                / f"seed_{int(seed)}"
            )
            _atomic_csv(oof_dir / "oof_predictions.csv", oof)
            _atomic_json(
                oof_dir / "OOF_COMPLETE.json",
                {
                    "schema": SCHEMA,
                    "stage": stage,
                    "variant": variant,
                    "seed": int(seed),
                    "folds": len(specs),
                    "patients": len(pool_ids),
                    "depths": depths,
                    "complete": True,
                    **policies,
                },
            )

    metrics = pd.DataFrame(metric_rows)
    _atomic_csv(Path(output_dir) / f"{stage}_oof_metrics_all.csv", metrics)
    seed_macro = (
        metrics.groupby(["stage", "variant", "seed"], sort=False)
        .agg(
            macro_oof_auroc=("auroc", "mean"),
            macro_oof_prauc=("prauc", "mean"),
            macro_fold_train_auroc=("mean_fold_train_auroc", "mean"),
            macro_train_oof_gap=("train_oof_gap", "mean"),
            checkpoint_policy=("checkpoint_policy", "first"),
            mean_checkpoint_epoch=("mean_checkpoint_epoch", "first"),
            mean_best_epoch=("mean_best_epoch", "first"),
            parameters=("parameters", "first"),
            depths=("temporal_depth", "count"),
        )
        .reset_index()
    )
    if not (seed_macro["depths"] == len(depths)).all():
        raise RuntimeError("macro OOF score does not contain all temporal depths")
    _atomic_csv(Path(output_dir) / f"{stage}_oof_macro_per_seed.csv", seed_macro)
    summary = (
        seed_macro.groupby("variant", sort=False)
        .agg(
            macro_oof_auroc_mean=("macro_oof_auroc", "mean"),
            macro_oof_auroc_std=("macro_oof_auroc", "std"),
            macro_oof_prauc_mean=("macro_oof_prauc", "mean"),
            macro_fold_train_auroc_mean=("macro_fold_train_auroc", "mean"),
            train_oof_gap_mean=("macro_train_oof_gap", "mean"),
            checkpoint_policy=("checkpoint_policy", "first"),
            mean_checkpoint_epoch=("mean_checkpoint_epoch", "mean"),
            mean_best_epoch=("mean_best_epoch", "mean"),
            parameters=("parameters", "first"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    summary["positive_train_oof_gap"] = summary["train_oof_gap_mean"].clip(
        lower=0
    )
    selection = None
    if select:
        selected, summary = _select_variant(
            summary, variants, _section(config)["selection_tolerance"]
        )
        selection = {
            "schema": SCHEMA,
            "stage": stage,
            "criterion": _section(config)["selection"],
            "selection_tolerance": float(_section(config)["selection_tolerance"]),
            "selected_variant": selected,
            "selected_effective_config": _effective_config(
                config, selected, smoke=smoke
            ),
            "variant_configs": {
                variant: _effective_config(config, variant, smoke=smoke)
                for variant in variants
            },
            "tune_seeds": [int(seed) for seed in seeds],
            "folds": len(specs),
            "test_data_used": False,
            "test_embeddings_or_labels_loaded": False,
            **policies,
        }
        _atomic_json(Path(output_dir) / "selected_variant.json", selection)
    _atomic_csv(Path(output_dir) / f"{stage}_variant_summary.csv", summary)
    return selection


def _validate_selection(config, selection, variants, tune_seeds, folds, smoke):
    policies = _policies(config)
    selected = selection.get("selected_variant")
    return (
        selection.get("schema") == SCHEMA
        and selection.get("stage") == "tune"
        and selection.get("criterion") == SELECTION_CRITERION
        and float(selection.get("selection_tolerance", -1))
        == float(_section(config)["selection_tolerance"])
        and selection.get("test_data_used") is False
        and selection.get("test_embeddings_or_labels_loaded") is False
        and selected in variants
        and int(selection.get("folds", -1)) == int(folds)
        and [int(value) for value in selection.get("tune_seeds", [])]
        == [int(value) for value in tune_seeds]
        and selection.get("variant_configs")
        == {
            variant: _effective_config(config, variant, smoke=smoke)
            for variant in variants
        }
        and selection.get("selected_effective_config")
        == _effective_config(config, selected, smoke=smoke)
        and _artifact_policies_match(
            selection,
            policies,
            _allows_legacy_checkpoint_policy(config),
        )
    )


def _retained_future_tokens(split):
    _, masks, _ = canonicalize_temporal_prefix(
        np.asarray(split["embs"]),
        np.asarray(split["masks"]),
        np.asarray(split["days"]),
        4,
    )
    return int((masks[:, 1:] > 0).sum())


def _ensemble_fold_predictions(fold_predictions, test_ids, depths, folds):
    expected_folds = set(range(int(folds)))
    for depth in depths:
        selected = fold_predictions[
            fold_predictions["temporal_depth"] == depth["name"]
        ]
        if (
            len(selected) != len(test_ids) * int(folds)
            or set(selected["fold"]) != expected_folds
            or not (selected.groupby("patient_id").size() == int(folds)).all()
            or set(selected["patient_id"]) != set(test_ids)
        ):
            raise RuntimeError("fold predictions do not cover the locked test exactly")
    group_columns = [
        "patient_id",
        "label",
        "seed",
        "strategy",
        "temporal_depth",
        "max_tp",
        "source",
    ]
    ensemble = (
        fold_predictions.groupby(group_columns, as_index=False, sort=False)[
            ["probability", "residual_logit", "clinical_prior_probability"]
        ]
        .mean()
    )
    ensemble["fold"] = -1
    ensemble["split"] = "test_fold_ensemble"
    patient_order = {patient_id: index for index, patient_id in enumerate(test_ids)}
    depth_order = {depth["name"]: index for index, depth in enumerate(depths)}
    ensemble["_patient_order"] = ensemble["patient_id"].map(patient_order)
    ensemble["_depth_order"] = ensemble["temporal_depth"].map(depth_order)
    ensemble = ensemble.sort_values(["_depth_order", "_patient_order"]).drop(
        columns=["_depth_order", "_patient_order"]
    )
    return ensemble[
        [
            "patient_id",
            "label",
            "probability",
            "residual_logit",
            "clinical_prior_probability",
            "seed",
            "fold",
            "split",
            "strategy",
            "temporal_depth",
            "max_tp",
            "source",
        ]
    ]


def _evaluation_dir(output_dir, source, seed):
    return Path(output_dir) / "evaluation" / source / f"seed_{int(seed)}"


def _evaluate_task(task):
    (
        config_path,
        output_dir,
        selection,
        source,
        seed,
        specs,
        test_ids,
        device_text,
        smoke,
    ) = task
    config = yaml.safe_load(Path(config_path).read_text())
    depths = _depths(config)
    policies = _policies(config)
    variant = str(selection["selected_variant"])
    effective = _effective_config(config, variant, smoke=smoke)

    # Validate every formal model before this worker loads any test embedding.
    for spec in specs:
        expected = {
            "stage": "formal",
            "variant": variant,
            "seed": int(seed),
            "fold": int(spec["fold"]),
            "effective_config": effective,
            "train_ids": spec["train_ids"],
            "validation_ids": spec["val_ids"],
            "depths": depths,
            "policies": policies,
            "allow_legacy_checkpoint_policy": _allows_legacy_checkpoint_policy(config),
        }
        run_dir = _training_dir(
            output_dir, "formal", variant, seed, spec["fold"]
        )
        if not _training_complete(run_dir, expected):
            raise RuntimeError(f"formal training contract mismatch: {run_dir}")

    test = _load_test_source(config, source, test_ids)
    retained = _retained_future_tokens(test)
    expected_retained = int(config["data"]["expected_contiguous_future_embeddings"])
    if retained != expected_retained:
        raise ValueError(
            f"contiguous future-token count mismatch: expected {expected_retained}, "
            f"found {retained}"
        )
    device = _resolve_device(device_text)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fold_frames = []
    for spec in specs:
        fold = int(spec["fold"])
        checkpoint_path = (
            _training_dir(output_dir, "formal", variant, seed, fold) / "best.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model = TDN({"downstream": effective}).to(device)
        model.load_state_dict(checkpoint["model_state"])
        prior = _prior_from_state(test["clinical"], checkpoint["clinical_prior"])
        frame = _prediction_frame(
            model,
            test,
            prior,
            depths,
            seed,
            fold,
            "test_fold",
            device,
            effective["batch_size"],
        )
        frame["source"] = source
        fold_frames.append(frame)
        del model
    fold_predictions = pd.concat(fold_frames, ignore_index=True)
    ensemble = _ensemble_fold_predictions(
        fold_predictions, test_ids, depths, len(specs)
    )
    run_dir = _evaluation_dir(output_dir, source, seed)
    _atomic_csv(run_dir / "fold_predictions.csv", fold_predictions)
    _atomic_csv(run_dir / "test_predictions.csv", ensemble)
    _atomic_json(
        run_dir / "EVALUATION_COMPLETE.json",
        {
            "schema": SCHEMA,
            "source": source,
            "source_contract": _evaluation_source_contract(config, source),
            "variant": variant,
            "seed": int(seed),
            "folds_ensembled": len(specs),
            "test_ids": list(test_ids),
            "temporal_depths": depths,
            "retained_contiguous_future_tokens": retained,
            "test_used_for_selection": False,
            "complete": True,
            **policies,
        },
    )
    return f"done evaluation {source} seed={seed}"


def _evaluation_complete(
    run_dir, config, source, seed, variant, test_ids, folds
):
    run_dir = Path(run_dir)
    paths = {
        "fold": run_dir / "fold_predictions.csv",
        "ensemble": run_dir / "test_predictions.csv",
        "sentinel": run_dir / "EVALUATION_COMPLETE.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        sentinel = json.loads(paths["sentinel"].read_text())
        fold_frame = pd.read_csv(paths["fold"], dtype={"patient_id": str})
        ensemble = pd.read_csv(paths["ensemble"], dtype={"patient_id": str})
        depths = _depths(config)
        policies = _policies(config)
        if not (
            sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("source") == source
            and sentinel.get("source_contract")
            == _evaluation_source_contract(config, source)
            and sentinel.get("variant") == variant
            and int(sentinel.get("seed", -1)) == int(seed)
            and int(sentinel.get("folds_ensembled", -1)) == int(folds)
            and sentinel.get("test_ids") == list(test_ids)
            and sentinel.get("temporal_depths") == depths
            and int(sentinel.get("retained_contiguous_future_tokens", -1))
            == int(config["data"]["expected_contiguous_future_embeddings"])
            and sentinel.get("test_used_for_selection") is False
            and _artifact_policies_match(
                sentinel,
                policies,
                _allows_legacy_checkpoint_policy(config),
            )
            and _valid_prediction_frame(
                ensemble,
                test_ids,
                seed,
                -1,
                "test_fold_ensemble",
                depths,
                source=source,
            )
        ):
            return False
        for fold in range(int(folds)):
            selected = fold_frame[fold_frame["fold"] == fold]
            if not _valid_prediction_frame(
                selected,
                test_ids,
                seed,
                fold,
                "test_fold",
                depths,
                source=source,
            ):
                return False
        return len(fold_frame) == len(test_ids) * len(depths) * int(folds)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False


def _aggregate_evaluation(config, output_dir, selection, sources, seeds, test_ids):
    depths = _depths(config)
    folds = int(_section(config)["folds"])
    variant = str(selection["selected_variant"])
    for source in sources:
        frames = []
        metric_rows = []
        for seed in seeds:
            run_dir = _evaluation_dir(output_dir, source, seed)
            if not _evaluation_complete(
                run_dir, config, source, seed, variant, test_ids, folds
            ):
                raise RuntimeError(f"invalid evaluation artifact: {run_dir}")
            frame = pd.read_csv(
                run_dir / "test_predictions.csv", dtype={"patient_id": str}
            )
            frames.append(frame)
            for row in _metrics_by_depth(frame, depths):
                metric_rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "variant": variant,
                        **row,
                    }
                )
        metrics = pd.DataFrame(metric_rows)
        summary_rows = []
        for depth in depths:
            selected = metrics[metrics["temporal_depth"] == depth["name"]]
            row = {
                "source": source,
                "temporal_depth": depth["name"],
                "max_tp": int(depth["max_tp"]),
                "variant": variant,
                "n_seeds": len(seeds),
                "folds_per_seed": folds,
            }
            for metric in METRIC_KEYS:
                values = selected[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_std"] = (
                    float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                )
            summary_rows.append(row)
        source_dir = Path(output_dir) / "evaluation" / source
        _atomic_csv(
            source_dir / "all_test_predictions.csv",
            pd.concat(frames, ignore_index=True),
        )
        _atomic_csv(source_dir / "metrics_per_seed.csv", metrics)
        _atomic_csv(source_dir / "summary.csv", pd.DataFrame(summary_rows))


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
        "--config",
        default="configs/mewm_ispy2_full978_locked102_shared_contiguous_cv.yaml",
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="train")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--tune-seeds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    section = _section(config)
    _depths(config)
    policies = _policies(config)
    if section["selection"] != SELECTION_CRITERION:
        raise SystemExit("selection criterion does not match the runner contract")
    if float(section["selection_tolerance"]) != 0.005:
        raise SystemExit("formal shared contiguous CV requires 0.005 tolerance")

    declared_variants = list(config["variants"])
    declared_tune_seeds = [int(seed) for seed in section["tune_seeds"]]
    declared_seeds = [int(seed) for seed in section["formal_seeds"]]
    declared_sources = list(config["evaluation_sources"])
    variants = args.variants or declared_variants
    tune_seeds = args.tune_seeds or (
        [declared_tune_seeds[0]] if args.smoke else declared_tune_seeds
    )
    seeds = args.seeds or ([declared_seeds[0]] if args.smoke else declared_seeds)
    sources = args.sources or declared_sources
    if (
        len(variants) != len(set(variants))
        or set(variants) - set(declared_variants)
        or len(tune_seeds) != len(set(tune_seeds))
        or set(tune_seeds) - set(declared_tune_seeds)
        or len(seeds) != len(set(seeds))
        or set(seeds) - set(declared_seeds)
        or len(sources) != len(set(sources))
        or set(sources) - set(declared_sources)
    ):
        raise SystemExit("variants, seeds, and sources must be unique and declared")
    if not args.smoke and (
        variants != declared_variants
        or tune_seeds != declared_tune_seeds
        or seeds != declared_seeds
        or sources != declared_sources
    ):
        raise SystemExit("formal runs require every declared variant, seed, and source")

    output_dir = _repo_path(args.output_dir or section["output_dir"])
    canonical_output = _repo_path(section["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = canonical_output / "smoke"
    if args.smoke and output_dir == canonical_output:
        raise SystemExit("smoke runs may not write to the formal output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, pool_ids, test_ids, fold_audit = _fold_specs(config)
    jobs = int(args.jobs or section.get("jobs", 10))
    if jobs < 1:
        raise SystemExit("jobs must be positive")
    if len(specs) != 5 or len(pool_ids) != 876 or len(test_ids) != 102:
        raise SystemExit("formal cohort must be 876 development / 102 locked test / 5 folds")
    _write_fold_manifests(output_dir, specs, config, fold_audit)
    _atomic_text(output_dir / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=True))
    _atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema": SCHEMA,
            "config": str(config_path),
            "phase": args.phase,
            "development_patients": len(pool_ids),
            "locked_test_patients": len(test_ids),
            "temporal_depths": _depths(config),
            "variants": variants,
            "tune_seeds": tune_seeds,
            "formal_seeds": seeds,
            "sources": sources,
            "folds": len(specs),
            "jobs": jobs,
            "parallel_start_method": "spawn",
            "test_policy": "loaded_only_by_evaluate_after_oof_selection_and_formal_training",
            "smoke": bool(args.smoke),
            **policies,
        },
    )

    selection = None
    if args.phase in ("train", "all"):
        tune_tasks = []
        for variant in variants:
            for seed in tune_seeds:
                for spec in specs:
                    tune_tasks.append(
                        (
                            _train_fold_task,
                            str(config_path),
                            str(output_dir),
                            "tune",
                            variant,
                            int(seed),
                            spec,
                            args.device,
                            bool(args.smoke),
                            bool(args.force),
                        )
                    )
        _run_tasks(tune_tasks, jobs, "tune")
        selection = _build_oof(
            config,
            output_dir,
            specs,
            "tune",
            variants,
            tune_seeds,
            True,
            bool(args.smoke),
        )
        selected = str(selection["selected_variant"])
        formal_tasks = []
        for seed in seeds:
            for spec in specs:
                formal_tasks.append(
                    (
                        _train_fold_task,
                        str(config_path),
                        str(output_dir),
                        "formal",
                        selected,
                        int(seed),
                        spec,
                        args.device,
                        bool(args.smoke),
                        bool(args.force),
                    )
                )
        _run_tasks(formal_tasks, jobs, "formal")
        _build_oof(
            config,
            output_dir,
            specs,
            "formal",
            [selected],
            seeds,
            False,
            bool(args.smoke),
        )
        _atomic_json(
            output_dir / "TRAINING_PHASE_COMPLETE.json",
            {
                "schema": SCHEMA,
                "selected_variant": selected,
                "tune_checkpoints": len(variants) * len(tune_seeds) * len(specs),
                "formal_checkpoints": len(seeds) * len(specs),
                "test_data_loaded": False,
                "test_embeddings_or_labels_loaded": False,
                "complete": True,
                **policies,
            },
        )

    if args.phase in ("evaluate", "all"):
        if selection is None:
            selection_path = output_dir / "selected_variant.json"
            if not selection_path.is_file():
                raise SystemExit("evaluate requires completed OOF variant selection")
            selection = json.loads(selection_path.read_text())
        if not _validate_selection(
            config, selection, variants, tune_seeds, len(specs), bool(args.smoke)
        ):
            raise SystemExit("selected variant does not match the experiment contract")
        tasks = []
        for source in sources:
            for seed in seeds:
                tasks.append(
                    (
                        _evaluate_task,
                        str(config_path),
                        str(output_dir),
                        selection,
                        source,
                        int(seed),
                        specs,
                        test_ids,
                        args.device,
                        bool(args.smoke),
                    )
                )
        _run_tasks(tasks, jobs, "evaluate")
        _aggregate_evaluation(
            config, output_dir, selection, sources, seeds, test_ids
        )
        _atomic_json(
            output_dir / "EXPERIMENT_COMPLETE.json",
            {
                "schema": SCHEMA,
                "selected_variant": selection["selected_variant"],
                "formal_seeds": seeds,
                "folds_per_seed": len(specs),
                "evaluation_sources": sources,
                "locked_test_patients": len(test_ids),
                "test_used_for_selection": False,
                "complete": True,
                **policies,
            },
        )
        print(f"complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

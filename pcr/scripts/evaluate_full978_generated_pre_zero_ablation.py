from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.evaluate_full978_frozen_mixed_biflow_trajectories as frozen_runner
from scripts.evaluate_generated_dce0_full978 import (
    _atomic_csv,
    _atomic_json,
    _atomic_torch_save,
    _cohort_inputs,
    _predict_checkpoint,
    _prefetched,
    _registered_timepoints,
    _repo_path,
)
from scripts.extract_pillar_mewm import validate_embedding
from scripts.run_experiments import _resolve_device
from src.data import EmbStore, load_all_splits
from src.metrics import METRIC_KEYS, compute_metrics
from src.mewm_data import build_registered_volume, create_registered_source, patient_key


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pillar_full978_generated_pre_zero_ablation_v1"
ZERO_SOURCE = "zero_pre"
REFERENCE_SOURCES = ("real", *frozen_runner.EXPECTED_STRATEGIES)
EXPECTED_SOURCES = (*REFERENCE_SOURCES, ZERO_SOURCE)
DISPLAY_NAMES = {
    "real": "Real",
    "direct_t0": "BiFlow direct-T0",
    "rollout_generated_dce0_real_ser": "BiFlow rollout",
    ZERO_SOURCE: "Zero-pre",
}


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    config["_config_path"] = str(config_path)
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    zero = config.get("zero_pre", {})
    reference = config.get("reference", {})
    bootstrap = config.get("evaluation", {}).get("paired_patient_bootstrap", {})
    if not str(config.get("frozen_config", "")):
        raise ValueError("frozen_config is required")
    if tuple(reference.get("sources", ())) != REFERENCE_SOURCES:
        raise ValueError("reference sources changed")
    if (
        int(zero.get("expected_patients", -1)) != 102
        or int(zero.get("expected_future_embeddings", -1)) != 296
        or zero.get("expected_target_counts") != {"T1": 101, "T2": 98, "T3": 97}
        or int(zero.get("batch_size", -1)) != 6
    ):
        raise ValueError("zero-pre extraction contract changed")
    expected_comparisons = [
        ["direct_t0", ZERO_SOURCE],
        ["rollout_generated_dce0_real_ser", ZERO_SOURCE],
        ["real", ZERO_SOURCE],
    ]
    if (
        int(bootstrap.get("repetitions", -1)) != 10_000
        or bootstrap.get("stratify_by_label") is not True
        or bootstrap.get("comparisons") != expected_comparisons
    ):
        raise ValueError("paired bootstrap contract changed")
    output_dir = _repo_path(config["evaluation"]["output_dir"])
    reference_dir = _repo_path(reference["result_dir"])
    if output_dir == reference_dir:
        raise ValueError("zero-pre output must not overwrite the reference evaluation")


def _frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return frozen_runner._load_config(config["frozen_config"])


def zero_pre_channel(volume: torch.Tensor) -> torch.Tensor:
    """Return a copy with only Pillar channel 0 replaced by zeros."""
    if volume.ndim != 5 or tuple(volume.shape[:2]) != (1, 3) or not bool(
        torch.isfinite(volume).all()
    ):
        raise ValueError("zero-pre requires one finite Pillar input volume")
    output = volume.clone()
    output[:, 0].zero_()
    if not torch.equal(output[:, 1:], volume[:, 1:]) or bool(output[:, 0].count_nonzero()):
        raise RuntimeError("zero-pre changed a postcontrast channel")
    return output


def _availability_frame(test_ids: Sequence[str], by_key: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for patient_id in test_ids:
        metadata = by_key[patient_key(patient_id)]
        for timepoint in _registered_timepoints(metadata.registered_timepoints):
            rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": f"T{timepoint}",
                    "timepoint_index": timepoint,
                    "pcr_input_available": True,
                    "pre_source": "real_dce0" if timepoint == 0 else "all_zero",
                    "post_early_source": "real",
                    "post_late_source": "real",
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("timepoint").size().astype(int).to_dict()
    if len(frame) != 398 or counts != {"T0": 102, "T1": 101, "T2": 98, "T3": 97}:
        raise ValueError("zero-pre availability does not match locked102")
    return frame


def _future_tasks(
    test_ids: Sequence[str], by_key: Mapping[str, Any], output_dir: Path
) -> list[tuple[str, int, Mapping[str, Any], Path]]:
    tasks = []
    for patient_id in test_ids:
        metadata = by_key[patient_key(patient_id)]
        row = metadata._asdict()
        for timepoint in _registered_timepoints(metadata.registered_timepoints):
            if timepoint == 0:
                continue
            destination = output_dir / patient_id / f"{patient_id}_T{timepoint}.pt"
            tasks.append((patient_id, timepoint, row, destination))
    counts = Counter(timepoint for _, timepoint, _, _ in tasks)
    if len(tasks) != 296 or counts != {1: 101, 2: 98, 3: 97}:
        raise ValueError("zero-pre future visit inventory changed")
    return tasks


def _build_zero_pre_volume(
    task: tuple[str, int, Mapping[str, Any], Path], source: Any
) -> torch.Tensor:
    patient_id, timepoint, metadata, _ = task
    members, policy = source.phase_selection(patient_id, timepoint, metadata)
    if members is None or policy != "metadata_pre_early_late" or len(members) != 3:
        raise ValueError("zero-pre requires metadata-selected pre/early/late phases")
    volume = build_registered_volume(source, patient_id, timepoint, metadata)
    if volume is None:
        raise ValueError(f"registered visit unexpectedly unavailable: {patient_id}:T{timepoint}")
    if tuple(volume.shape) != (1, 3, 384, 384, 192):
        raise ValueError("registered visit violates the Pillar input shape")
    return zero_pre_channel(volume)


def _batches(values: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _extract_batch(model: Any, volumes: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    batch = torch.cat(list(volumes), dim=0)
    expected = (len(volumes), 3, 384, 384, 192)
    if tuple(batch.shape) != expected or not bool(torch.isfinite(batch).all()):
        raise ValueError("batched zero-pre Pillar input is invalid")
    with torch.inference_mode():
        embeddings = model.extract_vision_feats({"breast_mr": batch.to(device)})
    embeddings = embeddings.float().cpu()
    if tuple(embeddings.shape) != (len(volumes), 1152) or not bool(
        torch.isfinite(embeddings).all()
    ):
        raise ValueError("Pillar returned invalid batched zero-pre embeddings")
    return embeddings


def extract_embeddings(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    limit: int,
    prefetch_workers: int,
    dry_run: bool,
    force: bool,
    batch_size: int,
) -> Path:
    frozen = _frozen_config(config)
    base = frozen_runner._base_config(frozen)
    adapter, cohort_dir, ids, _, by_key = _cohort_inputs(base)
    test_ids = ids["test"]
    output_dir = _repo_path(config["zero_pre"]["embeddings_dir"])
    availability = _availability_frame(test_ids, by_key)
    all_tasks = _future_tasks(test_ids, by_key, output_dir)
    source = create_registered_source(adapter)
    contract = {
        "schema": SCHEMA,
        "ablation": "future pre channel set to zero after standard Pillar preprocessing",
        "cohort_dir": str(cohort_dir),
        "test_patient_count": 102,
        "future_embedding_count": 296,
        "target_counts": {"T1": 101, "T2": 98, "T3": 97},
        "real_t0_reused_at_evaluation": True,
        "postcontrast": "unchanged real metadata-selected early and late phases",
        "pillar_preprocessing": "1mm, per-channel p1-p99 clip/min-max, center pad/crop; then pre=0",
        "pillar_model": "YalaLab/Pillar0-BreastMRI",
        "pillar_revision": str(adapter.get("model_revision", "main")),
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = output_dir / "extraction_contract.json"
        if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("zero-pre extraction contract changed")
        _atomic_json(contract_path, contract)
        _atomic_csv(output_dir / "availability.csv", availability)

    pending = []
    skipped = 0
    for task in all_tasks:
        destination = task[-1]
        if destination.is_file() and not force and not dry_run:
            validate_embedding(destination)
            skipped += 1
        else:
            pending.append(task)
    if limit:
        pending = pending[:limit]

    model = None
    if not dry_run:
        model = frozen_runner.load_pillar(
            device, revision=str(adapter.get("model_revision", "main"))
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    built = _prefetched(
        pending,
        lambda task: _build_zero_pre_volume(task, source),
        prefetch_workers,
    )
    processed = 0
    for group in tqdm(
        _batches(built, batch_size),
        total=math.ceil(len(pending) / batch_size),
        desc="zero-pre Pillar",
    ):
        tasks = [item[0] for item in group]
        volumes = [item[1] for item in group]
        if dry_run:
            for task, volume in zip(tasks, volumes):
                print(
                    f"zero-pre {task[0]} T{task[1]}: shape={tuple(volume.shape)} "
                    f"pre_nonzero={int(volume[:, 0].count_nonzero())} "
                    f"early=[{float(volume[:, 1].min()):.4f},{float(volume[:, 1].max()):.4f}] "
                    f"late=[{float(volume[:, 2].min()):.4f},{float(volume[:, 2].max()):.4f}]"
                )
        else:
            embeddings = _extract_batch(model, volumes, device)
            for task, embedding in zip(tasks, embeddings):
                _atomic_torch_save(task[-1], embedding)
        processed += len(group)
        del volumes

    if dry_run:
        print(f"dry-run validated {processed} zero-pre volume(s)")
        return output_dir

    expected = {(task[0], task[1]) for task in all_tasks}
    actual = {
        (path.parent.name, int(path.stem.rsplit("_T", 1)[1]))
        for path in output_dir.glob("*/*.pt")
    }
    complete = skipped + processed == len(all_tasks) and actual == expected
    peak_gib = (
        float(torch.cuda.max_memory_allocated(device)) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    _atomic_json(
        output_dir / "extraction_summary.json",
        {
            "schema": SCHEMA,
            "complete": complete,
            "expected_future_embeddings": 296,
            "processed_future_embeddings": processed,
            "validated_existing_future_embeddings": skipped,
            "real_t0_embeddings_reused_at_evaluation": 102,
            "batch_size": batch_size,
            "peak_model_allocated_memory_gib": peak_gib,
        },
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not complete and not limit:
        raise RuntimeError("zero-pre embedding extraction is incomplete")
    return output_dir


def _overlay_zero_pre(original: Mapping[str, Any], embedding_dir: Path) -> dict[str, Any]:
    result = copy.deepcopy(original)
    store = EmbStore(str(embedding_dir))
    expected = {
        (str(original["pids"][index]), timepoint)
        for index in range(len(original["pids"]))
        for timepoint in range(1, 4)
        if original["masks"][index, timepoint] == 1
    }
    actual = {
        (path.parent.name, int(path.stem.rsplit("_T", 1)[1]))
        for path in embedding_dir.glob("*/*.pt")
    }
    if actual != expected or len(actual) != 296:
        raise ValueError("zero-pre embedding inventory does not match real future visits")
    pid_index = {str(pid): index for index, pid in enumerate(original["pids"])}
    for patient_id, timepoint in sorted(expected):
        result["embs"][pid_index[patient_id], timepoint] = store.load(
            patient_id, timepoint, require_finite=True
        )
    for key in ("masks", "days", "clinical", "labels"):
        if not np.array_equal(result[key], original[key]):
            raise ValueError(f"zero-pre overlay changed {key}")
    if not np.array_equal(result["embs"][:, 0], original["embs"][:, 0]):
        raise ValueError("zero-pre overlay changed T0")
    return result


def _validate_fold_partition(
    training_sets: Sequence[set[str]], validation_sets: Sequence[set[str]]
) -> None:
    development = set().union(*validation_sets)
    if (
        len(development) != 876
        or any(
            validation_sets[i] & validation_sets[j]
            for i in range(5)
            for j in range(i + 1, 5)
        )
        or any(training_sets[index] | validation_sets[index] != development for index in range(5))
    ):
        raise ValueError("five-fold development partition changed")


def _zero_predictions(
    config: Mapping[str, Any], frozen: Mapping[str, Any], *, device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frozen_runner._base_config(frozen)
    adapter, cohort_dir, ids, _, _ = _cohort_inputs(base)
    embedding_dir = _repo_path(config["zero_pre"]["embeddings_dir"])
    extraction = json.loads((embedding_dir / "extraction_summary.json").read_text())
    if (
        extraction.get("schema") != SCHEMA
        or extraction.get("complete") is not True
        or int(extraction.get("expected_future_embeddings", -1)) != 296
    ):
        raise ValueError("zero-pre embeddings are incomplete")
    real = load_all_splits(
        _repo_path(adapter["embeddings_dir"]),
        cohort_dir / "metadata_enriched.csv",
        ids["train"],
        ids["val"],
        ids["test"],
    )["test"]
    zero = _overlay_zero_pre(real, embedding_dir)
    rows = []
    replay_rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for spec in frozen["frozen_models"]["depths"]:
        name = str(spec["name"])
        split = frozen_runner._canonical_split(zero, int(spec["max_tp"]))
        for seed in frozen["frozen_models"]["seeds"]:
            fold_probabilities = []
            training_sets = []
            validation_sets = []
            for fold, checkpoint_path in enumerate(frozen_runner._checkpoint_paths(spec, seed)):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                frozen_runner._validate_checkpoint(checkpoint, checkpoint_path, spec, seed, fold)
                if int(spec["folds"]) == 5:
                    train_ids = set(checkpoint["train_ids"])
                    validation_ids = set(checkpoint["validation_ids"])
                    if train_ids & validation_ids or (train_ids | validation_ids) & set(ids["test"]):
                        raise ValueError("five-fold checkpoint patient boundary changed")
                    training_sets.append(train_ids)
                    validation_sets.append(validation_ids)
                fold_probabilities.append(_predict_checkpoint(checkpoint, split, device))
                del checkpoint
            if int(spec["folds"]) == 5:
                _validate_fold_partition(training_sets, validation_sets)
            probabilities = np.mean(np.stack(fold_probabilities, axis=0), axis=0)
            if name == "T0":
                baseline = frozen_runner._baseline_probabilities(
                    spec, int(seed), list(real["pids"]), real["labels"]
                )
                error = float(np.max(np.abs(probabilities.astype(float) - baseline)))
                if error > 5e-4:
                    raise ValueError(f"zero-pre T0 replay failed for seed {seed}: {error}")
                replay_rows.append({"seed": int(seed), "max_absolute_probability_error": error})
            rows.append(
                pd.DataFrame(
                    {
                        "patient_id": real["pids"],
                        "label": real["labels"].astype(int),
                        "probability": probabilities,
                        "seed": int(seed),
                        "split": "test",
                        "source": ZERO_SOURCE,
                        "temporal_depth": name,
                        "max_tp": int(spec["max_tp"]),
                        "protocol": spec["protocol"],
                        "folds_ensembled": int(spec["folds"]),
                        "variant": spec["variant"],
                        "residual_l2_weight": 0.0,
                    }
                )
            )
            print(f"evaluated zero-pre {name} seed={seed}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(replay_rows)


def _reference_predictions(
    config: Mapping[str, Any], frozen: Mapping[str, Any]
) -> pd.DataFrame:
    result_dir = _repo_path(config["reference"]["result_dir"])
    complete = json.loads((result_dir / "EXPERIMENT_COMPLETE.json").read_text())
    if (
        complete.get("schema") != frozen_runner.SUMMARY_SCHEMA
        or complete.get("complete") is not True
        or complete.get("sources") != list(REFERENCE_SOURCES)
        or int(complete.get("prediction_rows", -1)) != 12_240
    ):
        raise ValueError("reference trajectory evaluation contract changed")
    frame = pd.read_csv(result_dir / "all_test_predictions.csv", dtype={"patient_id": str})
    expected_depths = [name for name, _ in frozen_runner.EXPECTED_DEPTHS]
    expected_seeds = [int(value) for value in frozen["frozen_models"]["seeds"]]
    if (
        len(frame) != 12_240
        or set(frame["source"]) != set(REFERENCE_SOURCES)
        or set(frame["temporal_depth"]) != set(expected_depths)
        or set(frame["seed"].astype(int)) != set(expected_seeds)
        or set(frame.groupby(["source", "temporal_depth", "seed"]).size()) != {102}
        or frame.groupby("patient_id")["label"].nunique().max() != 1
        or not np.isfinite(frame["probability"]).all()
    ):
        raise ValueError("reference trajectory predictions are incomplete")
    return frame


def _metric_frames(predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    order = {name: index for index, name in enumerate(EXPECTED_SOURCES)}
    depth_order = {name: index for index, (name, _) in enumerate(frozen_runner.EXPECTED_DEPTHS)}
    predictions = predictions.assign(
        _source_order=predictions["source"].map(order),
        _depth_order=predictions["temporal_depth"].map(depth_order),
    ).sort_values(["_source_order", "_depth_order", "seed", "patient_id"])
    for (source, depth, seed), group in predictions.groupby(
        ["source", "temporal_depth", "seed"], sort=False
    ):
        first = group.iloc[0]
        rows.append(
            {
                "source": source,
                "temporal_depth": depth,
                "max_tp": int(first["max_tp"]),
                "seed": int(seed),
                "protocol": first["protocol"],
                "folds_per_seed": int(first["folds_ensembled"]),
                "variant": first["variant"],
                **compute_metrics(group["label"], group["probability"], threshold=threshold),
            }
        )
    metrics = pd.DataFrame(rows)
    return metrics, frozen_runner._summary_frame(metrics)


def _paired_seed_deltas(
    metrics: pd.DataFrame, comparisons: Sequence[Sequence[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = metrics.set_index(["source", "temporal_depth", "seed"])
    rows = []
    for depth, _ in frozen_runner.EXPECTED_DEPTHS:
        for seed in range(42, 52):
            for left, right in comparisons:
                a = keyed.loc[(left, depth, seed)]
                b = keyed.loc[(right, depth, seed)]
                rows.append(
                    {
                        "temporal_depth": depth,
                        "seed": seed,
                        "comparison": f"{left}_minus_{right}",
                        **{metric: float(a[metric] - b[metric]) for metric in METRIC_KEYS},
                    }
                )
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for (depth, comparison), group in per_seed.groupby(
        ["temporal_depth", "comparison"], sort=False
    ):
        row = {"temporal_depth": depth, "comparison": comparison, "n_seeds": len(group)}
        for metric in METRIC_KEYS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def paired_patient_bootstrap(
    predictions: pd.DataFrame,
    comparisons: Sequence[Sequence[str]],
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient = (
        predictions.groupby(["source", "temporal_depth", "patient_id", "label"], as_index=False)[
            "probability"
        ]
        .mean()
        .sort_values(["temporal_depth", "source", "patient_id"])
    )
    labels = (
        patient[["patient_id", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if len(labels) != 102 or labels["label"].value_counts().to_dict() != {0: 70, 1: 32}:
        raise ValueError("bootstrap cohort changed")
    patient_ids = labels["patient_id"].tolist()
    y = labels["label"].to_numpy(dtype=int)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(int(seed))
    draws = np.concatenate(
        [
            rng.choice(negative, size=(repetitions, len(negative)), replace=True),
            rng.choice(positive, size=(repetitions, len(positive)), replace=True),
        ],
        axis=1,
    )
    probability = {
        (source, depth): group.set_index("patient_id").loc[patient_ids, "probability"].to_numpy()
        for (source, depth), group in patient.groupby(["source", "temporal_depth"], sort=False)
    }
    sample_rows = []
    summary_rows = []
    for depth, _ in frozen_runner.EXPECTED_DEPTHS:
        for left, right in comparisons:
            left_prob = probability[(left, depth)]
            right_prob = probability[(right, depth)]
            observed_left_auc = float(roc_auc_score(y, left_prob))
            observed_right_auc = float(roc_auc_score(y, right_prob))
            observed_left_ap = float(average_precision_score(y, left_prob))
            observed_right_ap = float(average_precision_score(y, right_prob))
            auc_deltas = np.empty(repetitions, dtype=np.float64)
            ap_deltas = np.empty(repetitions, dtype=np.float64)
            for index, draw in enumerate(draws):
                sampled_y = y[draw]
                auc_deltas[index] = roc_auc_score(sampled_y, left_prob[draw]) - roc_auc_score(
                    sampled_y, right_prob[draw]
                )
                ap_deltas[index] = average_precision_score(
                    sampled_y, left_prob[draw]
                ) - average_precision_score(sampled_y, right_prob[draw])
            comparison = f"{left}_minus_{right}"
            sample_rows.append(
                pd.DataFrame(
                    {
                        "temporal_depth": depth,
                        "comparison": comparison,
                        "bootstrap_index": np.arange(repetitions, dtype=int),
                        "auroc_delta": auc_deltas,
                        "prauc_delta": ap_deltas,
                    }
                )
            )
            summary_rows.append(
                {
                    "temporal_depth": depth,
                    "comparison": comparison,
                    "patient_count": len(y),
                    "pcr_count": int(y.sum()),
                    "bootstrap_repetitions": repetitions,
                    "left_auroc": observed_left_auc,
                    "right_auroc": observed_right_auc,
                    "auroc_delta": observed_left_auc - observed_right_auc,
                    "auroc_delta_ci_low": float(np.quantile(auc_deltas, 0.025)),
                    "auroc_delta_ci_high": float(np.quantile(auc_deltas, 0.975)),
                    "left_prauc": observed_left_ap,
                    "right_prauc": observed_right_ap,
                    "prauc_delta": observed_left_ap - observed_right_ap,
                    "prauc_delta_ci_low": float(np.quantile(ap_deltas, 0.025)),
                    "prauc_delta_ci_high": float(np.quantile(ap_deltas, 0.975)),
                }
            )
    return patient, pd.concat(sample_rows, ignore_index=True), pd.DataFrame(summary_rows)


def _write_report(
    path: Path, summary: pd.DataFrame, bootstrap_summary: pd.DataFrame
) -> None:
    lines = [
        "# Future Pre Zero Ablation",
        "",
        "Future T1-T3 pre/DCE0 is zero. T0, real early/late, clinical features,",
        "elapsed days, masks, patient membership, and frozen pCR checkpoints are unchanged.",
        "Values below are ten-seed mean +/- sample standard deviation.",
        "",
        "| Source | Depth | Models/seed | AUROC | PR-AUC | Accuracy | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES[str(row.source)]} | {row.temporal_depth} | "
            f"{int(row.folds_per_seed)} | {row.auroc_mean:.5f} +/- {row.auroc_std:.5f} | "
            f"{row.prauc_mean:.5f} +/- {row.prauc_std:.5f} | "
            f"{row.acc_mean:.5f} +/- {row.acc_std:.5f} | "
            f"{row.sens_mean:.5f} +/- {row.sens_std:.5f} | "
            f"{row.spec_mean:.5f} +/- {row.spec_std:.5f} | "
            f"{row.bacc_mean:.5f} +/- {row.bacc_std:.5f} |"
        )
    lines.extend(
        [
            "",
            "Patient-level comparison after averaging the ten seed probabilities; 95% CIs use",
            "10,000 paired bootstrap resamples stratified by pCR label.",
            "",
            "| Depth | Comparison | AUROC delta (95% CI) | PR-AUC delta (95% CI) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in bootstrap_summary.itertuples(index=False):
        lines.append(
            f"| {row.temporal_depth} | {row.comparison} | {row.auroc_delta:.5f} "
            f"[{row.auroc_delta_ci_low:.5f}, {row.auroc_delta_ci_high:.5f}] | "
            f"{row.prauc_delta:.5f} [{row.prauc_delta_ci_low:.5f}, "
            f"{row.prauc_delta_ci_high:.5f}] |"
        )
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def evaluate(config: Mapping[str, Any], *, device: torch.device) -> Path:
    frozen = _frozen_config(config)
    zero, replay = _zero_predictions(config, frozen, device=device)
    reference = _reference_predictions(config, frozen)
    if set(zero["patient_id"]) != set(reference["patient_id"]):
        raise ValueError("zero-pre and reference patient cohorts differ")
    predictions = pd.concat([reference, zero], ignore_index=True)
    threshold = float(config["evaluation"]["threshold"])
    metrics, summary = _metric_frames(predictions, threshold)
    comparisons = config["evaluation"]["paired_patient_bootstrap"]["comparisons"]
    deltas, delta_summary = _paired_seed_deltas(metrics, comparisons)
    bootstrap = config["evaluation"]["paired_patient_bootstrap"]
    patient, bootstrap_samples, bootstrap_summary = paired_patient_bootstrap(
        predictions,
        comparisons,
        repetitions=int(bootstrap["repetitions"]),
        seed=int(bootstrap["seed"]),
    )
    output_dir = _repo_path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "all_test_predictions.csv", predictions)
    _atomic_csv(output_dir / "metrics_per_seed.csv", metrics)
    _atomic_csv(output_dir / "summary.csv", summary)
    _atomic_csv(output_dir / "paired_deltas_per_seed.csv", deltas)
    _atomic_csv(output_dir / "paired_delta_summary.csv", delta_summary)
    _atomic_csv(output_dir / "patient_ensemble_predictions.csv", patient)
    _atomic_csv(output_dir / "paired_patient_bootstrap_samples.csv", bootstrap_samples)
    _atomic_csv(output_dir / "paired_patient_bootstrap_summary.csv", bootstrap_summary)
    _atomic_csv(output_dir / "zero_pre_t0_replay.csv", replay)
    _write_report(output_dir / "report.md", summary, bootstrap_summary)

    expected_rows = 102 * 10 * 4 * len(EXPECTED_SOURCES)
    if (
        len(predictions) != expected_rows
        or len(metrics) != 10 * 4 * len(EXPECTED_SOURCES)
        or set(predictions.groupby(["source", "temporal_depth", "seed"]).size()) != {102}
        or predictions["patient_id"].nunique() != 102
        or predictions["label"].value_counts().to_dict() != {0: expected_rows * 70 // 102, 1: expected_rows * 32 // 102}
        or not np.isfinite(predictions["probability"]).all()
        or len(bootstrap_samples) != int(bootstrap["repetitions"]) * 12
    ):
        raise RuntimeError("zero-pre evaluation output is incomplete")
    t0 = predictions[predictions["temporal_depth"] == "T0"].pivot_table(
        index=["patient_id", "seed"], columns="source", values="probability"
    )
    if float(t0.max(axis=1).sub(t0.min(axis=1)).max()) > 5e-4:
        raise RuntimeError("T0 predictions changed across future-pre sources")
    _atomic_json(
        output_dir / "EXPERIMENT_COMPLETE.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "config": str(config["_config_path"]),
            "test_patients": 102,
            "test_labels": {"pCR": 32, "non_pCR": 70},
            "sources": list(EXPECTED_SOURCES),
            "seeds": list(range(42, 52)),
            "pcr_models_retrained": False,
            "biflow_rerun": False,
            "shuffled_pre_evaluated": False,
            "zeroed_channels": ["future_pre"],
            "unchanged_inputs": [
                "T0",
                "future_early",
                "future_late",
                "clinical",
                "elapsed_days",
                "visit_masks",
            ],
            "prediction_rows": len(predictions),
            "metric_rows": len(metrics),
            "bootstrap_repetitions": int(bootstrap["repetitions"]),
            "bootstrap_stratified_by_label": True,
            "maximum_zero_pre_t0_replay_error": float(
                replay["max_absolute_probability_error"].max()
            ),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate the locked102 future pre-channel zero ablation."
    )
    parser.add_argument(
        "--config",
        default="configs/mewm_ispy2_full978_locked102_generated_pre_zero_ablation.yaml",
    )
    parser.add_argument("--phase", choices=("extract", "evaluate", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.prefetch_workers < 1:
        raise SystemExit("limit must be non-negative and prefetch-workers must be positive")
    if args.dry_run and args.phase == "evaluate":
        raise SystemExit("--dry-run is only valid for extraction")
    config = _load_config(args.config)
    batch_size = int(config["zero_pre"]["batch_size"] if args.batch_size is None else args.batch_size)
    if batch_size < 1:
        raise SystemExit("batch-size must be positive")
    device = _resolve_device(args.device)
    if args.phase in ("extract", "all"):
        extract_embeddings(
            config,
            device=device,
            limit=args.limit,
            prefetch_workers=args.prefetch_workers,
            dry_run=args.dry_run,
            force=args.force,
            batch_size=batch_size,
        )
    if args.phase in ("evaluate", "all") and not args.dry_run and args.limit == 0:
        evaluate(config, device=device)


if __name__ == "__main__":
    main()

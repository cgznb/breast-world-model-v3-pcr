from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_generated_dce0_full978 import (
    _atomic_csv,
    _atomic_json,
    _atomic_torch_save,
    _cohort_inputs,
    _load_world_assets,
    _predict_checkpoint,
    _prefetched,
    _registered_timepoints,
    _repo_path,
)
from scripts.extract_pillar_mewm import validate_embedding
from scripts.run_experiments import _resolve_device
from src.data import EmbStore, TABULAR_FEATURE_NAMES, load_all_splits
from src.generated_dce0 import (
    build_hybrid_pillar_volume,
    generated_roi_to_native,
    native_geometry_from_meta,
    native_valid_foreground_roi,
)
from src.metrics import METRIC_KEYS, compute_metrics
from src.mewm_data import create_registered_source, patient_key
from src.pillar import global_embedding, load_pillar
from src.temporal import canonicalize_temporal_prefix


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCHEMA = "pillar_full978_frozen_mixed_biflow_trajectory_evaluation_v1"
EXPECTED_DEPTHS = (("T0", 1), ("T0-T1", 2), ("T0-T2", 3), ("T0-T3", 4))
EXPECTED_STRATEGIES = ("direct_t0", "rollout_generated_dce0_real_ser")
DISPLAY_NAMES = {
    "real": "Real",
    "direct_t0": "BiFlow direct-T0",
    "rollout_generated_dce0_real_ser": "BiFlow rollout",
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
    trajectory = config.get("trajectory", {})
    strategies = trajectory.get("strategies", {})
    models = config.get("frozen_models", {})
    evaluation = config.get("evaluation", {})
    depths = models.get("depths", [])
    seeds = models.get("seeds", [])
    if tuple(strategies) != EXPECTED_STRATEGIES:
        raise ValueError("trajectory strategies must be direct-T0 then rollout")
    if [(item.get("name"), item.get("max_tp")) for item in depths] != list(EXPECTED_DEPTHS):
        raise ValueError("frozen model depths changed")
    if list(seeds) != list(range(42, 52)):
        raise ValueError("formal seeds must be 42-51")
    if evaluation.get("sources") != ["real", *EXPECTED_STRATEGIES]:
        raise ValueError("evaluation sources changed")
    for index, spec in enumerate(depths):
        expected_folds = 1 if index < 2 else 5
        expected_protocol = (
            "fixed_train_validation_split"
            if expected_folds == 1
            else "five_fold_probability_ensemble"
        )
        if (
            int(spec.get("folds", -1)) != expected_folds
            or spec.get("protocol") != expected_protocol
            or float(spec.get("residual_l2_weight", math.nan)) != 0.0
        ):
            raise ValueError(f"frozen model policy changed for {spec.get('name')}")
        template = str(spec.get("checkpoint_template", ""))
        if "{seed}" not in template or (expected_folds == 5) != ("{fold}" in template):
            raise ValueError(f"checkpoint template changed for {spec.get('name')}")


def _base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _repo_path(config["base_config"])
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or "mewm_adapter" not in value:
        raise ValueError("base full978 config is invalid")
    return value


def _trajectory_root(config: Mapping[str, Any]) -> Path:
    return _repo_path(config["trajectory"]["root"])


def _decoded_path(root: Path, strategy: str, patient_id: str, target_stage: int) -> Path:
    return root / strategy / "decoded" / patient_id / f"T{target_stage}.pt"


def _load_decoded_payload(
    path: Path,
    *,
    strategy: str,
    patient_id: str,
    source_stage: int,
    target_stage: int,
    decoded_schema: str,
    solver_steps: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prediction = payload.get("prediction") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != decoded_schema
        or payload.get("strategy") != strategy
        or payload.get("patient_id") != patient_id
        or int(payload.get("source_stage", -1)) != source_stage
        or int(payload.get("target_stage", -1)) != target_stage
        or payload.get("source_visit_id") != f"{patient_id}:T{source_stage}"
        or payload.get("target_visit_id") != f"{patient_id}:T{target_stage}"
        or payload.get("target_dce0_read") is not False
        or int(payload.get("solver_steps", -1)) != solver_steps
        or not isinstance(prediction, torch.Tensor)
        or prediction.dtype != torch.float16
        or tuple(prediction.shape) != (1, 96, 256, 256)
        or not bool(torch.isfinite(prediction).all())
    ):
        raise ValueError(f"invalid BiFlow trajectory decoded payload: {path}")
    return payload


def _trajectory_index(
    config: Mapping[str, Any], strategy: str, test_ids: list[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    section = config["trajectory"]
    root = _trajectory_root(config)
    overall = json.loads((root / "generation_summary.json").read_text())
    expected_counts = {
        key: int(value) for key, value in section["expected_target_counts"].items()
    }
    if (
        overall.get("schema") != section["schema"]
        or overall.get("complete") is not True
        or overall.get("locked_full_run") is not True
        or overall.get("strategies") != list(EXPECTED_STRATEGIES)
        or int(overall.get("patient_count", -1)) != int(section["expected_patients"])
        or int(overall.get("future_target_count_per_strategy", -1))
        != int(section["expected_future_targets"])
        or int(overall.get("shared_real_t0_target_count", -1))
        != int(section["shared_real_t0_targets"])
        or overall.get("shared_real_t0_condition_and_tensor_equal") is not True
    ):
        raise ValueError("BiFlow trajectory generation summary changed")

    strategy_summary = json.loads((root / strategy / "summary.json").read_text())
    if (
        strategy_summary.get("schema") != section["schema"]
        or strategy_summary.get("complete") is not True
        or strategy_summary.get("strategy") != strategy
        or int(strategy_summary.get("decoded_count", -1))
        != int(section["expected_future_targets"])
        or strategy_summary.get("target_counts") != expected_counts
        or strategy_summary.get("target_dce0_read_for_current_generation_step") is not False
    ):
        raise ValueError(f"BiFlow trajectory summary changed for {strategy}")

    routes = pd.read_csv(root / strategy / "routes.csv", dtype={"patient_id": str})
    required = {
        "strategy",
        "patient_id",
        "source_stage",
        "target_stage",
        "source_visit_id",
        "target_visit_id",
        "seed",
        "route_id",
    }
    if required - set(routes.columns):
        raise ValueError(f"trajectory routes are missing columns for {strategy}")
    if (
        len(routes) != int(section["expected_future_targets"])
        or not routes["strategy"].eq(strategy).all()
        or routes[["patient_id", "target_stage"]].duplicated().any()
        or set(routes["patient_id"]) != set(test_ids)
        or not routes["target_stage"].isin([1, 2, 3]).all()
        or not (routes["source_stage"] < routes["target_stage"]).all()
    ):
        raise ValueError(f"trajectory route inventory changed for {strategy}")
    actual_counts = Counter(f"T{value}" for value in routes["target_stage"].astype(int))
    if dict(actual_counts) != expected_counts:
        raise ValueError(f"trajectory target counts changed for {strategy}")

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in routes.to_dict("records"):
        patient_id = str(row["patient_id"])
        source_stage = int(row["source_stage"])
        target_stage = int(row["target_stage"])
        if (
            row["source_visit_id"] != f"{patient_id}:T{source_stage}"
            or row["target_visit_id"] != f"{patient_id}:T{target_stage}"
        ):
            raise ValueError(f"trajectory route identity changed for {strategy}")
        path = _decoded_path(root, strategy, patient_id, target_stage)
        if not path.is_file():
            raise FileNotFoundError(f"trajectory decoded tensor is missing: {path}")
        result[(patient_id, target_stage)] = {**row, "decoded_path": path}
    return result


def _availability_frame(
    test_ids: list[str],
    by_key: Mapping[str, Any],
    index: Mapping[tuple[str, int], Mapping[str, Any]],
    strategy: str,
) -> pd.DataFrame:
    rows = []
    for patient_id in test_ids:
        row = by_key[patient_key(patient_id)]
        for timepoint in _registered_timepoints(row.registered_timepoints):
            key = (patient_id, timepoint)
            if timepoint > 0 and key not in index:
                raise ValueError(f"trajectory target is missing: {patient_id} T{timepoint}")
            rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": f"T{timepoint}",
                    "timepoint_index": timepoint,
                    "pcr_input_available": True,
                    "pre_source": "real_dce0" if timepoint == 0 else strategy,
                    "post_early_source": "real",
                    "post_late_source": "real",
                    "decoded_path": "" if timepoint == 0 else str(index[key]["decoded_path"]),
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("timepoint")["pcr_input_available"].sum().astype(int).to_dict()
    if len(frame) != 398 or counts != {"T0": 102, "T1": 101, "T2": 98, "T3": 97}:
        raise ValueError("trajectory availability does not match locked102")
    return frame


def _build_hybrid_volume(
    task: tuple[str, int, Mapping[str, Any], Path, Path],
    *,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    by_key: Mapping[str, Any],
    source: Any,
    loaded: Any,
    roi_cache: Any,
    crop_plans: Mapping[str, Any],
    dce0_mean: float,
    dce0_std: float,
    strict_spacing: tuple[float, float, float],
) -> torch.Tensor:
    patient_id, timepoint, route, decoded_path, _ = task
    row = by_key[patient_key(patient_id)]
    members, policy = source.phase_selection(patient_id, timepoint, row._asdict())
    if members is None or policy != "metadata_pre_early_late":
        raise ValueError("full978 target phase selection changed")
    real_pre, spacing = source.load(patient_id, timepoint, members[0])
    visit = source.visits[(patient_key(patient_id), timepoint)]
    meta = json.loads(Path(visit.meta_path).read_text())
    native = native_geometry_from_meta(meta, real_pre.shape)
    if not np.allclose(native.spacing_zyx, spacing, rtol=0, atol=1e-5):
        raise ValueError("registered source and metadata spacing differ")

    payload = _load_decoded_payload(
        decoded_path,
        strategy=str(route["strategy"]),
        patient_id=patient_id,
        source_stage=int(route["source_stage"]),
        target_stage=timepoint,
        decoded_schema=str(config["trajectory"]["decoded_schema"]),
        solver_steps=int(config["trajectory"]["solver_steps"]),
    )
    crop_plan = crop_plans[patient_id]
    target_visit = loaded.visits.get(f"{patient_id}:T{timepoint}")
    if target_visit is None:
        valid_foreground = native_valid_foreground_roi(
            real_pre,
            crop_plan,
            native,
            strict_spacing_xyz=strict_spacing,
        )
    else:
        prepared = roi_cache.load(target_visit)
        if prepared.metadata["crop"] != crop_plan:
            raise ValueError("target foreground and crop-plan identities differ")
        valid_foreground = prepared.valid_foreground[0].numpy()
    generated_native = generated_roi_to_native(
        payload["prediction"][0].float().numpy(),
        valid_foreground,
        crop_plan,
        native,
        dce0_mean=dce0_mean,
        dce0_std=dce0_std,
        strict_spacing_xyz=strict_spacing,
    )
    return build_hybrid_pillar_volume(
        source, patient_id, timepoint, row._asdict(), generated_native
    )


def _same_decoded_target(
    config: Mapping[str, Any],
    direct_route: Mapping[str, Any],
    rollout_route: Mapping[str, Any],
) -> bool:
    patient_id = str(rollout_route["patient_id"])
    target_stage = int(rollout_route["target_stage"])
    if int(rollout_route["source_stage"]) != 0:
        return False
    direct = _load_decoded_payload(
        Path(direct_route["decoded_path"]),
        strategy="direct_t0",
        patient_id=patient_id,
        source_stage=0,
        target_stage=target_stage,
        decoded_schema=str(config["trajectory"]["decoded_schema"]),
        solver_steps=int(config["trajectory"]["solver_steps"]),
    )
    rollout = _load_decoded_payload(
        Path(rollout_route["decoded_path"]),
        strategy="rollout_generated_dce0_real_ser",
        patient_id=patient_id,
        source_stage=0,
        target_stage=target_stage,
        decoded_schema=str(config["trajectory"]["decoded_schema"]),
        solver_steps=int(config["trajectory"]["solver_steps"]),
    )
    return torch.equal(direct["prediction"], rollout["prediction"])


def _extraction_task(
    patient_id: str,
    timepoint: int,
    route: Mapping[str, Any],
    destination: Path,
) -> tuple[str, int, Mapping[str, Any], Path, Path]:
    """Keep the immutable decoded input separate from the embedding output."""
    return patient_id, timepoint, route, Path(route["decoded_path"]), destination


def extract_embeddings(
    config: Mapping[str, Any],
    *,
    strategies: tuple[str, ...],
    device: torch.device,
    limit: int,
    prefetch_workers: int,
    dry_run: bool,
    force: bool,
) -> dict[str, Path]:
    base_config = _base_config(config)
    adapter, cohort_dir, ids, _, by_key = _cohort_inputs(base_config)
    test_ids = ids["test"]
    indexes = {
        strategy: _trajectory_index(config, strategy, test_ids)
        for strategy in EXPECTED_STRATEGIES
    }
    shared_rollout = {
        key
        for key, route in indexes["rollout_generated_dce0_real_ser"].items()
        if int(route["source_stage"]) == 0
    }
    if len(shared_rollout) != int(config["trajectory"]["shared_real_t0_targets"]):
        raise ValueError("shared real-T0 trajectory count changed")

    world = base_config["generated_dce0_test"]
    crop_plans = json.loads(_repo_path(world["crop_plans_json"]).read_text())
    normalization = json.loads(_repo_path(world["normalization_json"]).read_text())
    dce0_stats = normalization.get("channels", {}).get("dce0", {})
    dce0_mean = float(dce0_stats.get("mean", math.nan))
    dce0_std = float(dce0_stats.get("std", math.nan))
    if not math.isfinite(dce0_mean) or not math.isfinite(dce0_std) or dce0_std <= 0:
        raise ValueError("BiFlow DCE0 normalization changed")
    strict_spacing = tuple(float(value) for value in world["strict_spacing_xyz"])
    source = create_registered_source(adapter)
    loaded, roi_cache = _load_world_assets(world)
    model = None if dry_run else load_pillar(
        device, revision=str(adapter.get("model_revision", "main"))
    )
    output_paths = {
        strategy: _repo_path(config["trajectory"]["strategies"][strategy]["embeddings_dir"])
        for strategy in strategies
    }

    for strategy in strategies:
        index = indexes[strategy]
        output_dir = output_paths[strategy]
        availability = _availability_frame(test_ids, by_key, index, strategy)
        contract = {
            "schema": SUMMARY_SCHEMA,
            "strategy": strategy,
            "trajectory_root": str(_trajectory_root(config)),
            "cohort_dir": str(cohort_dir),
            "test_patient_count": 102,
            "future_embedding_count": 296,
            "target_counts": {"T1": 101, "T2": 98, "T3": 97},
            "real_t0_reused_at_evaluation": True,
            "postcontrast": "unchanged real metadata-selected early and late phases",
            "native_regridding": "metadata physical geometry, linear image and nearest support",
            "pillar_preprocessing": "1mm, per-channel p1-p99 clip/min-max, center pad/crop",
            "pillar_model": "YalaLab/Pillar0-BreastMRI",
            "pillar_revision": str(adapter.get("model_revision", "main")),
            "roi_outside_policy": "zero; no real pre pixels retained outside generator ROI",
        }
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            contract_path = output_dir / "extraction_contract.json"
            if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
                raise ValueError(f"embedding extraction contract changed: {output_dir}")
            _atomic_json(contract_path, contract)
            _atomic_csv(output_dir / "availability.csv", availability)

        tasks: list[tuple[str, int, Mapping[str, Any], Path, Path]] = []
        skipped = copied = 0
        for key, route in sorted(index.items()):
            patient_id, timepoint = key
            destination = output_dir / patient_id / f"{patient_id}_T{timepoint}.pt"
            if destination.is_file() and not force and not dry_run:
                validate_embedding(destination)
                skipped += 1
                continue
            if (
                strategy == "rollout_generated_dce0_real_ser"
                and key in shared_rollout
                and not dry_run
            ):
                direct_route = indexes["direct_t0"][key]
                direct_embedding = _repo_path(
                    config["trajectory"]["strategies"]["direct_t0"]["embeddings_dir"]
                ) / patient_id / f"{patient_id}_T{timepoint}.pt"
                if direct_embedding.is_file():
                    validate_embedding(direct_embedding)
                    if not _same_decoded_target(config, direct_route, route):
                        raise ValueError("shared real-T0 decoded tensors differ between strategies")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(
                        destination.suffix + f".tmp.{os.getpid()}"
                    )
                    shutil.copyfile(direct_embedding, temporary)
                    os.replace(temporary, destination)
                    validate_embedding(destination)
                    copied += 1
                    continue
            tasks.append(_extraction_task(patient_id, timepoint, route, destination))
        if limit:
            tasks = tasks[:limit]

        def build(task):
            return _build_hybrid_volume(
                task,
                config=config,
                base_config=base_config,
                by_key=by_key,
                source=source,
                loaded=loaded,
                roi_cache=roi_cache,
                crop_plans=crop_plans,
                dce0_mean=dce0_mean,
                dce0_std=dce0_std,
                strict_spacing=strict_spacing,
            )

        processed = 0
        for task, volume in tqdm(
            _prefetched(tasks, build, prefetch_workers),
            total=len(tasks),
            desc=f"{strategy} Pillar",
        ):
            patient_id, timepoint, _, _, destination = task
            if tuple(volume.shape) != (1, 3, 384, 384, 192) or not bool(
                torch.isfinite(volume).all()
            ):
                raise ValueError("hybrid Pillar input is invalid")
            if dry_run:
                print(
                    f"{strategy} {patient_id} T{timepoint}: "
                    f"shape={tuple(volume.shape)} pre=[{float(volume[:, 0].min()):.4f},"
                    f"{float(volume[:, 0].max()):.4f}]"
                )
            else:
                embedding = global_embedding(model, volume.to(device))
                if tuple(embedding.shape) != (1152,) or not bool(
                    torch.isfinite(embedding).all()
                ):
                    raise ValueError("Pillar returned an invalid trajectory embedding")
                _atomic_torch_save(destination, embedding)
            processed += 1

        if dry_run:
            print(f"dry-run validated {processed} {strategy} hybrid volumes")
            continue
        complete = skipped + copied + processed == len(index)
        actual = {
            (path.parent.name, int(path.stem.rsplit("_T", 1)[1]))
            for path in output_dir.glob("*/*.pt")
        }
        if actual != set(index):
            complete = False
        _atomic_json(
            output_dir / "extraction_summary.json",
            {
                "schema": SUMMARY_SCHEMA,
                "strategy": strategy,
                "expected_future_embeddings": len(index),
                "processed_future_embeddings": processed,
                "copied_tensor_equal_direct_embeddings": copied,
                "validated_existing_future_embeddings": skipped,
                "real_t0_embeddings_reused_at_evaluation": 102,
                "complete": complete,
            },
        )
        if not complete:
            raise RuntimeError(f"trajectory embedding extraction is incomplete: {strategy}")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_paths


def _overlay_generated_test(
    original: Mapping[str, Any], embedding_dir: Path
) -> dict[str, Any]:
    generated = copy.deepcopy(original)
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
        raise ValueError("trajectory embedding inventory does not match real future visits")
    pid_index = {str(pid): index for index, pid in enumerate(original["pids"])}
    for patient_id, timepoint in sorted(expected):
        generated["embs"][pid_index[patient_id], timepoint] = store.load(
            patient_id, timepoint, require_finite=True
        )
    if not (
        np.array_equal(generated["embs"][:, 0], original["embs"][:, 0])
        and np.array_equal(generated["masks"], original["masks"])
        and np.array_equal(generated["days"], original["days"])
        and np.array_equal(generated["clinical"], original["clinical"])
        and np.array_equal(generated["labels"], original["labels"])
    ):
        raise ValueError("trajectory overlay changed T0, masks, days, clinical data, or labels")
    return generated


def _canonical_split(split: Mapping[str, Any], max_tp: int) -> dict[str, Any]:
    """Replay the contiguous T0-starting prefix used by the frozen models."""
    result = dict(split)
    result["embs"], result["masks"], result["days"] = canonicalize_temporal_prefix(
        np.asarray(split["embs"]),
        np.asarray(split["masks"]),
        np.asarray(split["days"]),
        int(max_tp),
    )
    result["clinical"] = np.asarray(split["clinical"]).copy()
    result["labels"] = np.asarray(split["labels"]).copy()
    result["pids"] = list(split["pids"])
    return result


def _checkpoint_paths(spec: Mapping[str, Any], seed: int) -> list[Path]:
    folds = int(spec["folds"])
    template = str(spec["checkpoint_template"])
    return [
        _repo_path(template.format(seed=int(seed), fold=fold))
        for fold in range(folds)
    ]


def _checkpoint_depth(checkpoint: Mapping[str, Any]) -> str:
    return str(
        checkpoint.get("temporal_depth")
        or checkpoint.get("run_context", {}).get("temporal_depth")
        or ""
    )


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    path: Path,
    spec: Mapping[str, Any],
    seed: int,
    fold: int,
) -> None:
    effective = checkpoint.get("effective_config", {})
    actual_fold = checkpoint.get("fold")
    if (
        int(checkpoint.get("seed", -1)) != int(seed)
        or _checkpoint_depth(checkpoint) != spec["name"]
        or effective.get("model_type") != "tdn"
        or int(effective.get("input_dim", -1)) != 1152
        or int(effective.get("clinical_dim", -1)) != len(TABULAR_FEATURE_NAMES)
        or effective.get("use_prior") is not True
        or float(effective.get("residual_l2_weight", 0.0)) != 0.0
        or not path.is_file()
    ):
        raise ValueError(f"frozen checkpoint contract changed: {path}")
    if int(spec["folds"]) == 1:
        if actual_fold is not None or spec["name"] == "T0-T1" and checkpoint.get("variant") != spec["variant"]:
            raise ValueError(f"single-split checkpoint contract changed: {path}")
    elif (
        int(actual_fold if actual_fold is not None else -1) != fold
        or checkpoint.get("variant") != spec["variant"]
        or checkpoint.get("schema") != "pillar_full978_independent_depth_cv_v1"
        or checkpoint.get("test_data_loaded_during_training") is not False
    ):
        raise ValueError(f"five-fold checkpoint contract changed: {path}")


def _baseline_probabilities(
    spec: Mapping[str, Any],
    seed: int,
    patient_ids: list[str],
    labels: np.ndarray,
) -> np.ndarray:
    path = _repo_path(str(spec["baseline_predictions_template"]).format(seed=int(seed)))
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if "temporal_depth" in frame:
        frame = frame[frame["temporal_depth"].astype(str) == str(spec["name"])].copy()
    if (
        len(frame) != len(patient_ids)
        or frame["patient_id"].duplicated().any()
        or set(frame["patient_id"]) != set(patient_ids)
    ):
        raise ValueError(f"baseline prediction cohort changed: {path}")
    indexed = frame.set_index("patient_id").loc[patient_ids]
    if not np.array_equal(indexed["label"].to_numpy(dtype=int), labels.astype(int)):
        raise ValueError(f"baseline prediction labels changed: {path}")
    return indexed["probability"].to_numpy(dtype=float)


def _summary_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, depth, max_tp, protocol, folds), group in metrics.groupby(
        ["source", "temporal_depth", "max_tp", "protocol", "folds_per_seed"],
        sort=False,
    ):
        row = {
            "source": source,
            "temporal_depth": depth,
            "max_tp": int(max_tp),
            "protocol": protocol,
            "folds_per_seed": int(folds),
            "n_seeds": len(group),
        }
        for metric in METRIC_KEYS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_frames(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    comparisons = (
        ("direct_t0", "real"),
        ("rollout_generated_dce0_real_ser", "real"),
        ("rollout_generated_dce0_real_ser", "direct_t0"),
    )
    keyed = metrics.set_index(["source", "temporal_depth", "seed"])
    for depth, _ in EXPECTED_DEPTHS:
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
        row = {
            "temporal_depth": depth,
            "comparison": comparison,
            "n_seeds": len(group),
        }
        for metric in METRIC_KEYS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Frozen Mixed No-Residual BiFlow Trajectory Evaluation",
        "",
        "All values are ten-seed mean +/- sample standard deviation. T0 and T0-T1",
        "use one fixed-split checkpoint per seed. T0-T2 and T0-T3 average five",
        "fold probabilities within each seed. No pCR model was retrained.",
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
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def evaluate(config: Mapping[str, Any], *, device: torch.device) -> Path:
    base_config = _base_config(config)
    adapter, cohort_dir, ids, _, _ = _cohort_inputs(base_config)
    real = load_all_splits(
        _repo_path(adapter["embeddings_dir"]),
        cohort_dir / "metadata_enriched.csv",
        ids["train"],
        ids["val"],
        ids["test"],
    )["test"]
    sources: dict[str, Mapping[str, Any]] = {"real": real}
    for strategy in EXPECTED_STRATEGIES:
        embedding_dir = _repo_path(
            config["trajectory"]["strategies"][strategy]["embeddings_dir"]
        )
        extraction = json.loads((embedding_dir / "extraction_summary.json").read_text())
        if (
            extraction.get("schema") != SUMMARY_SCHEMA
            or extraction.get("strategy") != strategy
            or extraction.get("complete") is not True
            or int(extraction.get("expected_future_embeddings", -1)) != 296
        ):
            raise ValueError(f"trajectory embeddings are incomplete: {strategy}")
        sources[strategy] = _overlay_generated_test(real, embedding_dir)

    seeds = [int(value) for value in config["frozen_models"]["seeds"]]
    depths = list(config["frozen_models"]["depths"])
    threshold = float(config["evaluation"]["threshold"])
    predictions = []
    metric_rows = []
    replay_rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    for spec in depths:
        name = str(spec["name"])
        max_tp = int(spec["max_tp"])
        truncated = {
            source_name: _canonical_split(split, max_tp)
            for source_name, split in sources.items()
        }
        for seed in seeds:
            fold_probabilities = {source_name: [] for source_name in sources}
            fold_validation_sets = []
            fold_training_sets = []
            for fold, checkpoint_path in enumerate(_checkpoint_paths(spec, seed)):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                _validate_checkpoint(checkpoint, checkpoint_path, spec, seed, fold)
                if int(spec["folds"]) == 5:
                    train_ids = set(checkpoint["train_ids"])
                    validation_ids = set(checkpoint["validation_ids"])
                    development_ids = train_ids | validation_ids
                    if train_ids & validation_ids or development_ids & set(ids["test"]):
                        raise ValueError("five-fold checkpoint patient boundary changed")
                    fold_training_sets.append(train_ids)
                    fold_validation_sets.append(validation_ids)
                for source_name, split in truncated.items():
                    fold_probabilities[source_name].append(
                        _predict_checkpoint(checkpoint, split, device)
                    )
                del checkpoint
            if int(spec["folds"]) == 5:
                if (
                    len(set().union(*fold_validation_sets)) != 876
                    or any(
                        fold_validation_sets[i] & fold_validation_sets[j]
                        for i in range(5)
                        for j in range(i + 1, 5)
                    )
                    or any(
                        fold_training_sets[index] | fold_validation_sets[index]
                        != set().union(*fold_validation_sets)
                        for index in range(5)
                    )
                ):
                    raise ValueError("five-fold development partition changed")

            mean_probabilities = {
                source_name: np.mean(np.stack(values, axis=0), axis=0)
                for source_name, values in fold_probabilities.items()
            }
            baseline = _baseline_probabilities(
                spec, seed, list(real["pids"]), real["labels"]
            )
            replay_error = float(
                np.max(np.abs(mean_probabilities["real"].astype(float) - baseline))
            )
            if replay_error > 5e-4:
                raise ValueError(
                    f"real baseline replay failed for {name} seed {seed}: {replay_error}"
                )
            replay_rows.append(
                {
                    "temporal_depth": name,
                    "seed": seed,
                    "max_absolute_probability_error": replay_error,
                }
            )

            for source_name, probabilities in mean_probabilities.items():
                metric = compute_metrics(real["labels"], probabilities, threshold=threshold)
                metric_rows.append(
                    {
                        "source": source_name,
                        "temporal_depth": name,
                        "max_tp": max_tp,
                        "seed": seed,
                        "protocol": spec["protocol"],
                        "folds_per_seed": int(spec["folds"]),
                        "variant": spec["variant"],
                        **metric,
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "patient_id": real["pids"],
                            "label": real["labels"].astype(int),
                            "probability": probabilities,
                            "seed": seed,
                            "split": "test",
                            "source": source_name,
                            "temporal_depth": name,
                            "max_tp": max_tp,
                            "protocol": spec["protocol"],
                            "folds_ensembled": int(spec["folds"]),
                            "variant": spec["variant"],
                            "residual_l2_weight": 0.0,
                        }
                    )
                )
            print(f"evaluated {name} seed={seed}", flush=True)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = _summary_frame(metrics)
    deltas, delta_summary = _delta_frames(metrics)
    replay = pd.DataFrame(replay_rows)
    output_dir = _repo_path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "all_test_predictions.csv", prediction_frame)
    _atomic_csv(output_dir / "metrics_per_seed.csv", metrics)
    _atomic_csv(output_dir / "summary.csv", summary)
    _atomic_csv(output_dir / "paired_deltas_per_seed.csv", deltas)
    _atomic_csv(output_dir / "paired_delta_summary.csv", delta_summary)
    _atomic_csv(output_dir / "real_baseline_replay.csv", replay)
    _write_report(output_dir / "report.md", summary)

    expected_predictions = 102 * len(seeds) * len(depths) * len(sources)
    if (
        len(prediction_frame) != expected_predictions
        or len(metrics) != len(seeds) * len(depths) * len(sources)
        or not np.isfinite(prediction_frame["probability"]).all()
        or prediction_frame["patient_id"].nunique() != 102
        or set(prediction_frame.groupby(["source", "temporal_depth", "seed"]).size())
        != {102}
    ):
        raise RuntimeError("frozen mixed evaluation output is incomplete")
    _atomic_json(
        output_dir / "EXPERIMENT_COMPLETE.json",
        {
            "schema": SUMMARY_SCHEMA,
            "complete": True,
            "config": str(config["_config_path"]),
            "test_patients": 102,
            "test_labels": {"pCR": 32, "non_pCR": 70},
            "seeds": seeds,
            "sources": list(sources),
            "depths": [
                {
                    "name": spec["name"],
                    "max_tp": int(spec["max_tp"]),
                    "protocol": spec["protocol"],
                    "folds_per_seed": int(spec["folds"]),
                    "variant": spec["variant"],
                    "residual_l2_weight": 0.0,
                }
                for spec in depths
            ],
            "pcr_models_retrained": False,
            "test_used_for_model_selection": False,
            "future_embeddings_per_strategy": 296,
            "prediction_rows": len(prediction_frame),
            "metric_rows": len(metrics),
            "threshold": threshold,
            "standard_deviation": "sample_ddof_1",
            "maximum_real_replay_error": float(replay["max_absolute_probability_error"].max()),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate locked102 direct-T0 and rollout BiFlow trajectories."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/mewm_ispy2_full978_locked102_frozen_mixed_no_residual_"
            "biflow_trajectories.yaml"
        ),
    )
    parser.add_argument("--phase", choices=("extract", "evaluate", "all"), default="all")
    parser.add_argument("--strategies", nargs="+", choices=EXPECTED_STRATEGIES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefetch-workers", type=int, default=2)
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
    device = _resolve_device(args.device)
    strategies = tuple(args.strategies or EXPECTED_STRATEGIES)
    if args.phase in ("extract", "all"):
        extract_embeddings(
            config,
            strategies=strategies,
            device=device,
            limit=args.limit,
            prefetch_workers=args.prefetch_workers,
            dry_run=args.dry_run,
            force=args.force,
        )
    if args.phase in ("evaluate", "all") and not args.dry_run and args.limit == 0:
        evaluate(config, device=device)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_full978_table2 import (
    AUXILIARY_METRICS,
    METRIC_COLUMNS,
    METRIC_NAMES,
    PRIMARY_METRICS,
    depth_slug,
)
from scripts.extract_pillar_mewm import validate_embedding
from scripts.run_experiments import _resolve_device
from src.data import EmbStore, TABULAR_FEATURE_NAMES, load_all_splits, load_ids
from src.generated_dce0 import (
    build_hybrid_pillar_volume,
    generated_roi_to_native,
    native_geometry_from_meta,
    native_valid_foreground_roi,
)
from src.metrics import compute_metrics
from src.mewm_data import (
    build_registered_volume,
    create_registered_source,
    patient_key,
)
from src.pillar import global_embedding, load_pillar
from src.tdn import TDN
from src.temporal import truncate_temporal_splits


REPO_ROOT = Path(__file__).resolve().parents[1]
DECODED_SCHEMA = "mewm_ispy2_dce0_biflow_cohort_decoded_v1"
SUPPLEMENTAL_LATENT_SCHEMA = "pillar_full978_biflow_supplemental_latent_v1"
SUPPLEMENTAL_DECODED_SCHEMA = "pillar_full978_biflow_supplemental_decoded_v1"
SUPPLEMENTAL_SUMMARY_SCHEMA = "pillar_full978_biflow_supplemental_summary_v1"
INPUT_VARIANTS = (
    "real_full",
    "real_generation_availability_matched",
    "real_t0_generated_future_dce0_available278",
    "real_t0_generated_future_dce0",
)
PAIR_PATTERN = re.compile(r"^(ISPY2-\d+):T([0-2])->T([1-3])$")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _registered_timepoints(value: Any) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    try:
        result = tuple(int(token[1:]) for token in tokens if re.fullmatch(r"T[0-3]", token))
    except ValueError:
        result = ()
    if len(result) != len(tokens) or not result or len(result) != len(set(result)):
        raise ValueError("registered_timepoints is invalid")
    return result


def _load_adjacent_decoded_index(
    evaluation_dir: Path, test_ids: set[str]
) -> dict[tuple[str, int], Path]:
    summary = json.loads((evaluation_dir / "summary.json").read_text())
    if (
        int(summary.get("solver_steps", -1)) != 20
        or int(summary.get("base_seed", -1)) != 2026
        or summary.get("pair_mode") != "adjacent"
    ):
        raise ValueError("BiFlow decoded directory is not the locked Euler-20 adjacent run")
    result: dict[tuple[str, int], Path] = {}
    for path in sorted((evaluation_dir / "decoded").glob("*/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        match = PAIR_PATTERN.fullmatch(str(payload.get("pair_id", "")))
        prediction = payload.get("prediction")
        if (
            payload.get("schema") != DECODED_SCHEMA
            or match is None
            or payload.get("patient_id") != match.group(1)
            or path.parent.name != match.group(1)
            or int(match.group(3)) != int(match.group(2)) + 1
            or not isinstance(prediction, torch.Tensor)
            or prediction.dtype != torch.float16
            or tuple(prediction.shape) != (1, 96, 256, 256)
            or not bool(torch.isfinite(prediction).all())
        ):
            raise ValueError(f"invalid decoded BiFlow artifact: {path}")
        patient_id = match.group(1)
        key = (patient_id, int(match.group(3)))
        if patient_id not in test_ids or key in result:
            raise ValueError("decoded BiFlow identities do not match the locked test set")
        result[key] = path
    counts = Counter(timepoint for _, timepoint in result)
    if len(result) != 278 or counts != {1: 100, 2: 92, 3: 86}:
        raise ValueError(f"decoded BiFlow inventory changed: {dict(counts)}")
    return result


def _load_supplemental_decoded_index(
    supplemental_dir: Path, test_ids: set[str]
) -> dict[tuple[str, int], Path]:
    summary = json.loads((supplemental_dir / "summary.json").read_text())
    if (
        summary.get("schema") != SUPPLEMENTAL_SUMMARY_SCHEMA
        or int(summary.get("solver_steps", -1)) != 20
        or int(summary.get("base_seed", -1)) != 12026
        or int(summary.get("decoded_count", -1)) != 18
        or summary.get("complete") is not True
    ):
        raise ValueError("supplemental BiFlow generation summary is invalid")
    result: dict[tuple[str, int], Path] = {}
    for path in sorted((supplemental_dir / "decoded").glob("*/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        patient_id = str(payload.get("patient_id", ""))
        target_stage = payload.get("target_stage")
        source_stage = payload.get("source_stage")
        prediction = payload.get("prediction")
        if (
            payload.get("schema") != SUPPLEMENTAL_DECODED_SCHEMA
            or patient_id not in test_ids
            or path.parent.name != patient_id
            or type(source_stage) is not int
            or type(target_stage) is not int
            or not 0 <= source_stage < target_stage <= 3
            or payload.get("pair_id")
            != f"{patient_id}:T{source_stage}->T{target_stage}"
            or not isinstance(prediction, torch.Tensor)
            or prediction.dtype != torch.float16
            or tuple(prediction.shape) != (1, 96, 256, 256)
            or not bool(torch.isfinite(prediction).all())
        ):
            raise ValueError(f"invalid supplemental BiFlow artifact: {path}")
        key = (patient_id, target_stage)
        if key in result:
            raise ValueError("supplemental BiFlow target identities are duplicated")
        result[key] = path
    counts = Counter(timepoint for _, timepoint in result)
    if len(result) != 18 or counts != {1: 1, 2: 6, 3: 11}:
        raise ValueError(f"supplemental BiFlow inventory changed: {dict(counts)}")
    return result


def _load_world_assets(section: Mapping[str, Any]):
    source_repo = _repo_path(section["biflow_repo"])
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    backend = importlib.import_module("mewm_ispy2.backend")
    cache_module = importlib.import_module("mewm_ispy2.cache")
    if not Path(backend.__file__).resolve().is_relative_to(source_repo):
        raise RuntimeError("mewm_ispy2 was imported from the wrong repository")
    bundle = _repo_path(section["bundle_json"])
    phase_manifest = _repo_path(section["phase_manifest_csv"])
    loaded = backend.load_transition_records(
        bundle, phase_manifest, backend="registered_t0"
    )
    roi_cache = cache_module.RegisteredStrictAROICache(
        _repo_path(section["roi_cache_dir"]), bundle_json=bundle
    )
    return loaded, roi_cache


def _import_biflow_module(section: Mapping[str, Any], name: str):
    source_repo = _repo_path(section["biflow_repo"])
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    module = importlib.import_module(f"mewm_ispy2.{name}")
    if not Path(module.__file__).resolve().is_relative_to(source_repo):
        raise RuntimeError("mewm_ispy2 was imported from the wrong repository")
    return module


def _load_locked_bundle_visit_index(section: Mapping[str, Any]):
    backend = _import_biflow_module(section, "backend")
    bundle_path = _repo_path(section["bundle_json"])
    payload = json.loads(bundle_path.read_text())
    visits_path = backend._load_locked_artifact(bundle_path, payload, "visits")
    frame = pd.read_csv(visits_path)
    visit_ids = set(frame["visit_id"].astype(str))
    return backend._strict_a_visit_index(frame, required_visit_ids=visit_ids)


def _prefetched(tasks, build, workers):
    if workers == 1:
        for task in tasks:
            yield task, build(task)
        return
    pending = deque()
    iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            try:
                task = next(iterator)
            except StopIteration:
                break
            pending.append((task, pool.submit(build, task)))
        while pending:
            task, future = pending.popleft()
            try:
                next_task = next(iterator)
            except StopIteration:
                next_task = None
            if next_task is not None:
                pending.append((next_task, pool.submit(build, next_task)))
            yield task, future.result()


def _cohort_inputs(config: Mapping[str, Any]):
    adapter = config["mewm_adapter"]
    cohort_dir = _repo_path(adapter["output_dir"])
    ids = {
        split: load_ids(cohort_dir / "splits" / f"{split}_ids.txt")
        for split in ("train", "val", "test")
    }
    if len(ids["test"]) != 102 or len(set(ids["test"])) != 102:
        raise ValueError("locked full978 test split must contain 102 unique patients")
    metadata = pd.read_csv(cohort_dir / "metadata_enriched.csv", dtype={"pid": str})
    if metadata["pid"].duplicated().any():
        raise ValueError("full978 metadata contains duplicate patient IDs")
    by_key = {patient_key(row.pid): row for row in metadata.itertuples(index=False)}
    return adapter, cohort_dir, ids, metadata, by_key


def _supplemental_routes(
    test_ids: list[str],
    by_key: Mapping[str, Any],
    adjacent: Mapping[tuple[str, int], Path],
    loaded: Any,
    source_visits: Mapping[str, Any],
) -> list[dict[str, Any]]:
    conditions: dict[str, tuple[str, str]] = {}
    for record in loaded.records:
        value = (str(record.clinical_text), str(record.action_text))
        existing = conditions.setdefault(str(record.patient_id), value)
        if existing != value:
            raise ValueError("BiFlow patient text conditions are inconsistent")

    routes = []
    for patient_id in test_ids:
        row = by_key[patient_key(patient_id)]
        for target_stage in _registered_timepoints(row.registered_timepoints):
            if target_stage == 0 or (patient_id, target_stage) in adjacent:
                continue
            candidates = [
                stage
                for stage in range(target_stage)
                if f"{patient_id}:T{stage}" in source_visits
            ]
            if not candidates or patient_id not in conditions:
                raise ValueError(
                    f"no Strict-A source is available for supplemental target "
                    f"{patient_id}:T{target_stage}"
                )
            source_stage = max(candidates)
            source_day = float(getattr(row, f"days_T{source_stage}"))
            target_day = float(getattr(row, f"days_T{target_stage}"))
            delta_days = int(round(target_day - source_day))
            if (
                not math.isfinite(source_day)
                or not math.isfinite(target_day)
                or delta_days <= 0
                or not np.isclose(target_day - source_day, delta_days, atol=1e-5)
            ):
                raise ValueError("supplemental BiFlow elapsed days are invalid")
            clinical_text, treatment_text = conditions[patient_id]
            routes.append(
                {
                    "patient_id": patient_id,
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "source_visit_id": f"{patient_id}:T{source_stage}",
                    "target_visit_id": f"{patient_id}:T{target_stage}",
                    "pair_id": f"{patient_id}:T{source_stage}->T{target_stage}",
                    "transition_type": f"T{source_stage}->T{target_stage}",
                    "delta_days": delta_days,
                    "clinical_text": clinical_text,
                    "treatment_text": treatment_text,
                    "route_policy": "latest_earlier_Strict-A_source",
                    "contract_status": "target_outside_locked_Strict-A_transition_set",
                }
            )
    counts = Counter(route["target_stage"] for route in routes)
    nonadjacent = sum(
        route["target_stage"] != route["source_stage"] + 1 for route in routes
    )
    if len(routes) != 18 or counts != {1: 1, 2: 6, 3: 11} or nonadjacent != 8:
        raise ValueError(
            "supplemental BiFlow route inventory changed: "
            f"counts={dict(counts)}, nonadjacent={nonadjacent}"
        )
    return routes


def _supplemental_artifact_path(
    root: Path, kind: str, route: Mapping[str, Any]
) -> Path:
    patient_id = str(route["patient_id"])
    source_stage = int(route["source_stage"])
    target_stage = int(route["target_stage"])
    return root / kind / patient_id / f"{patient_id}_T{source_stage}-_T{target_stage}.pt"


def _valid_supplemental_latent(
    path: Path, route: Mapping[str, Any], seed: int
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        return False
    latent = payload.get("predicted_normalized_latent") if isinstance(payload, dict) else None
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == SUPPLEMENTAL_LATENT_SCHEMA
        and payload.get("pair_id") == route["pair_id"]
        and payload.get("patient_id") == route["patient_id"]
        and payload.get("source_stage") == route["source_stage"]
        and payload.get("target_stage") == route["target_stage"]
        and payload.get("seed") == seed
        and payload.get("solver_steps") == 20
        and isinstance(latent, torch.Tensor)
        and latent.dtype == torch.float16
        and tuple(latent.shape) == (1, 1, 8, 24, 64, 64)
        and bool(torch.isfinite(latent).all())
    )


def _valid_supplemental_decoded(path: Path, route: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        return False
    prediction = payload.get("prediction") if isinstance(payload, dict) else None
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == SUPPLEMENTAL_DECODED_SCHEMA
        and payload.get("pair_id") == route["pair_id"]
        and payload.get("patient_id") == route["patient_id"]
        and payload.get("source_stage") == route["source_stage"]
        and payload.get("target_stage") == route["target_stage"]
        and isinstance(prediction, torch.Tensor)
        and prediction.dtype == torch.float16
        and tuple(prediction.shape) == (1, 96, 256, 256)
        and bool(torch.isfinite(prediction).all())
    )


def _load_biflow_config_and_latent(section: Mapping[str, Any]):
    config_module = _import_biflow_module(section, "ispy2_biflow_config")
    latent_module = _import_biflow_module(section, "ispy2_dce0_world_latents")
    config = config_module.load_ispy2_biflow_config(
        _repo_path(section["biflow_config"])
    )
    if (
        config.base.model.latent_normalization
        != latent_module.ISPY2_DCE0_CONTINUOUS_NORMALIZATION
        or int(config.runtime.seed) != 2026
    ):
        raise ValueError("supplemental generation requires the locked original BiFlow config")
    latent_cache = latent_module.ISPY2DCE0ContinuousLatentCache(
        config.base.data.continuous_root
    )
    return config, latent_cache


def _load_biflow_system_without_dataset(
    section: Mapping[str, Any], device: torch.device
):
    workflow = _import_biflow_module(section, "ispy2_biflow_workflow")
    training = _import_biflow_module(section, "ispy2_biflow_training")
    config, latent_cache = _load_biflow_config_and_latent(section)
    model = workflow._build_model(config)
    identity = workflow._identity(config, model=model, latent_cache=latent_cache)
    system = training.ISPY2BiFlowTrainingSystem(
        model, config=config, checkpoint_identity=identity
    )
    checkpoint_path = _repo_path(section["biflow_checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    if checkpoint.get("ispy2_biflow_schema") != training.ISPY2_BIFLOW_CHECKPOINT_SCHEMA:
        raise ValueError("supplemental checkpoint is not the locked BiFlow schema")
    system.on_load_checkpoint(checkpoint)
    incompatible = system.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = set(checkpoint["ispy2_biflow_omitted_state_keys"])
    if not set(incompatible.missing_keys) <= expected_missing or incompatible.unexpected_keys:
        raise ValueError("supplemental BiFlow checkpoint restore was incomplete")
    del checkpoint
    return config, system.to(device).eval(), latent_cache


def generate_supplemental_predictions(
    config: Mapping[str, Any], *, device: torch.device, force: bool
) -> Path:
    _, _, ids, _, by_key = _cohort_inputs(config)
    section = config["generated_dce0_test"]
    adjacent = _load_adjacent_decoded_index(
        _repo_path(section["biflow_evaluation_dir"]), set(ids["test"])
    )
    loaded, roi_cache = _load_world_assets(section)
    source_visits = _load_locked_bundle_visit_index(section)
    routes = _supplemental_routes(
        ids["test"], by_key, adjacent, loaded, source_visits
    )
    output = _repo_path(section["biflow_supplemental_dir"])
    route_frame = pd.DataFrame(
        [
            {key: value for key, value in route.items() if not key.endswith("_text")}
            for route in routes
        ]
    )
    _atomic_csv(output / "routes.csv", route_frame)

    base_seed = 12026
    seeds = {route["pair_id"]: base_seed + index for index, route in enumerate(routes)}
    pending = [
        route
        for route in routes
        if force
        or not _valid_supplemental_latent(
            _supplemental_artifact_path(output, "latents", route),
            route,
            seeds[route["pair_id"]],
        )
    ]
    if pending:
        biflow_config, system, latent_cache = _load_biflow_system_without_dataset(
            section, device
        )
        integrate = _import_biflow_module(
            section, "ispy2_biflow_training"
        ).integrate_ispy2_biflow_euler
        model = system.model.eval()
        for index, route in enumerate(pending, start=1):
            source_visit = source_visits[route["source_visit_id"]]
            source_mri = roi_cache.load_source_mri(source_visit).unsqueeze(0).to(device)
            case_seed = seeds[route["pair_id"]]
            generator = torch.Generator(device=device).manual_seed(case_seed)
            noise = torch.randn(
                (1, 1, 8, 24, 64, 64),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = integrate(
                    model,
                    source_mri=source_mri,
                    clinical_text=[route["clinical_text"]],
                    treatment_text=[route["treatment_text"]],
                    delta_days=torch.tensor([route["delta_days"]], device=device),
                    target_stage=torch.tensor([route["target_stage"]], device=device),
                    solver_steps=20,
                    noise=noise,
                )
            if tuple(prediction.shape) != (1, 1, 8, 24, 64, 64) or not bool(
                torch.isfinite(prediction).all()
            ):
                raise ValueError("supplemental BiFlow latent is invalid")
            _atomic_torch_save(
                _supplemental_artifact_path(output, "latents", route),
                {
                    "schema": SUPPLEMENTAL_LATENT_SCHEMA,
                    "pair_id": route["pair_id"],
                    "patient_id": route["patient_id"],
                    "source_stage": route["source_stage"],
                    "target_stage": route["target_stage"],
                    "delta_days": route["delta_days"],
                    "seed": case_seed,
                    "solver_steps": 20,
                    "predicted_normalized_latent": prediction.cpu().half(),
                },
            )
            print(f"supplemental latent {index}/{len(pending)}: {route['pair_id']}")
            del source_mri, noise, prediction
        del model, system
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        biflow_config, latent_cache = _load_biflow_config_and_latent(section)

    pending_decode = [
        route
        for route in routes
        if force
        or not _valid_supplemental_decoded(
            _supplemental_artifact_path(output, "decoded", route), route
        )
    ]
    if pending_decode:
        workflows = _import_biflow_module(section, "workflows")
        vqgan_contract = _import_biflow_module(section, "vqgan")
        latent_contract = _import_biflow_module(
            section, "ispy2_biflow_latent_contract"
        )
        vqgan = (
            workflows.load_mri_vqgan(
                biflow_config.base.data.vqgan_checkpoint,
                expected_numeric_contract=vqgan_contract.REGISTERED_VQGAN_NUMERIC_CONTRACT,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        for index, route in enumerate(pending_decode, start=1):
            latent_payload = torch.load(
                _supplemental_artifact_path(output, "latents", route),
                map_location="cpu",
                weights_only=True,
            )
            normalized = latent_payload["predicted_normalized_latent"].float().to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                decoded = latent_contract.decode_ispy2_biflow_continuous(
                    vqgan, latent_cache.denormalize(normalized)
                )
            prediction = decoded[0, 0].cpu().half()
            if tuple(prediction.shape) != (1, 96, 256, 256) or not bool(
                torch.isfinite(prediction).all()
            ):
                raise ValueError("supplemental decoded BiFlow prediction is invalid")
            _atomic_torch_save(
                _supplemental_artifact_path(output, "decoded", route),
                {
                    "schema": SUPPLEMENTAL_DECODED_SCHEMA,
                    "pair_id": route["pair_id"],
                    "patient_id": route["patient_id"],
                    "source_stage": route["source_stage"],
                    "target_stage": route["target_stage"],
                    "prediction": prediction,
                },
            )
            print(f"supplemental decode {index}/{len(pending_decode)}: {route['pair_id']}")
            del latent_payload, normalized, decoded, prediction
        del vqgan
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    complete = all(
        _valid_supplemental_latent(
            _supplemental_artifact_path(output, "latents", route),
            route,
            seeds[route["pair_id"]],
        )
        and _valid_supplemental_decoded(
            _supplemental_artifact_path(output, "decoded", route), route
        )
        for route in routes
    )
    _atomic_json(
        output / "summary.json",
        {
            "schema": SUPPLEMENTAL_SUMMARY_SCHEMA,
            "base_seed": base_seed,
            "solver_steps": 20,
            "route_count": len(routes),
            "adjacent_route_count": 10,
            "nonadjacent_route_count": 8,
            "latent_count": len(routes),
            "decoded_count": len(routes),
            "target_counts": {"T1": 1, "T2": 6, "T3": 11},
            "target_latent_used_for_generation": False,
            "source_policy": "latest earlier Strict-A-passing real DCE0+SER",
            "contract_scope": (
                "supplemental targets are outside the locked Strict-A transition set"
            ),
            "complete": complete,
        },
    )
    if not complete:
        raise RuntimeError("supplemental BiFlow generation is incomplete")
    return output


def _availability_frame(
    test_ids: list[str],
    by_key: Mapping[str, Any],
    decoded: Mapping[tuple[str, int], Path],
    supplemental_keys: set[tuple[str, int]],
) -> pd.DataFrame:
    rows = []
    for patient_id in test_ids:
        for timepoint in _registered_timepoints(
            getattr(by_key[patient_key(patient_id)], "registered_timepoints")
        ):
            key = (patient_id, timepoint)
            generated = timepoint == 0 or key in decoded
            rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": f"T{timepoint}",
                    "timepoint_index": timepoint,
                    "pcr_input_available": generated,
                    "pre_source": (
                        "real_dce0"
                        if timepoint == 0
                        else "biflow_euler20_supplemental"
                        if key in supplemental_keys
                        else "biflow_euler20"
                        if generated
                        else "unavailable_generation"
                    ),
                    "post_early_source": "real",
                    "post_late_source": "real",
                    "decoded_path": (
                        "" if timepoint == 0 or not generated else str(decoded[(patient_id, timepoint)])
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("timepoint")["pcr_input_available"].agg(["size", "sum"])
    if len(frame) != 398 or counts["sum"].astype(int).to_dict() != {
        "T0": 102,
        "T1": 101,
        "T2": 98,
        "T3": 97,
    }:
        raise ValueError("hybrid input availability no longer matches the audited cohort")
    return frame


def extract_embeddings(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    limit: int,
    prefetch_workers: int,
    dry_run: bool,
    force: bool,
) -> Path:
    adapter, cohort_dir, ids, _, by_key = _cohort_inputs(config)
    section = config["generated_dce0_test"]
    evaluation_dir = _repo_path(section["biflow_evaluation_dir"])
    adjacent = _load_adjacent_decoded_index(evaluation_dir, set(ids["test"]))
    supplemental = _load_supplemental_decoded_index(
        _repo_path(section["biflow_supplemental_dir"]), set(ids["test"])
    )
    if set(adjacent).intersection(supplemental):
        raise ValueError("adjacent and supplemental BiFlow targets overlap")
    decoded = {**adjacent, **supplemental}
    supplemental_keys = set(supplemental)
    availability = _availability_frame(
        ids["test"], by_key, decoded, supplemental_keys
    )
    output_dir = _repo_path(section["embeddings_dir"])
    if not dry_run:
        _atomic_csv(output_dir / "availability.csv", availability)

    crop_plans = json.loads(_repo_path(section["crop_plans_json"]).read_text())
    normalization = json.loads(_repo_path(section["normalization_json"]).read_text())
    dce0_stats = normalization.get("channels", {}).get("dce0", {})
    dce0_mean = float(dce0_stats.get("mean", math.nan))
    dce0_std = float(dce0_stats.get("std", math.nan))
    strict_spacing = tuple(float(value) for value in section["strict_spacing_xyz"])
    source = create_registered_source(adapter)
    loaded, roi_cache = _load_world_assets(section)
    tasks = [
        (patient_id, timepoint, path)
        for (patient_id, timepoint), path in sorted(decoded.items())
    ]
    if any(
        f"{patient_id}:T{timepoint}" not in loaded.visits
        for patient_id, timepoint, _ in tasks
        if (patient_id, timepoint) not in supplemental_keys
    ):
        raise ValueError("generated target visit is absent from the locked Strict-A bundle")

    contract = {
        "schema": "pillar_full978_generated_future_dce0_embeddings_v2",
        "cohort_dir": str(cohort_dir),
        "test_patient_count": len(ids["test"]),
        "real_t0_reused": True,
        "future_embedding_count": len(tasks),
        "future_counts": {
            f"T{timepoint}": sum(task[1] == timepoint for task in tasks)
            for timepoint in (1, 2, 3)
        },
        "generated_pre": (
            "BiFlow Euler-20: 278 locked adjacent plus 18 supplemental targets"
        ),
        "locked_adjacent_embedding_count": len(adjacent),
        "supplemental_embedding_count": len(supplemental),
        "supplemental_route_policy": "latest earlier Strict-A-passing source",
        "supplemental_nonadjacent_route_count": 8,
        "supplemental_contract_scope": "outside locked Strict-A transition set",
        "postcontrast": "unchanged real metadata-selected early and late phases",
        "roi_outside_policy": "zero; no real pre pixels retained outside generator ROI",
        "foreground_policy": (
            "target real-DCE0 FOV support from locked ROI cache or equivalently rebuilt "
            "from target physical geometry"
        ),
        "native_regridding": "metadata physical geometry, linear image and nearest support",
        "pillar_preprocessing": "1mm, per-channel p1-p99 clip/min-max, center pad/crop",
        "pillar_model": "YalaLab/Pillar0-BreastMRI",
        "pillar_revision": str(adapter.get("model_revision", "main")),
        "missing_future_policy": "none; all 296 real future visits have generated pre",
    }
    if not dry_run:
        contract_path = output_dir / "extraction_contract.json"
        if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("generated embedding directory has a different contract")
        _atomic_json(contract_path, contract)

    extraction_tasks = []
    skipped = 0
    reused = 0
    reuse_dir = _repo_path(section["reuse_embeddings_dir"])
    for patient_id, timepoint, path in tasks:
        destination = output_dir / patient_id / f"{patient_id}_T{timepoint}.pt"
        if destination.is_file() and not force and not dry_run:
            validate_embedding(destination)
            skipped += 1
        elif (
            not force
            and not dry_run
            and (reuse_dir / patient_id / f"{patient_id}_T{timepoint}.pt").is_file()
        ):
            source_embedding = reuse_dir / patient_id / f"{patient_id}_T{timepoint}.pt"
            validate_embedding(source_embedding)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
            shutil.copyfile(source_embedding, temporary)
            os.replace(temporary, destination)
            validate_embedding(destination)
            reused += 1
        else:
            extraction_tasks.append((patient_id, timepoint, path, destination))
    if limit:
        extraction_tasks = extraction_tasks[:limit]

    def build(task):
        patient_id, timepoint, decoded_path, _ = task
        row = by_key[patient_key(patient_id)]
        members, policy = source.phase_selection(patient_id, timepoint, row._asdict())
        if members is None or policy != "metadata_pre_early_late":
            raise ValueError("full978 target phase selection changed")
        real_pre, spacing = source.load(patient_id, timepoint, members[0])
        meta = json.loads(Path(source.visits[(patient_key(patient_id), timepoint)].meta_path).read_text())
        native = native_geometry_from_meta(meta, real_pre.shape)
        if not np.allclose(native.spacing_zyx, spacing, rtol=0, atol=1e-5):
            raise ValueError("registered source and metadata spacing differ")
        payload = torch.load(decoded_path, map_location="cpu", weights_only=True)
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
        volume = build_hybrid_pillar_volume(
            source, patient_id, timepoint, row._asdict(), generated_native
        )
        return volume

    model = None if dry_run else load_pillar(
        device, revision=str(adapter.get("model_revision", "main"))
    )
    processed = 0
    for task, volume in tqdm(
        _prefetched(extraction_tasks, build, prefetch_workers),
        total=len(extraction_tasks),
        desc="generated-pre Pillar",
    ):
        patient_id, timepoint, _, destination = task
        if dry_run:
            real = build_registered_volume(
                source,
                patient_id,
                timepoint,
                by_key[patient_key(patient_id)]._asdict(),
                target_spacing=(1.0, 1.0, 1.0),
                target_depth=192,
                target_hw=384,
            )
            if real is None or not torch.equal(volume[:, 1:], real[:, 1:]):
                raise ValueError("hybrid early/late channels differ from the real input path")
            print(
                f"{patient_id} T{timepoint}: {tuple(volume.shape)} "
                f"pre=[{float(volume[:, 0].min()):.4f},{float(volume[:, 0].max()):.4f}]"
            )
        else:
            embedding = global_embedding(model, volume.to(device))
            if tuple(embedding.shape) != (1152,) or not bool(torch.isfinite(embedding).all()):
                raise ValueError("Pillar returned an invalid generated-pre embedding")
            _atomic_torch_save(destination, embedding)
        processed += 1
    if dry_run:
        print(f"dry-run validated {processed} hybrid volume(s)")
        return output_dir

    complete = skipped + reused + processed == len(tasks)
    _atomic_json(
        output_dir / "extraction_summary.json",
        {
            "schema": "pillar_full978_generated_future_dce0_extraction_summary_v2",
            "expected_future_embeddings": len(tasks),
            "processed_future_embeddings": processed,
            "reused_locked_adjacent_embeddings": reused,
            "validated_existing_future_embeddings": skipped,
            "real_t0_embeddings_reused": 102,
            "locked_adjacent_future_embeddings": 278,
            "supplemental_future_embeddings": 18,
            "missing_future_tokens": 0,
            "complete": complete,
        },
    )
    if not complete:
        raise RuntimeError("generated-pre embedding extraction is incomplete")
    return output_dir


def _prior_logits(state: Mapping[str, Any], clinical: np.ndarray) -> np.ndarray:
    if (
        state.get("estimator") != "sklearn.linear_model.LogisticRegression"
        or state.get("feature_names") != list(TABULAR_FEATURE_NAMES)
    ):
        raise ValueError("checkpoint clinical prior contract changed")
    coef = np.asarray(state["coef"], dtype=np.float64)
    intercept = np.asarray(state["intercept"], dtype=np.float64)
    if coef.shape != (1, clinical.shape[1]) or intercept.shape != (1,):
        raise ValueError("checkpoint clinical prior shape is invalid")
    logits = clinical.astype(np.float64) @ coef[0] + intercept[0]
    return np.clip(np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0), -30, 30).astype(
        np.float32
    )


def _predict_checkpoint(
    checkpoint: Mapping[str, Any], split: Mapping[str, Any], device: torch.device
) -> np.ndarray:
    effective = checkpoint.get("effective_config")
    if not isinstance(effective, dict) or effective.get("model_type") != "tdn":
        raise ValueError("checkpoint is not a TDN model")
    model = TDN({"downstream": effective}).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    prior = _prior_logits(checkpoint["clinical_prior"], split["clinical"])
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(split["embs"]).to(device),
            torch.from_numpy(split["masks"]).to(device),
            torch.from_numpy(split["clinical"]).to(device),
            days=torch.from_numpy(split["days"]).to(device),
            prior_logit=torch.from_numpy(prior).to(device),
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.all(
        (probabilities >= 0) & (probabilities <= 1)
    ):
        raise ValueError("checkpoint produced invalid probabilities")
    return probabilities


def _aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = PRIMARY_METRICS + AUXILIARY_METRICS
    for (depth, max_tp, variant), group in per_seed.groupby(
        ["temporal_depth", "max_tp", "input_variant"], sort=False
    ):
        row = {
            "temporal_depth": depth,
            "max_tp": int(max_tp),
            "input_variant": variant,
            "n_seeds": len(group),
        }
        for metric in ordered:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            column = METRIC_COLUMNS[metric]
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_checkpoints(config: Mapping[str, Any], *, device: torch.device) -> Path:
    adapter, cohort_dir, ids, _, _ = _cohort_inputs(config)
    section = config["generated_dce0_test"]
    embedding_dir = _repo_path(section["embeddings_dir"])
    summary = json.loads((embedding_dir / "extraction_summary.json").read_text())
    if summary.get("complete") is not True or int(
        summary.get("expected_future_embeddings", -1)
    ) != 296:
        raise ValueError("generated-pre embeddings are incomplete")
    availability = pd.read_csv(embedding_dir / "availability.csv", dtype={"patient_id": str})
    if len(availability) != 398:
        raise ValueError("generated-pre availability manifest is incomplete")

    original = load_all_splits(
        _repo_path(adapter["embeddings_dir"]),
        cohort_dir / "metadata_enriched.csv",
        ids["train"],
        ids["val"],
        ids["test"],
    )["test"]
    generated = copy.deepcopy(original)
    generated_available = copy.deepcopy(original)
    matched = copy.deepcopy(original)
    store = EmbStore(str(embedding_dir))
    pid_index = {patient_id: index for index, patient_id in enumerate(original["pids"])}
    supplemental = 0
    replaced = 0
    for row in availability.itertuples(index=False):
        index = pid_index[str(row.patient_id)]
        timepoint = int(row.timepoint_index)
        if original["masks"][index, timepoint] != 1:
            raise ValueError("availability row points to a missing real test visit")
        if timepoint == 0:
            continue
        if not bool(row.pcr_input_available):
            raise ValueError("full296 availability contains a missing generated target")
        embedding = store.load(row.patient_id, timepoint, require_finite=True)
        generated["embs"][index, timepoint] = embedding
        replaced += 1
        if row.pre_source == "biflow_euler20_supplemental":
            supplemental += 1
            for split in (generated_available, matched):
                split["embs"][index, timepoint] = 0
                split["masks"][index, timepoint] = 0
                split["days"][index, timepoint] = 0
        else:
            generated_available["embs"][index, timepoint] = embedding
    if replaced != 296 or supplemental != 18:
        raise ValueError("generated-pre replacement accounting changed")
    if not (
        np.array_equal(generated["embs"][:, 0], original["embs"][:, 0])
        and np.array_equal(generated["clinical"], original["clinical"])
        and np.array_equal(generated["labels"], original["labels"])
    ):
        raise ValueError("T0, clinical features, or labels changed")

    output_dir = _repo_path(section["results_dir"])
    baseline_dir = _repo_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames = []
    metric_rows = []
    delta_rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for depth in config["experiment"]["temporal_depths"]:
        name = str(depth["name"])
        max_tp = int(depth["max_tp"])
        variants = {
            "real_full": truncate_temporal_splits({"test": original}, max_tp)["test"],
            "real_generation_availability_matched": truncate_temporal_splits(
                {"test": matched}, max_tp
            )["test"],
            "real_t0_generated_future_dce0_available278": truncate_temporal_splits(
                {"test": generated_available}, max_tp
            )["test"],
            "real_t0_generated_future_dce0": truncate_temporal_splits(
                {"test": generated}, max_tp
            )["test"],
        }
        for seed in config["experiment"]["seeds"]:
            checkpoint_path = (
                baseline_dir
                / depth_slug(name)
                / f"seed_{seed}"
                / "global_tab_temporal"
                / "best.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                int(checkpoint.get("seed", -1)) != int(seed)
                or int(checkpoint.get("effective_config", {}).get("max_tp", -1))
                != max_tp
                or checkpoint.get("selection", {}).get("criterion")
                != "validation_auroc"
            ):
                raise ValueError(f"checkpoint contract mismatch: {checkpoint_path}")
            probabilities = {
                variant: _predict_checkpoint(checkpoint, split, device)
                for variant, split in variants.items()
            }
            saved = pd.read_csv(
                checkpoint_path.parent / "test_predictions.csv",
                dtype={"patient_id": str},
            )
            if (
                saved["patient_id"].tolist() != original["pids"]
                or saved["label"].astype(int).tolist()
                != original["labels"].astype(int).tolist()
                or float(
                    np.max(
                        np.abs(
                            probabilities["real_full"]
                            - saved["probability"].to_numpy(dtype=float)
                        )
                    )
                )
                > 5e-4
            ):
                raise ValueError("checkpoint replay does not reproduce the real test baseline")
            metrics = {}
            for variant in INPUT_VARIANTS:
                metric = compute_metrics(original["labels"], probabilities[variant])
                metrics[variant] = metric
                metric_rows.append(
                    {
                        "temporal_depth": name,
                        "max_tp": max_tp,
                        "seed": int(seed),
                        "input_variant": variant,
                        **metric,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "patient_id": original["pids"],
                            "label": original["labels"].astype(int),
                            "probability": probabilities[variant],
                            "seed": int(seed),
                            "split": "test",
                            "temporal_depth": name,
                            "input_variant": variant,
                            "source_checkpoint": str(checkpoint_path),
                        }
                    )
                )
            for generated_variant, comparator in (
                ("real_t0_generated_future_dce0", "real_full"),
                (
                    "real_t0_generated_future_dce0_available278",
                    "real_generation_availability_matched",
                ),
                (
                    "real_t0_generated_future_dce0",
                    "real_t0_generated_future_dce0_available278",
                ),
            ):
                delta_rows.append(
                    {
                        "temporal_depth": name,
                        "max_tp": max_tp,
                        "seed": int(seed),
                        "comparison": f"{generated_variant}_minus_{comparator}",
                        **{
                            metric: metrics[generated_variant][metric]
                            - metrics[comparator][metric]
                            for metric in PRIMARY_METRICS + AUXILIARY_METRICS
                        },
                    }
                )
            print(
                f"{name} seed {seed}: generated AUROC="
                f"{metrics['real_t0_generated_future_dce0']['auroc']:.4f}"
            )

    per_seed = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    deltas = pd.DataFrame(delta_rows)
    aggregate = _aggregate_metrics(per_seed)
    delta_summary = _aggregate_metrics(
        deltas.rename(columns={"comparison": "input_variant"})
    ).rename(columns={"input_variant": "comparison"})
    _atomic_csv(output_dir / "metrics_per_seed.csv", per_seed)
    _atomic_csv(output_dir / "summary.csv", aggregate)
    _atomic_csv(output_dir / "metric_deltas_per_seed.csv", deltas)
    _atomic_csv(output_dir / "metric_delta_summary.csv", delta_summary)
    _atomic_csv(output_dir / "all_test_predictions.csv", predictions)
    _atomic_json(
        output_dir / "run_summary.json",
        {
            "schema": "pillar_full978_generated_future_dce0_checkpoint_evaluation_v2",
            "test_patient_count": 102,
            "test_patient_set_unchanged": True,
            "pcr_checkpoints_retrained": False,
            "pcr_checkpoint_count": 40,
            "t0_source": "real",
            "generated_future_embeddings": 296,
            "locked_Strict-A_adjacent_generated_embeddings": 278,
            "supplemental_generated_embeddings": 18,
            "supplemental_contract_scope": "outside locked Strict-A transition set",
            "missing_generated_future_tokens": 0,
            "missing_policy": "none for real visits",
            "controls": list(INPUT_VARIANTS),
            "threshold": 0.5,
            "standard_deviation": "sample_ddof_1",
            "primary_metrics": [METRIC_NAMES[key] for key in PRIMARY_METRICS],
            "auxiliary_metrics": [METRIC_NAMES[key] for key in AUXILIARY_METRICS],
            "prediction_rows": len(predictions),
        },
    )
    if len(predictions) != 102 * 40 * len(INPUT_VARIANTS):
        raise RuntimeError("checkpoint evaluation prediction count is incomplete")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate real T0 plus BiFlow-generated future DCE0 with unchanged real "
            "early/late phases using the trained full978 TDN checkpoints."
        )
    )
    parser.add_argument(
        "--config", default="configs/mewm_ispy2_full978_locked102_table2.yaml"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefetch_workers < 1 or args.limit < 0:
        raise SystemExit("prefetch workers must be positive and limit non-negative")
    modes = (args.generate_only, args.extract_only, args.evaluate_only)
    if sum(bool(value) for value in modes) > 1:
        raise SystemExit("generate-only, extract-only, and evaluate-only are mutually exclusive")
    if args.generate_only and args.dry_run:
        raise SystemExit("generate-only cannot be combined with dry-run")
    with _repo_path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config.get("generated_dce0_test"), dict):
        raise SystemExit("config is missing generated_dce0_test")
    device = _resolve_device(args.device)
    if not args.evaluate_only:
        if not args.dry_run:
            supplemental = generate_supplemental_predictions(
                config, device=device, force=args.force
            )
            print(f"supplemental generation -> {supplemental}")
        if args.generate_only:
            return
        extract_embeddings(
            config,
            device=device,
            limit=args.limit,
            prefetch_workers=args.prefetch_workers,
            dry_run=args.dry_run,
            force=args.force,
        )
    if not args.extract_only and not args.dry_run and args.limit == 0:
        output = evaluate_checkpoints(config, device=device)
        print(f"results -> {output}")


if __name__ == "__main__":
    main()

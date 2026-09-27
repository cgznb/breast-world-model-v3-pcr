#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.biflow_trajectories import (
    DIRECT_T0,
    EXPECTED_LOCKED_FUTURE_VISITS,
    ROLLOUT_GENERATED_DCE0_REAL_SER,
    TRAJECTORY_STRATEGIES,
    TrajectoryRoute,
    batched,
    build_trajectory_routes,
    compose_rollout_source_mri,
    index_registered_source_assets,
    prepare_real_source_mri,
    shared_target_seed_schedule,
    validate_locked102_routes,
)
from src.mewm_data import load_ids, patient_key


CONFIG_SCHEMA = "pillar_full978_locked102_biflow_trajectories_v1"
LATENT_SCHEMA = "pillar_full978_biflow_trajectory_latent_v1"
DECODED_SCHEMA = "pillar_full978_biflow_trajectory_decoded_v1"
SUMMARY_SCHEMA = "pillar_full978_biflow_trajectory_summary_v1"
LATENT_SHAPE = (1, 1, 8, 24, 64, 64)
DECODED_SHAPE = (1, 96, 256, 256)
SOURCE_SHAPE = (2, 96, 256, 256)
SOURCE_PROVENANCE_VALUES = frozenset(
    {
        "locked_strict_a_cache",
        "registered_manifest_fallback_outside_strict_a",
    }
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("trajectory generation config schema is invalid")
    for section in ("cohort", "biflow", "generation"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"trajectory generation config is missing {section}")
    if tuple(payload["generation"].get("strategies", ())) != TRAJECTORY_STRATEGIES:
        raise ValueError("trajectory strategy order or names changed")
    payload["_config_path"] = config_path
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty route table")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _import_biflow(config: Mapping[str, Any], name: str):
    source_repo = _repo_path(config["biflow"]["repo"])
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    module = importlib.import_module(f"mewm_ispy2.{name}")
    if not Path(module.__file__).resolve().is_relative_to(source_repo):
        raise RuntimeError("mewm_ispy2 was imported from the wrong repository")
    return module


def _cohort_inputs(
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    section = config["cohort"]
    patient_ids = load_ids(_repo_path(section["test_ids"]))
    metadata = pd.read_csv(_repo_path(section["metadata_csv"]), dtype={"pid": str})
    if metadata["pid"].duplicated().any():
        raise ValueError("full978 metadata contains duplicate patient IDs")
    rows = {patient_key(row["pid"]): row for row in metadata.to_dict(orient="records")}
    if any(patient_key(patient_id) not in rows for patient_id in patient_ids):
        raise ValueError("full978 metadata does not cover every locked test patient")
    return patient_ids, rows


def _load_world_inputs(config: Mapping[str, Any]):
    section = config["biflow"]
    backend = _import_biflow(config, "backend")
    cache_module = _import_biflow(config, "cache")
    bundle = _repo_path(section["bundle_json"])
    loaded = backend.load_transition_records(
        bundle,
        _repo_path(section["phase_manifest_csv"]),
        backend="registered_t0",
    )
    roi_cache = cache_module.RegisteredStrictAROICache(
        _repo_path(section["roi_cache_dir"]), bundle_json=bundle
    )
    return loaded, roi_cache


def _text_conditions(loaded: Any, patient_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
    conditions: dict[str, tuple[str, str]] = {}
    val_patients = set()
    for record in loaded.records:
        key = patient_key(record.patient_id)
        value = (str(record.clinical_text), str(record.action_text))
        previous = conditions.setdefault(key, value)
        if previous != value:
            raise ValueError("BiFlow text conditions differ within one patient")
        if record.fold == "val":
            val_patients.add(key)
    expected = {patient_key(value) for value in patient_ids}
    if val_patients != expected or not expected <= set(conditions):
        raise ValueError("full978 locked test patients differ from the BiFlow validation fold")
    return conditions


def _load_json_object(path: str | Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{name} is unreadable") from None
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty object")
    return value


class RealSourceLoader:
    def __init__(
        self,
        *,
        loaded: Any,
        roi_cache: Any,
        assets: Mapping[tuple[str, int], Any],
        crop_plans: Mapping[str, Mapping[str, Any]],
        channel_statistics: Mapping[str, Mapping[str, Any]],
        target_spacing_xyz: Sequence[float],
    ) -> None:
        self.roi_cache = roi_cache
        self.assets = dict(assets)
        self.crop_plans = crop_plans
        self.channel_statistics = channel_statistics
        self.target_spacing_xyz = tuple(float(value) for value in target_spacing_xyz)
        self.locked_visits = {
            (patient_key(visit.patient_id), int(str(visit.visit)[1:])): visit
            for visit in loaded.visits.values()
        }
        self.provenance_counts: Counter[str] = Counter()

    def load(self, patient_id: str, stage: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        key = (patient_key(patient_id), int(stage))
        asset = self.assets.get(key)
        if asset is None:
            raise ValueError(f"registered source asset is unavailable: {patient_id}:T{stage}")
        visit = self.locked_visits.get(key)
        if visit is not None and Path(visit.dce0.path).resolve() == asset.dce0_path:
            try:
                payload, _ = self.roi_cache._load_payload(visit)
            except ValueError:
                payload = None
            if payload is not None:
                source = payload["mri"].clone().contiguous()
                foreground = payload["valid_foreground"][0].bool().clone().contiguous()
                provenance = "locked_strict_a_cache"
                self.provenance_counts[provenance] += 1
                return source, foreground, provenance
        try:
            crop_plan = self.crop_plans[patient_id]
        except KeyError:
            raise ValueError(f"T0 crop plan is unavailable: {patient_id}") from None
        source, foreground = prepare_real_source_mri(
            asset,
            crop_plan,
            channel_statistics=self.channel_statistics,
            target_spacing_xyz=self.target_spacing_xyz,
        )
        provenance = "registered_manifest_fallback_outside_strict_a"
        self.provenance_counts[provenance] += 1
        return source, foreground, provenance


def _artifact_path(root: Path, route: TrajectoryRoute, kind: str) -> Path:
    return root / route.strategy / kind / route.patient_id / f"T{route.target_stage}.pt"


def _load_payload(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"trajectory artifact must be a mapping: {path}")
    return value


def _artifact_identity(route: TrajectoryRoute, seed: int, solver_steps: int) -> dict[str, Any]:
    return {
        "strategy": route.strategy,
        "route_id": route.route_id,
        "patient_id": route.patient_id,
        "source_stage": route.source_stage,
        "target_stage": route.target_stage,
        "source_visit_id": route.source_visit_id,
        "target_visit_id": route.target_visit_id,
        "delta_days": route.delta_days,
        "seed": int(seed),
        "solver_steps": int(solver_steps),
        "source_dce0": route.source_dce0,
        "source_ser": route.source_ser,
    }


def _valid_latent(path: Path, route: TrajectoryRoute, seed: int, solver_steps: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_payload(path)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        return False
    identity = _artifact_identity(route, seed, solver_steps)
    expected_keys = {
        "schema",
        *identity,
        "source_input_provenance",
        "target_latent_read",
        "predicted_normalized_latent",
    }
    latent = payload.get("predicted_normalized_latent")
    return bool(
        set(payload) == expected_keys
        and payload.get("schema") == LATENT_SCHEMA
        and all(payload.get(key) == value for key, value in identity.items())
        and payload.get("source_input_provenance") in SOURCE_PROVENANCE_VALUES
        and payload.get("target_latent_read") is False
        and isinstance(latent, torch.Tensor)
        and latent.dtype == torch.float16
        and tuple(latent.shape) == LATENT_SHAPE
        and bool(torch.isfinite(latent).all())
    )


def _valid_decoded(path: Path, route: TrajectoryRoute, seed: int, solver_steps: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_payload(path)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        return False
    identity = _artifact_identity(route, seed, solver_steps)
    expected_keys = {
        "schema",
        *identity,
        "source_input_provenance",
        "target_dce0_read",
        "prediction",
    }
    prediction = payload.get("prediction")
    return bool(
        set(payload) == expected_keys
        and payload.get("schema") == DECODED_SCHEMA
        and all(payload.get(key) == value for key, value in identity.items())
        and payload.get("source_input_provenance") in SOURCE_PROVENANCE_VALUES
        and payload.get("target_dce0_read") is False
        and isinstance(prediction, torch.Tensor)
        and prediction.dtype == torch.float16
        and tuple(prediction.shape) == DECODED_SHAPE
        and bool(torch.isfinite(prediction).all())
    )


def _load_biflow_system(config: Mapping[str, Any], device: torch.device):
    section = config["biflow"]
    config_module = _import_biflow(config, "ispy2_biflow_config")
    latent_module = _import_biflow(config, "ispy2_dce0_world_latents")
    workflow = _import_biflow(config, "ispy2_biflow_workflow")
    training = _import_biflow(config, "ispy2_biflow_training")
    biflow_config = config_module.load_ispy2_biflow_config(_repo_path(section["config"]))
    if (
        biflow_config.base.model.latent_normalization
        != latent_module.ISPY2_DCE0_CONTINUOUS_NORMALIZATION
        or int(biflow_config.runtime.seed) != 2026
    ):
        raise ValueError("BiFlow config is not the locked original experiment")
    latent_cache = latent_module.ISPY2DCE0ContinuousLatentCache(
        biflow_config.base.data.continuous_root
    )
    model = workflow._build_model(biflow_config)
    identity = workflow._identity(biflow_config, model=model, latent_cache=latent_cache)
    system = training.ISPY2BiFlowTrainingSystem(
        model, config=biflow_config, checkpoint_identity=identity
    )
    checkpoint = torch.load(
        _repo_path(section["checkpoint"]), map_location="cpu", mmap=True, weights_only=False
    )
    expected_epoch = int(config["generation"]["expected_checkpoint_epoch"])
    expected_step = int(config["generation"]["expected_checkpoint_global_step"])
    if (
        checkpoint.get("ispy2_biflow_schema") != training.ISPY2_BIFLOW_CHECKPOINT_SCHEMA
        or int(checkpoint.get("epoch", -1)) != expected_epoch
        or int(checkpoint.get("global_step", -1)) != expected_step
    ):
        raise ValueError("BiFlow checkpoint metadata changed")
    system.on_load_checkpoint(checkpoint)
    incompatible = system.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = set(checkpoint["ispy2_biflow_omitted_state_keys"])
    if not set(incompatible.missing_keys) <= expected_missing or incompatible.unexpected_keys:
        raise ValueError("BiFlow checkpoint restore was incomplete")
    del checkpoint
    return system.to(device).eval(), latent_cache, training.integrate_ispy2_biflow_euler


def _load_vqgan(config: Mapping[str, Any], device: torch.device):
    config_module = _import_biflow(config, "ispy2_biflow_config")
    workflows = _import_biflow(config, "workflows")
    contract = _import_biflow(config, "vqgan")
    decoder = _import_biflow(config, "ispy2_biflow_latent_contract")
    biflow_config = config_module.load_ispy2_biflow_config(
        _repo_path(config["biflow"]["config"])
    )
    vqgan = (
        workflows.load_mri_vqgan(
            biflow_config.base.data.vqgan_checkpoint,
            expected_numeric_contract=contract.REGISTERED_VQGAN_NUMERIC_CONTRACT,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    return vqgan, decoder.decode_ispy2_biflow_continuous


def _noise_for_route(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        LATENT_SHAPE,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def _source_for_route(
    route: TrajectoryRoute,
    *,
    root: Path,
    route_index: Mapping[tuple[str, str, int], TrajectoryRoute],
    seeds: Mapping[tuple[str, int], int],
    solver_steps: int,
    source_loader: RealSourceLoader,
) -> tuple[torch.Tensor, str]:
    real_source, foreground, provenance = source_loader.load(
        route.patient_id, route.source_stage
    )
    if route.source_dce0 == "real":
        return real_source, provenance
    previous = route_index.get(
        (ROLLOUT_GENERATED_DCE0_REAL_SER, route.patient_id, route.source_stage)
    )
    if previous is None:
        raise ValueError(f"rollout predecessor is missing: {route.route_id}")
    decoded_path = _artifact_path(root, previous, "decoded")
    previous_seed = seeds[previous.target_key]
    if not _valid_decoded(decoded_path, previous, previous_seed, solver_steps):
        raise ValueError(f"rollout predecessor is incomplete: {previous.route_id}")
    prediction = _load_payload(decoded_path)["prediction"]
    return compose_rollout_source_mri(prediction, real_source, foreground), provenance


def _save_latent(
    root: Path,
    route: TrajectoryRoute,
    seed: int,
    solver_steps: int,
    prediction: torch.Tensor,
    source_provenance: str,
) -> None:
    _atomic_torch_save(
        _artifact_path(root, route, "latents"),
        {
            "schema": LATENT_SCHEMA,
            **_artifact_identity(route, seed, solver_steps),
            "source_input_provenance": source_provenance,
            "target_latent_read": False,
            "predicted_normalized_latent": prediction.cpu().half().contiguous(),
        },
    )


def _infer_batch(
    routes: Sequence[TrajectoryRoute],
    *,
    root: Path,
    route_index: Mapping[tuple[str, str, int], TrajectoryRoute],
    seeds: Mapping[tuple[str, int], int],
    solver_steps: int,
    source_loader: RealSourceLoader,
    model: Any,
    integrate: Any,
    device: torch.device,
) -> None:
    source_items = [
        _source_for_route(
            route,
            root=root,
            route_index=route_index,
            seeds=seeds,
            solver_steps=solver_steps,
            source_loader=source_loader,
        )
        for route in routes
    ]
    source_mri = torch.stack([item[0] for item in source_items]).to(device)
    noise = torch.cat([_noise_for_route(seeds[route.target_key], device) for route in routes])
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        prediction = integrate(
            model,
            source_mri=source_mri,
            clinical_text=[route.clinical_text for route in routes],
            treatment_text=[route.treatment_text for route in routes],
            delta_days=torch.tensor(
                [route.delta_days for route in routes], device=device, dtype=torch.float32
            ),
            target_stage=torch.tensor(
                [route.target_stage for route in routes], device=device, dtype=torch.long
            ),
            solver_steps=solver_steps,
            noise=noise,
        )
    if tuple(prediction.shape) != (len(routes), *LATENT_SHAPE[1:]) or not bool(
        torch.isfinite(prediction).all()
    ):
        raise ValueError("BiFlow returned an invalid trajectory latent batch")
    for index, route in enumerate(routes):
        _save_latent(
            root,
            route,
            seeds[route.target_key],
            solver_steps,
            prediction[index : index + 1],
            source_items[index][1],
        )


def _generate_pending(
    routes: Sequence[TrajectoryRoute],
    *,
    batch_size: int,
    root: Path,
    route_index: Mapping[tuple[str, str, int], TrajectoryRoute],
    seeds: Mapping[tuple[str, int], int],
    solver_steps: int,
    source_loader: RealSourceLoader,
    model: Any,
    integrate: Any,
    device: torch.device,
) -> int:
    pending = [
        route
        for route in routes
        if not _valid_latent(
            _artifact_path(root, route, "latents"),
            route,
            seeds[route.target_key],
            solver_steps,
        )
    ]
    completed = 0
    for group in batched(pending, batch_size):
        try:
            _infer_batch(
                group,
                root=root,
                route_index=route_index,
                seeds=seeds,
                solver_steps=solver_steps,
                source_loader=source_loader,
                model=model,
                integrate=integrate,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            print(
                f"CUDA OOM for latent batch of {len(group)}; retrying individually",
                flush=True,
            )
            torch.cuda.empty_cache()
            for route in group:
                _infer_batch(
                    [route],
                    root=root,
                    route_index=route_index,
                    seeds=seeds,
                    solver_steps=solver_steps,
                    source_loader=source_loader,
                    model=model,
                    integrate=integrate,
                    device=device,
                )
        completed += len(group)
        print(f"generated latent {completed}/{len(pending)} pending", flush=True)
    return len(pending)


def _decode_batch(
    routes: Sequence[TrajectoryRoute],
    *,
    root: Path,
    seeds: Mapping[tuple[str, int], int],
    solver_steps: int,
    latent_cache: Any,
    vqgan: Any,
    decode: Any,
    device: torch.device,
) -> None:
    payloads = [_load_payload(_artifact_path(root, route, "latents")) for route in routes]
    normalized = torch.cat(
        [payload["predicted_normalized_latent"].float() for payload in payloads]
    ).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        decoded = decode(vqgan, latent_cache.denormalize(normalized))
    if tuple(decoded.shape) != (len(routes), 1, *DECODED_SHAPE) or not bool(
        torch.isfinite(decoded).all()
    ):
        raise ValueError("VQGAN returned an invalid trajectory DCE0 batch")
    for index, route in enumerate(routes):
        _atomic_torch_save(
            _artifact_path(root, route, "decoded"),
            {
                "schema": DECODED_SCHEMA,
                **_artifact_identity(route, seeds[route.target_key], solver_steps),
                "source_input_provenance": payloads[index]["source_input_provenance"],
                "target_dce0_read": False,
                "prediction": decoded[index, 0].cpu().half().contiguous(),
            },
        )


def _decode_pending(
    routes: Sequence[TrajectoryRoute],
    *,
    batch_size: int,
    root: Path,
    seeds: Mapping[tuple[str, int], int],
    solver_steps: int,
    latent_cache: Any,
    vqgan: Any,
    decode: Any,
    device: torch.device,
) -> int:
    pending = [
        route
        for route in routes
        if not _valid_decoded(
            _artifact_path(root, route, "decoded"),
            route,
            seeds[route.target_key],
            solver_steps,
        )
    ]
    completed = 0
    for group in batched(pending, batch_size):
        try:
            _decode_batch(
                group,
                root=root,
                seeds=seeds,
                solver_steps=solver_steps,
                latent_cache=latent_cache,
                vqgan=vqgan,
                decode=decode,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            print(
                f"CUDA OOM for decode batch of {len(group)}; retrying individually",
                flush=True,
            )
            torch.cuda.empty_cache()
            for route in group:
                _decode_batch(
                    [route],
                    root=root,
                    seeds=seeds,
                    solver_steps=solver_steps,
                    latent_cache=latent_cache,
                    vqgan=vqgan,
                    decode=decode,
                    device=device,
                )
        completed += len(group)
        print(f"decoded DCE0 {completed}/{len(pending)} pending", flush=True)
    return len(pending)


def _clone_shared_real_t0_artifacts(
    direct_routes: Sequence[TrajectoryRoute],
    rollout_routes: Sequence[TrajectoryRoute],
    *,
    root: Path,
    seeds: Mapping[tuple[str, int], int],
    solver_steps: int,
) -> None:
    direct = {route.target_key: route for route in direct_routes}
    rollout = {
        route.target_key: route for route in rollout_routes if route.source_stage == 0
    }
    if not set(rollout) <= set(direct):
        raise ValueError("shared real-T0 rollout targets differ from direct targets")
    for target_key in sorted(rollout):
        source_route = direct[target_key]
        target_route = rollout[target_key]
        source_condition = (
            source_route.patient_id,
            source_route.source_stage,
            source_route.target_stage,
            source_route.source_visit_id,
            source_route.target_visit_id,
            source_route.delta_days,
            source_route.clinical_text,
            source_route.treatment_text,
            source_route.source_dce0,
            source_route.source_ser,
        )
        target_condition = (
            target_route.patient_id,
            target_route.source_stage,
            target_route.target_stage,
            target_route.source_visit_id,
            target_route.target_visit_id,
            target_route.delta_days,
            target_route.clinical_text,
            target_route.treatment_text,
            target_route.source_dce0,
            target_route.source_ser,
        )
        if source_condition != target_condition or source_route.source_stage != 0:
            raise ValueError("shared real-T0 generation conditions differ")
        seed = seeds[target_key]
        for kind, schema, tensor_key, target_read_key in (
            ("latents", LATENT_SCHEMA, "predicted_normalized_latent", "target_latent_read"),
            ("decoded", DECODED_SCHEMA, "prediction", "target_dce0_read"),
        ):
            source_path = _artifact_path(root, source_route, kind)
            validator = _valid_latent if kind == "latents" else _valid_decoded
            if not validator(source_path, source_route, seed, solver_steps):
                raise RuntimeError(f"shared real-T0 source artifact is invalid: {source_route.route_id}")
            source_payload = _load_payload(source_path)
            target_payload = {
                "schema": schema,
                **_artifact_identity(target_route, seed, solver_steps),
                "source_input_provenance": source_payload["source_input_provenance"],
                target_read_key: False,
                tensor_key: source_payload[tensor_key].clone(),
            }
            target_path = _artifact_path(root, target_route, kind)
            _atomic_torch_save(target_path, target_payload)
            if not validator(target_path, target_route, seed, solver_steps):
                raise RuntimeError(f"shared real-T0 target artifact is invalid: {target_route.route_id}")
            if not torch.equal(
                source_payload[tensor_key],
                _load_payload(target_path)[tensor_key],
            ):
                raise RuntimeError("shared real-T0 artifacts are not tensor-equal")


def _write_contracts(
    config: Mapping[str, Any],
    routes: Mapping[str, Sequence[TrajectoryRoute]],
    seeds: Mapping[tuple[str, int], int],
    route_summary: Mapping[str, Any],
    *,
    output: Path,
    solver_steps: int,
    base_seed: int,
    batch_size: int,
    patient_count: int,
    locked_full_run: bool,
) -> None:
    generation = config["generation"]
    common = {
        "schema": CONFIG_SCHEMA,
        "config": str(config["_config_path"]),
        "checkpoint": str(_repo_path(config["biflow"]["checkpoint"])),
        "checkpoint_epoch": int(generation["expected_checkpoint_epoch"]),
        "checkpoint_global_step": int(generation["expected_checkpoint_global_step"]),
        "solver": "explicit_euler",
        "solver_steps": solver_steps,
        "base_seed": base_seed,
        "requested_batch_size": batch_size,
        "oom_fallback_policy": "retry_failed_multi-item_batch_as_single-item_batches",
        "shared_real_t0_policy": "generate_direct_once_and_clone_tensor-exact_to_rollout",
        "seed_policy": "base_seed_plus_sorted_patient_target_index_shared_between_strategies",
        "patient_count": patient_count,
        "locked_full_run": locked_full_run,
        "t0_policy": "real_not_generated",
        "target_latent_policy": "never_read",
        "target_dce0_policy": "never_read_for_the_current_generation_step",
        "latent_normalization": "continuous_codebook_minmax_v1",
        "source_tensor": {"shape": list(SOURCE_SHAPE), "channels": ["DCE0", "SER"]},
        "latent_tensor": {"shape": list(LATENT_SHAPE), "saved_dtype": "float16"},
        "decoded_tensor": {"shape": list(DECODED_SHAPE), "saved_dtype": "float16"},
        "rollout_foreground_policy": (
            "real source-visit DCE0 support masks generated feedback outside the valid FOV; "
            "real DCE0 intensities are not supplied to the model"
        ),
        "fallback_warning": (
            "registered_manifest_fallback sources are constructible but outside the locked "
            "Strict-A visit-quality contract"
        ),
    }
    for strategy in TRAJECTORY_STRATEGIES:
        strategy_dir = output / strategy
        contract = {
            **common,
            "strategy": strategy,
            "route_count": len(routes[strategy]),
            "route_summary": route_summary[strategy],
            "source_policy": (
                "real T0 DCE0 plus real T0 SER for every target"
                if strategy == DIRECT_T0
                else "latest generated DCE0 plus real same-visit SER; real T0 starts each trajectory"
            ),
        }
        path = strategy_dir / "run_contract.json"
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"existing trajectory output has a different contract: {strategy}")
        route_rows = [route.public_payload(seed=seeds[route.target_key]) for route in routes[strategy]]
        _atomic_csv(strategy_dir / "routes.csv", route_rows)
        _atomic_json(path, contract)


def _audit_outputs(
    routes: Mapping[str, Sequence[TrajectoryRoute]],
    seeds: Mapping[tuple[str, int], int],
    *,
    root: Path,
    solver_steps: int,
    locked_full_run: bool,
) -> dict[str, Any]:
    summaries = {}
    for strategy in TRAJECTORY_STRATEGIES:
        provenance = Counter()
        for route in routes[strategy]:
            seed = seeds[route.target_key]
            latent_path = _artifact_path(root, route, "latents")
            decoded_path = _artifact_path(root, route, "decoded")
            if not _valid_latent(latent_path, route, seed, solver_steps):
                raise RuntimeError(f"invalid or missing trajectory latent: {route.route_id}")
            if not _valid_decoded(decoded_path, route, seed, solver_steps):
                raise RuntimeError(f"invalid or missing decoded DCE0: {route.route_id}")
            latent_payload = _load_payload(latent_path)
            decoded_payload = _load_payload(decoded_path)
            if latent_payload["source_input_provenance"] != decoded_payload["source_input_provenance"]:
                raise RuntimeError("trajectory latent and decoded provenance differ")
            provenance[str(latent_payload["source_input_provenance"])] += 1
        summary = {
            "schema": SUMMARY_SCHEMA,
            "strategy": strategy,
            "patients": len({route.patient_id for route in routes[strategy]}),
            "routes": len(routes[strategy]),
            "latent_count": len(routes[strategy]),
            "decoded_count": len(routes[strategy]),
            "target_counts": dict(
                sorted(Counter(f"T{route.target_stage}" for route in routes[strategy]).items())
            ),
            "source_input_provenance_counts": dict(sorted(provenance.items())),
            "target_latent_read": False,
            "target_dce0_read_for_current_generation_step": False,
            "complete": True,
        }
        if locked_full_run:
            expected_provenance = (
                Counter(
                    {
                        "locked_strict_a_cache": 291,
                        "registered_manifest_fallback_outside_strict_a": 5,
                    }
                )
                if strategy == DIRECT_T0
                else Counter(
                    {
                        "locked_strict_a_cache": 289,
                        "registered_manifest_fallback_outside_strict_a": 7,
                    }
                )
            )
            if summary["routes"] != EXPECTED_LOCKED_FUTURE_VISITS:
                raise RuntimeError("locked trajectory output count changed")
            if provenance != expected_provenance:
                raise RuntimeError(f"locked trajectory source provenance changed: {strategy}")
        summaries[strategy] = summary

    direct_by_target = {
        route.target_key: route for route in routes[DIRECT_T0]
    }
    shared_rollout = {
        route.target_key: route
        for route in routes[ROLLOUT_GENERATED_DCE0_REAL_SER]
        if route.source_stage == 0
    }
    if not set(shared_rollout) <= set(direct_by_target):
        raise RuntimeError("shared real-T0 target sets differ")
    if locked_full_run and len(shared_rollout) != 102:
        raise RuntimeError("locked shared real-T0 target count changed")
    for target_key in sorted(shared_rollout):
        direct_route = direct_by_target[target_key]
        rollout_route = shared_rollout[target_key]
        conditions = (
            direct_route.source_stage,
            direct_route.target_stage,
            direct_route.delta_days,
            direct_route.clinical_text,
            direct_route.treatment_text,
        )
        rollout_conditions = (
            rollout_route.source_stage,
            rollout_route.target_stage,
            rollout_route.delta_days,
            rollout_route.clinical_text,
            rollout_route.treatment_text,
        )
        if conditions != rollout_conditions:
            raise RuntimeError("shared real-T0 generation conditions differ")
        for kind, tensor_key in (
            ("latents", "predicted_normalized_latent"),
            ("decoded", "prediction"),
        ):
            direct_value = _load_payload(_artifact_path(root, direct_route, kind))[tensor_key]
            rollout_value = _load_payload(_artifact_path(root, rollout_route, kind))[tensor_key]
            if not torch.equal(direct_value, rollout_value):
                raise RuntimeError("shared real-T0 outputs are not tensor-equal")
    for strategy, summary in summaries.items():
        _atomic_json(root / strategy / "summary.json", summary)
    overall = {
        "schema": SUMMARY_SCHEMA,
        "strategies": list(TRAJECTORY_STRATEGIES),
        "patient_count": len({route.patient_id for values in routes.values() for route in values}),
        "future_target_count_per_strategy": len(routes[DIRECT_T0]),
        "artifact_count": 2 * sum(len(values) for values in routes.values()),
        "shared_real_t0_target_count": len(shared_rollout),
        "shared_real_t0_condition_and_tensor_equal": True,
        "locked_full_run": locked_full_run,
        "strategy_summaries": summaries,
        "complete": True,
    }
    _atomic_json(root / "generation_summary.json", overall)
    return overall


def generate(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    output: Path,
    solver_steps: int,
    base_seed: int,
    batch_size: int,
    selected_patients: Sequence[str] | None,
    max_target_stage: int,
    preflight_only: bool,
    audit_only: bool,
) -> Path:
    if solver_steps <= 0 or base_seed < 0 or batch_size <= 0 or not 1 <= max_target_stage <= 3:
        raise ValueError("trajectory runtime parameters are invalid")
    patient_ids, rows = _cohort_inputs(config)
    loaded, roi_cache = _load_world_inputs(config)
    conditions = _text_conditions(loaded, patient_ids)
    full_routes = build_trajectory_routes(patient_ids, rows, conditions)
    full_route_summary = validate_locked102_routes(patient_ids, full_routes)
    full_seeds = shared_target_seed_schedule(full_routes, base_seed=base_seed)

    if selected_patients:
        requested = [str(value) for value in selected_patients]
        if len(requested) != len(set(requested)) or not set(requested) <= set(patient_ids):
            raise ValueError("selected trajectory patients must be unique locked test IDs")
        active_patients = requested
    else:
        active_patients = patient_ids
    active_set = set(active_patients)
    routes = {
        strategy: tuple(
            route
            for route in full_routes[strategy]
            if route.patient_id in active_set and route.target_stage <= max_target_stage
        )
        for strategy in TRAJECTORY_STRATEGIES
    }
    if any(not values for values in routes.values()):
        raise ValueError("trajectory route subset is empty")
    seeds = {route.target_key: full_seeds[route.target_key] for route in routes[DIRECT_T0]}
    locked_full_run = (
        not selected_patients
        and max_target_stage == 3
        and solver_steps == int(config["generation"]["solver_steps"])
        and base_seed == int(config["generation"]["base_seed"])
        and batch_size == int(config["generation"]["batch_size"])
    )
    if output == _repo_path(config["generation"]["output_dir"]) and not locked_full_run:
        raise ValueError("runtime overrides require a separate output directory")

    source_assets = index_registered_source_assets(
        _repo_path(config["cohort"]["selected_registered_manifest"]),
        [_repo_path(value) for value in config["cohort"]["registered_source_manifests"]],
        patient_ids=active_patients,
    )
    normalization = _load_json_object(config["biflow"]["normalization_json"], "normalization")
    crop_plans = _load_json_object(config["biflow"]["crop_plans_json"], "crop plans")
    channel_statistics = normalization.get("channels")
    if not isinstance(channel_statistics, dict) or set(channel_statistics) != {"dce0", "ser"}:
        raise ValueError("BiFlow channel normalization statistics are invalid")
    route_summary = (
        full_route_summary
        if locked_full_run
        else {
            strategy: {
                "routes": len(routes[strategy]),
                "patients": len({route.patient_id for route in routes[strategy]}),
                "target_counts": dict(
                    sorted(Counter(f"T{route.target_stage}" for route in routes[strategy]).items())
                ),
                "transition_counts": dict(
                    sorted(Counter(route.transition_type for route in routes[strategy]).items())
                ),
            }
            for strategy in TRAJECTORY_STRATEGIES
        }
    )
    _write_contracts(
        config,
        routes,
        seeds,
        route_summary,
        output=output,
        solver_steps=solver_steps,
        base_seed=base_seed,
        batch_size=batch_size,
        patient_count=len(active_patients),
        locked_full_run=locked_full_run,
    )
    print(json.dumps({"routes": route_summary, "output": str(output)}, sort_keys=True), flush=True)
    if preflight_only:
        return output
    if audit_only:
        _audit_outputs(
            routes,
            seeds,
            root=output,
            solver_steps=solver_steps,
            locked_full_run=locked_full_run,
        )
        return output

    source_loader = RealSourceLoader(
        loaded=loaded,
        roi_cache=roi_cache,
        assets=source_assets,
        crop_plans=crop_plans,
        channel_statistics=channel_statistics,
        target_spacing_xyz=config["biflow"]["strict_spacing_xyz"],
    )
    route_index = {
        (route.strategy, route.patient_id, route.target_stage): route
        for values in routes.values()
        for route in values
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    for target_stage in range(1, max_target_stage + 1):
        direct = tuple(
            route for route in routes[DIRECT_T0] if route.target_stage == target_stage
        )
        rollout = tuple(
            route
            for route in routes[ROLLOUT_GENERATED_DCE0_REAL_SER]
            if route.target_stage == target_stage
        )
        shared_real_t0 = tuple(route for route in rollout if route.source_stage == 0)
        rollout_to_generate = tuple(
            route for route in rollout if route.source_stage != 0
        )
        stage_routes = tuple(
            sorted(
                (*direct, *rollout_to_generate),
                key=lambda route: (route.patient_id, route.strategy),
            )
        )
        if not stage_routes:
            continue
        print(
            f"T{target_stage}: loading BiFlow model for {len(stage_routes)} route(s)",
            flush=True,
        )
        system, latent_cache, integrate = _load_biflow_system(config, device)
        generated_count = _generate_pending(
            stage_routes,
            batch_size=batch_size,
            root=output,
            route_index=route_index,
            seeds=seeds,
            solver_steps=solver_steps,
            source_loader=source_loader,
            model=system.model.eval(),
            integrate=integrate,
            device=device,
        )
        del system
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"T{target_stage}: loading VQGAN decoder", flush=True)
        vqgan, decode = _load_vqgan(config, device)
        decoded_count = _decode_pending(
            stage_routes,
            batch_size=batch_size,
            root=output,
            seeds=seeds,
            solver_steps=solver_steps,
            latent_cache=latent_cache,
            vqgan=vqgan,
            decode=decode,
            device=device,
        )
        del vqgan, latent_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if shared_real_t0:
            _clone_shared_real_t0_artifacts(
                direct,
                shared_real_t0,
                root=output,
                seeds=seeds,
                solver_steps=solver_steps,
            )
        print(
            f"T{target_stage}: generated {generated_count}, decoded {decoded_count}, stage complete",
            flush=True,
        )

    summary = _audit_outputs(
        routes,
        seeds,
        root=output,
        solver_steps=solver_steps,
        locked_full_run=locked_full_run,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate locked102 direct-T0 and generated-DCE0/real-SER rollout trajectories."
    )
    parser.add_argument(
        "--config",
        default="configs/mewm_ispy2_full978_locked102_biflow_trajectories.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--solver-steps", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patient-id", action="append")
    parser.add_argument("--max-target-stage", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only and args.audit_only:
        raise ValueError("preflight-only and audit-only are mutually exclusive")
    config = _load_config(args.config)
    generation = config["generation"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    generate(
        config,
        device=device,
        output=_repo_path(
            generation["output_dir"] if args.output_dir is None else args.output_dir
        ),
        solver_steps=int(
            generation["solver_steps"] if args.solver_steps is None else args.solver_steps
        ),
        base_seed=int(
            generation["base_seed"] if args.base_seed is None else args.base_seed
        ),
        batch_size=int(
            generation["batch_size"] if args.batch_size is None else args.batch_size
        ),
        selected_patients=args.patient_id,
        max_target_stage=int(args.max_target_stage),
        preflight_only=bool(args.preflight_only),
        audit_only=bool(args.audit_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

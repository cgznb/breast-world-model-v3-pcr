"""Read archived previous-real predictions without accessing real MRI pixels."""

from __future__ import annotations

import builtins
import gc
import io
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.first_post_pcr_data import disk_gate, identity, read_json, repo_path, save_tensor, write_json
from src.single_phase_fmbcmri_data import (
    ROI_SHAPE, SCHEMA, encode, feature_complete, feature_path, load_encoder, prepare_volume,
    progress, store_feature, unchanged_json,
)

REGISTERED_SWEEP = "results/ispy2_biflow_symmflow_steps_20260911"
REGISTERED_SYMM = "results/ispy2_generator_comparison_20260911/symmflow/adjacent_real/decoded"
ROUTE_FIELDS = ("patient_id", "source_stage", "target_stage", "source_visit_id", "target_visit_id", "seed")


@contextmanager
def deny_real_pixels():
    def wrapper(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = os.fsdecode(file).lower()
                if path.endswith((".nii", ".nii.gz", ".npy")) or "/roi_cache/" in path:
                    raise PermissionError("Generated feature extraction cannot read real MRI, ROI or mask pixels")
            return original(file, *args, **kwargs)
        return guarded
    with patch.object(builtins, "open", wrapper(builtins.open)), patch.object(io, "open", wrapper(io.open)), \
            patch.object(os, "open", wrapper(os.open)):
        yield


def routes(cfg, cohort):
    if cfg["phase"] == "dce0":
        frame = pd.read_csv(repo_path(REGISTERED_SWEEP) / "routes.csv", dtype={"patient_id": str})
        frame = frame[(frame.model == "biflow") & (frame.steps == 20) & (frame.strategy == "adjacent_real")]
        result = [{k: row[k] for k in ROUTE_FIELDS} for row in frame.to_dict("records")]
    else:
        result = read_json(Path(cfg["source_cohort"]) / "routes.json")
    visits = {(v["canonical_patient_id"], v["timepoint"]) for v in cohort["visits"] if v["split"] == "val"}
    expected = {(pid, stage) for pid, stage in visits if stage > 0}
    actual = {(r["patient_id"], r["target_stage"]) for r in result}
    if len(result) != cfg["expected_targets"] or len(actual) != len(result) or actual != expected:
        raise ValueError("Archived forecast targets differ from the full holdout cohort")
    for route in result:
        pid, stage = route["patient_id"], route["target_stage"]
        previous = max(t for p, t in visits if p == pid and t < stage)
        if route["source_stage"] != previous or (pid, 0) not in visits:
            raise ValueError("Archived generation did not condition on the previous available real visit")
    unchanged_json(Path(cfg["output_dir"]) / "generation_routes.json", result)
    return result


def archived_path(cfg, source, route, kind="decoded"):
    pid, stage = route["patient_id"], route["target_stage"]
    if cfg["phase"] == "first_post":
        return Path(cfg["source_cohort"]) / "predictions" / source / f"{pid}_T{stage}.pt"
    base = (repo_path(REGISTERED_SYMM) if source == "symm" else
            repo_path(REGISTERED_SWEEP) / "biflow/euler_20/adjacent_real" / kind)
    return base / pid / f"T{stage}.pt"


def registered_contract(source, route):
    return {**{k: route[k] for k in ROUTE_FIELDS}, "strategy": "adjacent_real", "source_dce0": "real",
            "checkpoint_step": 100000 if source == "symm" else 97104,
            "model": "symmflow" if source == "symm" else "biflow", "solver": "euler",
            "solver_steps": 20, "candidate": 0, "target_dce0_read": False}


def read_registered(path, source, route, *, latent=False):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if any(value.get(k) != v for k, v in registered_contract(source, route).items()):
        raise ValueError("Registered cache checkpoint, phase or previous-real route differs")
    if latent:
        array = value["endpoint_latent"]
        if value["normalization"] != "codebook_minmax" or array.shape != (1, 1, 8, 24, 64, 64):
            raise ValueError("Invalid original BiFM endpoint latent")
    else:
        array = value["prediction"]
        if array.shape != (1, *ROI_SHAPE):
            raise ValueError("Invalid registered generated ROI shape")
    if array.dtype != torch.float16 or not torch.isfinite(array).all():
        raise ValueError("Invalid archived tensor values or precision")
    return array


def first_post_snapshots(cfg):
    snapshots = read_json(Path(cfg["source_cohort"]) / "world_checkpoints/snapshots.json")
    for source, step in (("symm", 80000), ("bifm", 10000)):
        snapshot = snapshots[source]
        if (snapshot["optimizer_step"] != step or identity(snapshot["path"]) != snapshot["identity"]
                or snapshot["evaluation_weights"] != ("ema" if source == "symm" else "ordinary")):
            raise ValueError("First-post world-model snapshot changed")
    return snapshots


def read_first_post(path, route, snapshot):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "first_post_unregistered_tumor_roi_pcr_v1" or payload["route"] != route
            or payload["checkpoint_identity"] != snapshot["identity"] or payload["solver_steps"] != 20
            or payload["optimizer_step"] != snapshot["optimizer_step"]
            or payload["image_space"] != "fixed_vqgan_first_post_global_zscore"):
        raise ValueError("First-post cached forecast provenance differs")
    array = payload["image"]
    if array.shape != ROI_SHAPE or not torch.isfinite(array).all():
        raise ValueError("Invalid first-post generated image")
    return array.float()


def preprocess_forecast(args):
    torch.set_num_threads(1)
    phase, source, route, path, snapshot = args
    with deny_real_pixels():
        array = (read_registered(path, source, route)[0].float() if phase == "dce0"
                 else read_first_post(path, route, snapshot))
        return prepare_volume(array.numpy())


def restore_bifm_images(cfg, route_list, device):
    output = Path(cfg["output_dir"]) / "recovered_bifm"
    retained = [r for r in route_list if archived_path(cfg, "bifm", r).exists()]
    missing = [r for r in route_list if not archived_path(cfg, "bifm", r).exists()]
    probe_routes = [retained[i] for i in sorted({0, len(retained) // 2, len(retained) - 1})] if retained else []
    if not probe_routes:
        raise ValueError("No retained BiFM images are available for decoder parity verification")
    from scripts.generate_biflow_full978_trajectories import _import_biflow, _load_vqgan
    config = yaml.safe_load(repo_path("configs/mewm_ispy2_full978_locked102_biflow_trajectories.yaml").read_text())
    module = _import_biflow(config, "ispy2_biflow_config")
    base = module.load_ispy2_biflow_config(config["biflow"]["config"])
    cache = _import_biflow(config, "ispy2_dce0_world_latents").ISPY2DCE0ContinuousLatentCache(base.base.data.continuous_root)
    codec, decoder = _load_vqgan(config, torch.device(device))
    codec_binding = {"checkpoint": str(base.base.data.vqgan_checkpoint),
                     "identity": identity(base.base.data.vqgan_checkpoint),
                     "normalization_identity": identity(cache.root / "cache_identity.json"),
                     "rounding": "FP16_endpoint_to_FP32_denormalization_BF16_decode_FP16_image"}
    errors = []
    for index, route in enumerate(probe_routes + missing):
        latent_path = archived_path(cfg, "bifm", route, "latents")
        latent_identity = identity(latent_path)
        path = output / route["patient_id"] / f"T{route['target_stage']}.pt"
        audit = {"endpoint": latent_identity, "codec": codec_binding, "route": route}
        if route in missing and path.exists() and path.with_suffix(".json").exists():
            previous = read_json(path.with_suffix(".json"))
            if previous["provenance"] != audit or previous["image_identity"] != identity(path):
                raise ValueError("Recovered BiFM volume provenance changed")
            read_registered(path, "bifm", route)
            continue
        latent = read_registered(latent_path, "bifm", route, latent=True).float().to(device)
        with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                                                   enabled=torch.device(device).type == "cuda"):
            decoded = decoder(codec, cache.denormalize(latent))
        image = decoded[0, 0].detach().cpu().half().contiguous()
        if image.shape != (1, *ROI_SHAPE) or not torch.isfinite(image).all():
            raise ValueError("VQ decoder returned an invalid volume")
        if route in probe_routes:
            original = read_registered(archived_path(cfg, "bifm", route), "bifm", route)
            error = float((original.float() - image.float()).abs().max())
            errors.append(error)
            if error > 1e-6:
                raise ValueError(f"Original VQ decoder parity failed: maximum voxel error {error}")
        else:
            save_tensor(path, {"schema": SCHEMA, **registered_contract("bifm", route), "prediction": image})
            write_json(path.with_suffix(".json"), {"provenance": audit, "image_identity": identity(path)})
        if index % 8 == 0:
            progress(cfg, "restoring_bifm_images", completed=index + 1, total=len(probe_routes) + len(missing))
        disk_gate(cfg)
        if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
            raise InterruptedError("Paused after restoring an endpoint latent")
    write_json(Path(cfg["output_dir"]) / "decoder_parity.json", {"passed": True, "probes": len(errors),
               "max_voxel_error": max(errors), "retained_images": len(retained), "recovered_images": len(missing),
               "codec": codec_binding, "generator_executed": False, "real_target_pixels_read": False})
    del codec
    gc.collect()
    torch.cuda.empty_cache()


def extract_generated(cfg, cohort, device):
    if not (Path(cfg["output_dir"]) / "frozen_models.json").exists():
        raise ValueError("Freeze all 200 real-only classifiers before generated evaluation")
    route_list = routes(cfg, cohort)
    snapshots = first_post_snapshots(cfg) if cfg["phase"] == "first_post" else None
    visits = {(v["canonical_patient_id"], v["timepoint"]): v for v in cohort["visits"] if v["split"] == "val"}
    with deny_real_pixels():
        if cfg["phase"] == "dce0":
            restore_bifm_images(cfg, route_list, device)
        model = load_encoder(cfg, device)
        for source in ("symm", "bifm", "copy"):
            pending = []
            for route in route_list:
                visit = visits[route["patient_id"], route["target_stage"]]
                if source == "copy":
                    path = feature_path(cfg, "real", visits[route["patient_id"], route["source_stage"]])
                else:
                    path = archived_path(cfg, source, route)
                    if not path.exists() and source == "bifm" and cfg["phase"] == "dce0":
                        path = Path(cfg["output_dir"]) / "recovered_bifm" / route["patient_id"] / f"T{route['target_stage']}.pt"
                provenance = {"archived_path": str(path), "identity": identity(path), "route": route,
                              "encoder_identity": identity(cfg["encoder_checkpoint"]), "real_target_pixels_read": False}
                if not feature_complete(cfg, source, visit, provenance):
                    pending.append((route, visit, path, provenance))
            with ProcessPoolExecutor(max_workers=cfg["preprocess_workers"], mp_context=mp.get_context("spawn")) as pool:
                for start in range(0, len(pending), cfg["extraction_batch_size"]):
                    batch = pending[start:start + cfg["extraction_batch_size"]]
                    if source == "copy":
                        vectors = [torch.load(path, map_location="cpu", weights_only=True) for _, _, path, _ in batch]
                    else:
                        snapshot = snapshots[source] if snapshots is not None else None
                        values = list(pool.map(preprocess_forecast,
                                      [(cfg["phase"], source, route, str(path), snapshot) for route, _, path, _ in batch]))
                        vectors = encode(model, torch.stack(values).to(device))
                    for (_, visit, _, provenance), vector in zip(batch, vectors, strict=True):
                        store_feature(cfg, source, visit, vector, provenance)
                    progress(cfg, f"extracting_{source}", completed=len(route_list) - len(pending) + start + len(batch),
                             total=len(route_list))
                    if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                        raise InterruptedError("Paused after a generated feature batch")
        del model
    gc.collect()
    torch.cuda.empty_cache()
    unchanged_json(Path(cfg["output_dir"]) / "GENERATED_FEATURES_COMPLETE.json",
                   {"complete": True, "sources": ["symm", "bifm", "copy"], "targets_per_source": len(route_list),
                    "real_t0_preserved": True, "real_image_pixel_reads_blocked": True, "new_generator_inference": False})

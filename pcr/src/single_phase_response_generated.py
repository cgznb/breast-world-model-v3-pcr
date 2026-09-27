"""Archived forecasts cropped only with localization from an observed source visit."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from src.first_post_pcr_data import disk_gate, read_json, save_tensor, write_json
from src.single_phase_fmbcmri_data import SPACING_XYZ, load_config as baseline_config, load_encoder
from src.single_phase_fmbcmri_generated import (
    archived_path, deny_real_pixels as legacy_read_guard, first_post_snapshots, read_first_post, read_registered,
)
from src.single_phase_response_data import (
    artifact_path, check_pause, feature_binding, feature_complete, frozen_tokens, physical_to_zyx,
    progress, sample_volume,
)


@contextmanager
def deny_real_pixels():
    import SimpleITK as sitk

    def guarded(original):
        def call(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = os.fsdecode(file).lower()
                if path.endswith((".nii", ".nii.gz", ".npy", ".npz", ".mha", ".mhd", ".dcm")) or "/roi_cache/" in path:
                    raise PermissionError("Generated features cannot read real MRI or mask pixels")
            return original(file, *args, **kwargs)
        return call

    # Python 3.10 Path.open caches an accessor; SimpleITK opens images in C++.
    with legacy_read_guard(), patch.object(Path, "open", guarded(Path.open)), \
            patch.object(nib, "load", guarded(nib.load)), patch.object(np, "load", guarded(np.load)), \
            patch.object(sitk, "ReadImage", guarded(sitk.ReadImage)):
        yield


def observed_crop(branch, source_record, source_localization):
    if source_localization.get("no_t0"):
        raise ValueError("Forecast requires an observed T0")
    if branch == "registered_dce0":
        center = np.asarray(source_localization["center_full_zyx"]) - np.asarray(source_record["crop_plan"]["crop_start_zyx"])
        spacing = SPACING_XYZ[::-1]
    else:
        grid = source_record["crop_geometry"]
        center = physical_to_zyx(source_localization["center_lps_mm"], grid)
        spacing = grid["spacing_xyz_mm"][::-1]
    return dict(center_zyx=list(center), spacing_zyx=list(spacing), sides_mm=source_localization["sides_mm"],
                localization_timepoint=source_record["timepoint"], scale_source_timepoint=0,
                native_archive_geometry_assumption="observed_source_crop_canvas" if branch != "registered_dce0" else None)


def preprocess_generated(args):
    branch, source, route, path, snapshot, crop = args
    torch.set_num_threads(1)
    with deny_real_pixels():
        image = (read_registered(path, source, route)[0].float() if branch == "registered_dce0"
                 else read_first_post(path, route, snapshot))
        return {roi: sample_volume(image.numpy(), crop["center_zyx"], crop["spacing_zyx"], side)
                for roi, side in crop["sides_mm"].items()}


def extract_generated(cfg, study, device="cuda:0"):
    root = Path(cfg["output_dir"])
    if not (root / "FROZEN_MODELS.json").exists():
        raise ValueError("All real-only models must be frozen before holdout generation evaluation")
    old = baseline_config(cfg["baseline_config"], cfg["branch"], 0)
    route_list = read_json(Path(cfg["baseline_run"]) / "generation_routes.json")
    visits = {(r["canonical_patient_id"], r["timepoint"]): r for r in study["records"]
              if r["canonical_patient_id"] in set(study["split"]["val"])}
    expected = {(p, t) for p, t in visits if t > 0}
    if {(r["patient_id"], r["target_stage"]) for r in route_list} != expected or len(route_list) != len(expected):
        raise ValueError("Archived routes differ from the holdout visits")
    snapshots = first_post_snapshots(old) if cfg["branch"] != "registered_dce0" else None
    encoder = None
    with deny_real_pixels():
        for source in ("symm", "bifm", "copy"):
            pending = []
            for route in route_list:
                pid, tp, previous = route["patient_id"], route["target_stage"], route["source_stage"]
                if previous != max(t for p, t in visits if p == pid and t < tp):
                    raise ValueError("Forecast route does not use the previous available observed visit")
                observed = visits[pid, previous]
                localization = study["localization"][f"{pid}_T{previous}"]
                crop = observed_crop(cfg["branch"], observed, localization)
                if source == "copy":
                    for roi in cfg["rois"]:
                        path = artifact_path(cfg, roi, "real", pid, previous)
                        binding = feature_binding(cfg, observed, crop, source, path)
                        destination = artifact_path(cfg, roi, source, pid, tp)
                        if not feature_complete(destination, binding, roi == "context"):
                            payload = torch.load(path, map_location="cpu", weights_only=False)
                            payload["binding"] = binding
                            save_tensor(destination, payload)
                    continue
                path = archived_path(old, source, route)
                if not path.exists() and source == "bifm" and cfg["branch"] == "registered_dce0":
                    path = Path(cfg["baseline_run"]) / "recovered_bifm" / pid / f"T{tp}.pt"
                binding = feature_binding(cfg, observed, crop, source, path)
                if not all(feature_complete(artifact_path(cfg, roi, source, pid, tp), binding, roi == "context") for roi in cfg["rois"]):
                    pending.append((route, path, crop, binding))
            if pending:
                if encoder is None:
                    encoder = load_encoder(cfg, device)
                with ProcessPoolExecutor(max_workers=cfg["preprocess_workers"], mp_context=mp.get_context("spawn")) as pool:
                    for start in range(0, len(pending), cfg["extraction_batch_size"]):
                        batch = pending[start:start + cfg["extraction_batch_size"]]
                        snapshot = snapshots[source] if snapshots else None
                        values = list(pool.map(preprocess_generated, [(cfg["branch"], source, route, str(path), snapshot, crop)
                                                                     for route, path, crop, _ in batch]))
                        for roi in cfg["rois"]:
                            tokens, features = frozen_tokens(encoder, torch.stack([v[roi][0] for v in values]).to(device))
                            for i, (route, _, _, binding) in enumerate(batch):
                                payload = dict(binding=binding, feature=features[i].clone(), sampled_image_coverage=values[i][roi][1])
                                if roi == "context":
                                    payload["tokens"] = tokens[i].clone()
                                save_tensor(artifact_path(cfg, roi, source, route["patient_id"], route["target_stage"]), payload)
                        progress(cfg, f"extracting_{source}", completed=len(route_list) - len(pending) + start + len(batch), total=len(route_list))
                        disk_gate(cfg)
                        check_pause(cfg)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    write_json(root / "GENERATED_FEATURES_COMPLETE.json", dict(complete=True, targets_per_source=len(route_list),
               real_target_pixels_read=False, target_localization_used=False, source_localization_only=True,
               real_T0_preserved=True, new_generator_training=False, new_generator_sampling=False,
               registered_recovered_images="reuse_v1_verified_original_VQ_decoder_outputs",
               native_limitation="Archived models learned retrospectively localized per-visit grids; assigning the observed source canvas cannot remove that upstream limitation."))

"""Same-support ROI scaling ablation; no additional real tissue or phase inputs."""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward, staged_chunks
from src.first_post_pcr_data import (
    IMAGE_SHAPE, PILLAR_SHAPE, disk_gate, embedding_path, identity, load_config,
    now, read_json, resample_native_roi, save_tensor, validate_embedding, world_imports, write_json,
)
from src.pillar import normalize_volume


def resize_roi_channel(roi, shape=(192, 384, 384)):
    roi = np.asarray(roi, dtype=np.float32)
    if roi.ndim != 3 or not np.isfinite(roi).all():
        raise ValueError("Invalid finite ZYX ROI")
    values = torch.from_numpy(normalize_volume(roi)).float()[None, None]
    scaled = F.interpolate(values, size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]
    return scaled.permute(1, 2, 0).contiguous()


def replacement_roi(value, support, valid=None):
    value = np.asarray(value, dtype=np.float32)
    support = np.asarray(support)
    if value.shape != IMAGE_SHAPE or support.shape != IMAGE_SHAPE or not np.isfinite(value).all():
        raise ValueError("Invalid replacement ROI or support")
    roi = value * 283.804038 + 142.555409
    roi[support == 0] = 0
    if valid is not None:
        if np.asarray(valid).shape != IMAGE_SHAPE:
            raise ValueError("Invalid generated ROI coverage")
        roi[~np.asarray(valid, dtype=bool)] = 0
    return roi


def build_resized_volume(cfg, visit, replacement=None, replacement_valid=None):
    world_imports(cfg)
    if identity(visit["native_first_post"]) != visit["native_first_post_identity"]:
        raise ValueError("Native first-post changed")
    root = Path(cfg["output_dir"]) / "staging"
    channels, crop_error = [], None
    for phase in ("pre", "first_post", "late"):
        if phase == "first_post" and replacement is not None:
            support = resample_native_roi(visit["native_first_post"], visit, support=True)
            roi = replacement_roi(replacement, support, replacement_valid)
        else:
            path = (Path(visit["native_first_post"]) if phase == "first_post" else
                    root / Path(visit["remote_phases"][phase]).relative_to(cfg["remote_root"]))
            roi = resample_native_roi(path, visit)
            if phase == "first_post":
                if identity(visit["image_path"]) != visit["crop_identity"]:
                    raise ValueError("VQ first-post crop changed")
                cached = np.load(visit["image_path"], allow_pickle=False).astype(np.float32)
                normalized = np.zeros_like(roi)
                nonzero = roi != 0
                normalized[nonzero] = (roi[nonzero] - 142.555409) / 283.804038
                if cached.shape != IMAGE_SHAPE or not np.allclose(cached, normalized, rtol=0.001, atol=0.0001):
                    raise ValueError("Resized ROI differs from frozen native crop before scaling")
                crop_error = float(np.abs(cached - normalized).max())
        channels.append(resize_roi_channel(roi))
    volume = torch.stack(channels)
    if tuple(volume.shape) != PILLAR_SHAPE or not torch.isfinite(volume).all():
        raise ValueError("Invalid resized Pillar volume")
    return volume, {"phase_indices": [0, 1, visit["late_index"]],
                    "shape_chwd": list(volume.shape), "vq_crop_max_absolute_difference": crop_error,
                    "policy": "same_native_roi_p1_p99_then_trilinear_resize",
                    "changes_physical_scale": True, "additional_tissue": False}


def extraction_config(study):
    cfg = load_config(study["input_config"])
    cfg.update(study["preprocessing"])
    cfg["output_dir"] = str(Path(study["output_dir"]) / "resized_roi")
    return cfg


def pin_pillar(output):
    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE) / "models--YalaLab--Pillar0-BreastMRI"
    revision = (cache / "refs/main").read_text().strip()
    snapshot = cache / "snapshots" / revision
    files = {p.name: identity(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    if not files:
        raise FileNotFoundError("Cached Pillar files are absent")
    path = Path(output) / "pillar_files.json"
    if path.exists() and read_json(path) != files:
        raise ValueError("Frozen Pillar files changed")
    write_json(path, files)


def extract_resized(study, update, limit=None):
    import SimpleITK as sitk
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    cfg = extraction_config(study)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    pin_pillar(output)
    cohort = read_json(Path(study["baseline_run"]) / "cohort.json")
    development = set(cohort["split"]["train"])
    visits = [v for v in cohort["visits"] if v["split"] == "train"]
    if any(v["canonical_patient_id"] not in development for v in visits):
        raise ValueError("Extraction received a non-development visit")
    visits.sort(key=lambda v: (v["canonical_patient_id"], v["timepoint"]))
    if limit is not None:
        visits = visits[:limit]
    pending = []
    for visit in visits:
        path = embedding_path(cfg, "real", visit)
        audit = output / "geometry_audit" / f"{visit['canonical_patient_id']}_T{visit['timepoint']}.json"
        if path.exists() and audit.exists():
            validate_embedding(path)
        else:
            pending.append(visit)
    completed = len(visits) - len(pending)
    started, initial = time.perf_counter(), completed
    if pending:
        update("loading_roi_pillar", completed=completed, total=len(visits))
        model = load_frozen_pillar()
        for chunk in staged_chunks(cfg, pending):
            with ThreadPoolExecutor(max_workers=int(cfg["preprocess_workers"])) as pool:
                queue, submitted = {}, 0
                for index in range(len(chunk)):
                    while submitted < min(len(chunk), index + 2):
                        queue[submitted] = pool.submit(build_resized_volume, cfg, chunk[submitted])
                        submitted += 1
                    volume, audit = queue.pop(index).result()
                    vector = pillar_forward(model, volume[None].to("cuda:0"))[0]
                    visit = chunk[index]
                    save_tensor(embedding_path(cfg, "real", visit), vector.clone())
                    write_json(output / "geometry_audit" / f"{visit['canonical_patient_id']}_T{visit['timepoint']}.json", audit)
                    completed += 1
                    elapsed = time.perf_counter() - started
                    rate = (completed - initial) / max(elapsed, 1e-6)
                    update("extracting_resized_roi", completed=completed, total=len(visits),
                           visits_per_second=rate, remaining_seconds=(len(visits) - completed) / max(rate, 1e-6))
                    del volume, vector
            disk_gate(cfg)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if limit is None:
        write_json(output / "REAL_FEATURES_COMPLETE.json", {"complete": True, "visits": len(visits),
                   "patients": len(development), "holdout_loaded": False, "completed_utc": now(),
                   "spatial_policy": study["preprocessing"]})
    return {"visits": len(visits), "new_visits": len(pending), "seconds": time.perf_counter() - started}

"""Frozen single-channel breast-MRI features for two independent pCR studies."""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from nibabel.processing import resample_to_output

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from scripts.run_full978_independent_cv import _effective_config
from src.biflow_trajectories import _resample_crop
from src.data import load_ids
from src.first_post_optimization import nested_folds, validate_partitions
from src.first_post_pcr_data import disk_gate, identity, now, public, read_json, repo_path, save_tensor, write_json
from src.generated_dce0 import native_geometry_from_meta

SCHEMA = "single_phase_fmbcmri_pcr_v1"
DEPTHS = {"T0": 1, "T0-T1": 2, "T0-T2": 3, "T0-T3": 4}
BRANCHES = ("registered_dce0", "unregistered_first_post")
ROI_SHAPE = (96, 256, 256)
SPACING_XYZ = (0.7032, 0.7032, 2.0)


def load_config(path, branch, gpu=None):
    path = repo_path(path).resolve()
    master = yaml.safe_load(path.read_text())
    if master.get("schema") != SCHEMA or branch not in BRANCHES:
        raise ValueError("Unexpected single-phase study")
    cfg = {k: copy.deepcopy(v) for k, v in master.items() if k != "branches"}
    cfg.update(master["branches"][branch], branch=branch, config_path=str(path))
    cfg["output_dir"] = str(repo_path(cfg["output_dir"]) / branch)
    for key in ("source_cohort", "source_config", "recipes_config", "source_folds"):
        if key in cfg:
            cfg[key] = str(repo_path(cfg[key]).resolve())
    if gpu is not None:
        cfg["gpu"] = int(gpu)
    if cfg["embedding_dim"] != 768 or cfg["phase"] != ("dce0" if branch == BRANCHES[0] else "first_post"):
        raise ValueError("Branch phase or feature dimension changed")
    return cfg


def progress(cfg, stage, **fields):
    value = {"schema": SCHEMA, "updated_utc": now(), "pid": os.getpid(),
             "branch": cfg["branch"], "gpu": cfg["gpu"], "stage": stage, **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", value)
    print(json.dumps(public(value)), flush=True)


def unchanged_json(path, value):
    if Path(path).exists():
        if read_json(path) != public(value):
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        write_json(path, value)


def local_mask_path(path):
    value = Path(path)
    if value.is_file():
        return value
    old = Path("/path/to/data/ispy2_acrin_t0_registration_v1/registered")
    if value.is_relative_to(old):
        value = Path("/path/to/research/Datasets/qingyuan_acrin_registered_minimal_20260804/data") / value.relative_to(old)
    if not value.is_file():
        raise FileNotFoundError("Registered T0 localization mask is unavailable")
    return value


def crop_builder():
    name = "single_phase_original_crop_rules"
    if name not in sys.modules:
        path = repo_path("../MOTFM-tumor-first/utils/ispy2_registered.py").resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].t0_fixed_crop_plan


def rebuild_crop(visit):
    meta = read_json(visit["meta_path"])
    path = local_mask_path(meta["mask_path"])
    mask = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    native = native_geometry_from_meta(meta, mask.shape)
    image = nib.Nifti1Image(mask.transpose(2, 1, 0), native.affine_ras)
    resampled = resample_to_output(image, voxel_sizes=SPACING_XYZ, order=0, mode="constant", cval=0)
    array = np.asarray(resampled.dataobj).transpose(2, 1, 0)
    return crop_builder()(array, output_shape_zyx=ROI_SHAPE).to_dict(), str(path)


def registered_inputs(cfg):
    base = Path(cfg["source_cohort"])
    metadata = pd.read_csv(base / "metadata_enriched.csv", dtype={"pid": str})
    development = sorted(load_ids(base / "splits/train_ids.txt") + load_ids(base / "splits/val_ids.txt"))
    heldout = load_ids(base / "splits/test_ids.txt")
    folds = [{"fold": fold, **{f"{role}_ids": load_ids(Path(cfg["source_folds"]) / f"fold_{fold}/{role}_ids.txt")
                               for role in ("train", "val")}} for fold in range(5)]
    original = yaml.safe_load(Path(cfg["source_config"]).read_text())["generated_dce0_test"]
    crops = read_json(original["crop_plans_json"])
    stats = read_json(original["normalization_json"])["channels"]["dce0"]
    frame = pd.read_csv(base / "registered_visits.csv")
    visits = []
    for row in frame.to_dict("records"):
        paths = json.loads(row["dce_paths"]) if row["dce_paths"].startswith("[") else row["dce_paths"].split(";")
        path = paths[0]
        if not path.endswith("_dce_aqc_0.nii.gz"):
            raise ValueError("Registered branch must select acquisition zero")
        visits.append({"canonical_patient_id": str(row["patient_id"]), "timepoint": int(row["visit"][1:]),
                       "split": "val" if row["patient_id"] in heldout else "train",
                       "input_path": path, "input_identity": identity(path),
                       "meta_path": row["meta_path"], "meta_identity": identity(row["meta_path"])})
    output = Path(cfg["output_dir"])
    prepared = output / "crop_plans.json"
    audit_path = output / "crop_preparation.json"
    if prepared.exists():
        additions = read_json(prepared)
        audit = read_json(audit_path)
        for path, previous in audit["localization_inputs"].items():
            if identity(path) != previous:
                raise ValueError("Supplemental T0 mask changed")
        crops.update(additions)
    else:
        additions, mask_inputs = {}, {}
        for visit in visits:
            pid = visit["canonical_patient_id"]
            if visit["timepoint"] == 0 and pid not in crops:
                plan, path = rebuild_crop(visit)
                additions[pid] = crops[pid] = plan
                mask_inputs[path] = identity(path)
        audit = {"rebuilt_t0_crop_plans": len(additions), "localization_inputs": mask_inputs}
        unchanged_json(prepared, additions)
        unchanged_json(audit_path, audit)
    t0 = {v["canonical_patient_id"] for v in visits if v["timepoint"] == 0}
    for visit in visits:
        pid = visit["canonical_patient_id"]
        visit["no_t0_masked_placeholder"] = pid not in t0
        if pid in t0:
            visit["crop_plan"] = crops[pid]
        visit["normalization"] = {"mean": float(stats["mean"]), "std": float(stats["std"])}
    return metadata, development, heldout, folds, visits


def first_post_inputs(cfg):
    base = Path(cfg["source_cohort"])
    original = read_json(base / "cohort.json")
    metadata = pd.read_csv(base / "metadata_enriched.csv", dtype={"pid": str})
    folds = read_json(base / "folds.json")
    if isinstance(folds, dict):
        folds = list(folds.values())
    visits = []
    for record in original["visits"]:
        if identity(record["image_path"]) != record["crop_identity"]:
            raise ValueError("Original normalized first-post ROI changed")
        visits.append({**{k: record[k] for k in ("canonical_patient_id", "timepoint", "split", "visit_id", "crop_geometry")},
                       "input_path": record["image_path"], "input_identity": record["crop_identity"],
                       "no_t0_masked_placeholder": False})
    return metadata, original["split"]["train"], original["split"]["val"], folds, visits


def recipes(cfg):
    source = yaml.safe_load(Path(cfg["recipes_config"]).read_text())["recipes"]
    result = {}
    for name in DEPTHS:
        spec = source[name]
        original = yaml.safe_load(repo_path(spec["path"]).read_text())
        effective = _effective_config(original, spec["variant"], name)
        effective.update(input_dim=768, clinical_dim=17, epoch_selection="logloss",
                         patience=cfg["early_stopping_patience"], min_delta=cfg["early_stopping_min_delta"])
        result[name] = effective
    return result


def prepare(cfg):
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    values = registered_inputs(cfg) if cfg["phase"] == "dce0" else first_post_inputs(cfg)
    metadata, development, heldout, folds, visits = values
    validate_partitions(development, heldout, folds)
    if (len(development), len(heldout), len(visits)) != (cfg["expected_development"], cfg["expected_holdout"], cfg["expected_visits"]):
        raise ValueError("Patient or visit cohort changed")
    if metadata.pid.duplicated().any() or set(metadata.pid) != set(development) | set(heldout):
        raise ValueError("Metadata patient identities differ")
    if len({(v["canonical_patient_id"], v["timepoint"]) for v in visits}) != len(visits):
        raise ValueError("Duplicate image visit")
    for visit in visits:
        if visit["canonical_patient_id"] not in (development if visit["split"] == "train" else heldout):
            raise ValueError("Image crosses the patient partition")
    labels = metadata.set_index("pid").pCR.to_dict()
    nested = nested_folds(development, heldout, folds, labels, cfg["inner_folds"], cfg["inner_fold_seed"])
    cohort = {"schema": SCHEMA, "branch": cfg["branch"], "phase": cfg["phase"],
              "split": {"train": development, "val": heldout}, "visits": visits,
              "test_role": "historical_internal_holdout_previously_used_for_upstream_validation"}
    sources = ["src/single_phase_fmbcmri_data.py", "src/single_phase_fmbcmri_training.py",
               "src/single_phase_fmbcmri_generated.py", "src/single_phase_fmbcmri_evaluation.py",
               "scripts/run_single_phase_fmbcmri_pcr.py", "src/tdn.py", "src/data.py", "src/temporal.py",
               "src/first_post_pcr_data.py", "src/first_post_optimization.py",
               "scripts/run_full978_independent_cv.py", "scripts/run_full978_anti_overfit.py"]
    contract = {"schema": SCHEMA, "config": cfg, "encoder": identity(cfg["encoder_checkpoint"]),
                "runtime_sources": {p: identity(repo_path(p)) for p in sources}, "recipes": recipes(cfg),
                "feature_policy": "world_normalized_single_ROI_trilinear48_then_volume_zscore_CLS768",
                "fit_images": "real_only", "frozen_encoder": True, "input_dim": 768,
                "selection": "inner_three_folds_two_seeds_logloss_then_ceil_median_epochs",
                "statistics": "ten_seed_mean_and_sample_standard_deviation"}
    for name, value in (("contract", contract), ("cohort", cohort), ("folds", folds), ("nested_folds", nested),
                        ("recipes", contract["recipes"])):
        unchanged_json(output / f"{name}.json", value)
    for filename, ids in (("development_metadata", development), ("holdout_metadata", heldout)):
        frame = metadata[metadata.pid.isin(ids)].reset_index(drop=True)
        path = output / f"{filename}.csv"
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path, dtype={"pid": str}), frame, check_dtype=False)
        else:
            _atomic_csv(path, frame)
    progress(cfg, "prepared", development_patients=len(development), holdout_patients=len(heldout), visits=len(visits))
    return cohort


def prepare_volume(roi):
    value = torch.as_tensor(np.asarray(roi, dtype=np.float32)).reshape(1, 1, *ROI_SHAPE)
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite single-phase ROI")
    value = F.interpolate(value, size=(48, 48, 48), mode="trilinear", align_corners=False)
    std = value.std()
    return ((value - value.mean()) / std.clamp_min(1e-8))[0].contiguous()


def real_roi(cfg, visit):
    if identity(visit["input_path"]) != visit["input_identity"]:
        raise ValueError("Single-phase image changed")
    if cfg["phase"] == "first_post":
        roi = np.load(visit["input_path"], allow_pickle=False).astype(np.float32)
    else:
        if identity(visit["meta_path"]) != visit["meta_identity"]:
            raise ValueError("Registered geometry metadata changed")
        raw = np.asarray(nib.load(visit["input_path"]).dataobj, dtype=np.float32)
        cropped = _resample_crop(raw, read_json(visit["meta_path"]), visit["crop_plan"], SPACING_XYZ)
        valid = cropped != 0
        roi = np.zeros(ROI_SHAPE, dtype=np.float32)
        stats = visit["normalization"]
        roi[valid] = (cropped[valid] - stats["mean"]) / stats["std"]
        # Match the original Strict-A cache's stored intensity precision.
        roi = roi.astype(np.float16).astype(np.float32)
    if roi.shape != ROI_SHAPE or not np.isfinite(roi).all():
        raise ValueError("Wrong single-phase ROI shape or values")
    return roi


def load_encoder(cfg, device):
    if cfg["encoder_root"] not in sys.path:
        sys.path.insert(0, cfg["encoder_root"])
    from fmbcmri.lib.models import vit_3d_base_patchsize8
    model = vit_3d_base_patchsize8(img_size=48, in_chans=1, num_classes=0)
    checkpoint = torch.load(cfg["encoder_checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("state_dict", checkpoint)
    encoder = {}
    for key, value in state.items():
        for prefix in ("module.base_encoder.", "base_encoder."):
            if key.startswith(prefix):
                name = key[len(prefix):]
                if not name.startswith("head."):
                    encoder[name] = value
                break
    model.load_state_dict(encoder, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


@torch.inference_mode()
def encode(model, volume):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Foundation encoder must be frozen and in eval mode")
    if tuple(volume.shape[1:]) != (1, 48, 48, 48):
        raise ValueError("Encoder input must be single-channel 48-cube")
    result = model.forward_features(volume)[:, 0, :].float().cpu()
    if result.shape != (len(volume), 768) or not torch.isfinite(result).all():
        raise ValueError("Invalid foundation features")
    return result


def feature_path(cfg, source, visit):
    pid = visit["canonical_patient_id"]
    return Path(cfg["output_dir"]) / "embeddings" / source / pid / f"{pid}_T{visit['timepoint']}.pt"


def feature_complete(cfg, source, visit, provenance):
    path = feature_path(cfg, source, visit)
    audit = path.with_suffix(".json")
    if not path.exists() or not audit.exists():
        return False
    value = read_json(audit)
    if value["provenance"] != public(provenance) or value["identity"] != identity(path):
        raise ValueError("Cached feature provenance changed")
    vector = torch.load(path, map_location="cpu", weights_only=True)
    if vector.shape != (768,) or vector.dtype != torch.float32 or not torch.isfinite(vector).all():
        raise ValueError("Invalid cached single-phase feature")
    return True


def store_feature(cfg, source, visit, vector, provenance):
    path = feature_path(cfg, source, visit)
    save_tensor(path, vector.detach().cpu().float().clone())
    write_json(path.with_suffix(".json"), {"schema": SCHEMA, "phase": cfg["phase"], "source": source,
               "provenance": provenance, "identity": identity(path), "frozen_encoder": True, "embedding_dim": 768})


def preprocess_visit(args):
    torch.set_num_threads(1)
    cfg, visit = args
    return prepare_volume(real_roi(cfg, visit))


def real_provenance(cfg, visit):
    return {"input": visit["input_identity"], "phase": cfg["phase"],
            "no_t0_masked_placeholder": visit["no_t0_masked_placeholder"],
            "encoder_identity": identity(cfg["encoder_checkpoint"]),
            "preprocessing_source": identity(Path(__file__))}


def extract_real(cfg, cohort, role, device, patient_ids=None):
    visits = [v for v in cohort["visits"] if v["split"] == role
              and (patient_ids is None or v["canonical_patient_id"] in patient_ids)]
    provenance = lambda v: real_provenance(cfg, v)
    pending = [v for v in visits if not feature_complete(cfg, "real", v, provenance(v))]
    completed, started = len(visits) - len(pending), time.monotonic()
    progress(cfg, f"extracting_{role}", completed=completed, total=len(visits))
    if pending:
        model = load_encoder(cfg, device)
        with ProcessPoolExecutor(max_workers=cfg["preprocess_workers"], mp_context=mp.get_context("spawn")) as pool:
            batch_size = cfg["extraction_batch_size"]
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                usable = [v for v in batch if not v["no_t0_masked_placeholder"]]
                values = list(pool.map(preprocess_visit, [(cfg, v) for v in usable]))
                vectors = encode(model, torch.stack(values).to(device)) if values else []
                for visit, vector in zip(usable, vectors, strict=True):
                    store_feature(cfg, "real", visit, vector, provenance(visit))
                for visit in batch:
                    if visit["no_t0_masked_placeholder"]:
                        store_feature(cfg, "real", visit, torch.zeros(768), provenance(visit))
                completed += len(batch)
                progress(cfg, f"extracting_{role}", completed=completed, total=len(visits),
                         elapsed_seconds=time.monotonic() - started)
                disk_gate(cfg)
                if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                    raise InterruptedError("Pause requested after a completed feature batch")
        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if patient_ids is None:
        unchanged_json(Path(cfg["output_dir"]) / f"{role.upper()}_FEATURES_COMPLETE.json",
                       {"complete": True, "visits": len(visits), "dimension": 768})

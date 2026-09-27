"""Physical single-phase crops and label provenance for the response study."""

from __future__ import annotations

import copy
import gc
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

from scripts.run_full978_anti_overfit import _atomic_csv
from src.data import TABULAR_FEATURE_NAMES, build_tabular_full
from src.first_post_optimization import validate_partitions
from src.first_post_pcr_data import disk_gate, identity, now, public, read_json, repo_path, save_tensor, write_json
from src.generated_dce0 import native_geometry_from_meta
from src.single_phase_fmbcmri_data import BRANCHES, SPACING_XYZ, load_encoder, local_mask_path, unchanged_json

SCHEMA = "single_phase_response_pcr_v2"


def load_config(path, branch):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema"] != SCHEMA or branch not in BRANCHES or cfg["gpu"] != 0:
        raise ValueError("Unexpected study, branch or GPU; this study is restricted to GPU0")
    cfg.update(branch=branch, config_path=str(path))
    cfg["study_dir"] = str(repo_path(cfg["output_dir"]))
    cfg["output_dir"] = str(Path(cfg["study_dir"]) / branch)
    cfg["baseline_run"] = str(repo_path(cfg["baseline_run"]) / branch)
    return cfg


def progress(cfg, stage, **fields):
    record = dict(schema=SCHEMA, updated_utc=now(), pid=os.getpid(), branch=cfg["branch"],
                  gpu=0, stage=stage, **fields)
    write_json(Path(cfg["output_dir"]) / "progress.json", record)
    print(json.dumps(public(record)), flush=True)


def check_pause(cfg):
    if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
        raise InterruptedError("Paused at a recoverable boundary")


def audit_labels(cfg, metadata, cohort):
    primary = pd.read_excel(cfg["primary_labels"])
    primary["Patient_ID"] = primary.Patient_ID.map(lambda value: f"ISPY2-{int(value)}")
    primary = primary.set_index("Patient_ID").pCR
    if primary.index.duplicated().any():
        raise ValueError("Duplicate patient in the primary clinical release")
    matched = metadata.pid.isin(primary.index)
    if not matched.all() or not np.array_equal(metadata.loc[matched, "pCR"].astype(int),
                          primary.reindex(metadata.loc[matched, "pid"]).astype(int)):
        raise ValueError("Cohort labels differ from the primary I-SPY2 clinical release")
    ancillary = pd.read_excel(cfg["ancillary_labels"])
    labels = metadata.set_index("pid").pCR.to_dict()
    conflicts, checked = [], 0
    for _, row in ancillary.iterrows():
        if pd.isna(row["I-SPY 2 Research ID"]) or pd.isna(row.pcr):
            continue
        pid = f"ISPY2-{int(row['I-SPY 2 Research ID'])}"
        if pid not in labels:
            continue
        target = {"Non-pCR": 0, "pCR": 1}[row.pcr]
        checked += 1
        if int(labels[pid]) != target:
            if pid not in primary.index or int(primary[pid]) != int(labels[pid]):
                raise ValueError("Conflict is not corroborated by the primary release")
            conflicts.append(dict(patient_id=pid, primary_pcr=int(labels[pid]), ancillary_pcr=target,
                                  partition="development" if pid in cohort["split"]["train"] else "holdout"))
    if len(conflicts) != 2 or any(r["partition"] != "development" for r in conflicts):
        raise ValueError("The two known development label conflicts changed")
    root = Path(cfg["output_dir"])
    private = root / "restricted"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_csv(private / "label_conflicts.csv", pd.DataFrame(conflicts))
    (private / "label_conflicts.csv").chmod(0o600)
    write_json(root / "label_audit.json", dict(
        primary_source=cfg["primary_labels"], primary_identity=identity(cfg["primary_labels"]),
        ancillary_source=cfg["ancillary_labels"], ancillary_identity=identity(cfg["ancillary_labels"]),
        primary_matched=int(matched.sum()), ancillary_matched=checked, conflicting_patients=len(conflicts),
        primary_labels_changed=0, adjudication="source_conflict_unresolved_primary_cohort_endpoint_retained",
        sensitivity="exclude_conflicts_from_fitting_only_keep_all_evaluation_patients",
        note="Two released clinical sources disagree; no pathology-level adjudication is available."))
    return [r["patient_id"] for r in conflicts]


def _sitk():
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    return sitk


def geometry_center(grid):
    spacing = np.asarray(grid["spacing_xyz_mm"])
    direction = np.asarray(grid["direction_lps"]).reshape(3, 3)
    return np.asarray(grid["origin_lps_mm"]) + direction @ (
        (np.asarray(grid["shape_zyx"])[::-1] - 1) * spacing / 2)


def physical_to_zyx(point, grid):
    direction = np.asarray(grid["direction_lps"]).reshape(3, 3)
    return (np.linalg.solve(direction, np.asarray(point) - np.asarray(grid["origin_lps_mm"]))
            / np.asarray(grid["spacing_xyz_mm"]))[::-1]


def describe_native_mask(record):
    sitk = _sitk()
    if identity(record["mask_path"]) != record["mask_identity"]:
        raise ValueError("Native localization mask changed")
    mask = sitk.DICOMOrient(sitk.ReadImage(record["mask_path"]), "LPS")
    indices = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)
    if not len(indices):
        grid = record["crop_geometry"]
        return dict(center_lps_mm=geometry_center(record["crop_geometry"]).tolist(),
                    extent_xyz_mm=(np.asarray(grid["shape_zyx"])[::-1] * np.asarray(grid["spacing_xyz_mm"])).tolist(), mask_voxels=0,
                    localization="observed_world_roi_empty_mask_fallback")
    spacing = np.asarray(mask.GetSpacing())
    low, high = indices.min(0)[::-1], indices.max(0)[::-1]
    center = mask.TransformContinuousIndexToPhysicalPoint(tuple((low + high).astype(float) / 2))
    return dict(center_lps_mm=list(center), extent_xyz_mm=((high - low + 1) * spacing).tolist(),
                mask_voxels=len(indices), localization="observed_visit_mask_union",
                direction_lps=list(mask.GetDirection()),
                mask_identity=identity(record["mask_path"]))


def _describe_visit(args):
    cfg, record = args
    if cfg["branch"] == "unregistered_first_post":
        return describe_native_mask(record)
    plan = record.get("crop_plan")
    if plan is None:
        return {"no_t0": True}
    lo, hi = np.asarray(plan["bbox_min_zyx"]), np.asarray(plan["bbox_max_zyx"])
    return dict(center_full_zyx=((lo + hi) / 2).tolist(),
                extent_xyz_mm=((hi - lo + 1)[::-1] * np.asarray(SPACING_XYZ)).tolist(),
                mask_voxels=plan["t0_mask_voxel_count"], localization="t0_fixed_registered_mask")


def prepare(cfg):
    root, base = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    root.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    if not (base / "COMPLETE.json").exists():
        raise ValueError("The original single-phase experiment must be complete")
    cohort = read_json(base / "cohort.json")
    metadata = pd.concat([pd.read_csv(base / f"{role}_metadata.csv", dtype={"pid": str})
                          for role in ("development", "holdout")], ignore_index=True)
    conflicts = audit_labels(cfg, metadata, cohort)
    folds = read_json(base / "folds.json")
    validate_partitions(cohort["split"]["train"], cohort["split"]["val"], folds)
    if cfg["branch"] == "unregistered_first_post":
        original = read_json(repo_path("results/first_post_pcr_v1_20260914/cohort.json"))
        records = original["visits"]
    else:
        records = cohort["visits"]
    geometry_file = root / "localization.json"
    geometry_binding = dict(policy="physical_cube_v2_empty_t0_preserves_world_fov", rois=cfg["rois"],
                            baseline=identity(base / "cohort.json"))
    geometry_contract = root / "localization_contract.json"
    refresh = not geometry_file.exists() or not geometry_contract.exists() or read_json(geometry_contract) != geometry_binding
    pristine = not any((root / name).exists() for name in ("features", "training", "smoke"))
    if not refresh:
        localizations = read_json(geometry_file)
    else:
        if not pristine:
            raise ValueError("Input geometry cannot be changed after feature extraction or model fitting")
        with ProcessPoolExecutor(max_workers=cfg["preprocess_workers"], mp_context=mp.get_context("spawn")) as pool:
            descriptions = list(pool.map(_describe_visit, [(cfg, r) for r in records]))
        localizations = {}
        by_t0 = {r["canonical_patient_id"]: d for r, d in zip(records, descriptions, strict=True)
                 if r["timepoint"] == 0}
        for record, desc in zip(records, descriptions, strict=True):
            pid, tp = record["canonical_patient_id"], record["timepoint"]
            if pid not in by_t0:
                localizations[f"{pid}_T{tp}"] = {"no_t0": True}
                continue
            extent = max(by_t0[pid]["extent_xyz_mm"])
            desc["sides_mm"] = {name: max(extent + 2 * v["margin_mm"], v["minimum_side_mm"])
                                for name, v in cfg["rois"].items()}
            desc["scale_source_timepoint"] = 0
            localizations[f"{pid}_T{tp}"] = desc
        write_json(geometry_file, localizations)
        write_json(geometry_contract, geometry_binding)
    if cfg["branch"] == "unregistered_first_post":
        # Revalidate input dependencies when reusing geometry, without reading target pixels.
        for record in records:
            for p, stamp in (("mask_path", "mask_identity"), ("native_first_post", "native_first_post_identity")):
                if identity(record[p]) != record[stamp]:
                    raise ValueError("A native source/localization dependency changed")
    study = dict(schema=SCHEMA, split=cohort["split"], records=records, conflicts=conflicts,
                 localization=localizations, branch=cfg["branch"])
    contract = dict(config=cfg, encoder=identity(cfg["encoder_checkpoint"]),
                    baseline_contract=identity(base / "contract.json"),
                    policy="T0_fixed_physical_cube_observed_localization_trilinear48_roi_zscore",
                    generated_policy="previous_observed_real_localization_only",
                    token_cache="FP32_output_of_frozen_first_11_blocks_no_fitting",
                    windows="one_shared_causal_model_four_prefix_outputs",
                    objective="ordinary_BCE_average_valid_prefixes_within_patient_then_patients",
                    statistics="ten_seeds_mean_sample_SD_no_bootstrap")
    unchanged_json(root / "contract.json", contract)
    if refresh and pristine:
        write_json(root / "cohort.json", study)
    else:
        unchanged_json(root / "cohort.json", study)
    unchanged_json(root / "folds.json", folds)
    unchanged_json(root / "nested_folds.json", read_json(base / "nested_folds.json"))
    _atomic_csv(root / "metadata.csv", metadata)
    progress(cfg, "prepared", development=len(study["split"]["train"]), holdout=len(study["split"]["val"]),
             visits=len(records), unresolved_label_conflicts=len(conflicts))
    return study


def normalized(array, stats):
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all() or stats["std"] <= 0:
        raise ValueError("Invalid image or normalization")
    return np.where(value != 0, (value - stats["mean"]) / stats["std"], 0).astype(np.float32)


def sample_volume(array, center_zyx, spacing_zyx, side_mm, *, zscore=True):
    """The same physical trilinear sampler and ROI z-score for real and generated MRI."""
    array = torch.as_tensor(np.asarray(array, dtype=np.float32))
    if array.ndim != 3 or not torch.isfinite(array).all() or side_mm <= 0:
        raise ValueError("Invalid normalized MRI")
    center, spacing = np.asarray(center_zyx), np.asarray(spacing_zyx)
    if center.shape != (3,) or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Invalid physical crop geometry")
    offset = (torch.arange(48, dtype=torch.float32) + 0.5) / 48 - 0.5
    coordinates = [float(c) + offset * float(side_mm / s) for c, s in zip(center, spacing)]
    axes = [(x + 0.5) * (2 / size) - 1 for x, size in zip(coordinates, array.shape)]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1)[None]
    value = F.grid_sample(array[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    coverage = float(((grid >= -1) & (grid <= 1)).all(-1).float().mean())
    std = value.std()
    if not torch.isfinite(std):
        raise ValueError("Nonfinite local crop")
    if zscore:
        value = (value - value.mean()) / std.clamp_min(1e-8)
    return value.contiguous(), coverage


def real_input(cfg, record, localization):
    if cfg["branch"] == "registered_dce0":
        if not record["input_path"].endswith("_dce_aqc_0.nii.gz"):
            raise ValueError("Registered input must be DCE0 only")
        if identity(record["input_path"]) != record["input_identity"] or identity(record["meta_path"]) != record["meta_identity"]:
            raise ValueError("Registered source changed")
        meta = read_json(record["meta_path"])
        raw = np.asarray(nib.load(record["input_path"]).dataobj, dtype=np.float32)
        native = native_geometry_from_meta(meta, raw.shape)
        image = resample_to_output(nib.Nifti1Image(raw.transpose(2, 1, 0), native.affine_ras),
                                   voxel_sizes=SPACING_XYZ, order=1, mode="constant", cval=0)
        value = normalized(np.asarray(image.dataobj).transpose(2, 1, 0), record["normalization"])
        return value, localization["center_full_zyx"], SPACING_XYZ[::-1]
    if cfg["native_world_repo"] not in sys.path:
        sys.path.insert(0, cfg["native_world_repo"])
    from src.first_post_pcr_data import native_image
    if not record["native_first_post"].endswith("_dce_aqc_1.nii.gz"):
        raise ValueError("Native input must be first-post only")
    image = native_image(record["native_first_post"], record)
    if not np.allclose(image.GetDirection(), record["crop_geometry"]["direction_lps"], atol=1e-4):
        raise ValueError("Native and archived observed crop axes differ")
    stats = read_json(cfg["native_world_bundle"])["normalization"]
    value = normalized(_sitk().GetArrayFromImage(image), {"mean": stats["image_mean"], "std": stats["image_std"]})
    center = image.TransformPhysicalPointToContinuousIndex(tuple(localization["center_lps_mm"]))[::-1]
    return value, center, image.GetSpacing()[::-1]


def preprocess_real(args):
    cfg, record, localization = args
    torch.set_num_threads(1)
    if localization.get("no_t0"):
        return None
    value, center, spacing = real_input(cfg, record, localization)
    return {name: sample_volume(value, center, spacing, side)
            for name, side in localization["sides_mm"].items()}


@torch.no_grad()
def frozen_tokens(encoder, volumes):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Shared token cache requires a completely frozen encoder")
    x = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(volumes))))
    for block in list(encoder.blocks)[:-1]:
        x = block(x)
    features = encoder.norm(encoder.blocks[-1](x))[:, 0]
    if (x.shape[1:] != (217, 768) or features.shape != (len(volumes), 768)
            or not torch.isfinite(x).all() or not torch.isfinite(features).all()):
        raise ValueError("Unexpected FM-BCMRI tokens")
    return x.float().cpu(), features.float().cpu()


def artifact_path(cfg, roi, source, pid, tp):
    return Path(cfg["output_dir"]) / "features" / roi / source / pid / f"T{tp}.pt"


def feature_binding(cfg, record, localization, source="real", archive=None):
    if source == "real":
        inputs = {k: record[k] for k in ("input_identity", "meta_identity", "native_first_post_identity", "mask_identity") if k in record}
    else:
        inputs = {"archive": str(archive), "archive_identity": identity(archive)}
    return dict(schema=SCHEMA, encoder=identity(cfg["encoder_checkpoint"]), input=inputs,
                localization=localization, source=source, preprocessing=identity(Path(__file__)))


def feature_complete(path, binding, tokens):
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload["binding"] != binding or payload["feature"].shape != (768,) or not torch.isfinite(payload["feature"]).all():
        raise ValueError("Invalid feature cache or changed provenance")
    if tokens and (payload["tokens"].shape != (217, 768) or not torch.isfinite(payload["tokens"]).all()):
        raise ValueError("Invalid frozen token cache")
    return True


def extract_real(cfg, study, role, device="cuda:0", limit=None, patient_ids=None):
    records = [r for r in study["records"] if r["canonical_patient_id"] in set(study["split"][role])]
    if patient_ids is not None:
        records = [r for r in records if r["canonical_patient_id"] in set(patient_ids)]
    if limit is not None:
        records = records[:limit]
    pending = []
    for r in records:
        pid, tp = r["canonical_patient_id"], r["timepoint"]
        loc = study["localization"][f"{pid}_T{tp}"]
        binding = feature_binding(cfg, r, loc)
        if not all(feature_complete(artifact_path(cfg, roi, "real", pid, tp), binding, roi == "context") for roi in cfg["rois"]):
            pending.append((r, loc, binding))
    started = time.monotonic()
    progress(cfg, f"extracting_{role}", completed=len(records) - len(pending), total=len(records))
    if pending:
        encoder = load_encoder(cfg, device)
        with ProcessPoolExecutor(max_workers=cfg["preprocess_workers"], mp_context=mp.get_context("spawn")) as pool:
            for start in range(0, len(pending), cfg["extraction_batch_size"]):
                batch = pending[start:start + cfg["extraction_batch_size"]]
                values = list(pool.map(preprocess_real, [(cfg, r, loc) for r, loc, _ in batch]))
                for roi in cfg["rois"]:
                    valid = [i for i, v in enumerate(values) if v is not None]
                    if valid:
                        tokens, features = frozen_tokens(encoder, torch.stack([values[i][roi][0] for i in valid]).to(device))
                    lookup = {idx: j for j, idx in enumerate(valid)}
                    for i, (record, loc, binding) in enumerate(batch):
                        pid, tp = record["canonical_patient_id"], record["timepoint"]
                        j = lookup.get(i)
                        payload = dict(binding=binding, feature=features[j].clone() if j is not None else torch.zeros(768),
                                       sampled_image_coverage=values[i][roi][1] if j is not None else None)
                        if roi == "context":
                            payload["tokens"] = tokens[j].clone() if j is not None else torch.zeros(217, 768)
                        save_tensor(artifact_path(cfg, roi, "real", pid, tp), payload)
                progress(cfg, f"extracting_{role}", completed=len(records) - len(pending) + start + len(batch), total=len(records),
                         elapsed_seconds=time.monotonic() - started)
                disk_gate(cfg)
                check_pause(cfg)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    if limit is None and patient_ids is None:
        write_json(Path(cfg["output_dir"]) / f"{role.upper()}_FEATURES_COMPLETE.json", dict(complete=True, visits=len(records)))


def load_data(cfg, study, roi, role, source="real", patient_ids=None):
    ids = study["split"][role] if patient_ids is None else list(patient_ids)
    if not set(ids) <= set(study["split"][role]):
        raise ValueError("Requested patients cross the declared partition")
    meta = pd.read_csv(Path(cfg["output_dir"]) / "metadata.csv", dtype={"pid": str}).set_index("pid")
    records = {(r["canonical_patient_id"], r["timepoint"]): r for r in study["records"]}
    embs = np.zeros((len(ids), 4, 768), np.float32)
    masks, days = np.zeros((len(ids), 4), np.float32), np.zeros((len(ids), 4), np.float32)
    clinical, paths = [], []
    for i, pid in enumerate(ids):
        row = meta.loc[pid]
        c = build_tabular_full(row)
        for col, j in (("HR", 0), ("HER2", 1), ("age", 2), ("menopause", 3), ("MP", 4)):
            if pd.isna(row.get(col)):
                c[j] = np.nan
        clinical.append(c)
        patient_paths, active = [], True
        for tp in range(4):
            record = records.get((pid, tp))
            if record is None or study["localization"][f"{pid}_T{tp}"].get("no_t0"):
                active = False
            path = artifact_path(cfg, roi, "real" if tp == 0 else source, pid, tp)
            patient_paths.append(str(path) if record is not None else None)
            if record is None:
                continue
            # Expected visits must be present even if the contiguous-prefix policy masks them.
            if not path.exists():
                raise FileNotFoundError("A planned visit is missing its feature cache")
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            vector = payload["feature"].numpy()
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError("Invalid feature vector")
            if active:
                embs[i, tp] = vector / max(float(np.linalg.norm(vector)), 1e-8)
                masks[i, tp] = 1
                value = row.get(f"days_T{tp}", np.nan)
                if pd.isna(value) or not np.isfinite(value) or value < 0:
                    raise ValueError("Missing or invalid elapsed time")
                days[i, tp] = value
                if tp and days[i, tp] <= days[i, tp - 1]:
                    raise ValueError("Non-increasing observed dates")
        paths.append(patient_paths)
    return dict(pids=list(ids), embs=embs, masks=masks, days=days, clinical=np.asarray(clinical, np.float32),
                labels=meta.pCR.reindex(ids).to_numpy(np.float32), token_paths=paths,
                conflicted=np.asarray([p in study["conflicts"] for p in ids]))


def subset(data, ids):
    index = {p: i for i, p in enumerate(data["pids"])}
    indices = [index[p] for p in ids]
    return {k: [v[i] for i in indices] if isinstance(v, list) else v[indices] for k, v in data.items()}

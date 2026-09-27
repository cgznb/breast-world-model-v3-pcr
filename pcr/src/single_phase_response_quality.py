"""Development-only tumor coverage and inspectable MRI input checks."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_to_output

from scripts.run_full978_anti_overfit import _atomic_csv
from src.first_post_pcr_data import identity, read_json, write_json
from src.generated_dce0 import native_geometry_from_meta
from src.single_phase_fmbcmri_data import SPACING_XYZ, local_mask_path
from src.single_phase_response_data import _sitk, progress, real_input, sample_volume


def mask_input(cfg, record, loc):
    if cfg["branch"] == "registered_dce0":
        meta = read_json(record["meta_path"])
        mask = np.asarray(nib.load(local_mask_path(meta["mask_path"])).dataobj, np.float32)
        native = native_geometry_from_meta(meta, mask.shape)
        image = resample_to_output(nib.Nifti1Image(mask.transpose(2, 1, 0), native.affine_ras),
                                   voxel_sizes=SPACING_XYZ, order=0, mode="constant", cval=0)
        value = np.asarray(image.dataobj).transpose(2, 1, 0)
        points = np.argwhere(value > 0)
        plan = record["crop_plan"]
        world_points = points - np.asarray(plan["crop_start_zyx"])
        return value, np.asarray(loc["center_full_zyx"]), np.asarray(SPACING_XYZ[::-1]), points, world_points
    sitk = _sitk()
    mask = sitk.DICOMOrient(sitk.ReadImage(record["mask_path"]), "LPS")
    value = sitk.GetArrayFromImage(mask)
    points = np.argwhere(value > 0)
    center = np.asarray(mask.TransformPhysicalPointToContinuousIndex(tuple(loc["center_lps_mm"]))[::-1])
    grid = record["crop_geometry"]
    physical = (points[:, ::-1] * np.asarray(mask.GetSpacing())) @ np.asarray(mask.GetDirection()).reshape(3, 3).T + np.asarray(mask.GetOrigin())
    world_points = ((physical - np.asarray(grid["origin_lps_mm"])) @ np.linalg.inv(np.asarray(grid["direction_lps"]).reshape(3, 3)).T
                    / np.asarray(grid["spacing_xyz_mm"]))[:, ::-1]
    return value, center, np.asarray(mask.GetSpacing()[::-1]), points, world_points


def coverage_record(args):
    cfg, record, loc = args
    _, center, spacing, points, world_points = mask_input(cfg, record, loc)
    inside_world = ((world_points >= -0.5) & (world_points < np.asarray([96, 256, 256]) - 0.5)).all(1)
    result = dict(patient_id=record["canonical_patient_id"], mask_voxels=len(points),
                  old_world_coverage=float(inside_world.mean()) if len(points) else None)
    for roi, side in loc["sides_mm"].items():
        inside = (np.abs((points - center) * spacing) <= side / 2 + 1e-5).all(1)
        result[f"{roi}_coverage"] = float(inside.mean()) if len(points) else None
        result[f"{roi}_side_mm"] = side
        result[f"{roi}_minimum_bbox_voxels48"] = min(loc["extent_xyz_mm"]) / side * 48 if len(points) else None
    return result


def quality(cfg, study):
    root = Path(cfg["output_dir"]) / "quality"
    geometry = identity(Path(cfg["output_dir"]) / "localization.json")
    if ((root / "COMPLETE.json").exists()
            and read_json(root / "COMPLETE.json").get("localization_identity") == geometry):
        return read_json(root / "summary.json")
    records = [r for r in study["records"] if r["timepoint"] == 0 and r["canonical_patient_id"] in set(study["split"]["train"])]
    arguments = [(cfg, r, study["localization"][f"{r['canonical_patient_id']}_T0"]) for r in records]
    progress(cfg, "checking_development_tumor_coverage", patients=len(records))
    with ProcessPoolExecutor(max_workers=cfg["preprocess_workers"], mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(coverage_record, arguments))
    frame = pd.DataFrame(rows)
    _atomic_csv(root / "t0_coverage.csv", frame)
    summary = dict(patients=len(records), empty_t0_masks=int((frame.mask_voxels == 0).sum()),
                   old_world_truncated_masks=int((frame.old_world_coverage < 1 - 1e-6).sum()),
                   holdout_used=False, later_visit_scale_source="T0_only", image_crops_read_from="full_single_phase_image",
                   roi_summary={})
    for roi in cfg["rois"]:
        summary["roi_summary"][roi] = dict(minimum_mask_coverage=float(frame[f"{roi}_coverage"].min()),
            truncated_masks=int((frame[f"{roi}_coverage"] < 1 - 1e-6).sum()),
            side_mm_median=float(frame[f"{roi}_side_mm"].median()),
            minimum_bbox_voxels48_median=float(frame[f"{roi}_minimum_bbox_voxels48"].median()),
            minimum_bbox_below_one_patch=int((frame[f"{roi}_minimum_bbox_voxels48"] < 8).sum()))
    if any(v["truncated_masks"] for v in summary["roi_summary"].values()):
        write_json(root / "summary.json", summary)
        raise ValueError("A new T0 crop does not contain its localization mask; inspect coverage before training")
    selected = frame.sort_values("tumor_minimum_bbox_voxels48")
    positions = sorted({0, len(selected) // 3, 2 * len(selected) // 3, len(selected) - 1})
    examples = selected.iloc[positions].patient_id.tolist()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 2, figsize=(7, 3 * len(examples)), squeeze=False)
    by_id = {r["canonical_patient_id"]: r for r in records}
    for i, pid in enumerate(examples):
        record, loc = by_id[pid], study["localization"][f"{pid}_T0"]
        image, center, spacing = real_input(cfg, record, loc)
        mask, mc, ms, _, _ = mask_input(cfg, record, loc)
        for j, (roi, side) in enumerate(loc["sides_mm"].items()):
            volume = sample_volume(image, center, spacing, side)[0][0].numpy()
            tumor = sample_volume(mask, mc, ms, side, zscore=False)[0][0].numpy()
            section = int(np.argmax((tumor > 0.5).sum((1, 2))))
            ax = axes[i, j]
            ax.imshow(volume[section], cmap="gray", vmin=-1, vmax=3)
            if (tumor[section] > 0.5).any():
                ax.contour(tumor[section], levels=[0.5], colors=["#30cfaa"], linewidths=0.7)
            title = f"Example {i + 1} | {roi} | {side:.0f} mm"
            if not loc["mask_voxels"]:
                title += "\nNo T0 localization: original FOV fallback"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "development_roi_examples.png", dpi=170)
    fig.savefig(root / "development_roi_examples.pdf")
    plt.close(fig)
    write_json(root / "summary.json", summary)
    write_json(root / "COMPLETE.json", dict(passed=True, holdout_used=False, localization_identity=geometry))
    progress(cfg, "development_input_quality_passed", **summary)
    return summary

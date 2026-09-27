from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to, resample_to_output

from src.mewm_data import patient_key
from src.pillar import _to_model_axes, normalize_volume, pad_or_crop, resample_volume


LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class NativeGeometry:
    shape_zyx: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    affine_ras: np.ndarray


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain {length} finite numbers") from None
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def native_geometry_from_meta(
    meta: Mapping[str, Any], shape_zyx: Sequence[int]
) -> NativeGeometry:
    """Reconstruct the physical registered grid kept outside the identity NIfTI affine."""
    shape = tuple(int(value) for value in shape_zyx)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("native shape must contain three positive integers")
    declared_shape = (meta.get("n_slices"), meta.get("rows"), meta.get("cols"))
    try:
        declared_shape = tuple(int(value) for value in declared_shape)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("registered metadata shape is invalid") from None
    if declared_shape != shape:
        raise ValueError("registered metadata and NIfTI shapes differ")

    pixel_spacing = _finite_vector(meta.get("pixel_spacing"), 2, "pixel_spacing")
    slice_spacing = float(meta.get("spacing_between_slices"))
    if (
        np.any(pixel_spacing <= 0)
        or not math.isfinite(slice_spacing)
        or slice_spacing <= 0
    ):
        raise ValueError("registered metadata spacing is invalid")
    orientation = _finite_vector(
        meta.get("image_orientation_patient"), 6, "image_orientation_patient"
    )
    origin_lps = _finite_vector(
        meta.get("image_position_patient_first"), 3, "image_position_patient_first"
    )
    x_axis = orientation[:3]
    y_axis = orientation[3:]
    if (
        not np.isclose(np.linalg.norm(x_axis), 1.0, atol=1e-4)
        or not np.isclose(np.linalg.norm(y_axis), 1.0, atol=1e-4)
        or not np.isclose(float(np.dot(x_axis, y_axis)), 0.0, atol=1e-4)
    ):
        raise ValueError("registered metadata orientation is invalid")
    z_axis = np.cross(x_axis, y_axis)
    direction_lps = np.column_stack((x_axis, y_axis, z_axis))
    if not np.isclose(np.linalg.det(direction_lps), 1.0, atol=1e-4):
        raise ValueError("registered metadata orientation is not a proper rotation")

    spacing_xyz = np.asarray(
        [pixel_spacing[1], pixel_spacing[0], slice_spacing], dtype=np.float64
    )
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = (LPS_TO_RAS @ direction_lps) @ np.diag(spacing_xyz)
    affine[:3, 3] = LPS_TO_RAS @ origin_lps
    return NativeGeometry(
        shape_zyx=shape,
        spacing_zyx=(slice_spacing, float(pixel_spacing[0]), float(pixel_spacing[1])),
        affine_ras=affine,
    )


def crop_slices(
    input_shape_zyx: Sequence[int],
    crop_start_zyx: Sequence[int],
    output_shape_zyx: Sequence[int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    input_shape = tuple(int(value) for value in input_shape_zyx)
    starts = tuple(int(value) for value in crop_start_zyx)
    output_shape = tuple(int(value) for value in output_shape_zyx)
    if (
        len(input_shape) != 3
        or len(starts) != 3
        or len(output_shape) != 3
        or any(value <= 0 for value in input_shape + output_shape)
    ):
        raise ValueError("crop shapes and starts must be three-dimensional")
    full_slices = []
    roi_slices = []
    for available, start, requested in zip(
        input_shape, starts, output_shape, strict=True
    ):
        full_start = max(start, 0)
        full_stop = min(start + requested, available)
        roi_start = full_start - start
        roi_stop = roi_start + max(full_stop - full_start, 0)
        full_slices.append(slice(full_start, full_stop))
        roi_slices.append(slice(roi_start, roi_stop))
    return tuple(full_slices), tuple(roi_slices)  # type: ignore[return-value]


def paste_roi_zyx(roi: np.ndarray, crop_plan: Mapping[str, Any]) -> np.ndarray:
    input_shape = tuple(int(value) for value in crop_plan["input_shape_zyx"])
    output_shape = tuple(int(value) for value in crop_plan["output_shape_zyx"])
    source = np.asarray(roi)
    if source.shape != output_shape:
        raise ValueError("generated ROI shape does not match its crop plan")
    full_slices, roi_slices = crop_slices(
        input_shape, crop_plan["crop_start_zyx"], output_shape
    )
    output = np.zeros(input_shape, dtype=source.dtype)
    output[full_slices] = source[roi_slices]
    return output


def crop_full_zyx(full: np.ndarray, crop_plan: Mapping[str, Any]) -> np.ndarray:
    input_shape = tuple(int(value) for value in crop_plan["input_shape_zyx"])
    output_shape = tuple(int(value) for value in crop_plan["output_shape_zyx"])
    source = np.asarray(full)
    if source.shape != input_shape:
        raise ValueError("full volume shape does not match its crop plan")
    full_slices, roi_slices = crop_slices(
        input_shape, crop_plan["crop_start_zyx"], output_shape
    )
    output = np.zeros(output_shape, dtype=source.dtype)
    output[roi_slices] = source[full_slices]
    return output


def strict_grid_reference(
    native: NativeGeometry, target_spacing_xyz: Sequence[float]
) -> tuple[tuple[int, int, int], np.ndarray]:
    spacing = tuple(float(value) for value in target_spacing_xyz)
    if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("Strict-A spacing must contain three positive finite values")
    # Only the grid geometry is used. The original producer uses the same nibabel
    # resample_to_output operation on the registered image payload.
    native_xyz = np.zeros(tuple(reversed(native.shape_zyx)), dtype=np.uint8)
    reference = resample_to_output(
        nib.Nifti1Image(native_xyz, native.affine_ras),
        voxel_sizes=spacing,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return tuple(int(value) for value in reversed(reference.shape)), np.asarray(
        reference.affine, dtype=np.float64
    )


def native_valid_foreground_roi(
    volume_zyx: np.ndarray,
    crop_plan: Mapping[str, Any],
    native: NativeGeometry,
    *,
    strict_spacing_xyz: Sequence[float] = (0.7032, 0.7032, 2.0),
) -> np.ndarray:
    """Rebuild the Strict-A target FOV support without using target intensities."""
    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.shape != native.shape_zyx or not np.isfinite(volume).all():
        raise ValueError("native foreground source must match its finite native grid")
    spacing = tuple(float(value) for value in strict_spacing_xyz)
    if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("Strict-A spacing must contain three positive finite values")
    resampled = resample_to_output(
        nib.Nifti1Image(np.transpose(volume, (2, 1, 0)), native.affine_ras),
        voxel_sizes=spacing,
        order=1,
        mode="constant",
        cval=0.0,
    )
    full = np.transpose(np.asarray(resampled.dataobj, dtype=np.float32), (2, 1, 0))
    expected = tuple(int(value) for value in crop_plan["input_shape_zyx"])
    if full.shape != expected:
        raise ValueError("native foreground Strict-A grid does not match the crop plan")
    valid = crop_full_zyx(np.isfinite(full) & (full != 0), crop_plan).astype(bool)
    if not valid.any():
        raise ValueError("native foreground is empty after Strict-A resampling and crop")
    return valid


def generated_roi_to_native(
    prediction_zyx: np.ndarray,
    valid_foreground_zyx: np.ndarray,
    crop_plan: Mapping[str, Any],
    native: NativeGeometry,
    *,
    dce0_mean: float,
    dce0_std: float,
    strict_spacing_xyz: Sequence[float] = (0.7032, 0.7032, 2.0),
) -> np.ndarray:
    """Map one decoded global-zscore ROI back to the registered native ZYX grid."""
    prediction = np.asarray(prediction_zyx, dtype=np.float32)
    valid = np.asarray(valid_foreground_zyx, dtype=bool)
    output_shape = tuple(int(value) for value in crop_plan["output_shape_zyx"])
    if prediction.shape != output_shape or valid.shape != output_shape:
        raise ValueError("prediction and foreground must match the Strict-A ROI shape")
    if (
        not np.isfinite(prediction).all()
        or not math.isfinite(float(dce0_mean))
        or not math.isfinite(float(dce0_std))
        or float(dce0_std) <= 0
        or not valid.any()
    ):
        raise ValueError("generated ROI values or normalization statistics are invalid")

    raw_roi = np.zeros(output_shape, dtype=np.float32)
    raw_roi[valid] = prediction[valid] * float(dce0_std) + float(dce0_mean)
    full_raw = paste_roi_zyx(raw_roi, crop_plan)
    full_valid = paste_roi_zyx(valid.astype(np.uint8), crop_plan)
    strict_shape, strict_affine = strict_grid_reference(native, strict_spacing_xyz)
    if strict_shape != tuple(int(value) for value in crop_plan["input_shape_zyx"]):
        raise ValueError("reconstructed Strict-A grid does not match the crop plan")

    source_raw = nib.Nifti1Image(np.transpose(full_raw, (2, 1, 0)), strict_affine)
    source_valid = nib.Nifti1Image(
        np.transpose(full_valid, (2, 1, 0)), strict_affine
    )
    native_target = (tuple(reversed(native.shape_zyx)), native.affine_ras)
    raw_xyz = np.asarray(
        resample_from_to(
            source_raw, native_target, order=1, mode="constant", cval=0.0
        ).dataobj,
        dtype=np.float32,
    )
    valid_xyz = np.asarray(
        resample_from_to(
            source_valid, native_target, order=0, mode="constant", cval=0.0
        ).dataobj
    ) > 0.5
    raw_xyz[~valid_xyz] = 0.0
    output = np.transpose(raw_xyz, (2, 1, 0)).astype(np.float32, copy=False)
    if output.shape != native.shape_zyx or not np.isfinite(output).all():
        raise ValueError("native generated DCE0 is invalid")
    return output


def pillar_channel(volume_zyx: np.ndarray, spacing_zyx: Sequence[float]) -> torch.Tensor:
    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Pillar channel must be a finite 3D volume")
    prepared = normalize_volume(
        resample_volume(volume, spacing_zyx, target_spacing=(1.0, 1.0, 1.0))
    )
    return _to_model_axes(pad_or_crop(torch.from_numpy(prepared).float()))


def build_hybrid_pillar_volume(
    source: Any,
    patient_id: str,
    timepoint: int,
    metadata_row: Mapping[str, Any],
    generated_pre_native_zyx: np.ndarray,
) -> torch.Tensor:
    members, policy = source.phase_selection(patient_id, timepoint, metadata_row)
    if members is None or policy != "metadata_pre_early_late":
        raise ValueError("hybrid volume requires metadata-selected pre/early/late phases")
    visit = source.visits[(patient_key(patient_id), int(timepoint))]
    generated = np.asarray(generated_pre_native_zyx, dtype=np.float32)
    if generated.shape != visit.shape_zyx:
        raise ValueError("generated pre and registered postcontrast grids differ")
    channels = [pillar_channel(generated, visit.spacing_zyx)]
    for path in members[1:]:
        volume, spacing = source.load(patient_id, timepoint, path)
        channels.append(pillar_channel(volume, spacing))
    output = torch.stack(channels, dim=0).unsqueeze(0)
    if tuple(output.shape) != (1, 3, 384, 384, 192) or not bool(
        torch.isfinite(output).all()
    ):
        raise ValueError("hybrid Pillar input violates the model contract")
    return output

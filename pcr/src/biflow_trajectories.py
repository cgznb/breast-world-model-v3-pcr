from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_to_output

from src.generated_dce0 import crop_full_zyx, native_geometry_from_meta
from src.mewm_data import patient_key, visit_index


DIRECT_T0 = "direct_t0"
ADJACENT_REAL = "adjacent_real"
ROLLOUT_GENERATED_DCE0_REAL_SER = "rollout_generated_dce0_real_ser"
TRAJECTORY_STRATEGIES = (DIRECT_T0, ROLLOUT_GENERATED_DCE0_REAL_SER)
EXPECTED_LOCKED_TEST_PATIENTS = 102
EXPECTED_LOCKED_FUTURE_VISITS = 296
EXPECTED_TARGET_COUNTS = {1: 101, 2: 98, 3: 97}
EXPECTED_DIRECT_TRANSITIONS = {(0, 1): 101, (0, 2): 98, (0, 3): 97}
EXPECTED_ROLLOUT_TRANSITIONS = {
    (0, 1): 101,
    (0, 2): 1,
    (1, 2): 97,
    (1, 3): 2,
    (2, 3): 95,
}


@dataclass(frozen=True)
class TrajectoryRoute:
    strategy: str
    patient_id: str
    source_stage: int
    target_stage: int
    source_visit_id: str
    target_visit_id: str
    delta_days: int
    clinical_text: str
    treatment_text: str
    source_dce0: str
    source_ser: str

    def __post_init__(self) -> None:
        if self.strategy not in (*TRAJECTORY_STRATEGIES, ADJACENT_REAL):
            raise ValueError("trajectory strategy is invalid")
        if not self.patient_id.strip():
            raise ValueError("trajectory patient ID is empty")
        if not 0 <= self.source_stage < self.target_stage <= 3:
            raise ValueError("trajectory stages are invalid")
        if self.source_visit_id != f"{self.patient_id}:T{self.source_stage}":
            raise ValueError("trajectory source visit ID is invalid")
        if self.target_visit_id != f"{self.patient_id}:T{self.target_stage}":
            raise ValueError("trajectory target visit ID is invalid")
        if type(self.delta_days) is not int or self.delta_days <= 0:
            raise ValueError("trajectory elapsed days must be a positive integer")
        if not self.clinical_text.strip() or not self.treatment_text.strip():
            raise ValueError("trajectory text condition is empty")
        expected_dce0 = "real" if self.source_stage == 0 else "generated"
        if self.strategy in (DIRECT_T0, ADJACENT_REAL):
            expected_dce0 = "real"
        if self.source_dce0 != expected_dce0 or self.source_ser != "real":
            raise ValueError("trajectory source modality policy is invalid")

    @property
    def transition_type(self) -> str:
        return f"T{self.source_stage}->T{self.target_stage}"

    @property
    def target_key(self) -> tuple[str, int]:
        return self.patient_id, self.target_stage

    @property
    def route_id(self) -> str:
        return f"{self.strategy}:{self.patient_id}:T{self.source_stage}->T{self.target_stage}"

    def public_payload(self, *, seed: int) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("clinical_text")
        payload.pop("treatment_text")
        payload.update(
            {
                "route_id": self.route_id,
                "transition_type": self.transition_type,
                "seed": int(seed),
                "condition_policy": "locked_BiFlow_patient_text_and_elapsed_days",
            }
        )
        return payload


@dataclass(frozen=True)
class RegisteredSourceAsset:
    patient_id: str
    stage: int
    dce0_path: Path
    ser_path: Path
    meta_path: Path

    @property
    def visit_id(self) -> str:
        return f"{self.patient_id}:T{self.stage}"


def parse_registered_timepoints(value: Any) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    try:
        result = tuple(int(token[1:]) for token in tokens if token in {"T0", "T1", "T2", "T3"})
    except (TypeError, ValueError):
        result = ()
    if (
        len(result) != len(tokens)
        or not result
        or result[0] != 0
        or tuple(sorted(result)) != result
        or len(result) != len(set(result))
    ):
        raise ValueError("registered_timepoints must be an ordered unique T0-starting sequence")
    return result


def _integer_elapsed_day(row: Mapping[str, Any], source_stage: int, target_stage: int) -> int:
    try:
        source_day = float(row[f"days_T{source_stage}"])
        target_day = float(row[f"days_T{target_stage}"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("trajectory elapsed-day metadata is missing") from None
    difference = target_day - source_day
    rounded = int(round(difference)) if math.isfinite(difference) else -1
    if rounded <= 0 or not np.isclose(difference, rounded, rtol=0, atol=1e-5):
        raise ValueError("trajectory elapsed days must be finite positive integers")
    return rounded


def build_trajectory_routes(
    patient_ids: Sequence[str],
    patient_rows: Mapping[str, Mapping[str, Any]],
    text_conditions: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[TrajectoryRoute, ...]]:
    identifiers = tuple(str(value) for value in patient_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("trajectory patient IDs must be nonempty and unique")
    routes = {strategy: [] for strategy in TRAJECTORY_STRATEGIES}
    for patient_id in identifiers:
        row = patient_rows.get(patient_key(patient_id))
        conditions = text_conditions.get(patient_key(patient_id))
        if row is None or conditions is None:
            raise ValueError(f"trajectory metadata or conditions are missing: {patient_id}")
        clinical_text, treatment_text = (str(value).strip() for value in conditions)
        timepoints = parse_registered_timepoints(row["registered_timepoints"])
        for target_stage in timepoints[1:]:
            routes[DIRECT_T0].append(
                TrajectoryRoute(
                    strategy=DIRECT_T0,
                    patient_id=patient_id,
                    source_stage=0,
                    target_stage=target_stage,
                    source_visit_id=f"{patient_id}:T0",
                    target_visit_id=f"{patient_id}:T{target_stage}",
                    delta_days=_integer_elapsed_day(row, 0, target_stage),
                    clinical_text=clinical_text,
                    treatment_text=treatment_text,
                    source_dce0="real",
                    source_ser="real",
                )
            )
        for source_stage, target_stage in zip(timepoints[:-1], timepoints[1:], strict=True):
            routes[ROLLOUT_GENERATED_DCE0_REAL_SER].append(
                TrajectoryRoute(
                    strategy=ROLLOUT_GENERATED_DCE0_REAL_SER,
                    patient_id=patient_id,
                    source_stage=source_stage,
                    target_stage=target_stage,
                    source_visit_id=f"{patient_id}:T{source_stage}",
                    target_visit_id=f"{patient_id}:T{target_stage}",
                    delta_days=_integer_elapsed_day(row, source_stage, target_stage),
                    clinical_text=clinical_text,
                    treatment_text=treatment_text,
                    source_dce0="real" if source_stage == 0 else "generated",
                    source_ser="real",
                )
            )
    return {
        strategy: tuple(
            sorted(values, key=lambda route: (route.target_stage, route.patient_id))
        )
        for strategy, values in routes.items()
    }


def shared_target_seed_schedule(
    routes: Mapping[str, Sequence[TrajectoryRoute]], *, base_seed: int
) -> dict[tuple[str, int], int]:
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("trajectory base seed must be a nonnegative integer")
    target_sets = {
        strategy: {route.target_key for route in routes[strategy]}
        for strategy in TRAJECTORY_STRATEGIES
    }
    if target_sets[DIRECT_T0] != target_sets[ROLLOUT_GENERATED_DCE0_REAL_SER]:
        raise ValueError("trajectory strategies must cover the same future targets")
    return {
        target: base_seed + index
        for index, target in enumerate(sorted(target_sets[DIRECT_T0]))
    }


def previous_real_routes(
    rollout_routes: Sequence[TrajectoryRoute],
) -> tuple[TrajectoryRoute, ...]:
    """Keep the previous available visit and interval, with real source DCE0."""
    if any(route.strategy != ROLLOUT_GENERATED_DCE0_REAL_SER for route in rollout_routes):
        raise ValueError("previous-real routes require the gap-aware rollout schedule")
    return tuple(replace(route, strategy=ADJACENT_REAL, source_dce0="real")
                 for route in rollout_routes)


def validate_locked102_routes(
    patient_ids: Sequence[str], routes: Mapping[str, Sequence[TrajectoryRoute]]
) -> dict[str, Any]:
    if len(patient_ids) != EXPECTED_LOCKED_TEST_PATIENTS or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("locked trajectory cohort must contain 102 unique patients")
    if set(routes) != set(TRAJECTORY_STRATEGIES):
        raise ValueError("locked trajectory strategies changed")
    expected_patients = set(patient_ids)
    expected_transitions = {
        DIRECT_T0: EXPECTED_DIRECT_TRANSITIONS,
        ROLLOUT_GENERATED_DCE0_REAL_SER: EXPECTED_ROLLOUT_TRANSITIONS,
    }
    summary: dict[str, Any] = {}
    target_sets = []
    for strategy in TRAJECTORY_STRATEGIES:
        strategy_routes = tuple(routes.get(strategy, ()))
        targets = [route.target_key for route in strategy_routes]
        transition_counts = Counter(
            (route.source_stage, route.target_stage) for route in strategy_routes
        )
        target_counts = Counter(route.target_stage for route in strategy_routes)
        if (
            len(strategy_routes) != EXPECTED_LOCKED_FUTURE_VISITS
            or len(targets) != len(set(targets))
            or {route.patient_id for route in strategy_routes} != expected_patients
            or any(route.strategy != strategy for route in strategy_routes)
            or dict(target_counts) != EXPECTED_TARGET_COUNTS
            or dict(transition_counts) != expected_transitions[strategy]
        ):
            raise ValueError(f"locked trajectory route inventory changed: {strategy}")
        target_sets.append(set(targets))
        summary[strategy] = {
            "routes": len(strategy_routes),
            "patients": len({route.patient_id for route in strategy_routes}),
            "target_counts": {f"T{key}": value for key, value in sorted(target_counts.items())},
            "transition_counts": {
                f"T{source}->T{target}": count
                for (source, target), count in sorted(transition_counts.items())
            },
        }
    if target_sets[0] != target_sets[1]:
        raise ValueError("locked trajectory strategies do not share the same targets")
    return summary


def _phase_paths(value: Any) -> tuple[Path, ...]:
    if isinstance(value, (list, tuple)):
        raw = value
    elif pd.isna(value):
        raw = ()
    else:
        text = str(value).strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("registered DCE path JSON must contain a list")
            raw = parsed
        else:
            raw = text.split(";")
    return tuple(Path(str(item)).expanduser().resolve() for item in raw if str(item).strip())


def index_registered_source_assets(
    selected_manifest_csv: str | Path,
    source_manifest_csvs: Sequence[str | Path],
    *,
    patient_ids: Iterable[str],
) -> dict[tuple[str, int], RegisteredSourceAsset]:
    wanted_patients = {patient_key(value) for value in patient_ids}
    if not wanted_patients:
        raise ValueError("registered source patient filter is empty")
    selected = pd.read_csv(Path(selected_manifest_csv).expanduser().resolve())
    required_selected = {"patient_id", "visit", "dce_paths", "meta_path"}
    if not required_selected <= set(selected.columns):
        raise ValueError("selected registered manifest is missing required fields")
    selected_rows: dict[tuple[str, int], tuple[str, Path, Path]] = {}
    for row in selected.to_dict(orient="records"):
        key = (patient_key(row["patient_id"]), visit_index(row["visit"]))
        if key[0] not in wanted_patients:
            continue
        paths = _phase_paths(row["dce_paths"])
        if not paths:
            raise ValueError(f"selected registered visit has no DCE0: {key}")
        value = (
            str(row["patient_id"]),
            paths[0],
            Path(str(row["meta_path"])).expanduser().resolve(),
        )
        if key in selected_rows:
            raise ValueError(f"selected registered visit is duplicated: {key}")
        selected_rows[key] = value
    if {key[0] for key in selected_rows} != wanted_patients:
        raise ValueError("selected registered manifest does not cover every requested patient")

    result: dict[tuple[str, int], RegisteredSourceAsset] = {}
    for manifest_path in source_manifest_csvs:
        frame = pd.read_csv(Path(manifest_path).expanduser().resolve())
        required = {"patient_id", "visit", "dce_paths", "ser_path", "meta_path"}
        if not required <= set(frame.columns):
            raise ValueError(f"registered source manifest is missing fields: {manifest_path}")
        for row in frame.to_dict(orient="records"):
            key = (patient_key(row["patient_id"]), visit_index(row["visit"]))
            selected_value = selected_rows.get(key)
            if selected_value is None:
                continue
            paths = _phase_paths(row["dce_paths"])
            if not paths or paths[0] != selected_value[1]:
                continue
            asset = RegisteredSourceAsset(
                patient_id=selected_value[0],
                stage=key[1],
                dce0_path=paths[0],
                ser_path=Path(str(row["ser_path"])).expanduser().resolve(),
                meta_path=Path(str(row["meta_path"])).expanduser().resolve(),
            )
            previous = result.get(key)
            if previous is not None and previous != asset:
                raise ValueError(f"registered source manifests conflict: {key}")
            result[key] = asset
    missing = sorted(set(selected_rows).difference(result))
    if missing:
        raise ValueError(f"registered source assets are missing for {len(missing)} visits")
    for asset in result.values():
        for path in (asset.dce0_path, asset.ser_path, asset.meta_path):
            if not path.is_file():
                raise FileNotFoundError(f"registered source asset is missing: {path}")
    return result


def _load_native_volume(path: Path) -> np.ndarray:
    try:
        value = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    except (OSError, ValueError):
        raise ValueError(f"registered source volume is unreadable: {path}") from None
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError(f"registered source volume must be finite and 3D: {path}")
    return value


def _resample_crop(
    volume_zyx: np.ndarray,
    meta: Mapping[str, Any],
    crop_plan: Mapping[str, Any],
    target_spacing_xyz: Sequence[float],
) -> np.ndarray:
    native = native_geometry_from_meta(meta, volume_zyx.shape)
    resampled = resample_to_output(
        nib.Nifti1Image(np.transpose(volume_zyx, (2, 1, 0)), native.affine_ras),
        voxel_sizes=tuple(float(value) for value in target_spacing_xyz),
        order=1,
        mode="constant",
        cval=0.0,
    )
    full = np.transpose(np.asarray(resampled.dataobj, dtype=np.float32), (2, 1, 0))
    if full.shape != tuple(int(value) for value in crop_plan["input_shape_zyx"]):
        raise ValueError("registered source Strict-A grid does not match the crop plan")
    return crop_full_zyx(full, crop_plan).astype(np.float32, copy=False)


def prepare_real_source_mri(
    asset: RegisteredSourceAsset,
    crop_plan: Mapping[str, Any],
    *,
    channel_statistics: Mapping[str, Mapping[str, Any]],
    target_spacing_xyz: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        meta = json.loads(asset.meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"registered source metadata is unreadable: {asset.meta_path}") from None
    if not isinstance(meta, dict):
        raise ValueError("registered source metadata must be an object")
    dce0 = _resample_crop(
        _load_native_volume(asset.dce0_path), meta, crop_plan, target_spacing_xyz
    )
    ser = _resample_crop(
        _load_native_volume(asset.ser_path), meta, crop_plan, target_spacing_xyz
    )
    if dce0.shape != ser.shape:
        raise ValueError("registered DCE0 and SER do not share one Strict-A grid")
    valid = np.isfinite(dce0) & (dce0 != 0)
    if not valid.any() or not np.isfinite(ser[valid]).all():
        raise ValueError("registered source foreground or SER is invalid")
    output = np.zeros((2, *dce0.shape), dtype=np.float32)
    for index, (name, volume) in enumerate((("dce0", dce0), ("ser", ser))):
        try:
            mean = float(channel_statistics[name]["mean"])
            std = float(channel_statistics[name]["std"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("registered source normalization statistics are invalid") from None
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("registered source normalization statistics are invalid")
        output[index, valid] = (volume[valid] - mean) / std
    tensor = torch.from_numpy(output).to(dtype=torch.float16).contiguous()
    foreground = torch.from_numpy(valid).to(dtype=torch.bool).contiguous()
    if tuple(tensor.shape) != (2, 96, 256, 256) or not bool(torch.isfinite(tensor).all()):
        raise ValueError("registered source MRI violates the BiFlow tensor contract")
    return tensor, foreground


def compose_rollout_source_mri(
    generated_dce0: torch.Tensor,
    real_source_mri: torch.Tensor,
    valid_foreground: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(generated_dce0, torch.Tensor)
        or not generated_dce0.dtype.is_floating_point
        or not isinstance(real_source_mri, torch.Tensor)
        or real_source_mri.dtype != torch.float16
        or not isinstance(valid_foreground, torch.Tensor)
        or valid_foreground.dtype != torch.bool
    ):
        raise ValueError("rollout source components violate the BiFlow tensor contract")
    prediction = generated_dce0
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if (
        tuple(prediction.shape) != (96, 256, 256)
        or tuple(real_source_mri.shape) != (2, 96, 256, 256)
        or tuple(valid_foreground.shape) != (96, 256, 256)
        or not bool(torch.isfinite(prediction).all())
        or not bool(torch.isfinite(real_source_mri).all())
        or not bool(valid_foreground.any())
    ):
        raise ValueError("rollout source components violate the BiFlow tensor contract")
    output = real_source_mri.clone()
    output[0].zero_()
    output[0, valid_foreground] = prediction.to(dtype=output.dtype)[valid_foreground]
    if not bool(torch.isfinite(output).all()):
        raise ValueError("rollout source MRI contains non-finite values")
    return output.contiguous()


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


__all__ = [
    "DIRECT_T0",
    "EXPECTED_LOCKED_FUTURE_VISITS",
    "ROLLOUT_GENERATED_DCE0_REAL_SER",
    "TRAJECTORY_STRATEGIES",
    "RegisteredSourceAsset",
    "TrajectoryRoute",
    "batched",
    "build_trajectory_routes",
    "compose_rollout_source_mri",
    "index_registered_source_assets",
    "parse_registered_timepoints",
    "prepare_real_source_mri",
    "shared_target_seed_schedule",
    "validate_locked102_routes",
]

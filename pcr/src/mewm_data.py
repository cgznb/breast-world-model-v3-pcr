from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from src.pillar import normalize_volume, pad_or_crop, resample_volume


INPUT_MODES = ("registered_three_phase", "registered_dce0_repeat")
SPLIT_NAMES = ("train", "val", "test")
_PATIENT_SUFFIX = re.compile(r"(\d+)$")
_VISIT = re.compile(r"^T([0-3])$")
_ACQUISITION = re.compile(r"_dce_aqc_(\d+)\.nii(?:\.gz)?$")


def patient_key(patient_id: Any) -> str:
    """Return the numeric I-SPY2 identifier shared by ISPY2 and ACRIN aliases."""
    match = _PATIENT_SUFFIX.search(str(patient_id).strip())
    if match is None:
        raise ValueError(f"patient ID has no numeric suffix: {patient_id!r}")
    return match.group(1)


def visit_index(visit: Any) -> int:
    match = _VISIT.fullmatch(str(visit).strip())
    if match is None:
        raise ValueError(f"visit must be T0, T1, T2, or T3: {visit!r}")
    return int(match.group(1))


def _resolved(path: Any) -> str:
    return str(Path(str(path)).expanduser().resolve())


def _phase_paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        raw = value
    elif pd.isna(value):
        raw = ()
    else:
        text = str(value).strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("DCE phase JSON must be a list")
            raw = parsed
        else:
            raw = text.split(";")
    return tuple(_resolved(item) for item in raw if str(item).strip())


def _acquisition_index(path: str) -> int | None:
    match = _ACQUISITION.search(Path(path).name)
    return int(match.group(1)) if match is not None else None


def load_ids(path: str | Path) -> list[str]:
    with Path(path).open() as handle:
        return [line.strip() for line in handle if line.strip()]


@dataclass(frozen=True)
class RegisteredVisit:
    patient_id: str
    patient_key: str
    visit: str
    visit_index: int
    fold: str
    dce_paths: tuple[str, ...]
    meta_path: str
    spacing_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]


class RegisteredMewmSource:
    """Index the registered Strict-A images used by MeWM without importing MeWM code.

    The locked bundle defines which visit and DCE0 path belong to the experiment.
    Phase-manifest rows are accepted only when their first DCE path matches that
    locked DCE0 path, preventing an older duplicate registration from being mixed
    into a patient's sequence.
    """

    def __init__(
        self,
        bundle_json: str | Path,
        phase_manifest_csvs: Sequence[str | Path],
        *,
        input_mode: str = "registered_three_phase",
    ) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        if not phase_manifest_csvs:
            raise ValueError("at least one phase manifest is required")
        self.input_mode = input_mode
        self.bundle_json = Path(bundle_json).expanduser().resolve()
        self.phase_manifest_csvs = tuple(
            Path(path).expanduser().resolve() for path in phase_manifest_csvs
        )
        bundle_visits = self._load_bundle_visits()
        phase_rows = self._load_matching_phase_rows(bundle_visits)

        visits: dict[tuple[str, int], RegisteredVisit] = {}
        for key, bundle_row in bundle_visits.items():
            phase_paths = phase_rows.get(key, ())
            minimum_phases = 3 if input_mode == "registered_three_phase" else 1
            if len(phase_paths) < minimum_phases:
                continue
            bundle_spacing = (
                float(bundle_row["slice_spacing_mm"]),
                float(bundle_row["row_spacing_mm"]),
                float(bundle_row["column_spacing_mm"]),
            )
            if not all(np.isfinite(bundle_spacing)) or any(
                value <= 0 for value in bundle_spacing
            ):
                raise ValueError(f"invalid registered spacing for {bundle_row['visit_id']}")
            meta_path = _resolved(bundle_row["meta_path"])
            spacing, shape = self._registered_geometry(
                meta_path, bundle_spacing, str(bundle_row["visit_id"])
            )
            visits[key] = RegisteredVisit(
                patient_id=str(bundle_row["patient_id"]),
                patient_key=key[0],
                visit=str(bundle_row["visit"]),
                visit_index=key[1],
                fold=str(bundle_row["fold"]),
                dce_paths=phase_paths,
                meta_path=meta_path,
                spacing_zyx=spacing,
                shape_zyx=shape,
            )
        self.visits = visits

        patient_ids: dict[str, str] = {}
        for visit in visits.values():
            previous = patient_ids.setdefault(visit.patient_key, visit.patient_id)
            if previous != visit.patient_id:
                raise ValueError("bundle aliases one numeric patient ID more than once")
        self.patient_ids = patient_ids

    @staticmethod
    def _registered_geometry(
        meta_path: str,
        bundle_spacing_zyx: Sequence[float] | None,
        visit_id: str,
    ) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
        try:
            meta = json.loads(Path(meta_path).read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError(f"registered DCE metadata is unreadable: {visit_id}") from None
        if not isinstance(meta, dict):
            raise ValueError(f"registered DCE metadata is invalid: {visit_id}")
        if meta.get("array_shape_policy") not in {None, "zyx_slices_rows_cols"}:
            raise ValueError("registered DCE array-axis policy is unsupported")

        target = meta.get("target_geometry")
        source = meta.get("source_geometry")
        if bundle_spacing_zyx is None and (
            not isinstance(target, dict) or not isinstance(source, dict)
        ):
            pixel_spacing = meta.get("pixel_spacing")
            top_level_spacing = (
                meta.get("spacing_between_slices"),
                pixel_spacing[0]
                if isinstance(pixel_spacing, list) and len(pixel_spacing) == 2
                else None,
                pixel_spacing[1]
                if isinstance(pixel_spacing, list) and len(pixel_spacing) == 2
                else None,
            )
            top_level_shape = (meta.get("n_slices"), meta.get("rows"), meta.get("cols"))
            try:
                spacing = tuple(float(value) for value in top_level_spacing)
                shape = tuple(int(value) for value in top_level_shape)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"registered DCE geometry is missing: {visit_id}") from None
            if (
                not all(np.isfinite(spacing))
                or any(value <= 0 for value in spacing)
                or any(value <= 0 for value in shape)
            ):
                raise ValueError(f"registered DCE geometry is missing: {visit_id}")
            return spacing, shape
        if not isinstance(target, dict) or not isinstance(source, dict):
            raise ValueError(f"registered DCE geometry is missing: {visit_id}")

        def geometry_values(
            geometry: Mapping[str, Any], name: str
        ) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
            spacing_xyz = geometry.get("spacing_xyz")
            shape_zyx = geometry.get("shape_zyx")
            if (
                not isinstance(spacing_xyz, list)
                or len(spacing_xyz) != 3
                or not isinstance(shape_zyx, list)
                or len(shape_zyx) != 3
            ):
                raise ValueError(f"registered DCE {name} geometry is invalid: {visit_id}")
            try:
                spacing_zyx = tuple(float(value) for value in reversed(spacing_xyz))
                shape = tuple(int(value) for value in shape_zyx)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"registered DCE {name} geometry is invalid: {visit_id}"
                ) from None
            if (
                not all(np.isfinite(spacing_zyx))
                or any(value <= 0 for value in spacing_zyx)
                or any(value <= 0 for value in shape)
            ):
                raise ValueError(f"registered DCE {name} geometry is invalid: {visit_id}")
            return spacing_zyx, shape

        target_spacing, target_shape = geometry_values(target, "target")
        source_spacing, _ = geometry_values(source, "source")
        top_level_shape = (meta.get("n_slices"), meta.get("rows"), meta.get("cols"))
        try:
            top_level_shape = tuple(int(value) for value in top_level_shape)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"registered DCE shape is invalid: {visit_id}") from None
        pixel_spacing = meta.get("pixel_spacing")
        top_level_spacing = (
            meta.get("spacing_between_slices"),
            pixel_spacing[0]
            if isinstance(pixel_spacing, list) and len(pixel_spacing) == 2
            else None,
            pixel_spacing[1]
            if isinstance(pixel_spacing, list) and len(pixel_spacing) == 2
            else None,
        )
        try:
            top_level_spacing_array = np.asarray(top_level_spacing, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"registered DCE spacing is invalid: {visit_id}") from None
        if (
            top_level_shape != target_shape
            or not np.allclose(
                top_level_spacing_array,
                np.asarray(target_spacing),
                rtol=0,
                atol=1e-5,
            )
            or (
                bundle_spacing_zyx is not None
                and not np.allclose(
                    np.asarray(source_spacing),
                    np.asarray(bundle_spacing_zyx, dtype=float),
                    rtol=0,
                    atol=1e-5,
                )
            )
        ):
            raise ValueError(f"registered DCE geometry sources disagree: {visit_id}")
        return target_spacing, target_shape

    def _load_bundle_visits(self) -> dict[tuple[str, int], Mapping[str, Any]]:
        if not self.bundle_json.is_file():
            raise FileNotFoundError(f"MeWM bundle does not exist: {self.bundle_json}")
        bundle = json.loads(self.bundle_json.read_text())
        descriptor = bundle.get("artifacts", {}).get("visits")
        if not isinstance(descriptor, dict) or not descriptor.get("path"):
            raise ValueError("MeWM bundle has no visits artifact")
        visits_path = (self.bundle_json.parent / str(descriptor["path"])).resolve()
        if not visits_path.is_file():
            raise FileNotFoundError(f"MeWM visits table does not exist: {visits_path}")
        frame = pd.read_csv(visits_path)
        required = {
            "patient_id",
            "visit_id",
            "visit",
            "dce0_path",
            "meta_path",
            "fold",
            "slice_spacing_mm",
            "row_spacing_mm",
            "column_spacing_mm",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"MeWM visits table is missing columns: {sorted(missing)}")

        rows: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in frame.to_dict(orient="records"):
            key = (patient_key(row["patient_id"]), visit_index(row["visit"]))
            if key in rows:
                raise ValueError(f"duplicate MeWM bundle visit: {key}")
            rows[key] = row
        if not rows:
            raise ValueError("MeWM bundle visits table is empty")
        return rows

    def _load_matching_phase_rows(
        self, bundle_visits: Mapping[tuple[str, int], Mapping[str, Any]]
    ) -> dict[tuple[str, int], tuple[str, ...]]:
        matched: dict[tuple[str, int], tuple[str, ...]] = {}
        for manifest_path in self.phase_manifest_csvs:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"phase manifest does not exist: {manifest_path}")
            frame = pd.read_csv(manifest_path)
            required = {"patient_id", "visit", "dce_paths"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"phase manifest {manifest_path} is missing columns: {sorted(missing)}"
                )
            for row in frame.itertuples(index=False):
                key = (patient_key(row.patient_id), visit_index(row.visit))
                bundle_row = bundle_visits.get(key)
                if bundle_row is None:
                    continue
                paths = _phase_paths(row.dce_paths)
                if paths and paths[0] == _resolved(bundle_row["dce0_path"]):
                    # Later manifests have priority, matching the local merge policy.
                    matched[key] = paths
        return matched

    def source_patient_id(self, patient_id: Any) -> str | None:
        return self.patient_ids.get(patient_key(patient_id))

    def phase_members(
        self, patient_id: Any, timepoint: int, metadata_row: Mapping[str, Any]
    ) -> tuple[str, str, str] | None:
        members, _ = self.phase_selection(patient_id, timepoint, metadata_row)
        return members

    def phase_selection(
        self, patient_id: Any, timepoint: int, metadata_row: Mapping[str, Any]
    ) -> tuple[tuple[str, str, str] | None, str | None]:
        visit = self.visits.get((patient_key(patient_id), int(timepoint)))
        if visit is None:
            return None, None
        if self.input_mode == "registered_dce0_repeat":
            return (visit.dce_paths[0],) * 3, "dce0_repeat"

        indexed = {
            index: path
            for path in visit.dce_paths
            if (index := _acquisition_index(path)) is not None
        }
        desired: list[int] = []
        for prefix in ("pre", "post_early", "post_late"):
            value = metadata_row.get(f"{prefix}_T{timepoint}", np.nan)
            if pd.isna(value):
                desired = []
                break
            desired.append(int(value))
        if len(desired) == 3 and len(set(desired)) == 3 and all(i in indexed for i in desired):
            return (
                tuple(indexed[i] for i in desired),  # type: ignore[return-value]
                "metadata_pre_early_late",
            )

        if len(indexed) >= 3:
            first_three = [indexed[index] for index in sorted(indexed)[:3]]
        elif len(visit.dce_paths) >= 3:
            first_three = list(visit.dce_paths[:3])
        else:
            return None, None
        return tuple(first_three), "first_three_fallback"  # type: ignore[return-value]

    def available_timepoints(
        self,
        patient_id: Any,
        metadata_row: Mapping[str, Any],
        *,
        check_files: bool = True,
    ) -> tuple[int, ...]:
        available = []
        for timepoint in range(4):
            members = self.phase_members(patient_id, timepoint, metadata_row)
            if members is None:
                continue
            if check_files:
                visit = self.visits[(patient_key(patient_id), timepoint)]
                for path in set(members):
                    if not Path(path).is_file():
                        break
                    try:
                        image = nib.load(path)
                    except (OSError, ValueError):
                        raise ValueError(f"registered DCE header is unreadable: {path}") from None
                    if image.ndim != 3 or tuple(image.shape) != visit.shape_zyx:
                        raise ValueError(
                            f"registered DCE header does not match target geometry: {path}"
                        )
                else:
                    available.append(timepoint)
                continue
            available.append(timepoint)
        return tuple(available)

    def load(self, patient_id: Any, timepoint: int, path: str) -> tuple[np.ndarray, list[float]]:
        visit = self.visits.get((patient_key(patient_id), int(timepoint)))
        if visit is None or path not in visit.dce_paths:
            raise ValueError("image is not part of the selected registered visit")
        image = nib.load(path)
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError(f"registered DCE volume is invalid: {path}")
        if tuple(volume.shape) != visit.shape_zyx:
            raise ValueError(f"registered DCE volume shape does not match target geometry: {path}")
        # MeWM's registered NIfTI payloads are stored as (slice, row, column).
        return volume, list(visit.spacing_zyx)


class RegisteredManifestSource(RegisteredMewmSource):
    """Read registered visits directly from an independent, localized manifest.

    Unlike :class:`RegisteredMewmSource`, this source has no Strict-A bundle
    eligibility gate. It is intended for explicitly versioned cohort manifests
    whose rows already identify the registration and local image paths.
    """

    def __init__(
        self,
        registered_manifest_csv: str | Path,
        *,
        input_mode: str = "registered_three_phase",
    ) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        self.input_mode = input_mode
        self.registered_manifest_csv = Path(registered_manifest_csv).expanduser().resolve()
        if not self.registered_manifest_csv.is_file():
            raise FileNotFoundError(
                f"registered manifest does not exist: {self.registered_manifest_csv}"
            )
        frame = pd.read_csv(self.registered_manifest_csv)
        required = {"patient_id", "visit", "dce_paths", "meta_path"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"registered manifest is missing columns: {sorted(missing)}")

        visits: dict[tuple[str, int], RegisteredVisit] = {}
        for row in frame.to_dict(orient="records"):
            key = (patient_key(row["patient_id"]), visit_index(row["visit"]))
            if key in visits:
                raise ValueError(f"duplicate registered manifest visit: {key}")
            paths = _phase_paths(row["dce_paths"])
            minimum_phases = 3 if input_mode == "registered_three_phase" else 1
            if len(paths) < minimum_phases:
                raise ValueError(f"registered manifest visit has too few DCE phases: {key}")
            indices = [_acquisition_index(path) for path in paths]
            known = [index for index in indices if index is not None]
            if len(known) != len(set(known)):
                raise ValueError(f"registered manifest visit has duplicate DCE phases: {key}")
            meta_path = _resolved(row["meta_path"])
            spacing, shape = self._registered_geometry(
                meta_path, None, f"{row['patient_id']}:{row['visit']}"
            )
            visits[key] = RegisteredVisit(
                patient_id=str(row["patient_id"]),
                patient_key=key[0],
                visit=str(row["visit"]),
                visit_index=key[1],
                fold="" if pd.isna(row.get("fold")) else str(row.get("fold", "")),
                dce_paths=paths,
                meta_path=meta_path,
                spacing_zyx=spacing,
                shape_zyx=shape,
            )
        if not visits:
            raise ValueError("registered manifest is empty")
        self.visits = visits

        patient_ids: dict[str, str] = {}
        for visit in visits.values():
            previous = patient_ids.setdefault(visit.patient_key, visit.patient_id)
            if previous != visit.patient_id:
                raise ValueError("registered manifest aliases one numeric patient ID more than once")
        self.patient_ids = patient_ids

    def phase_selection(
        self, patient_id: Any, timepoint: int, metadata_row: Mapping[str, Any]
    ) -> tuple[tuple[str, str, str] | None, str | None]:
        visit = self.visits.get((patient_key(patient_id), int(timepoint)))
        if visit is None:
            return None, None
        if self.input_mode == "registered_dce0_repeat":
            return (visit.dce_paths[0],) * 3, "dce0_repeat"

        indexed = {
            index: path
            for path in visit.dce_paths
            if (index := _acquisition_index(path)) is not None
        }
        desired = []
        for prefix in ("pre", "post_early", "post_late"):
            value = metadata_row.get(f"{prefix}_T{timepoint}", np.nan)
            if pd.isna(value):
                return None, None
            desired.append(int(value))
        if len(set(desired)) != 3 or any(index not in indexed for index in desired):
            return None, None
        return (
            tuple(indexed[index] for index in desired),  # type: ignore[return-value]
            "metadata_pre_early_late",
        )


def create_registered_source(
    adapter: Mapping[str, Any], *, input_mode: str | None = None
) -> RegisteredMewmSource:
    """Construct the configured registered source while preserving legacy configs."""
    mode = input_mode or str(adapter.get("input_mode", "registered_three_phase"))
    source_type = str(adapter.get("source_type", "strict_a_bundle"))
    if source_type == "strict_a_bundle":
        return RegisteredMewmSource(
            adapter["bundle_json"], adapter["phase_manifest_csvs"], input_mode=mode
        )
    if source_type == "registered_manifest":
        return RegisteredManifestSource(adapter["registered_manifest_csv"], input_mode=mode)
    raise ValueError(f"unsupported registered source_type: {source_type}")


def registered_source_contract(adapter: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path provenance fields stored in an extraction contract."""
    source_type = str(adapter.get("source_type", "strict_a_bundle"))
    if source_type == "strict_a_bundle":
        return {
            "bundle_json": _resolved(adapter["bundle_json"]),
            "phase_manifest_csvs": [
                _resolved(path) for path in adapter["phase_manifest_csvs"]
            ],
        }
    if source_type == "registered_manifest":
        return {
            "source_type": source_type,
            "registered_manifest_csv": _resolved(adapter["registered_manifest_csv"]),
        }
    raise ValueError(f"unsupported registered source_type: {source_type}")


def build_registered_volume(
    source: RegisteredMewmSource,
    patient_id: Any,
    timepoint: int,
    metadata_row: Mapping[str, Any],
    *,
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    target_depth: int = 192,
    target_hw: int = 384,
) -> torch.Tensor | None:
    """Build Pillar-0 input `[1,3,H,W,D]` from one registered MeWM visit."""
    members = source.phase_members(patient_id, timepoint, metadata_row)
    if members is None:
        return None
    cached: dict[str, torch.Tensor] = {}
    channels = []
    for path in members:
        channel = cached.get(path)
        if channel is None:
            volume, spacing = source.load(patient_id, timepoint, path)
            volume = resample_volume(
                volume, spacing, target_spacing=tuple(float(x) for x in target_spacing)
            )
            volume = normalize_volume(volume)
            dhw = pad_or_crop(
                torch.from_numpy(volume).float(),
                target_d=int(target_depth),
                target_hw=int(target_hw),
            )
            channel = dhw.permute(1, 2, 0).contiguous()
            cached[path] = channel
        channels.append(channel)
    result = torch.stack(channels, dim=0).unsqueeze(0)
    expected = (1, 3, int(target_hw), int(target_hw), int(target_depth))
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise ValueError("constructed Pillar-0 volume is invalid")
    return result


def build_adapted_cohort(
    metadata_csv: str | Path,
    source_splits_dir: str | Path | None,
    source: RegisteredMewmSource,
    *,
    reference_manifest_json: str | Path | None = None,
    min_timepoints: int = 1,
    check_files: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    if not 1 <= int(min_timepoints) <= 4:
        raise ValueError("min_timepoints must be between one and four")
    metadata = pd.read_csv(metadata_csv)
    required = {"pid", "pCR"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if metadata["pid"].duplicated().any():
        raise ValueError("metadata contains duplicate patient IDs")

    by_key: dict[str, pd.Series] = {}
    for _, row in metadata.iterrows():
        key = patient_key(row["pid"])
        if key in by_key:
            raise ValueError("metadata contains duplicate numeric patient IDs")
        by_key[key] = row

    reference: dict[str, Mapping[str, Any]] = {}
    if reference_manifest_json is not None:
        reference_path = Path(reference_manifest_json)
        payload = json.loads(reference_path.read_text())
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("reference pCR manifest has no records")
        requested = {split: [] for split in SPLIT_NAMES}
        split_alias = {"train": "train", "validation": "val", "val": "val", "test": "test"}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("reference pCR manifest record is invalid")
            pid = str(record.get("patient_id", ""))
            key = patient_key(pid)
            split = split_alias.get(str(record.get("split", "")))
            if split is None or key in reference:
                raise ValueError("reference pCR manifest split or patient identity is invalid")
            valid = record.get("valid_timepoints")
            if (
                not isinstance(valid, list)
                or len(valid) != 4
                or any(type(value) is not bool for value in valid)
            ):
                raise ValueError("reference pCR validity mask is invalid")
            reference[key] = record
            requested[split].append(pid)
    else:
        if source_splits_dir is None:
            raise ValueError("source_splits_dir is required without a reference manifest")
        source_splits_dir = Path(source_splits_dir)
        requested = {
            split: load_ids(source_splits_dir / f"{split}_ids.txt")
            for split in SPLIT_NAMES
        }
    requested_sets = [{patient_key(value) for value in values} for values in requested.values()]
    if any(requested_sets[i] & requested_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("source patient splits overlap")

    selected: dict[str, list[str]] = {split: [] for split in SPLIT_NAMES}
    selected_rows: list[pd.Series] = []
    visit_counts: dict[str, dict[str, int]] = {split: {} for split in SPLIT_NAMES}
    phase_selection_counts: dict[str, int] = {}
    exclusion_counts = {"missing_metadata": 0, "insufficient_registered_timepoints": 0}
    for split in SPLIT_NAMES:
        for pid in requested[split]:
            row = by_key.get(patient_key(pid))
            if row is None:
                exclusion_counts["missing_metadata"] += 1
                continue
            available = source.available_timepoints(pid, row, check_files=check_files)
            if len(available) < int(min_timepoints):
                exclusion_counts["insufficient_registered_timepoints"] += 1
                continue
            record = reference.get(patient_key(pid))
            if record is not None:
                expected = tuple(
                    index for index, present in enumerate(record["valid_timepoints"]) if present
                )
                if available != expected:
                    raise ValueError(
                        f"registered visits do not match reference pCR manifest for {pid}: "
                        f"expected {expected}, found {available}"
                    )
                if int(record["label"]) != int(row["pCR"]):
                    raise ValueError(f"pCR label mismatch for {pid}")
            out_row = row.copy()
            canonical_pid = source.source_patient_id(pid)
            if canonical_pid is None:
                raise ValueError(f"patient is absent from the selected MeWM source: {pid}")
            if record is not None:
                canonical_pid = str(record["patient_id"])
            out_row["pid"] = canonical_pid
            out_row["mewm_patient_id"] = canonical_pid
            out_row["registered_timepoints"] = ";".join(f"T{t}" for t in available)
            out_row["n_registered_timepoints"] = len(available)
            selected[split].append(canonical_pid)
            selected_rows.append(out_row)
            key = str(len(available))
            visit_counts[split][key] = visit_counts[split].get(key, 0) + 1
            for timepoint in available:
                _, policy = source.phase_selection(pid, timepoint, row)
                if policy is None:
                    raise RuntimeError("available registered visit has no phase-selection policy")
                phase_selection_counts[policy] = phase_selection_counts.get(policy, 0) + 1

    if not selected_rows:
        raise ValueError("adapted cohort is empty")
    adapted = pd.DataFrame(selected_rows)
    labels = {str(pid): int(value) for pid, value in zip(adapted["pid"], adapted["pCR"])}
    if any(value not in (0, 1) for value in labels.values()):
        raise ValueError("adapted cohort labels must be binary")
    summary_splits = {}
    for split, pids in selected.items():
        positives = sum(labels[pid] for pid in pids)
        summary_splits[split] = {
            "patients": len(pids),
            "pcr_positive": positives,
            "pcr_negative": len(pids) - positives,
            "registered_timepoint_counts": visit_counts[split],
        }
    summary = {
        "input_mode": source.input_mode,
        "minimum_timepoints": int(min_timepoints),
        "files_checked": bool(check_files),
        "split_source": "reference_manifest" if reference else "source_split_files",
        "patients": len(adapted),
        "phase_selection_counts": phase_selection_counts,
        "splits": summary_splits,
        "excluded": exclusion_counts,
    }
    return adapted, selected, summary


def write_adapted_cohort(
    output_dir: str | Path,
    metadata: pd.DataFrame,
    splits: Mapping[str, Iterable[str]],
    summary: Mapping[str, Any],
) -> None:
    output_dir = Path(output_dir)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata_enriched.csv", index=False)
    for split in SPLIT_NAMES:
        values = list(splits[split])
        text = "\n".join(values) + ("\n" if values else "")
        (splits_dir / f"{split}_ids.txt").write_text(text)
    (output_dir / "cohort_summary.json").write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True) + "\n"
    )

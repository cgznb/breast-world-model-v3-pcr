"""Three registered phases on the existing fixed T0 ROI32 patient grids."""

from __future__ import annotations

import copy
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset

from . import registered_roi32_data as single

SCHEMA = "registered_three_phase_roi32_v1"
PHASES = ("pre_aqc0", "first_post_aqc1", "metadata_late")
LATENT_SHAPE = (24, 8, 32, 32)


def load_config(path):
    settings = yaml.safe_load(Path(path).read_text())
    if settings["schema"] != SCHEMA or tuple(settings["phase_order"]) != PHASES:
        raise ValueError("Expected the registered three-phase ROI32 contract")
    if settings["image_normalization"] != "frozen_training_dce0_shared_across_phases":
        raise ValueError("All phases must retain the pretrained codec intensity scale")
    if settings["codec_policy"] != "frozen_single_channel_dce0":
        raise ValueError("This configuration evaluates the frozen DCE0 codec")
    baseline = single.read_config(settings["baseline_config"])
    output, original = Path(settings["output_dir"]).resolve(), Path(baseline["output_dir"]).resolve()
    if output == original or output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError("Three-phase outputs must be separate from the single-phase run")
    config = copy.deepcopy(baseline)
    config.update(schema=SCHEMA, output_dir=str(output), three_phase=settings)
    config["fm"]["velocity"]["latent_channels"] = 24
    config["runtime"].update(settings["runtime"])
    return config, baseline


def phase_indices(row, visit):
    try:
        pre, late = (float(row[f"{field}_{visit}"]) for field in ("pre", "post_late"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing explicit pre/late phase metadata") from error
    if pre != 0 or not np.isfinite(late) or not late.is_integer() or late <= 1:
        raise ValueError("Invalid explicit pre/late phase metadata")
    return (0, 1, int(late))


def registered_phase_paths(record, metadata, indices):
    """Resolve only within the exact registered directory, using metadata IDs."""
    ids, paths = metadata["dce_phase_ids"], metadata["dce_paths"]
    if len(ids) != len(paths) or len(set(ids)) != len(ids):
        raise ValueError("Ambiguous registered acquisition phase metadata")
    if metadata.get("registered_to_visit") != "T0":
        raise ValueError("Phase source is not registered to T0")
    intra = metadata.get("intra_visit_registration", {})
    registered = {int(row["phase_id"]) for row in intra.get("registrations", [])
                  if row.get("registered_to_phase_id") == 0}
    if (intra.get("status") != "completed" or intra.get("reference_phase_id") != 0
            or not set(indices[1:]) <= registered):
        raise ValueError("Selected phases lack completed intravisit registration")
    directory = Path(record["image_source"]["path"]).parent.resolve()
    resolved = []
    for index in indices:
        if index not in ids:
            raise ValueError("Explicit selected phase absent from registered metadata")
        name = Path(paths[ids.index(index)]).name
        if not name.endswith(f"_aqc_{index}.nii.gz"):
            raise ValueError("Phase filename disagrees with its acquisition ID")
        path = directory / name
        if path.resolve().parent != directory:
            raise ValueError("Phase leaves its verified registered directory")
        resolved.append(path)
    if resolved[0] != Path(record["image_source"]["path"]).resolve():
        raise ValueError("Precontrast source differs from the baseline DCE0 source")
    return resolved


def restored_sources(config):
    path = Path(config["output_dir"]) / "source_supplement.json"
    if not path.is_file():
        return {}, None
    receipt = single.read_json(path)
    if (receipt["schema"] != "registered_three_phase_source_supplement_v1" or receipt["status"] != "verified"
            or not receipt["existing_anchors_equal"] or not receipt["restored_sources_equal"]):
        raise ValueError("Registered source supplement lacks verified anchors")
    for source in receipt["existing_anchors"]:
        single.verify_identity(source)
    result = {}
    for row in receipt["restored"]:
        key = (row["original_missing_path"], row["visit_id"], row["phase"])
        if key in result or not Path(row["source"]["path"]).is_relative_to(Path(config["output_dir"]) / "source_supplement"):
            raise ValueError("Ambiguous or external registered source supplement")
        single.verify_identity(row["source"])
        result[key] = row["source"]
    return result, single.file_identity(path)


def build_inventory(config, baseline):
    dataset = single.CropDataset(baseline)
    settings = config["three_phase"]
    supplements, supplement_receipt = restored_sources(config)
    mapping = pd.read_excel(settings["identity_mapping"])
    aliases = {}
    for _, row in mapping.iterrows():
        if pd.notna(row["I-SPY 2 Research ID"]):
            key, value = str(row["TCIA PATIENT ID"]), f"ISPY2-{int(row['I-SPY 2 Research ID'])}"
            if key in aliases and aliases[key] != value:
                raise ValueError("Conflicting patient identity aliases")
            aliases[key] = value
    phase_rows = {}
    with Path(settings["phase_metadata"]).open() as handle:
        for row in csv.DictReader(handle):
            key = aliases.get(row["pid"], row["pid"])
            if key in phase_rows:
                raise ValueError("Duplicate canonical phase metadata")
            phase_rows[key] = row
    base = Path(baseline["output_dir"]) / "data"
    visits, missing, reports = [], [], {}
    for patient in dataset.inventory["patients"]:
        path = base / "patients" / (patient["patient_id"] + ".json")
        report = single.read_json(path)
        for identity in report["cache_identities"]:
            single.verify_identity(identity)
        reports[patient["patient_id"]] = report
    canonical_folds = {}
    for original in dataset.records:
        row = copy.deepcopy(original)
        canonical = aliases.get(row["patient_id"], row["patient_id"])
        if canonical_folds.setdefault(canonical, row["fold"]) != row["fold"]:
            raise ValueError("Aliased patient crosses train/validation splits")
        indices = phase_indices(phase_rows.get(canonical, {}), row["visit"])
        single.verify_identity(row["metadata_source"])
        single.verify_identity(row["image_source"])
        metadata = single.read_json(row["metadata_source"]["path"])
        report = reports[row["patient_id"]]
        t0 = single.read_json(report["patient_contract"]["t0_metadata"]["path"])
        if (metadata["target_geometry"] != t0["target_geometry"]
                or not np.allclose(single.acquisition_affine(metadata), single.acquisition_affine(t0), atol=1e-6)):
            raise ValueError("Registered phases do not share the original T0 target grid")
        paths = registered_phase_paths(row, metadata, indices)
        sources = [single.file_identity(path) if path.is_file() else supplements.get((str(path), row["visit_id"], phase))
                   for phase, path in zip(PHASES, paths, strict=True)]
        for phase, path, identity in zip(PHASES, paths, sources, strict=True):
            if identity is None:
                missing.append({"visit_id": row["visit_id"], "fold": row["fold"], "phase": phase, "path": str(path)})
        row.update(phase_indices=list(indices), phase_sources=sources,
                   crop_report=single.file_identity(base / "patients" / (row["patient_id"] + ".json")))
        visits.append(row)
    sources = [base / name for name in ("inventory.json", "normalization.json", "COMPLETE.json")]
    sources += [Path(settings[key]) for key in ("baseline_config", "phase_metadata", "identity_mapping")]
    result = {"schema": SCHEMA, "phase_order": list(PHASES), "visits": visits,
            "patients": dataset.inventory["patients"], "pairs": dataset.inventory["pairs"],
            "sources": [single.file_identity(path) for path in sources], "missing_sources": missing,
            "counts": {"visits": dict(Counter(v["fold"] for v in visits)),
                       "pairs": dict(Counter(p["split"] for p in dataset.inventory["pairs"])),
                       "missing_files": dict(Counter(v["fold"] for v in missing))},
            "image_normalization": dataset.normalization, "test_split": None}
    if supplement_receipt is not None:
        result["supplement_receipt"] = supplement_receipt
    return result


def is_source_completion(previous, current):
    if not previous["missing_sources"] or not set(r["visit_id"] for r in current["visits"]) == set(r["visit_id"] for r in previous["visits"]):
        return False
    candidate = copy.deepcopy(current)
    old_rows = {r["visit_id"]: r for r in previous["visits"]}
    completed = 0
    for row in candidate["visits"]:
        for index, old in enumerate(old_rows[row["visit_id"]]["phase_sources"]):
            if old is None:
                completed += row["phase_sources"][index] is not None
                row["phase_sources"][index] = None
    candidate["missing_sources"] = previous["missing_sources"]
    candidate["counts"]["missing_files"] = previous["counts"]["missing_files"]
    candidate.pop("supplement_receipt", None)
    expected = copy.deepcopy(previous)
    expected.pop("supplement_receipt", None)
    return completed > 0 and candidate == expected


def inventory(config, baseline):
    result = build_inventory(config, baseline)
    path = Path(config["output_dir"]) / "inventory.json"
    if path.exists():
        previous = single.read_json(path)
        if previous != result:
            if not is_source_completion(previous, result) or (path.parent / "latents/binding.json").exists():
                raise ValueError("Three-phase inventory changed; use a new output directory")
            single.write_json(path, result)
    else:
        single.write_json(path, result)
    return result


def crop_phase(source, metadata, report):
    single.verify_identity(source)
    array = single.load_mri_payload(source["path"], metadata)
    mapping = (single.REVERSE @ np.linalg.inv(single.LPS_RAS @ single.acquisition_affine(metadata))
               @ np.asarray(report["crop_affine_ras"]) @ single.REVERSE)
    cropped = single.resample_array(array, mapping, 1)
    support = single.resample_array(np.ones(array.shape, dtype=np.uint8), mapping, 0).astype(bool)
    probes = np.array([[0, 0, 0], [15, 63, 63], [31, 127, 127], [9, 22, 117]], dtype=int)
    locations = probes @ mapping[:3, :3].T + mapping[:3, 3]
    expected = map_coordinates(array, locations.T, order=1, mode="constant", cval=0, prefilter=False)
    if not np.allclose(cropped[tuple(probes.T)], expected, atol=1e-3, rtol=2e-6):
        raise ValueError("Independent three-phase interpolation check failed")
    valid = support & (cropped != 0)
    if not valid.any() or not np.isfinite(cropped).all():
        raise ValueError("Empty or non-finite registered phase crop")
    return cropped, valid, support


def normalize_phase(raw, valid, normalization):
    if raw.shape != single.SHAPE or valid.shape != raw.shape or normalization["std"] <= 0:
        raise ValueError("Invalid phase crop or frozen intensity scale")
    result = np.zeros(raw.shape, dtype=np.float32)
    result[valid] = (raw[valid] - normalization["mean"]) / normalization["std"]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite normalized phase")
    return result


class ThreePhaseCrops(Dataset):
    def __init__(self, baseline, manifest, *, split=None, records=None):
        self.baseline = single.CropDataset(baseline)
        self.by_id = {row["visit_id"]: i for i, row in enumerate(self.baseline.records)}
        self.manifest = manifest
        self.records = list(records) if records is not None else [r for r in manifest["visits"] if split is None or r["fold"] == split]
        if not self.records or any(source is None for row in self.records for source in row["phase_sources"]):
            raise ValueError("Selected cohort is empty or has missing registered phases")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        single.verify_identity(row["metadata_source"])
        single.verify_identity(row["crop_report"])
        baseline = self.baseline[self.by_id[row["visit_id"]]]
        report = single.read_json(row["crop_report"]["path"])
        metadata = single.read_json(row["metadata_source"]["path"])
        images, valid, support = [baseline["image"][0]], [baseline["valid"][0]], [baseline["coverage"][0]]
        for source in row["phase_sources"][1:]:
            raw, foreground, covered = crop_phase(source, metadata, report)
            images.append(torch.from_numpy(normalize_phase(raw, foreground, self.baseline.normalization)))
            valid.append(torch.from_numpy(foreground))
            support.append(torch.from_numpy(covered))
        return {**baseline, "image": torch.stack(images), "valid": torch.stack(valid),
                "coverage": torch.stack(support), "fold": row["fold"], "largest_clipped": report["largest_clipped"]}

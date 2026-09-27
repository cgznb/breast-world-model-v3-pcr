"""Import the completed reduced three-phase cache without resampling patient data."""
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .conditioning import validate_condition
from .data import PHASES, VISIT_SCHEMA, PAIR_SCHEMA, VisitStore, PairStore, _legacy_conditions
from .utils import file_identity, codec_file_id, read_json, write_json, load_checkpoint


def verify_identity(identity):
    if file_identity(identity["path"]) != identity:
        raise ValueError(f"Source asset changed: {identity['path']}")


def convert_registered_roi32(root, output, codec_checkpoint):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inventory = read_json(root/"inventory.json")
    cache = read_json(root/"latents/cache_manifest.json")
    previous = read_json(root/"latents/statistics.json")
    complete = read_json(root/"latents/COMPLETE.json")
    if (inventory.get("schema") != "registered_three_phase_roi32_v1"
            or inventory.get("phase_order") != PHASES or inventory["missing_sources"]
            or complete["status"] != "passed"):
        raise ValueError("Expected the completed registered three-phase ROI32 cache")
    verify_identity(cache["codec"])
    codec_payload = load_checkpoint(codec_checkpoint)
    if (codec_payload.get("schema") != "symm_world_codec_v2"
            or codec_payload.get("source_identity") != cache["codec"]):
        raise ValueError("Safe codec does not originate from this cache's original codec")
    by_name = {}
    for asset in cache["files"]:
        verify_identity(asset)
        name = Path(asset["path"]).name
        if name in by_name:
            raise ValueError("Duplicate latent filename")
        by_name[name] = asset
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    crops, metadata, rows = {}, {}, {}
    registration = Counter()
    for visit in inventory["visits"]:
        vid = visit["visit_id"]
        indices = visit["phase_indices"]
        if len(indices) != 3 or indices[:2] != [0, 1] or indices[2] <= 1:
            raise ValueError("Invalid pre / first-post / metadata-late order")
        crop_path = visit["crop_report"]["path"]
        if crop_path not in crops:
            verify_identity(visit["crop_report"])
            crops[crop_path] = read_json(crop_path)
        crop = crops[crop_path]
        if crop["shape_zyx"] != [32, 128, 128] or crop["spacing_zyx_mm"] != [2.0, .7032, .7032]:
            raise ValueError("Expected reduced MRI shape 32x128x128 and original spacing")
        verify_identity(visit["metadata_source"])
        meta = read_json(visit["metadata_source"]["path"])
        if (meta["registered_to_visit"] != "T0"
                or meta["intra_visit_registration"]["status"] != "completed"
                or meta["registration_status"] not in {"fixed_reference", "deformable", "rigid_fallback"}):
            raise ValueError("Registration provenance is incomplete")
        metadata[vid] = meta
        registration[meta["registration_status"]] += 1
        latent = by_name[vid.replace(":", "_")+".npy"]
        z = np.load(latent["path"], allow_pickle=False)
        if z.shape != (24, 8, 32, 32) or z.dtype != np.float16 or not np.isfinite(z).all():
            raise ValueError("Expected finite raw float16 VQ latent [24,8,32,32]")
        # Relative crop transforms use the same array's ZYX lattice. Keep the
        # original physical geometry without guessing its affine convention.
        rows[vid] = {"id": vid, "patient_id": visit["patient_id"], "visit_id": vid,
                     "stage": visit["visit"], "split": visit["fold"], "latent_path": latent["path"],
                     "geometry": {k: crop[k] for k in ("shape_zyx", "spacing_zyx_mm", "crop_affine_ras")},
                     "phase_alignment_verified": True, "source_available_grid": True,
                     "support_assumed": True}
    if len(rows) != len(by_name):
        raise ValueError("Inventory and latent cache visit counts differ")
    pairs = []
    known_dates = {"tcia_metadata_study_uid", "tcia_study_uid_study_date", "local_dicom_study_date"}
    for pair in inventory["pairs"]:
        source, target = (visits[pair[k]] for k in ("earlier_visit_id", "later_visit_id"))
        if any(v["visit_date_source"] not in known_dates for v in (source, target)):
            raise ValueError("Unrecognized original clinical date source")
        delta = (datetime.fromisoformat(target["visit_date"])-datetime.fromisoformat(source["visit_date"])).days
        if pair["interval_missing"] or delta <= 0 or delta != pair["delta_days"]:
            raise ValueError("Clinical interval does not replay from the original visit dates")
        if metadata[source["visit_id"]]["target_geometry"] != metadata[target["visit_id"]]["target_geometry"]:
            raise ValueError("Source and target registration geometries differ")
        conditions = _legacy_conditions(pair)
        conditions.update(interval_verified=True, delta_days=delta)
        pairs.append({"id": pair["pair_id"], "patient_id": pair["patient_id"], "split": pair["split"],
                      "source": rows[source["visit_id"]], "target": rows[target["visit_id"]],
                      "conditions": validate_condition(conditions)})
    training_visits = sorted(v["visit_id"] for v in visits.values() if v["fold"] == "train")
    if (previous["fit_split"] != "train" or previous["fit_visit_ids"] != training_visits
            or previous["phase_order"] != PHASES):
        raise ValueError("Original latent normalization used different training observations")
    common = {"phase_order": PHASES, "latent_channels": 24, "codec_id": codec_file_id(codec_checkpoint),
              "provenance": {"adapter": "registered_three_phase_roi32_observation_v3",
                  "source_inventory": file_identity(root/"inventory.json"),
                  "source_cache": file_identity(root/"latents/cache_manifest.json"),
                  "codec": file_identity(codec_checkpoint), "original_codec": cache["codec"],
                  "registration_status": dict(registration), "image_shape_czyx": [3, 32, 128, 128],
                  "latent_shape_czyx": [24, 8, 32, 32], "array_space": "raw_continuous_VQ",
                  "image_normalization": {k: inventory["image_normalization"][k] for k in ("mean", "std", "fit_split", "scope")},
                  "support_note": "Full cached ROI support assumed; no measured coverage masks imported",
                  "crop_coordinates": "Same-visit array-index ZYX; original physical geometry retained",
                  "end_to_end_test_isolation_verified": False, "pcr_supervision": False}}
    write_json(output/"visits.json", {**common, "schema": VISIT_SCHEMA, "visits": list(rows.values())})
    write_json(output/"pairs.json", {**common, "schema": PAIR_SCHEMA, "pairs": pairs})
    a, b = VisitStore(output/"visits.json"), PairStore(output/"pairs.json")
    stats = a.fit_statistics()
    for key in ("mean", "std"):
        if not np.allclose(stats[key], previous[key], atol=1e-12, rtol=0):
            raise ValueError("Training-only normalization does not replay the original cache")
    if a.patient_splits != b.patient_splits:
        raise ValueError("A and B patient partitions differ")
    write_json(output/"statistics.json", stats)
    report = {**a.audit(), "pairs": len(pairs),
              "pairs_by_split": dict(Counter(p["split"] for p in pairs)),
              "image_shape_czyx": [3, 32, 128, 128], "latent_shape_czyx": [24, 8, 32, 32],
              "raw_arrays_checked": len(rows), "verified_intervals": len(pairs),
              "registration_status": dict(registration), "training_statistics_replayed": True,
              "A_contains_clinical_or_pcr": False, "B_contains_pcr": False}
    write_json(output/"audit.json", report)
    return report

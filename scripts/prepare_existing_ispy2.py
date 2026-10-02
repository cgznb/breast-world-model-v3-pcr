"""Adapt the audited, anonymous I-SPY2 development cache without copying arrays."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from responsewm.data import ManifestStore, PHASES
from responsewm.io import digest, read_json, write_json


def prepare(manifest_path, latent_root, codec_path, output):
    old = read_json(manifest_path)
    if old.get("schema") != "symm_world_manifest_v2" or old.get("phase_order") != PHASES:
        raise ValueError("Expected the audited three-phase V2 development manifest")
    latent_root, output = Path(latent_root).resolve(), Path(output).resolve()
    baseline_clinical = {}
    for pair in old["pairs"]:
        condition = pair["conditions"]
        age = condition.get("age")
        if age is not None and (not isinstance(age, (int, float)) or not np.isfinite(age) or not 0 < age < 120):
            raise ValueError("Invalid age at screening")
        values = [age]
        for field in ("hr_status", "her2_status", "mammaprint"):
            value = condition.get(field)
            if value is not None and str(value) not in {"0", "1"}:
                raise ValueError(f"Unknown baseline binary coding: {field}")
            values.append(None if value is None else int(value))
        pid = pair["patient_id"]
        if baseline_clinical.setdefault(pid, values) != values:
            raise ValueError("Static baseline covariates disagree across a patient's pairs")
    patients = defaultdict(dict)
    for view in old["views"]:
        pid, visit = view["patient_id"], view["visit"]
        if visit not in {"T0", "T1", "T2", "T3"} or visit in patients[pid]:
            raise ValueError("Unknown or duplicate longitudinal visit")
        if view.get("source_available_grid") is not True:
            raise ValueError("The cache does not document a source-available grid")
        path = latent_root / Path(view["latent"]).name
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (24, 8, 32, 32) or not np.isfinite(value).all():
            raise ValueError(f"Invalid raw continuous latent: {path.name}")
        patients[pid][visit] = {**view, "latent": str(path)}

    direct, longitudinal = [], []
    split_counts, labelled_counts, paired_counts = Counter(), Counter(), Counter()
    for pid, visits in sorted(patients.items()):
        if "T0" not in visits:
            raise ValueError("Source-only baseline grid requires an observed T0")
        baseline = visits["T0"]
        split = baseline["split"]
        label = baseline.get("labels", {}).get("pcr")
        if split not in {"train", "val"} or label not in (None, 0, 1):
            raise ValueError("Keep the existing train/validation split and missing pCR labels")
        geometry, grid = baseline["geometry"], baseline["grid_id"]
        for view in visits.values():
            if view["split"] != split or view.get("labels", {}).get("pcr") != label:
                raise ValueError("Patient split or outcome disagrees across visits")
            if view["geometry"] != geometry or view["grid_id"] != grid:
                raise ValueError("Prepared visits do not share the baseline coordinate grid")
        split_counts[split] += 1
        labelled_counts[split] += label is not None
        paired_counts[split] += "T3" in visits
        clinical = baseline_clinical.get(pid, [None] * 4)

        def make_case(landmark, queries, suffix):
            observed = [
                {"latent": visits[f"T{stage}"]["latent"], "day": stage, "available_at": stage}
                for stage in range(landmark + 1) if f"T{stage}" in visits
            ]
            future = [
                {"latent": visits[f"T{stage}"]["latent"], "day": stage,
                 "anatomy_comparable": False} if f"T{stage}" in visits else None
                for stage in queries
            ]
            return {
                "id": f"{pid}_{suffix}", "patient_id": pid, "split": split,
                "input": {"landmark_day": landmark, "observed": observed,
                          "clinical": clinical,
                          "clinical_known_at": [0 if value is not None else None for value in clinical],
                          "queries": [{"day": stage, "known_at": 0, "actions": [],
                                       "actions_known_at": []} for stage in queries],
                          "source_only_geometry": True},
                "target": {"pcr": label, "future": future},
            }

        direct.append(make_case(0, [3], "T0_to_T3"))
        for stage in range(3):
            if f"T{stage}" in visits:
                longitudinal.append(make_case(stage, list(range(stage + 1, 4)), f"T{stage}_trajectory"))

    provenance = {
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": digest(manifest_path),
        "cohort_role": "development_only",
        "independent_test": False,
        "split_policy": "Existing patient-disjoint training/validation assignments retained exactly",
        "geometry": "Same prepared baseline-derived registered coordinate grid; not expert-verified anatomical alignment",
        "anatomy_comparable": "False for every future target; historical flags are not expert verification",
        "query_policy": "Fixed protocol stage indices T1/T2/T3; no actual future scan dates used",
        "clinical_policy": "Age at screening, HR, HER2 and MP are locked-bundle static baseline fields; known_at=0 follows the source data dictionary, not an independent timestamp audit",
        "clinical_source_contract": "mewm_ispy2/contracts.py ClinicalTextPolicy and registered_roi32_data.py baseline_conditions/connected_pairs; consistency rechecked across all exported pairs",
        "action_policy": "No verified prospective treatment plan supplied; no treatment covariates used",
        "pcr_policy": "Existing final binary pCR copied only to supervision; missing remains null",
        "codec_policy": "Frozen original DCE0 codec independently applied to three phases; selected on existing validation cohort",
        "upstream_localizer_patient_overlap": "Unverified",
        "model_initialization": "New world model trained from scratch; no prior V2 generator or pCR weights loaded",
    }
    common = {
        "schema": "responsewm_manifest_v1", "phase_order": PHASES,
        "clinical_features": ["age_at_screening", "hr_positive", "her2_positive", "mammaprint_binary"],
        "action_features": [], "latent_shape": [24, 8, 32, 32],
        "vq_identity": "sha256:" + digest(codec_path), "time_basis": "stage_index",
        "shared_grid_verified": True, "synthetic": False, "provenance": provenance,
    }
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, cases in (("direct_t0_t3", direct), ("longitudinal", longitudinal)):
        path = output / f"{name}.json"
        write_json(path, {**common, "cases": cases})
        reports[name] = ManifestStore(path).audit(scan_arrays=False)
        write_json(output / f"{name}.audit.json", reports[name])
    report = {
        "patients": dict(split_counts), "labelled_patients": dict(labelled_counts),
        "paired_t0_t3_patients": dict(paired_counts), "visits_scanned": len(old["views"]),
        "all_latents_finite_shape_valid": True, "patient_overlap": 0,
        "independent_test": False, "clinical_features": 4, "treatment_features": 0,
        "outputs": {name: str(output / f"{name}.json") for name in reports},
    }
    write_json(output / "adaptation_report.json", report)
    return report


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.latent_root, args.codec, args.output), indent=2))

"""Create an audited v2 patient manifest from existing local v1 longitudinal data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from responsewm.data import ManifestStore
from responsewm.data_v2 import PatientTrajectoryStore, SCHEMA, stage_index
from responsewm.io import digest, write_json


def prepare(manifest_path, output, codec_path=None, scan_arrays=True, allow_synthetic=False):
    source = ManifestStore(manifest_path, allow_synthetic=allow_synthetic)
    if source.manifest["time_basis"] != "stage_index":
        raise ValueError("Cannot infer canonical stages from calendar days")
    if source.a:
        raise ValueError("Legacy query actions lack interval semantics; provide an audited v2 interval plan")
    output = Path(output).resolve()
    path = output / "patient_trajectories_v2.json"
    if path.exists():
        raise FileExistsError(f"Preserving existing v2 manifest: {path}")
    if codec_path is not None and source.manifest["vq_identity"] != "sha256:" + digest(codec_path):
        raise ValueError("Codec checkpoint hash differs from the existing latent contract")
    patients = {}
    visits = {}
    availability_evidence = set()
    for case in source.cases:
        pid, inp, target = case["patient_id"], case["input"], case["target"]
        baseline = {"values": inp["clinical"], "known_at": inp["clinical_known_at"]}
        patient = patients.setdefault(pid, {"patient_key": pid, "split": case["split"], "baseline": deepcopy(baseline),
                                           "visits": [], "events": [], "interval_plans": [],
                                           "target": {"pcr": target["pcr"], "label_source":
                                                      source.manifest.get("provenance", {}).get("pcr_policy", "Inherited existing v1 final binary pCR supervision")}})
        if patient["split"] != case["split"] or patient["baseline"] != baseline:
            raise ValueError("Patient split or static baseline differs across legacy cases")
        if target["pcr"] is not None:
            if patient["target"]["pcr"] not in (None, target["pcr"]):
                raise ValueError("Conflicting final labels")
            patient["target"]["pcr"] = target["pcr"]
        observed_aux = target.get("observed_auxiliary", [None] * len(inp["observed"]))
        observations = [(v, aux, True) for v, aux in zip(inp["observed"], observed_aux)]
        observations += [(v, v.get("auxiliary"), False) for v in target["future"] if v is not None]
        for old, aux, observed in observations:
            stage = stage_index(old["day"])
            key = (pid, stage)
            record = {"stage": stage, "latent": str(source.resolve(old["latent"])),
                      "available_at": old.get("available_at", stage),
                      "anatomy_comparable": old.get("anatomy_comparable", False)}
            if aux:
                record["auxiliary"] = str(source.resolve(aux))
            previous = visits.get(key)
            if previous is not None:
                if previous["latent"] != record["latent"]:
                    raise ValueError("Different latent files assigned to the same patient stage")
                if previous.get("auxiliary") and aux and previous["auxiliary"] != record["auxiliary"]:
                    raise ValueError("Conflicting auxiliary sidecars")
                if observed and key in availability_evidence and previous["available_at"] != record["available_at"]:
                    raise ValueError("Conflicting observed MRI availability declarations")
                if not observed or key in availability_evidence:
                    record["available_at"] = previous["available_at"]
                record["anatomy_comparable"] = previous["anatomy_comparable"] and record["anatomy_comparable"]
                if previous.get("auxiliary"):
                    record["auxiliary"] = previous["auxiliary"]
            visits[key] = record
            if observed:
                availability_evidence.add(key)
    for (pid, _), visit in sorted(visits.items()):
        patients[pid]["visits"].append(visit)
    manifest = {key: deepcopy(value) for key, value in source.manifest.items() if key not in {"schema", "cases"}}
    manifest.update(schema=SCHEMA, canonical_stages=[0, 1, 2, 3], patients=[patients[k] for k in sorted(patients)])
    manifest["provenance"] = {**manifest.get("provenance", {}),
                              "migration_source_manifest": str(source.path), "migration_source_sha256": source.manifest_digest,
                              "migration_policy": "Merge real visits and retain patient splits; no arrays copied or missing scans imputed",
                              "availability_policy": "Observed available_at retained. Target-only visits inherit protocol stage availability from v1; not an independently audited timestamp",
                              "target_only_availability_declarations": len(visits) - len(availability_evidence),
                              "codec_hash_verified_against_file": codec_path is not None,
                              "anatomy_invariance": False}
    write_json(path, manifest)
    store = PatientTrajectoryStore(path, allow_synthetic=allow_synthetic)
    audit = store.audit(scan_arrays=scan_arrays)
    write_json(output / "patient_trajectories_v2.audit.json", audit)
    stats = store.fit_statistics()
    write_json(output / "train_statistics_v2.json", stats)
    report = {"manifest": str(path), "audit": str(output / "patient_trajectories_v2.audit.json"),
              "statistics": str(output / "train_statistics_v2.json"), "source_preserved": True,
              "split_summary": audit["split_summary"], "codec_file_verified": codec_path is not None}
    write_json(output / "migration_report_v2.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codec")
    parser.add_argument("--skip-array-audit", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output, args.codec, not args.skip_array_audit, args.allow_synthetic), indent=2))

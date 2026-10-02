"""Lossless view-path resolution from an existing V2 manifest.

Prospective covariates/query schedules must be supplied explicitly. In
stage-index mode T0/T1/T2/T3 are categorical ordered stage coordinates, NEVER
claimed to be measured calendar days. This avoids using unknown future dates.
"""
from pathlib import Path
import copy
from .io import read_json,write_json,digest
from .data import ManifestStore,PHASES,exact_keys


def convert_v2(v2_manifest,landmark_spec,output):
    old_path = Path(v2_manifest).resolve(); old = read_json(old_path)
    if old.get("schema") != "symm_world_manifest_v2" or old.get("phase_order") != PHASES:
        raise ValueError("Expected the existing world.py convert-legacy V2 manifest")
    spec = read_json(landmark_spec)
    views = {v["id"]:v for v in old["views"]}
    new = copy.deepcopy(spec)
    new["schema"] = "responsewm_manifest_v1"
    new["phase_order"] = PHASES
    new["synthetic"] = False
    basis = new.get("time_basis","calendar_days")
    used = set()
    def resolve_view(view_id,pid,split,day):
        if view_id not in views:
            raise ValueError(f"Unknown V2 view_id: {view_id}")
        v = views[view_id]
        if v["patient_id"] != pid or v["split"] != split:
            raise ValueError("V2 view crosses patient/split boundary")
        if basis == "stage_index" and int(v["visit"].removeprefix("T")) != day:
            raise ValueError("V2 longitudinal stage disagrees with the requested stage index")
        used.add(view_id)
        p = Path(v["latent"])
        return str((old_path.parent/p).resolve()) if not p.is_absolute() else str(p)
    for case in new["cases"]:
        pid,split = case["patient_id"],case["split"]
        for v in case["input"]["observed"]:
            vid = v.pop("view_id")
            v["latent"] = resolve_view(vid,pid,split,v["day"])
        for v in case["target"]["future"]:
            if v is not None:
                vid = v.pop("view_id")
                v["latent"] = resolve_view(vid,pid,split,v["day"])
        # Relative sidecar paths are relative to the spec, not the output directory.
        for v in case["target"]["future"]:
            if v is not None and v.get("auxiliary"):
                v["auxiliary"] = str((Path(landmark_spec).resolve().parent/v["auxiliary"]).resolve())
        if "observed_auxiliary" in case["target"]:
            case["target"]["observed_auxiliary"] = [str((Path(landmark_spec).resolve().parent/p).resolve()) if p else None
                                                       for p in case["target"]["observed_auxiliary"]]
    new["provenance"] = {"v2_manifest_sha256":digest(v2_manifest),"landmark_spec_sha256":digest(landmark_spec),
                         "resolved_views":len(used),"inferred_treatments":False,"inferred_query_dates":False}
    write_json(output,new)
    return ManifestStore(output).audit(scan_arrays=True)

"""Separate manifests enforce single-visit A and longitudinal B responsibilities."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import torch
from .utils import file_digest, file_identity, read_json, write_json
from .conditioning import ALLOWED, validate_condition

PHASES = ["pre_aqc0", "first_post_aqc1", "metadata_late"]
VISIT_SCHEMA = "three_phase_visits_v3"
PAIR_SCHEMA = "three_phase_pairs_v3"
VISIT_KEYS = {"id", "patient_id", "visit_id", "stage", "split", "latent_path", "geometry",
              "valid_path", "image_path", "kinetic_path", "segmentation_path", "kinetic_provenance",
              "phenotype_label", "domain_label", "augmentations", "phase_alignment_verified",
              "support_assumed", "source_available_grid", "image_normalization"}


def resolve(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def array(path, key="latent"):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as obj:
            if key not in obj:
                raise ValueError(f"{path} has no array {key!r}")
            value = np.asarray(obj[key], dtype=np.float32)
    else:
        value = np.asarray(loaded, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite array at {path}")
    return torch.from_numpy(np.array(value, copy=True))


def stage_index(stage):
    if not isinstance(stage, str) or len(stage) < 2 or stage[0] != "T" or not stage[1:].isdigit():
        raise ValueError(f"Invalid clinical stage {stage!r}")
    return int(stage[1:])


def validate_visit(v):
    if not isinstance(v, dict) or set(v) - VISIT_KEYS:
        raise ValueError(f"Observation record contains unknown/forbidden fields: {sorted(set(v)-VISIT_KEYS)}")
    for key in ("id", "patient_id", "visit_id", "latent_path"):
        if not isinstance(v.get(key), str) or not v[key]:
            raise ValueError(f"Missing visit field {key}")
    if v.get("split") not in {"train", "val", "test"}:
        raise ValueError("Each visit must have a patient-level train/val/test split")
    if "stage" in v:
        stage_index(v["stage"])
    for key in ("phenotype_label", "domain_label"):
        label = v.get(key, -1)
        if isinstance(label, bool) or not isinstance(label, int) or label < -1:
            raise ValueError(f"{key} must be a declared class index or -1 (missing)")
    for key in ("phase_alignment_verified", "source_available_grid", "support_assumed"):
        if key in v and not isinstance(v[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    if v.get("kinetic_path") and v.get("kinetic_provenance") != "measured_same_visit_shared_normalization":
        raise ValueError("Kinetic supervision needs measured same-visit/shared-normalization provenance")
    for aug in v.get("augmentations", []):
        identity_key = "base_latent_identity" if "base_latent_identity" in aug else "base_latent_sha256"
        allowed = {"latent_path", "action", "transform_space", "visit_id", identity_key, "codec_id"}
        if set(aug) != allowed or aug["transform_space"] != "image" or aug["visit_id"] != v["visit_id"]:
            raise ValueError("Domain augmentation must originate from this visit in IMAGE space")
        action = np.asarray(aug["action"], dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all() or action[0] <= -1 or np.any(action[2:] < 0):
            raise ValueError("Invalid four-scalar image augmentation parameters")
    return v


def load_visit(row, base, *, include_aux=False, augmentation=None, strict_alignment=False):
    if strict_alignment and not row.get("phase_alignment_verified", False):
        raise ValueError("Same-visit DCE alignment is not verified")
    z = array(resolve(base, row["latent_path"]))
    if z.ndim != 4 or z.shape[0] != 24:
        raise ValueError("Three-phase latent must be [24,D,H,W]")
    if row.get("valid_path"):
        valid = array(resolve(base, row["valid_path"]), "valid")
        if valid.ndim == 3:
            valid = valid[None]
        if valid.shape[0] not in (1, 3) or valid.shape[1:] != z.shape[1:] or ((valid < 0) | (valid > 1)).any():
            raise ValueError("Validity mask must match the latent lattice")
    else:
        valid = torch.ones(1, *z.shape[1:])
    if not (valid.amin(0) > 0).any():
        raise ValueError("No common phase support")
    item = {"latent": z, "valid": valid, "record": row,
            "action": torch.zeros(4), "context_latent": z}
    if augmentation is not None:
        path = resolve(base, row["latent_path"])
        matches = (augmentation["base_latent_identity"] == file_identity(path)
                   if "base_latent_identity" in augmentation
                   else augmentation["base_latent_sha256"] == file_digest(path))
        if not matches:
            raise ValueError("Domain bank was made from a different raw latent")
        transformed = array(resolve(base, augmentation["latent_path"]))
        if transformed.shape != z.shape:
            raise ValueError("Domain bank changes image geometry")
        item.update(context_latent=transformed, action=torch.tensor(augmentation["action"]).float())
    if include_aux:
        for field, key, channels in (("kinetic_path", "kinetics", 2), ("segmentation_path", "segmentation", 1)):
            if row.get(field):
                y = array(resolve(base, row[field]), key)
                if y.ndim == 3:
                    y = y[None]
                if y.shape != (channels, *z.shape[1:]):
                    raise ValueError(f"{field} must already be on the matching latent lattice")
                if key == "segmentation" and ((y < 0) | (y > 1)).any():
                    raise ValueError("Segmentation target must lie in [0,1]")
                item[key] = y
    return item


class VisitStore:
    """Does not read pair manifests or longitudinal condition fields, even at initialization."""
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.base = self.path.parent
        self.manifest = read_json(self.path)
        m = self.manifest
        allowed = {"schema", "phase_order", "latent_channels", "codec_id", "visits", "provenance"}
        if set(m) - allowed or m.get("schema") != VISIT_SCHEMA or m.get("phase_order") != PHASES or m.get("latent_channels") != 24:
            raise ValueError("A requires a pure visits-v3 manifest, NOT a pair/future manifest")
        self.codec_id = m.get("codec_id")
        if not isinstance(self.codec_id, str) or not self.codec_id:
            raise ValueError("Declare the matching codec_id")
        self.visits = [validate_visit(copy.deepcopy(v)) for v in m["visits"]]
        if not self.visits or len({v["id"] for v in self.visits}) != len(self.visits):
            raise ValueError("Empty or duplicate visit IDs")
        if len({(v["patient_id"], v["visit_id"]) for v in self.visits}) != len(self.visits):
            raise ValueError("A manifest must deduplicate patient/visit, not overweight pair-specific views")
        self.patient_splits = {}
        for v in self.visits:
            if self.patient_splits.setdefault(v["patient_id"], v["split"]) != v["split"]:
                raise ValueError("A patient overlaps splits")
            for aug in v.get("augmentations", []):
                if aug["codec_id"] != self.codec_id:
                    raise ValueError("Augmentation codec_id mismatch")
        self.digest = file_digest(self.path)
        self.identity = file_identity(self.path)
        self._cache = {}

    def records(self, split):
        return [v for v in self.visits if v["split"] == split]

    def read(self, record, **kwargs):
        if record["id"] in self._cache and kwargs.get("augmentation") is None:
            if kwargs.get("strict_alignment") and not record.get("phase_alignment_verified", False):
                raise ValueError("Same-visit DCE alignment is not verified")
            return self._cache[record["id"]]
        return load_visit(record, self.base, **kwargs)

    def preload(self):
        self._cache = {r["id"]: load_visit(r, self.base, include_aux=True) for r in self.visits}

    def fit_statistics(self):
        """Train-only, per-visit moments with valid coverage; no future pairing is consulted."""
        total, square, count = np.zeros(24), np.zeros(24), np.zeros(24)
        n = 0
        for v in self.records("train"):
            item = self.read(v)
            z = item["latent"].numpy().astype(np.float64)
            weight = item["valid"].numpy().astype(np.float64)
            weight = np.repeat(weight, 24, axis=0) if weight.shape[0] == 1 else np.repeat(weight, 8, axis=0)
            total += (z*weight).sum((1, 2, 3))
            square += (z*z*weight).sum((1, 2, 3))
            count += weight.sum((1, 2, 3))
            n += 1
        if not n or (count <= 0).any():
            raise ValueError("No valid training data for normalization")
        mean = total/count
        std = np.sqrt(np.maximum(square/count - mean**2, 0))
        if not np.isfinite(std).all() or (std < 1e-8).any():
            raise ValueError("Degenerate training latent channel")
        return {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train", "visits": n,
                "patients": len({v["patient_id"] for v in self.records("train")})}

    def assets(self, domain_enabled=False):
        paths = set()
        for v in self.visits:
            for key in ("latent_path", "valid_path", "kinetic_path", "segmentation_path"):
                if v.get(key):
                    paths.add(str(resolve(self.base, v[key])))
            if domain_enabled:
                paths.update(str(resolve(self.base, a["latent_path"])) for a in v.get("augmentations", []))
        return {p: file_identity(p) for p in sorted(paths)}

    def audit(self):
        return {"schema": VISIT_SCHEMA, "visits": len(self.visits),
                "visits_by_split": dict(Counter(v["split"] for v in self.visits)),
                "patients_by_split": dict(Counter(self.patient_splits.values())),
                "unverified_phase_alignment": sum(not v.get("phase_alignment_verified", False) for v in self.visits),
                "assumed_support": sum(not v.get("valid_path") for v in self.visits),
                "future_predictor": False, "longitudinal_pairs_used": False}


class PairStore:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.base = self.path.parent
        self.manifest = read_json(self.path)
        m = self.manifest
        if m.get("schema") != PAIR_SCHEMA or m.get("phase_order") != PHASES or m.get("latent_channels") != 24:
            raise ValueError("B requires a pairs-v3 manifest")
        self.codec_id = m.get("codec_id")
        if not isinstance(self.codec_id, str) or not self.codec_id:
            raise ValueError("Declare the matching codec_id")
        self.pairs = copy.deepcopy(m["pairs"])
        self.patient_splits = {}
        if len({p["id"] for p in self.pairs}) != len(self.pairs) or not self.pairs:
            raise ValueError("Empty or duplicated pairs")
        for p in self.pairs:
            s, t = validate_visit(p["source"]), validate_visit(p["target"])
            if not s["patient_id"] == t["patient_id"] == p["patient_id"] or not s["split"] == t["split"] == p["split"]:
                raise ValueError("Cross-patient/split pair")
            if self.patient_splits.setdefault(p["patient_id"], p["split"]) != p["split"]:
                raise ValueError("B patient overlaps splits")
            if stage_index(s["stage"]) >= stage_index(t["stage"]):
                raise ValueError("Pair must be chronologically forward")
            p["conditions"] = validate_condition(p["conditions"])
            if p["conditions"].get("stage_i") != s["stage"] or p["conditions"].get("stage_j") != t["stage"]:
                raise ValueError("Pair conditions do not match clinical stages")
        self.digest = file_digest(self.path)
        self.identity = file_identity(self.path)
        self._cache = {}

    def records(self, split):
        return [p for p in self.pairs if p["split"] == split]

    def assets(self):
        paths = set()
        for p in self.pairs:
            for r in (p["source"], p["target"]):
                for key in ("latent_path", "valid_path"):
                    if r.get(key):
                        paths.add(str(resolve(self.base, r[key])))
        return {p: file_identity(p) for p in sorted(paths)}

    def preload(self):
        for pair in self.pairs:
            for row in (pair["source"], pair["target"]):
                key = (str(resolve(self.base, row["latent_path"])), row.get("valid_path"))
                if key not in self._cache:
                    self._cache[key] = load_visit(row, self.base)

    def read(self, row):
        key = (str(resolve(self.base, row["latent_path"])), row.get("valid_path"))
        return self._cache[key] if key in self._cache else load_visit(row, self.base)

    def source(self, pair):
        """No target file access is performed here."""
        return self.read(pair["source"])

    def batch(self, pairs, device):
        sources, targets = [], []
        for p in pairs:
            sources.append(self.source(p))
            targets.append(self.read(p["target"]))
        shapes = {tuple(x["latent"].shape) for x in sources + targets}
        if len(shapes) != 1:
            raise ValueError("Batch latent shapes must match; do not silently resize longitudinal targets")
        return {"source_raw": torch.stack([x["latent"] for x in sources]).to(device),
                "target_raw": torch.stack([x["latent"] for x in targets]).to(device),
                "source_valid": torch.stack([x["valid"].amin(0, keepdim=True) for x in sources]).to(device),
                "target_valid": torch.stack([x["valid"].amin(0, keepdim=True) for x in targets]).to(device),
                "conditions": [p["conditions"] for p in pairs], "records": pairs}


class PatientSampler:
    """Counter-based patient -> visit/pair sampling, exactly replayable after resume."""
    def __init__(self, records, seed):
        self.groups = defaultdict(list)
        for r in records:
            self.groups[r["patient_id"]].append(r)
        self.patients = sorted(self.groups)
        self.seed = seed
        if not self.patients:
            raise ValueError("Empty training split")

    def batch(self, counter, size):
        rng = np.random.default_rng(np.random.SeedSequence([int(self.seed), int(counter)]))
        selected = []
        for _ in range(size):
            pid = self.patients[int(rng.integers(len(self.patients)))]
            rows = self.groups[pid]
            selected.append(rows[int(rng.integers(len(rows)))])
        return selected


def _legacy_conditions(pair):
    values = {**(pair.get("baseline_clinical") or {}), **(pair.get("treatment") or {})}
    if any(any(s in k.lower() for s in ("pcr", "survival", "recurrence", "pathology", "outcome", "rcb")) for k in values):
        raise ValueError("Legacy clinical fields contain outcomes")
    kept = {k: v for k, v in values.items() if k in ALLOWED}
    kept.update(stage_i=pair["earlier_stage"], stage_j=pair["later_stage"], delta_days=pair.get("delta_days"),
                interval_verified=(pair.get("interval_source") == "verified_relative_dicom_study_date"
                                   and not pair.get("interval_missing", True)))
    return validate_condition(kept)


def convert_legacy(root, output, codec_id, qc_path=None):
    """Read the user's existing admitted inventory; write TWO distinct manifests.

    Isolated visits absent from the all-pairs inventory cannot be recovered here:
    add them through a separate visits manifest created from the source inventory.
    No patient arrays or original manifests are overwritten.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if any((output/name).exists() for name in ("visits.json", "pairs.json")):
        raise FileExistsError("Use a new manifest output directory")
    inv = read_json(root / "admitted_inventory.json")
    if inv.get("phase_order") != PHASES:
        raise ValueError("Unexpected legacy phase order")
    quality = read_json(qc_path) if qc_path else {}
    visits = {v["visit_id"]: v for v in inv["visits"]}
    rows = {}
    for view in inv["views"]:
        visit = visits[view["visit_id"]]
        q = quality.get("views", {}).get(view["view_id"], {})
        row = {"id": view["view_id"], "patient_id": view["patient_id"], "visit_id": view["visit_id"],
               "stage": visit["visit"], "split": view["split"],
               "latent_path": str(resolve(root, view["latent_file"])), "geometry": view.get("geometry", {}),
               "phase_alignment_verified": q.get("phase_alignment_verified", False),
               "source_available_grid": q.get("source_available_grid", False), "support_assumed": True}
        if view.get("reference_file"):
            ref = resolve(root, view["reference_file"])
            # References remain optional; stage A never reads image_path during ordinary cache training.
            row["image_path"] = str(ref)
            if ref.is_file():
                with np.load(ref, allow_pickle=False) as a:
                    if "support" in a and "images" in a:
                        support = a["support"]
                        shape = a["images"].shape
                        if support.ndim == 1:
                            support = np.unpackbits(support, count=int(np.prod(shape))).reshape(shape)
                        # All subvoxels need support for a latent cell; conservative 4x pooling.
                        import torch.nn.functional as F
                        sm = torch.from_numpy(np.asarray(support, np.float32))[None]
                        sm = (F.avg_pool3d(sm, 4, 4) >= .999).float()[0].numpy()
                        vp = output / "support" / (view["view_id"] + ".npy")
                        vp.parent.mkdir(parents=True, exist_ok=True)
                        np.save(vp, sm)
                        row["valid_path"], row["support_assumed"] = str(vp), False
        for key in ("image_path", "valid_path", "kinetic_path", "segmentation_path", "kinetic_provenance",
                    "phenotype_label", "domain_label", "augmentations", "image_normalization"):
            if key in q:
                row[key] = q[key]
        for key in ("image_path", "valid_path", "kinetic_path", "segmentation_path"):
            if q.get(key):
                row[key] = str(resolve(Path(qc_path).resolve().parent, q[key]))
        if "augmentations" in q:
            row["augmentations"] = copy.deepcopy(q["augmentations"])
            for aug in row["augmentations"]:
                aug["latent_path"] = str(resolve(Path(qc_path).resolve().parent, aug["latent_path"]))
        if "valid_path" in q:
            row["support_assumed"] = False
        rows[view["view_id"]] = validate_visit(row)
    grouped = defaultdict(list)
    source_match = {v["view_id"]: v["source_visit_id"] == v["visit_id"] for v in inv["views"]}
    for row in rows.values():
        grouped[row["patient_id"], row["visit_id"]].append(row)
    unique = [sorted(group, key=lambda r: (not source_match[r["id"]], r["id"]))[0]
              for _, group in sorted(grouped.items())]
    provenance = {"upstream_inventory": file_identity(root / "admitted_inventory.json"),
                  "geometry_note": "Native legacy ROI crops are NOT automatically registered or source-localizable",
                  "singletons_note": "Only visits present in admitted inventory are included"}
    common = {"phase_order": PHASES, "latent_channels": 24, "codec_id": codec_id, "provenance": provenance}
    a = {**common, "schema": VISIT_SCHEMA, "visits": unique}
    pairs = [{"id": str(p["pair_id"]), "patient_id": p["patient_id"], "split": p["split"],
              "source": rows[p["source_view"]], "target": rows[p["target_view"]],
              "conditions": _legacy_conditions(p)} for p in inv["pairs"]]
    b = {**common, "schema": PAIR_SCHEMA, "pairs": pairs}
    output.mkdir(parents=True, exist_ok=True)
    for path in (output/"visits.json", output/"pairs.json"):
        if path.exists():
            raise FileExistsError(f"Will not overwrite {path}")
    write_json(output/"visits.json", a)
    write_json(output/"pairs.json", b)
    VisitStore(output/"visits.json"); PairStore(output/"pairs.json")
    result = {"visits_manifest": str(output/"visits.json"), "pairs_manifest": str(output/"pairs.json"),
              "visits": len(unique), "pairs": len(pairs), "warnings": provenance}
    write_json(output/"conversion_report.json", result)
    return result

"""Auditable patient trajectories with prospective prefix reads and separate truth."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import re
import numpy as np
import torch

from .contracts_v2 import EdgeSupervision, PatientPrefix, TrajectorySupervision
from .data import FORBIDDEN, PHASES, ManifestStore, exact_keys, finite_number, normalize_values
from .io import digest, read_json, stable_hash

SCHEMA = "responsewm_patient_trajectory_v2"
STAGES = (0, 1, 2, 3)


def stage_index(value):
    if not finite_number(value) or value != int(value) or value not in STAGES:
        raise ValueError("R1 requires canonical integer stage indices 0, 1, 2, 3; calendar days unsupported")
    return int(value)


def known_values(values, known_at, as_of, dimension):
    if len(values) != dimension or len(known_at) != dimension:
        raise ValueError("Feature value/known_at dimensions differ")
    for value, when in zip(values, known_at):
        if value is not None and (not finite_number(value) or not finite_number(when) or when < 0):
            raise ValueError("Every finite feature value requires a nonnegative known_at")
        if value is None and when is not None:
            raise ValueError("Missing features must have null known_at")
    mask = np.asarray([v is not None and k <= as_of for v, k in zip(values, known_at)], bool)
    return np.asarray([v if m else 0 for v, m in zip(values, mask)], np.float32), mask


class PatientTrajectoryStore:
    resolve = ManifestStore.resolve
    read_latent = ManifestStore.read_latent
    read_auxiliary = ManifestStore.read_auxiliary
    asset_signature = ManifestStore.asset_signature

    def __init__(self, path, allow_synthetic=False, cache_size=32):
        self.path = Path(path).resolve()
        self.manifest = read_json(path)
        m = self.manifest
        required = {"schema", "time_basis", "canonical_stages", "clinical_features", "action_features",
                    "phase_order", "latent_shape", "vq_identity", "patients", "shared_grid_verified"}
        exact_keys(m, required | {"synthetic", "provenance"}, required)
        if m["schema"] != SCHEMA or m["time_basis"] != "stage_index" or m["canonical_stages"] != list(STAGES):
            raise ValueError("Expected v2 stage-index trajectory schema with canonical [0,1,2,3]")
        if m["phase_order"] != PHASES or m["shared_grid_verified"] is not True:
            raise ValueError("Audited phase order and shared source-available geometry required")
        if m.get("synthetic", False) and not allow_synthetic:
            raise ValueError("Synthetic trajectories require explicit allow_synthetic")
        shape = m["latent_shape"]
        if len(shape) != 4 or shape[0] != 24 or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in shape):
            raise ValueError("Expected continuous VQ latent shape [24,D,H,W]")
        if not isinstance(m["vq_identity"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", m["vq_identity"]):
            raise ValueError("Exact VQ sha256 identity required")
        for group in ("clinical_features", "action_features"):
            names = m[group]
            if any(not isinstance(v, str) or any(k in v.lower() for k in FORBIDDEN) for v in names) or len(names) != len(set(names)):
                raise ValueError("Duplicate or forbidden input feature names")
        self.c, self.a = len(m["clinical_features"]), len(m["action_features"])
        self.patients = m["patients"]
        self.by_split = defaultdict(list)
        self.cache, self.cache_size, self.statistics = OrderedDict(), cache_size, None
        self.manifest_digest = digest(path)
        patient_splits, assets = {}, {}
        for index, patient in enumerate(self.patients):
            required_patient = {"patient_key", "split", "baseline", "visits", "events", "interval_plans", "target"}
            exact_keys(patient, required_patient | {"stage_valid"}, required_patient)
            pid, split = patient["patient_key"], patient["split"]
            if not isinstance(pid, str) or not pid or split not in {"train", "val", "test"}:
                raise ValueError("Invalid patient key/split")
            if pid in patient_splits:
                if patient_splits[pid] != split:
                    raise ValueError("Patient overlaps training/validation/test")
                raise ValueError("Each patient must have exactly one trajectory")
            patient_splits[pid] = split
            exact_keys(patient["baseline"], {"values", "known_at"}, {"values", "known_at"})
            known_values(patient["baseline"]["values"], patient["baseline"]["known_at"], 3, self.c)
            stage_valid = patient.get("stage_valid", [True] * 4)
            if len(stage_valid) != 4 or any(type(v) is not bool for v in stage_valid) or not all(stage_valid):
                raise ValueError("R1 trajectory structurally includes all four canonical stages; missing visits are absent")
            seen = set()
            for visit in patient["visits"]:
                exact_keys(visit, {"stage", "latent", "available_at", "auxiliary", "anatomy_comparable", "event_id"},
                           {"stage", "latent", "available_at"})
                stage = stage_index(visit["stage"])
                if stage in seen or not isinstance(visit["latent"], str):
                    raise ValueError("Duplicate stage or invalid latent path")
                seen.add(stage)
                if not finite_number(visit["available_at"]) or visit["available_at"] < stage:
                    raise ValueError("MRI available_at cannot precede acquisition stage")
                if "anatomy_comparable" in visit and type(visit["anatomy_comparable"]) is not bool:
                    raise ValueError("Anatomical comparability must be explicitly audited")
                for path in (visit["latent"], visit.get("auxiliary")):
                    if path is not None and assets.setdefault(str(self.resolve(path)), pid) != pid:
                        raise ValueError("A patient asset is shared across patients/splits")
            if not seen or patient["visits"] != sorted(patient["visits"], key=lambda v: v["stage"]):
                raise ValueError("Nonempty visits must be sorted by actual stage")
            if 0 not in seen:
                raise ValueError("Source-available baseline geometry requires T0")
            if patient["events"]:
                raise ValueError("R1 supports MRI and baseline fields; clinical event dynamics require audited R2")
            plan_ids = set()
            for plan in patient["interval_plans"]:
                exact_keys(plan, {"source_stage", "target_stage", "values", "known_at", "kind", "units"},
                           {"source_stage", "target_stage", "values", "known_at", "kind", "units"})
                src, dst = stage_index(plan["source_stage"]), stage_index(plan["target_stage"])
                if dst != src + 1 or (src, dst) in plan_ids or plan["kind"] not in {"planned", "administered"}:
                    raise ValueError("Interval plans require distinct adjacent source/target stages and a declared kind")
                if len(plan["units"]) != self.a:
                    raise ValueError("Every action feature requires a declared unit")
                plan_ids.add((src, dst))
                known_values(plan["values"], plan["known_at"], 3, self.a)
            target = patient["target"]
            exact_keys(target, {"pcr", "label_source"}, {"pcr", "label_source"})
            if target["pcr"] is not None and (type(target["pcr"]) not in {int, float} or target["pcr"] not in (0, 1)):
                raise ValueError("pCR must be binary or null")
            if not isinstance(target["label_source"], str) or not target["label_source"]:
                raise ValueError("Supervision requires an explicit label source")
            self.by_split[split].append(index)
        if not self.patients:
            raise ValueError("Empty patient cohort")

    def all_assets(self):
        return sorted({str(self.resolve(v[key])) for p in self.patients for v in p["visits"]
                       for key in ("latent", "auxiliary") if v.get(key)})

    def landmarks(self, patient_index, include_terminal=True):
        p = self.patients[patient_index]
        return sorted({stage_index(v["available_at"]) for v in p["visits"]
                       if v["available_at"] in STAGES and (include_terminal or v["available_at"] < 3)})

    def edge_index(self, split="train", include_bridges=False):
        result = {(a, a + 1): [] for a in range(3)}
        if include_bridges:
            result.update({(0, 2): [], (0, 3): [], (1, 3): []})
        for i in self.by_split[split]:
            visits = {v["stage"]: v for v in self.patients[i]["visits"]}
            for edge in result:
                src, dst = edge
                if src in visits and dst in visits and visits[src]["available_at"] <= src:
                    result[edge].append(i)
        return result

    def fit_statistics(self):
        ids = self.by_split["train"]
        if not ids:
            raise ValueError("No training patients for statistics")
        files = {str(self.resolve(v["latent"])) for i in ids for v in self.patients[i]["visits"]}
        n, mean, m2 = 0, np.zeros(24), np.zeros(24)
        for path in sorted(files):
            x = self.read_latent(path).astype(np.float64).reshape(24, -1)
            count, mu = x.shape[1], x.mean(1)
            delta = mu - mean
            m2 += ((x - mu[:, None]) ** 2).sum(1) + delta ** 2 * n * count / (n + count)
            mean += delta * count / (n + count)
            n += count
        std = np.sqrt(m2 / n)
        if (std < 1e-8).any():
            raise ValueError("Degenerate latent channel on training data")
        def features(records, dim):
            if not records:
                return [0.] * dim, [1.] * dim
            values, masks = zip(*(known_values(r["values"], r["known_at"], 3, dim) for r in records))
            x, mask = np.asarray(values, np.float64), np.asarray(masks, bool)
            counts = mask.sum(0).clip(1)
            mu = (x * mask).sum(0) / counts
            sd = np.sqrt((((x - mu) * mask) ** 2).sum(0) / counts)
            return mu.tolist(), np.where(sd < 1e-6, 1, sd).tolist()
        cm, cs = features([self.patients[i]["baseline"] for i in ids], self.c)
        am, ast = features([p for i in ids for p in self.patients[i]["interval_plans"]], self.a)
        self.statistics = {"fit_split": "train", "latent_mean": mean.tolist(), "latent_std": std.tolist(),
                           "clinical_mean": cm, "clinical_std": cs, "action_mean": am, "action_std": ast,
                           "train_patient_hashes": sorted(stable_hash(self.patients[i]["patient_key"]) for i in ids),
                           "manifest_digest": self.manifest_digest, "vq_identity": self.manifest["vq_identity"],
                           "train_visit_count": len(files)}
        return self.statistics

    def set_statistics(self, value):
        expected = sorted(stable_hash(self.patients[i]["patient_key"]) for i in self.by_split["train"])
        if (value.get("fit_split") != "train" or value.get("manifest_digest") != self.manifest_digest
                or value.get("vq_identity") != self.manifest["vq_identity"]
                or value.get("train_patient_hashes") != expected):
            raise ValueError("Normalization must be fitted on this manifest's training split and codec")
        for name, count in (("latent", 24), ("clinical", self.c), ("action", self.a)):
            mu, sd = np.asarray(value[name + "_mean"]), np.asarray(value[name + "_std"])
            if mu.shape != (count,) or sd.shape != (count,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
                raise ValueError("Invalid normalization dimensions or values")
        self.statistics = value

    def make_prefix(self, patient_index, as_of, output_stages=None, device="cpu"):
        return self.batch([(patient_index, as_of)], device=device, supervised=False, output_stages=output_stages)

    def batch(self, tasks, device="cpu", supervised=True, output_stages=None):
        if self.statistics is None:
            raise ValueError("Fit/load TRAIN statistics before creating model inputs")
        if not tasks:
            raise ValueError("Empty prefix batch")
        tasks = [(i, stage_index(j)) for i, j in tasks]
        histories = [[v for v in self.patients[i]["visits"] if v["stage"] <= j and v["available_at"] <= j] for i, j in tasks]
        if any(not h for h in histories):
            raise ValueError("Every prefix needs at least one actually available MRI")
        b, t, f = len(tasks), max(map(len, histories)), max(3 - j for _, j in tasks)
        shape, st = self.manifest["latent_shape"], self.statistics
        z = torch.zeros(b, t, *shape); om = torch.zeros(b, t, dtype=torch.bool)
        od = torch.zeros(b, t); oa = torch.zeros(b, t)
        clinical = torch.zeros(b, self.c); cm = torch.zeros(b, self.c, dtype=torch.bool)
        ck = torch.full((b, self.c), -1.)
        fd = torch.zeros(b, f); fm = torch.zeros(b, f, dtype=torch.bool)
        actions = torch.zeros(b, f, self.a); am = torch.zeros(b, f, self.a, dtype=torch.bool)
        query = torch.zeros(b, 4, dtype=torch.bool)
        mean, std = (torch.tensor(st[key])[:, None, None, None] for key in ("latent_mean", "latent_std"))
        for row, ((index, as_of), history) in enumerate(zip(tasks, histories)):
            p = self.patients[index]
            for column, visit in enumerate(history):
                z[row, column] = (torch.from_numpy(self.read_latent(visit["latent"])) - mean) / std
                om[row, column], od[row, column], oa[row, column] = True, visit["stage"], visit["available_at"]
            values, mask = known_values(p["baseline"]["values"], p["baseline"]["known_at"], as_of, self.c)
            clinical[row] = torch.from_numpy(normalize_values(values, mask, st["clinical_mean"], st["clinical_std"]))
            cm[row] = torch.from_numpy(mask)
            ck[row] = torch.tensor([when if valid else -1. for when, valid in zip(p["baseline"]["known_at"], mask)])
            plans = {(v["source_stage"], v["target_stage"]): v for v in p["interval_plans"]}
            for column, stage in enumerate(range(as_of + 1, 4)):
                fd[row, column], fm[row, column] = stage, True
                plan = plans.get((stage - 1, stage))
                if plan is not None:
                    values, mask = known_values(plan["values"], plan["known_at"], as_of, self.a)
                    actions[row, column] = torch.from_numpy(normalize_values(values, mask, st["action_mean"], st["action_std"]))
                    am[row, column] = torch.from_numpy(mask)
            requested = list(range(as_of + 1, 4)) if output_stages is None else [stage_index(v) for v in output_stages]
            if len(requested) != len(set(requested)) or any(j <= as_of for j in requested):
                raise ValueError("Output requests must be distinct future canonical stages")
            query[row, requested] = True
        inp = PatientPrefix(z, om, od, clinical, cm, fd, fm, actions, am,
                            torch.tensor([j for _, j in tasks]), torch.ones(b, 4, dtype=torch.bool), query,
                            tuple(tuple(v["stage"] for v in h) for h in histories), oa, ck)
        if not supervised:
            return inp.to(device)
        future = torch.zeros(b, f, *shape); mask = torch.zeros(b, f, dtype=torch.bool)
        labels = torch.zeros(b); lm = torch.zeros(b, dtype=torch.bool)
        full_mask = torch.zeros(b, 4, dtype=torch.bool); anatomy = torch.zeros_like(mask); auxiliary = []
        for row, ((index, as_of), history) in enumerate(zip(tasks, histories)):
            p = self.patients[index]
            visits = {v["stage"]: v for v in p["visits"]}
            sidecars = [self.read_auxiliary(v.get("auxiliary")) for v in history] + [{} for _ in range(t - len(history))]
            for column in range(f):
                stage = as_of + 1 + column
                visit = visits.get(stage)
                sidecars.append(self.read_auxiliary(visit.get("auxiliary")) if visit is not None else {})
                if visit is not None:
                    future[row, column] = (torch.from_numpy(self.read_latent(visit["latent"])) - mean) / std
                    mask[row, column], full_mask[row, stage] = True, True
                    anatomy[row, column] = visit.get("anatomy_comparable", False)
            auxiliary.append(sidecars)
            if p["target"]["pcr"] is not None:
                labels[row], lm[row] = p["target"]["pcr"], True
        sup = TrajectorySupervision(future, mask, labels, lm, auxiliary, anatomy, full_mask)
        return inp.to(device), sup.to(device)

    def make_edge_task(self, edge, device="cpu"):
        index, src, dst = edge.patient_index, edge.source_stage, edge.target_stage
        src, dst = stage_index(src), stage_index(dst)
        visits = {v["stage"]: v for v in self.patients[index]["visits"]}
        if src >= dst or src not in visits or dst not in visits or visits[src]["available_at"] > src:
            raise ValueError("Edge does not have a legal observed source and real target")
        kind = "adjacent" if dst == src + 1 else "bridge_auxiliary"
        if edge.kind != kind:
            raise ValueError("Edge kind disagrees with actual stage distance")
        prefix = self.make_prefix(index, src, device=device)
        visit = visits[dst]
        mean, std = (torch.tensor(self.statistics[key])[:, None, None, None] for key in ("latent_mean", "latent_std"))
        latent = ((torch.from_numpy(self.read_latent(visit["latent"])) - mean) / std).unsqueeze(0)
        truth = EdgeSupervision(latent, src, dst, edge.kind,
                                torch.tensor([visit.get("anatomy_comparable", False)]),
                                [self.read_auxiliary(visit.get("auxiliary"))])
        return prefix, truth.to(device)

    def make_edge_batch(self, edges, device="cpu"):
        if not edges:
            raise ValueError("Empty edge batch")
        src, dst = stage_index(edges[0].source_stage), stage_index(edges[0].target_stage)
        kind = edges[0].kind
        targets, histories = [], []
        for edge in edges:
            if (stage_index(edge.source_stage), stage_index(edge.target_stage), edge.kind) != (src, dst, kind):
                raise ValueError("Edge batches require the same source, target, and kind")
            visits = {v["stage"]: v for v in self.patients[edge.patient_index]["visits"]}
            if src >= dst or src not in visits or dst not in visits or visits[src]["available_at"] > src:
                raise ValueError("Edge does not have a legal observed source and real target")
            if kind != ("adjacent" if dst == src + 1 else "bridge_auxiliary"):
                raise ValueError("Edge kind disagrees with actual stage distance")
            histories.append(tuple(v["stage"] for v in visits.values()
                                   if v["stage"] <= src and v["available_at"] <= src))
            targets.append(visits[dst])
        if any(history != histories[0] for history in histories[1:]):
            raise ValueError("Edge batches require the same observed stage prefix")
        prefix = self.batch([(edge.patient_index, src) for edge in edges], device=device, supervised=False)
        mean, std = (torch.tensor(self.statistics[key])[:, None, None, None]
                     for key in ("latent_mean", "latent_std"))
        latent = torch.stack([(torch.from_numpy(self.read_latent(visit["latent"])) - mean) / std
                              for visit in targets])
        truth = EdgeSupervision(latent, src, dst, kind,
                                torch.tensor([visit.get("anatomy_comparable", False) for visit in targets]),
                                [self.read_auxiliary(visit.get("auxiliary")) for visit in targets])
        return prefix, truth.to(device)

    def fit_prior(self, model):
        from sklearn.linear_model import LogisticRegression
        x, y = [], []
        for index in self.by_split["train"]:
            p = self.patients[index]
            if p["target"]["pcr"] is None:
                continue
            values, mask = known_values(p["baseline"]["values"], p["baseline"]["known_at"], 0, self.c)
            normalized = normalize_values(values, mask, self.statistics["clinical_mean"], self.statistics["clinical_std"])
            x.append(np.concatenate((normalized, mask.astype(np.float32)))); y.append(p["target"]["pcr"])
        if not y:
            return {"fitted": False, "reason": "No labelled training patients"}
        if len(set(y)) < 2 or self.c == 0:
            rate = (sum(y) + .5) / (len(y) + 1)
            coef, intercept = np.zeros(self.c * 2), float(np.log(rate / (1 - rate)))
        else:
            fit = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(np.asarray(x), y)
            coef, intercept = fit.coef_[0], float(fit.intercept_[0])
        with torch.no_grad():
            prior = model.pcr.prior
            prior.coefficient.copy_(torch.tensor(coef, dtype=torch.float32)); prior.intercept.fill_(intercept)
            prior.fitted.fill_(True)
        return {"fitted": True, "fit_split": "train", "patients": len(y), "patient_balanced": True, "as_of": 0}

    def audit(self, scan_arrays=False):
        summary = {}
        for split, indices in self.by_split.items():
            labels = [self.patients[i]["target"]["pcr"] for i in indices]
            visits = Counter(v["stage"] for i in indices for v in self.patients[i]["visits"])
            edges = self.edge_index(split, include_bridges=True)
            summary[split] = {"patients": len(indices), "labelled_patients": sum(v is not None for v in labels),
                              "positive_patients": sum(v == 1 for v in labels),
                              "visits_by_stage": {str(j): visits[j] for j in STAGES},
                              "adjacent_edges": {f"{a}{b}": len(v) for (a, b), v in edges.items() if b == a + 1},
                              "bridge_auxiliary_edges": {f"{a}{b}": len(v) for (a, b), v in edges.items() if b > a + 1},
                              "observed_landmarks": sum(len(self.landmarks(i)) for i in indices)}
        if scan_arrays:
            for p in self.patients:
                for v in p["visits"]:
                    self.read_latent(v["latent"])
                    self.read_auxiliary(v.get("auxiliary"))
        return {"schema": "responsewm_audit_v2", "manifest_digest": self.manifest_digest,
                "split_summary": summary, "patient_overlap": 0, "arrays_checked": scan_arrays,
                "synthetic": self.manifest.get("synthetic", False), "time_basis": "stage_index",
                "vq_identity": self.manifest["vq_identity"], "clinical_features": self.manifest["clinical_features"],
                "action_features": self.manifest["action_features"],
                "action_status": "ACTION_UNOBSERVED" if self.a == 0 else "availability_masked",
                "provenance": self.manifest.get("provenance", {}),
                "limitations": ["Availability declarations checked for consistency, not independently timestamp-audited",
                                "Shared geometry does not imply expert-verified anatomical correspondence",
                                "Clinical event dynamics are outside the stage-index R1 scope"]}

"""Prospective landmark manifests, train-only statistics and patient-balanced reads.

Data files stay local. This module does not infer a treatment plan from later
records, calculate ROI coordinates from future tumors, or download patient data.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict,OrderedDict
import math
import numpy as np
import torch
from .contracts import ForecastInput,Supervision
from .io import read_json,digest,stable_hash

PHASES = ["pre_aqc0","first_post_aqc1","metadata_late"]
FORBIDDEN = ("pcr","pathology","outcome","recurrence","survival","patient_id","death")


def exact_keys(value,allowed,required=()):
    if not isinstance(value,dict):
        raise ValueError("Expected a JSON object")
    extra,missing = set(value)-set(allowed),set(required)-set(value)
    if extra or missing:
        raise ValueError(f"Unexpected fields {sorted(extra)}, missing fields {sorted(missing)}")


def finite_number(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def _values(values,known,cutoff,count):
    if len(values) != count or len(known) != count:
        raise ValueError("Feature value/availability dimensions differ from schema")
    for value,when in zip(values,known):
        if value is not None and (not finite_number(value) or not finite_number(when) or when > cutoff):
            raise ValueError("A clinical/treatment value was unavailable at the prediction landmark")
    return np.asarray([0 if v is None else v for v in values],np.float32),np.asarray([v is not None for v in values],bool)


def validate_input(record,c,a):
    exact_keys(record,{"landmark_day","observed","clinical","clinical_known_at","queries","source_only_geometry"},
               {"landmark_day","observed","clinical","clinical_known_at","queries","source_only_geometry"})
    day = record["landmark_day"]
    if not finite_number(day) or day < 0 or record["source_only_geometry"] is not True:
        raise ValueError("A nonnegative landmark and audited source-only preprocessing are required")
    visits = record["observed"]
    if not visits:
        raise ValueError("Empty observed history")
    last = -1
    for v in visits:
        exact_keys(v,{"latent","day","available_at"},{"latent","day","available_at"})
        if not isinstance(v["latent"],str) or not finite_number(v["day"]) or not finite_number(v["available_at"]):
            raise ValueError("Invalid observed visit")
        if v["day"] < 0 or v["day"] <= last or v["day"] > day or not v["day"] <= v["available_at"] <= day:
            raise ValueError("Observed MRI chronology/availability invalid")
        last = v["day"]
    if last != day:
        raise ValueError("This version defines the landmark at the latest observed MRI day")
    _values(record["clinical"],record["clinical_known_at"],day,c)
    last = day
    for q in record["queries"]:
        exact_keys(q,{"day","known_at","actions","actions_known_at"},{"day","known_at","actions","actions_known_at"})
        if not finite_number(q["day"]) or not finite_number(q["known_at"]) or q["day"] <= last or q["known_at"] > day:
            raise ValueError("Future query schedule must be defined at the landmark, not inferred from future events")
        _values(q["actions"],q["actions_known_at"],day,a)
        last = q["day"]


def normalize_values(values,mask,mean,std):
    x = (values-np.asarray(mean,np.float32))/np.asarray(std,np.float32)
    return np.where(mask,x,0).astype(np.float32)


class ManifestStore:
    def __init__(self,path,allow_synthetic=False,cache_size=32):
        self.path = Path(path).resolve()
        self.manifest = read_json(path)
        m = self.manifest
        exact_keys(m,{"schema","phase_order","clinical_features","action_features","latent_shape","vq_identity",
                      "shared_grid_verified","synthetic","cases","provenance","time_basis"},
                     {"schema","phase_order","clinical_features","action_features","latent_shape","vq_identity",
                      "shared_grid_verified","cases"})
        m.setdefault("time_basis","calendar_days")
        if m["time_basis"] not in {"calendar_days","stage_index"}:
            raise ValueError("Unknown time basis")
        if m["schema"] != "responsewm_manifest_v1" or m["phase_order"] != PHASES:
            raise ValueError("Unsupported manifest/phase order")
        if m.get("synthetic",False) and not allow_synthetic:
            raise ValueError("Synthetic data require explicit allow_synthetic; never use as clinical evidence")
        if m["shared_grid_verified"] is not True:
            raise ValueError("Joint spatial trajectories require verified shared image coordinates")
        if len(m["latent_shape"]) != 4 or m["latent_shape"][0] != 24 or min(m["latent_shape"]) < 1:
            raise ValueError("Expected [24,D,H,W] latent shape")
        if not isinstance(m["vq_identity"],str) or not m["vq_identity"].strip():
            raise ValueError("Declare the exact VQ checkpoint/normalization identity")
        for group in ("clinical_features","action_features"):
            values = m[group]
            if len(set(values)) != len(values) or any(not isinstance(v,str) or any(k in v.lower() for k in FORBIDDEN) for v in values):
                raise ValueError("Duplicate/forbidden feature names")
        self.c,self.a = len(m["clinical_features"]),len(m["action_features"])
        self.cases = m["cases"]
        self.by_split = defaultdict(list)
        self.patients = {}
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.statistics = None
        ids,assets,labels = set(),{},{}
        for index,case in enumerate(self.cases):
            exact_keys(case,{"id","patient_id","split","input","target"},{"id","patient_id","split","input","target"})
            if not isinstance(case["id"],str) or case["id"] in ids:
                raise ValueError("Case IDs must be unique strings")
            ids.add(case["id"])
            pid,split = case["patient_id"],case["split"]
            if not isinstance(pid,str) or split not in {"train","val","test"}:
                raise ValueError("Invalid patient/split")
            if self.patients.setdefault(pid,split) != split:
                raise ValueError("Patient overlaps training/validation/test")
            inp,target = case["input"],case["target"]
            validate_input(inp,self.c,self.a)
            if m["time_basis"] == "stage_index":
                coordinates = [v["day"] for v in inp["observed"]]+[q["day"] for q in inp["queries"]]
                if any(v != int(v) for v in coordinates):
                    raise ValueError("Stage-index coordinates must be integers, not guessed elapsed days")
            exact_keys(target,{"pcr","future","observed_auxiliary"},{"pcr","future"})
            if target["pcr"] is not None and (isinstance(target["pcr"],bool) or target["pcr"] not in (0,1)):
                raise ValueError("Invalid final pCR label")
            if target["pcr"] is not None and labels.setdefault(pid,target["pcr"]) != target["pcr"]:
                raise ValueError("Contradictory final labels for one patient")
            if len(target["future"]) != len(inp["queries"]):
                raise ValueError("One target slot (possibly null) per query is required")
            if len(target.get("observed_auxiliary",[None]*len(inp["observed"]))) != len(inp["observed"]):
                raise ValueError("Observed sidecar slots mismatch")
            paths = [v["latent"] for v in inp["observed"]]
            for q,v in zip(inp["queries"],target["future"]):
                if v is None:
                    continue
                exact_keys(v,{"latent","day","auxiliary","anatomy_comparable"},{"latent","day"})
                if not isinstance(v["latent"],str) or v["day"] != q["day"]:
                    raise ValueError("A paired target must match the prospectively specified query day")
                if "anatomy_comparable" in v and not isinstance(v["anatomy_comparable"],bool):
                    raise ValueError("Anatomical comparability must be an audited boolean")
                paths.append(v["latent"])
            for p in paths:
                canonical = str(self.resolve(p))
                if assets.setdefault(canonical,pid) != pid:
                    raise ValueError("A latent file is shared across different patients/splits")
            self.by_split[split].append(index)
        if not self.cases:
            raise ValueError("Empty cohort")
        self.manifest_digest = digest(path)

    def resolve(self,path):
        p = Path(path)
        return (self.path.parent/p).resolve() if not p.is_absolute() else p.resolve()

    def read_latent(self,path):
        p = str(self.resolve(path))
        if p in self.cache:
            self.cache.move_to_end(p)
            return self.cache[p].copy()
        value = np.load(p,allow_pickle=False)
        if isinstance(value,np.lib.npyio.NpzFile):
            with value as archive:
                value = np.asarray(archive["latent"],np.float32)
        else:
            value = np.asarray(value,np.float32)
        if value.shape != tuple(self.manifest["latent_shape"]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid continuous VQ latent: {p}")
        if self.cache_size:
            self.cache[p] = value.copy()
            while len(self.cache)>self.cache_size:
                self.cache.popitem(last=False)
        return value.copy()

    def read_auxiliary(self,path):
        if path is None:
            return {}
        allowed = {"pillar","dense_teacher","segmentation","segmentation_mask","kinetics","kinetics_mask",
                   "biomarkers","biomarkers_mask"}
        with np.load(self.resolve(path),allow_pickle=False) as values:
            if set(values.files)-allowed:
                raise ValueError("Unexpected sidecar arrays (pCR belongs in final-label supervision, not feature caches)")
            result = {k:np.asarray(values[k],np.float32) for k in values.files}
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite auxiliary data")
        return {k:torch.from_numpy(v.copy()) for k,v in result.items()}

    def all_assets(self):
        paths = set()
        for c in self.cases:
            paths.update(str(self.resolve(v["latent"])) for v in c["input"]["observed"])
            for v in c["target"]["future"]:
                if v is not None:
                    paths.add(str(self.resolve(v["latent"])))
                    if v.get("auxiliary"):
                        paths.add(str(self.resolve(v["auxiliary"])))
            paths.update(str(self.resolve(v)) for v in c["target"].get("observed_auxiliary",[]) if v)
        return sorted(paths)

    def asset_signature(self):
        return {p:digest(p) for p in self.all_assets()}

    def fit_statistics(self):
        ids = self.by_split["train"]
        if not ids:
            raise ValueError("No training patients for fitting statistics")
        files = set()
        clinical,cm,actions,am = [],[],[],[]
        for i in ids:
            case = self.cases[i]; inp = case["input"]
            files.update(v["latent"] for v in inp["observed"])
            files.update(v["latent"] for v in case["target"]["future"] if v is not None)
            v,m = _values(inp["clinical"],inp["clinical_known_at"],inp["landmark_day"],self.c)
            clinical.append(v); cm.append(m)
            for q in inp["queries"]:
                v,m = _values(q["actions"],q["actions_known_at"],inp["landmark_day"],self.a)
                actions.append(v); am.append(m)
        # Channel-wise weighted Welford merging; all and only training visits.
        n,mean,m2 = 0,np.zeros(24),np.zeros(24)
        for path in sorted({str(self.resolve(p)) for p in files}):
            x = self.read_latent(path).astype(np.float64).reshape(24,-1)
            count = x.shape[1]
            mu = x.mean(1); central = ((x-mu[:,None])**2).sum(1)
            delta = mu-mean
            m2 += central+delta**2*n*count/(n+count)
            mean += delta*count/(n+count)
            n += count
        std = np.sqrt(m2/n)
        if (std < 1e-8).any():
            raise ValueError("Degenerate latent channel on training data")
        def feature_stats(values,masks,dim):
            if not values:
                return [0.]*dim,[1.]*dim
            x,m = np.asarray(values,np.float64),np.asarray(masks,bool)
            count = m.sum(0).clip(1)
            mu = (x*m).sum(0)/count
            std = np.sqrt((((x-mu)*m)**2).sum(0)/count)
            return mu.tolist(),np.where(std<1e-6,1,std).tolist()
        cmean,cstd = feature_stats(clinical,cm,self.c)
        amean,astd = feature_stats(actions,am,self.a)
        self.statistics = {"fit_split":"train","latent_mean":mean.tolist(),"latent_std":std.tolist(),
                           "clinical_mean":cmean,"clinical_std":cstd,"action_mean":amean,"action_std":astd,
                           "train_patient_hashes":sorted(stable_hash(p) for p,s in self.patients.items() if s == "train"),
                           "manifest_digest":self.manifest_digest}
        return self.statistics

    def set_statistics(self,value):
        if value["fit_split"] != "train" or value["manifest_digest"] != self.manifest_digest:
            raise ValueError("Normalization was not fitted on this manifest's training split")
        self.statistics = value

    def normalized_input(self,records):
        if self.statistics is None:
            raise ValueError("Fit/load TRAIN statistics before creating model inputs")
        st = self.statistics
        b = len(records); t = max(len(r["observed"]) for r in records); f = max(len(r["queries"]) for r in records)
        shape = tuple(self.manifest["latent_shape"])
        z = np.zeros((b,t,*shape),np.float32); om = np.zeros((b,t),bool); od = np.zeros((b,t),np.float32)
        c = np.zeros((b,self.c),np.float32); cm = np.zeros_like(c,dtype=bool)
        fd = np.zeros((b,f),np.float32); fm = np.zeros((b,f),bool)
        actions = np.zeros((b,f,self.a),np.float32); am = np.zeros_like(actions,dtype=bool)
        mean,std = np.asarray(st["latent_mean"],np.float32)[:,None,None,None],np.asarray(st["latent_std"],np.float32)[:,None,None,None]
        for i,r in enumerate(records):
            validate_input(r,self.c,self.a)
            for j,v in enumerate(r["observed"]):
                z[i,j] = (self.read_latent(v["latent"])-mean)/std
                od[i,j],om[i,j] = v["day"],True
            cv,cma = _values(r["clinical"],r["clinical_known_at"],r["landmark_day"],self.c)
            c[i],cm[i] = normalize_values(cv,cma,st["clinical_mean"],st["clinical_std"]),cma
            for j,q in enumerate(r["queries"]):
                fd[i,j],fm[i,j] = q["day"],True
                av,ama = _values(q["actions"],q["actions_known_at"],r["landmark_day"],self.a)
                actions[i,j],am[i,j] = normalize_values(av,ama,st["action_mean"],st["action_std"]),ama
        return ForecastInput(*[torch.from_numpy(x) for x in (z,om,od,c,cm,fd,fm,actions,am)])

    def batch(self,indices,device="cpu",supervised=True):
        records = [self.cases[i]["input"] for i in indices]
        inp = self.normalized_input(records)
        if not supervised:
            return inp.to(device)  # Does not open any future/label/sidecar file.
        b,t = inp.observed.shape[:2]; f = inp.future_days.shape[1]
        z = torch.zeros(b,f,*inp.observed.shape[2:]); mask = torch.zeros(b,f,dtype=torch.bool)
        labels = torch.zeros(b); lm = torch.zeros(b,dtype=torch.bool); comparable = torch.zeros_like(mask)
        mean = torch.tensor(self.statistics["latent_mean"])[:,None,None,None]
        std = torch.tensor(self.statistics["latent_std"])[:,None,None,None]
        aux = []
        for row,index in enumerate(indices):
            target = self.cases[index]["target"]
            old_t = len(records[row]["observed"])
            sidecars = target.get("observed_auxiliary",[None]*old_t)
            sidecars = [self.read_auxiliary(p) for p in sidecars]+[{} for _ in range(t-old_t)]
            for j in range(f):
                v = target["future"][j] if j < len(target["future"]) else None
                sidecars.append(self.read_auxiliary(v.get("auxiliary")) if v is not None else {})
                if v is not None:
                    z[row,j] = (torch.from_numpy(self.read_latent(v["latent"]))-mean)/std
                    mask[row,j] = True
                    comparable[row,j] = v.get("anatomy_comparable",False)
            aux.append(sidecars)
            if target["pcr"] is not None:
                labels[row],lm[row] = target["pcr"],True
        return inp.to(device),Supervision(z,mask,labels,lm,aux,comparable).to(device)

    def fit_prior(self,model):
        from sklearn.linear_model import LogisticRegression
        x,y,pids = [],[],[]
        for i in self.by_split["train"]:
            case = self.cases[i]
            if case["target"]["pcr"] is None:
                continue
            r = case["input"]
            cv,mask = _values(r["clinical"],r["clinical_known_at"],r["landmark_day"],self.c)
            cv = normalize_values(cv,mask,self.statistics["clinical_mean"],self.statistics["clinical_std"])
            x.append(np.concatenate((cv,mask.astype(np.float32))))
            y.append(case["target"]["pcr"]); pids.append(case["patient_id"])
        prior = model.pcr.prior
        if not y:
            return {"fitted":False,"reason":"No labelled training cases"}
        counts = {p:pids.count(p) for p in set(pids)}
        w = np.asarray([1/counts[p] for p in pids]); y = np.asarray(y)
        if len(set(y))<2 or self.c == 0:
            rate = (np.dot(w,y)+.5)/(w.sum()+1)
            coef = np.zeros(self.c*2); intercept = math.log(rate/(1-rate))
        else:
            fit = LogisticRegression(C=1.0,max_iter=1000,random_state=0).fit(np.asarray(x),y,sample_weight=w)
            coef,intercept = fit.coef_[0],float(fit.intercept_[0])
        with torch.no_grad():
            prior.coefficient.copy_(torch.tensor(coef,dtype=torch.float32))
            prior.intercept.fill_(intercept); prior.fitted.fill_(True)
        return {"fitted":True,"fit_split":"train","patients":len(counts),"patient_balanced":True,
                "regularization_C":1.0,"class_weight":None}

    def audit(self,scan_arrays=False):
        rows = {}
        for split,indices in self.by_split.items():
            pids = {self.cases[i]["patient_id"] for i in indices}
            labels = {self.cases[i]["patient_id"]:None for i in indices}
            for i in indices:
                if self.cases[i]["target"]["pcr"] is not None:
                    labels[self.cases[i]["patient_id"]] = self.cases[i]["target"]["pcr"]
            rows[split] = {"patients":len(pids),"landmarks":len(indices),
                           "labelled_patients":sum(v is not None for v in labels.values()),
                           "positive_patients":sum(v == 1 for v in labels.values())}
        if scan_arrays:
            for c in self.cases:
                for v in c["input"]["observed"]:
                    self.read_latent(v["latent"])
                for v in c["target"]["future"]:
                    if v is not None:
                        self.read_latent(v["latent"])
        return {"schema":"responsewm_audit_v1","split_summary":rows,"manifest_digest":self.manifest_digest,
                "synthetic":self.manifest.get("synthetic",False),"arrays_checked":scan_arrays,
                "clinical_features":self.manifest["clinical_features"],"action_features":self.manifest["action_features"],
                "note":"Availability/geometry declarations are validated for consistency, not independently proven."}


class PatientSampler:
    def __init__(self,store,seed,eligible_only=False):
        by_patient = defaultdict(list)
        for i in store.by_split["train"]:
            if eligible_only:
                future = store.cases[i]["target"]["future"]
                if not any(v is not None and (j == 0 or future[j-1] is not None) for j,v in enumerate(future)):
                    continue
            by_patient[store.cases[i]["patient_id"]].append(i)
        self.groups = [by_patient[k] for k in sorted(by_patient)]
        if not self.groups:
            raise ValueError("No training patients")
        self.generator = torch.Generator().manual_seed(seed)
    def sample(self,count):
        result = []
        for _ in range(count):
            group = self.groups[int(torch.randint(len(self.groups),(),generator=self.generator))]
            result.append(group[int(torch.randint(len(group),(),generator=self.generator))])
        return result

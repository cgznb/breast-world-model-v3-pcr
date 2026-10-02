"""Public synthetic fixtures only. Metrics from these are not medical results."""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
from .io import write_json
from .data import PHASES


def make_synthetic(output,shape=(24,4,8,8),patients=8):
    root = Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(321)
    cases=[]
    days=[0.,30.,60.,90.]
    for p in range(patients):
        pid=f"synthetic_{p:03d}"
        split="train" if p<patients-4 else ("val" if p<patients-2 else "test")
        label=p%2
        z0=rng.normal(0,1,shape).astype(np.float32)
        paths=[]
        for t in range(4):
            z=(z0*(1-.1*t*(1+label))+rng.normal(0,.1,shape)).astype(np.float32)
            path=root/f"{pid}_T{t}.npy"; np.save(path,z); paths.append(str(path))
        for origin in (0,1):
            observed=[{"latent":paths[t],"day":days[t],"available_at":days[t]} for t in range(origin+1)]
            queries=[{"day":days[t],"known_at":0.,"actions":[1.,float(p%2)],"actions_known_at":[0.,0.]}
                     for t in range(origin+1,4)]
            target=[{"latent":paths[t],"day":days[t],"anatomy_comparable":False} for t in range(origin+1,4)]
            # Genuine missing intermediate target in selected cases.
            if p%3 == 0 and len(target)>1:
                target[0]=None
            inp={"landmark_day":days[origin],"observed":observed,"clinical":[float(30+p),float(p%2),None],
                 "clinical_known_at":[0.,0.,None],"queries":queries,"source_only_geometry":True}
            cases.append({"id":pid+f"_at_T{origin}","patient_id":pid,"split":split,"input":inp,
                          "target":{"pcr":label,"future":target,"observed_auxiliary":[None]*len(observed)}})
    manifest={"schema":"responsewm_manifest_v1","phase_order":PHASES,"clinical_features":["age","receptor_indicator","missing_lab"],
              "action_features":["regimen_a","regimen_b"],"latent_shape":list(shape),"vq_identity":"synthetic:no-real-vq",
              "shared_grid_verified":True,"time_basis":"calendar_days","synthetic":True,"cases":cases,
              "provenance":{"warning":"SYNTHETIC NOISE; NOT PATIENT MRI OR CLINICAL EVIDENCE"}}
    path=root/"manifest.json"; write_json(path,manifest)
    first=next(c for c in cases if c["split"] == "test")
    request={k:manifest[k] for k in ("phase_order","clinical_features","action_features","latent_shape","vq_identity","time_basis")}
    request.update(schema="responsewm_request_v1",input=copy.deepcopy(first["input"]))
    write_json(root/"request.json",request)
    return path

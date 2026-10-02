#!/usr/bin/env python
"""Export ONLY one prospective input; no final label or future MRI is exported."""
from pathlib import Path
import sys,argparse,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from responsewm.data import ManifestStore
from responsewm.io import write_json
p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True);p.add_argument('--case-id',required=True)
p.add_argument('--output',required=True);p.add_argument('--allow-synthetic',action='store_true')
a=p.parse_args()
store=ManifestStore(a.manifest,a.allow_synthetic)
matched=[c for c in store.cases if c['id']==a.case_id]
if len(matched)!=1: raise ValueError('Case ID must identify one landmark')
request={k:store.manifest[k] for k in ('clinical_features','action_features','phase_order','latent_shape','vq_identity','time_basis')}
request.update(schema='responsewm_request_v1',input=copy.deepcopy(matched[0]['input']))
for v in request['input']['observed']: v['latent']=str(store.resolve(v['latent']))
write_json(a.output,request)

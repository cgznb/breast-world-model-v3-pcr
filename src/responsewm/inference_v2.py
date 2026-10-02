"""Strict input-only requests and weights-only persistent state archives."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .data import ManifestStore, exact_keys, validate_input
from .contracts_v2 import PatientPrefix
from .state import PatientBelief, Observation
from .rollout import NoiseLedger
from .training_v2 import load_trained
from .io import read_json, write_json, save_checkpoint, load_checkpoint, digest, autocast


def read_request(path,payload):
    request=read_json(path)
    keys={"schema","clinical_features","action_features","phase_order","latent_shape","vq_identity","time_basis","input"}
    exact_keys(request,keys,keys)
    if request["schema"]!="responsewm_request_v2":
        raise ValueError("Expected an independent v2 input request")
    for key,value in payload["metadata"]["data_contract"].items():
        if request[key]!=value:
            raise ValueError(f"Request {key} differs from checkpoint")
    record=request["input"]
    c,a=len(request["clinical_features"]),len(request["action_features"])
    if a:
        raise ValueError("Persistent CLI states currently require action_dim=0; use typed model plan API for audited actions")
    validate_input(record,c,a)
    stage=record["landmark_day"]
    coordinates=[stage]+[v["day"] for v in record["observed"]]+[q["day"] for q in record["queries"]]
    if any(type(v) is bool or v not in (0,1,2,3) for v in coordinates):
        raise ValueError("R1 requests use stages 0,1,2,3; arbitrary days are unsupported")
    requested=[int(q["day"]) for q in record["queries"]]
    if a and requested!=list(range(int(stage)+1,4)):
        raise ValueError("Action-bearing requests must explicitly specify every interval")
    canonical={q["day"]:q for q in record["queries"]}
    expanded=dict(record,queries=[canonical.get(j,{"day":j,"known_at":stage,"actions":[],"actions_known_at":[]})
                                 for j in range(int(stage)+1,4)])
    reader=ManifestStore.__new__(ManifestStore)
    reader.path=Path(path).resolve()
    reader.manifest=request
    reader.c,reader.a=c,a
    reader.statistics=payload["metadata"]["statistics"]
    reader.cache,reader.cache_size=OrderedDict(),0
    inp=reader.normalized_input([expanded])
    query=torch.zeros(1,4,dtype=torch.bool)
    query[0,requested]=True
    inp=PatientPrefix(**inp.__dict__,as_of=torch.tensor([float(stage)]),stage_valid=torch.ones(1,4,dtype=torch.bool),
                      output_query_mask=query,observed_stage_ids=(tuple(int(v["day"]) for v in record["observed"]),),
                      observed_available_at=torch.tensor([[v["available_at"] for v in record["observed"]]]),
                      clinical_known_at=torch.tensor([[float("inf") if v is None else v for v in record["clinical_known_at"]]]))
    inp.validate(c,a)
    return inp,requested


def save_state(output,belief,checkpoint,payload):
    save_checkpoint(output,{"schema":"responsewm_state_v2","state":belief.state_dict(),
                            "checkpoint":str(Path(checkpoint).resolve()),"checkpoint_sha256":digest(checkpoint),
                            "vq_identity":payload["metadata"]["data_contract"]["vq_identity"],
                            "config_digest":payload["config_digest"]})


def _to_device(value,device):
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:_to_device(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(_to_device(v,device) for v in value)
    return value


def load_state(path,device="cpu"):
    archive=load_checkpoint(path)
    if archive.get("schema")!="responsewm_state_v2":
        raise ValueError("Expected v2 state archive")
    if digest(archive["checkpoint"])!=archive["checkpoint_sha256"]:
        raise ValueError("State checkpoint changed since initialization")
    model,payload=load_trained(archive["checkpoint"],device)
    belief=PatientBelief.from_state_dict(_to_device(archive["state"],device))
    return model,payload,belief,archive


@torch.no_grad()
def initialize(checkpoint,request,output,device="cpu"):
    model,payload=load_trained(checkpoint,device)
    inp,_=read_request(request,payload)
    with autocast(device,model.cfg.training.precision if device!="cpu" else "fp32"):
        belief=model.initialize(inp.to(device))
    save_state(output,belief,checkpoint,payload)
    return {"state":str(output),"stage":belief.stage,"observed_stage_ids":belief.observed_stage_ids}


@torch.no_grad()
def forecast(state,output,*,output_stages=None,samples=8,steps=20,seed=0,device="cpu",codec=None,state_output_dir=None):
    model,payload,belief,archive=load_state(state,device)
    with autocast(device,model.cfg.training.precision if device!="cpu" else "fp32"):
        trace=model.forecast(belief,output_stages=output_stages,samples=samples,steps=steps,noise_ledger=NoiseLedger(seed))
    mean,std=model.encoder.latent_mean[None,None],model.encoder.latent_std[None,None]
    raw=trace.latent*std+mean
    arrays={"latent":raw.float().cpu().numpy(),"disease":trace.state.float().cpu().numpy(),
            "pcr_marginal":trace.probability.cpu().numpy(),"forecasted_stage_probabilities":trace.stage_probs.cpu().numpy(),
            "output_stages":np.asarray(trace.output_stages),"internal_stages":np.asarray(trace.stage_ids)}
    if codec:
        if "sha256:"+digest(codec)!=archive["vq_identity"]:
            raise ValueError("Codec SHA differs from original VQ identity")
        from .legacy.codec import load_codec
        decoder=load_codec(codec,device)
        flat=raw.flatten(0,2)
        decoded=torch.cat([decoder.decode(z[None]).cpu() for z in flat],0)
        arrays["images"]=decoded.reshape(*raw.shape[:3],*decoded.shape[1:]).numpy()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    if state_output_dir:
        for prior in trace.all_states:
            save_state(Path(state_output_dir)/f"T{prior.stage}_prior.pt",prior,archive["checkpoint"],payload)
    info={"origin_stage":belief.stage,"internal_stages":trace.stage_ids,"output_stages":trace.output_stages,
          "latent_coordinates":"original continuous VQ coordinates","samples":samples,"steps":steps,"seed":seed,
          "pcr_interpretation":"marginal_current","observed_stage_ids":belief.observed_stage_ids,
          "checkpoint_sha256":archive["checkpoint_sha256"]}
    write_json(Path(output).with_suffix(".json"),info)
    return info


@torch.no_grad()
def observe(state,observation,output,device="cpu"):
    model,payload,belief,archive=load_state(state,device)
    record=read_json(observation)
    exact_keys(record,{"schema","stage","latent","event_id","vq_identity","available_at"},
                      {"schema","stage","latent","event_id","vq_identity","available_at"})
    if record["schema"]!="responsewm_observation_v2" or record["vq_identity"]!=archive["vq_identity"]:
        raise ValueError("Invalid observation schema or codec identity")
    if record["stage"] not in range(4) or record["available_at"]!=record["stage"]:
        raise ValueError("R1 observation requires stage-aligned availability")
    path=Path(record["latent"])
    if not path.is_absolute():
        path=Path(observation).resolve().parent/path
    raw=np.load(path,allow_pickle=False)
    if isinstance(raw,np.lib.npyio.NpzFile):
        with raw as data:
            raw=np.asarray(data["latent"],dtype=np.float32)
    z=torch.as_tensor(raw,dtype=torch.float32,device=device)
    if z.shape!=belief.anchor_latent.shape[2:] or not torch.isfinite(z).all():
        raise ValueError("Observed MRI has wrong shape or nonfinite values")
    z=(z[None]-model.encoder.latent_mean)/model.encoder.latent_std
    with autocast(device,model.cfg.training.precision if device!="cpu" else "fp32"):
        posterior=model.observe(belief,Observation(int(record["stage"]),z,(record["event_id"],)))
    save_state(output,posterior,archive["checkpoint"],payload)
    return {"stage":posterior.stage,"version":posterior.version,"observed_stage_ids":posterior.observed_stage_ids}


@torch.no_grad()
def query(state,output,*,mode="observed_landmark",device="cpu",samples=8,steps=20,seed=0):
    model,_,belief,_=load_state(state,device)
    with autocast(device,model.cfg.training.precision if device!="cpu" else "fp32"):
        trace=(model.forecast(belief,samples=samples,steps=steps,noise_ledger=NoiseLedger(seed))
               if mode=="marginal_current" else None)
        result=model.query_pcr(belief,mode=mode,future_trace=trace)
    record={"probability":result.probability.tolist(),"interpretation":result.interpretation,
            "as_of":result.as_of,"observed_stage_ids":result.observed_stage_ids}
    write_json(output,record)
    return record

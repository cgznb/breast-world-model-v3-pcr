"""Prequential predictions recorded before each real MRI is loaded."""
from __future__ import annotations
import torch
from .conditioning import IntervalSpec
from .state import Observation
from .rollout import NoiseLedger
from .io import autocast
from .metrics import classification_metrics,patient_bootstrap
from .losses_v2 import interval_plan


@torch.no_grad()
def prequential_evaluation(model,store,cfg,split="val",bootstrap=0):
    device=next(model.parameters()).device
    rows=[]
    model.eval()
    for index in store.by_split[split]:
        inp=store.batch([(index,0)],device,supervised=False)
        plan=interval_plan(inp)
        with autocast(device,cfg.training.precision if device.type!="cpu" else "fp32"):
            state=model.initialize(inp)
            for stage in (1,2,3):
                trace=model.advance(state,plan[stage-1],samples=cfg.training.validate_samples,
                                    steps=cfg.sampling.inference_steps,noise_ledger=NoiseLedger(cfg.training.seed+200003))
                prior=trace.final_state
                row={"patient_key":store.patients[index]["patient_key"],"target_stage":stage,
                     "prediction_as_of":state.as_of,"prior_observed_stage_ids":list(state.observed_stage_ids[0]),
                     "forecasted_state_probability":float(trace.probability[0]),
                     "label":store.patients[index]["target"]["pcr"]}
                visits=[v for v in store.patients[index]["visits"] if v["stage"]==stage and v["available_at"]<=stage]
                if visits:
                    # Only after producing the prior may the actual new MRI be opened.
                    observed=store.make_prefix(index,stage,device=device)
                    column=observed.observed_stage_ids[0].index(stage)
                    obs=Observation(stage,observed.observed[:,column],(f"mri:T{stage}",),
                                    clinical=observed.clinical,clinical_mask=observed.clinical_mask)
                    state=model.observe(prior,obs)
                    row["observed_probability"]=float(model.query_pcr(state).probability[0])
                    row["observed_stage_ids"]=list(state.observed_stage_ids[0])
                    row["prior_latent_mae"]=float((prior.anchor_latent-observed.observed[:,column,None]).abs().mean())
                else:
                    state=prior
                    row["new_mri_received"]=False
                rows.append(row)
    grouped={}
    for row in rows:
        if "observed_probability" in row and row["label"] is not None:
            key="{"+",".join(map(str,row["observed_stage_ids"]))+"}"
            grouped.setdefault(key,[]).append(row)
    scores={}
    for key,group in grouped.items():
        y=[r["label"] for r in group];p=[r["observed_probability"] for r in group]
        scores[key]={"observed_landmark":classification_metrics(y,p),"patients":len(group)}
        if bootstrap:
            scores[key]["patient_bootstrap"]=patient_bootstrap(y,p,[r["patient_key"] for r in group],bootstrap,cfg.training.seed)
    return {"protocol":"advance_then_load_and_observe_real_MRI","by_observed_prefix":scores,"rows":rows,
            "independent_test":False,"clinical_time_basis":"stage_index"}

"""Four-stage patient trajectory training with auditable step-boundary resume."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from .config import from_dict
from .io import (write_json, read_json, save_checkpoint, load_checkpoint, seed_all,
                 rng_state, restore_rng, autocast, digest)
from .model_v2 import MultistageResponseWorldModel
from .samplers import PatientLandmarkSampler, BalancedEdgeSampler
from .losses_v2 import (representation_objective, prefix_objective, edge_objective,
                        free_rollout_objective, observed_update_objective,interval_plan)
from .metrics import classification_metrics, patient_bootstrap
from .rollout import NoiseLedger
from .training import STAGES, learning_rate


def initialize_run(store, cfg, root):
    if cfg.schema != "responsewm_v2":
        raise ValueError("train-v2 requires responsewm_v2 configuration")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signatures = store.asset_signature()
    path = root/"metadata.json"
    if path.exists():
        meta = read_json(path)
        if meta["config_digest"] != cfg.digest or meta["manifest_digest"] != store.manifest_digest or meta["asset_signatures"] != signatures:
            raise ValueError("Run identity changed; use a new output directory")
        store.set_statistics(meta["statistics"])
        return meta
    if not store.by_split["train"] or not store.by_split["val"]:
        raise ValueError("Patient-disjoint train and validation cohorts required")
    meta = {"schema": "responsewm_run_v2", "config": cfg.to_dict(), "config_digest": cfg.digest,
            "manifest_digest": store.manifest_digest, "statistics": store.fit_statistics(),
            "asset_signatures": signatures, "audit": store.audit(),
            "data_contract": {k:store.manifest[k] for k in ("clinical_features", "action_features",
                              "phase_order", "latent_shape", "vq_identity", "time_basis")},
            "synthetic": store.manifest.get("synthetic",False), "independent_test_available": bool(store.by_split["test"]),
            "selection_protocol": {"representation": "reconstruction_plus_JEPA_plus_0.25_patient_mean_prefix_NLL",
                                   "flow": "stage_balanced_adjacent_FM", "readout": "patient_mean_real_prefix_NLL",
                                   "joint": "patient_mean_real_prefix_NLL_with_fixed_generation_guard"}}
    source_root=Path(__file__).resolve().parents[2]
    meta["source_hashes"]={str(p.relative_to(source_root)):digest(p) for p in sorted((source_root/"src").rglob("*.py"))}
    source_manifest=source_root/"docs/source_manifest.json"
    if source_manifest.exists():
        meta["source_manifest"]=read_json(source_manifest)
        meta["source_manifest_sha256"]=digest(source_manifest)
    write_json(path,meta)
    return meta


def build_model(cfg, store):
    model = MultistageResponseWorldModel(cfg,store.c,store.a)
    for encoder in (model.encoder, model.target_encoder):
        encoder.set_normalization(store.statistics["latent_mean"],store.statistics["latent_std"])
    return model


def _append(path, value):
    with Path(path).open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")


@torch.no_grad()
def validate(model,store,cfg,stage,*,split="val",bootstrap=0):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = store.by_split[split][:cfg.training.validation_cases]
    rows, repr_scores, edge_losses = [], [], {}
    generation=[]
    devices=[device.index or 0] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.training.seed+100003)
        if devices:
            torch.cuda.manual_seed_all(cfg.training.seed+100003)
        for index in ids:
            patient=store.patients[index]
            for origin in store.landmarks(index):
                inp,sup=store.batch([(index,origin)],device)
                with autocast(device,cfg.training.precision):
                    belief=model.initialize(inp)
                    observed=model.query_pcr(belief)
                    row={"patient_key":patient["patient_key"],"origin_stage":origin,
                         "observed_stage_ids":list(belief.observed_stage_ids[0]),
                         "label":patient["target"]["pcr"],"observed_probability":float(observed.probability[0])}
                    if stage in {"readout","joint"} and origin<3:
                        trace=model.forecast(belief,samples=cfg.training.validate_samples,
                              steps=cfg.sampling.inference_steps,noise_ledger=NoiseLedger(cfg.training.seed+100003),plan=interval_plan(inp))
                        row["marginal_probability"]=float(trace.probability[0])
                        for j,s in enumerate(trace.stage_ids):
                            row[f"forecasted_state_probability_T{s}"]=float(trace.stage_probs[0,j])
                        for j,s in enumerate(trace.output_stages):
                            if bool(sup.future_mask[0,j]):
                                row[f"latent_MAE_T{s}"]=float((trace.latent[0,:,j].float()-sup.future[0,j].float()).abs().mean())
                        ground=float((trace.state.float()-trace.image_state.float()).square().mean())
                        generation.append(ground)
                    rows.append(row)
                    if stage=="representation" and origin==0:
                        _,terms=representation_objective(model,inp,sup)
                        repr_scores.append(terms["loss/reconstruction"]+terms["loss/masked_jepa"])
        if stage in {"flow","joint"}:
            from .samplers import EdgeSample
            for (src,dst),population in store.edge_index(split).items():
                for index in population:
                    if index not in ids:
                        continue
                    task=EdgeSample(index,src,dst,"adjacent",1.,1.,1.)
                    inp,truth=store.make_edge_task(task,device)
                    with autocast(device,cfg.training.precision):
                        loss,_=edge_objective(model,inp,truth)
                    edge_losses.setdefault(f"{src}{dst}",[]).append(float(loss))
    model.train(was_training)
    groups={}
    patient_nll={}
    for row in rows:
        if row["label"] is None:
            continue
        key="{"+",".join(map(str,row["observed_stage_ids"]))+"}"
        groups.setdefault(key,[]).append(row)
        p=min(max(row["observed_probability"],1e-7),1-1e-7)
        import math
        nll=-math.log(p if row["label"] else 1-p)
        patient_nll.setdefault(row["patient_key"],[]).append(nll)
    mean=lambda values:sum(values)/max(1,len(values))
    real_nll=mean([mean(v) for v in patient_nll.values()])
    metrics={}
    for key,group in groups.items():
        labels=[r["label"] for r in group]
        observed=[r["observed_probability"] for r in group]
        metrics[key]={"observed":classification_metrics(labels,observed),"patients":len(group)}
        marginal=[r for r in group if "marginal_probability" in r]
        if marginal:
            metrics[key]["marginal"]=classification_metrics([r["label"] for r in marginal],[r["marginal_probability"] for r in marginal])
        if bootstrap:
            metrics[key]["patient_bootstrap"]=patient_bootstrap(labels,observed,[r["patient_key"] for r in group],bootstrap,cfg.training.seed)
    edge_means={key:mean(values) for key,values in edge_losses.items()}
    selection=(mean(repr_scores)+cfg.multistage.representation_pcr_weight*real_nll if stage=="representation"
               else mean(list(edge_means.values())) if stage=="flow" else real_nll)
    return {"selection_score":selection,"real_prefix_patient_mean_nll":real_nll,"by_observed_prefix":metrics,
            "edge_objectives":edge_means,"edge_support":{k:len(v) for k,v in edge_losses.items()},
            "generation_grounding":mean(generation) if generation else None,
            "rows":rows,"split":split,"patients":len(ids),"synthetic":store.manifest.get("synthetic",False)}


def _gradient_norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _landmark_batch(sampler, count, *, origin_stage=None):
    if count == 1:
        tasks = sampler.sample(count)
        if origin_stage is not None:
            tasks[0] = (tasks[0][0], origin_stage)
        return tasks
    return sampler.sample_batch(count, origin_stage=origin_stage)


def _edge_batches(store, selected, batch_size):
    if batch_size == 1:
        return [[edge] for edge in selected]
    groups = {}
    for edge in selected:
        prefix = tuple(v["stage"] for v in store.patients[edge.patient_index]["visits"]
                       if v["stage"] <= edge.source_stage and v["available_at"] <= edge.source_stage)
        key = (edge.source_stage, edge.target_stage, edge.kind, prefix, edge.effective_weight)
        groups.setdefault(key, []).append(edge)
    return [group[start:start+batch_size] for group in groups.values()
            for start in range(0, len(group), batch_size)]


def train_stage(store,cfg,root,stage,*,resume=False,warm_start=None,stop_after=None):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    seed_all(cfg.training.seed,cfg.training.threads,cfg.training.strict_determinism)
    meta=initialize_run(store,cfg,root)
    folder=Path(root)/stage
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"last.pt"
    if path.exists() and not resume:
        raise FileExistsError("Stage already exists; pass --resume")
    model=build_model(cfg,store)
    if stage!="representation" and not resume:
        previous=Path(root)/STAGES[STAGES.index(stage)-1]
        previous_last=load_checkpoint(previous/"last.pt")
        if not previous_last.get("completed"):
            raise ValueError("Previous stage has not completed")
        payload=load_checkpoint(previous/"best.pt")
        model.load_state_dict(payload["model"],strict=True)
        if stage=="flow":
            model.freeze_representation()
    elif not resume:
        if warm_start:
            from .migration_v2 import migrate_checkpoint,MigrationError
            try:
                migration=migrate_checkpoint(model,warm_start,meta)
            except MigrationError as exc:
                write_json(Path(root)/"migration_report.json",exc.report)
                raise
            write_json(Path(root)/"migration_report.json",migration)
        else:
            write_json(Path(root)/"migration_report.json",{"mode":"fresh_v2","source_checkpoint":None,
                       "vq_identity":meta["data_contract"]["vq_identity"],"normalization":"current_training_patients_only",
                       "old_experiments_preserved":True,"reason":"New longitudinal representation and state protocol"})
        write_json(Path(root)/"clinical_prior.json",store.fit_prior(model))
    model.to(cfg.training.device)
    params=model.configure_stage(stage)
    lr=cfg.training.joint_lr if stage=="joint" else cfg.training.lr
    image_ids={id(p) for p in model.velocity.image.parameters()}
    if stage=="joint":
        groups=[{"params":[p for p in params if id(p) not in image_ids],"base_lr":lr},
                {"params":[p for p in params if id(p) in image_ids],"base_lr":cfg.multistage.joint_image_lr}]
    else:
        groups=[{"params":params,"base_lr":lr}]
    optimizer=torch.optim.AdamW(groups,lr=lr,weight_decay=cfg.training.weight_decay)
    landmarks=PatientLandmarkSampler(store,cfg.training.seed+17)
    origins=PatientLandmarkSampler(store,cfg.training.seed+19,include_terminal=False)
    edges=BalancedEdgeSampler(store,cfg.training.seed+23,include_bridges=cfg.multistage.bridge_auxiliary_weight>0,
                              bridge_weight=cfg.multistage.bridge_auxiliary_weight)
    total=getattr(cfg.training,stage+"_steps")
    start,best,stale=0,float("inf"),0
    rollout_updates=0
    generation_guard=None
    if resume:
        payload=load_checkpoint(path)
        if payload["config_digest"]!=cfg.digest or payload["manifest_digest"]!=store.manifest_digest or payload["stage"]!=stage:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(payload["model"],strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for sampler,key in ((landmarks,"landmarks"),(origins,"origins"),(edges,"edges")):
            sampler.load_state_dict(payload["samplers"][key])
        start,best,stale=payload["step"],payload["best_score"],payload["stale_validations"]
        rollout_updates=payload["rollout_updates"]
        generation_guard=payload.get("generation_guard")
        restore_rng(payload["rng"])
        if payload.get("completed"):
            return {"stage":stage,"steps":start,"completed":True,"checkpoint":str(path)}
    def snapshot(step,completed=False):
        return {"schema":"responsewm_checkpoint_v2","config":cfg.to_dict(),"config_digest":cfg.digest,
                "manifest_digest":store.manifest_digest,"metadata":meta,"stage":stage,"step":step,
                "completed":completed,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "best_score":best,"stale_validations":stale,"rollout_updates":rollout_updates,
                "generation_guard":generation_guard,"rng":rng_state(),
                "samplers":{"landmarks":landmarks.state_dict(),"origins":origins.state_dict(),"edges":edges.state_dict()}}
    def validation(step,completed=False):
        nonlocal best,stale,generation_guard
        result=validate(model,store,cfg,stage)
        result.update(stage=stage,step=step)
        write_json(folder/"validation_last.json",result)
        _append(folder/"validation.jsonl",result)
        generation=sum(result["edge_objectives"].values())/max(1,len(result["edge_objectives"]))
        generation+=(result["generation_grounding"] or 0)
        if stage=="joint" and generation_guard is None:
            generation_guard=max(generation,1e-6)*1.20
        eligible=(step>0 if stage in {"representation","readout"} else True)
        eligible=eligible and (stage!="flow" or (step>=max(1,int(total*.5)) and rollout_updates>0))
        eligible=eligible and (stage!="joint" or generation<=generation_guard)
        improved=result["selection_score"]<best-cfg.multistage.early_stopping_min_delta
        if eligible and (improved or not (folder/"best.pt").exists()):
            best=result["selection_score"]
            stale=0
            save_checkpoint(folder/"best.pt",snapshot(step,completed))
            write_json(folder/"validation_best.json",result)
        elif eligible:
            stale+=1
        print(f"{stage} validation step={step} score={result['selection_score']:.6f} eligible={eligible}",flush=True)
    if not resume:
        validation(0)
        save_checkpoint(path,snapshot(0))
    start_time=time.monotonic()
    for step in range(start,total):
        optimizer.zero_grad(set_to_none=True)
        factor=learning_rate(step,total,cfg.training.warmup)
        for group in optimizer.param_groups:
            group["lr"]=group["base_lr"]*factor
        aggregate={}
        total_loss=0.
        physical_batches=[]
        def backward(loss,terms,weight=1.):
            nonlocal total_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss at step {step+1}")
            scaled=loss*(weight/cfg.training.accumulation)
            if scaled.requires_grad:
                scaled.backward()
            total_loss+=float(scaled.detach())
            for key,value in terms.items():
                aggregate[key]=aggregate.get(key,0.)+float(value)*weight/cfg.training.accumulation
        for micro in range(cfg.training.accumulation):
            tasks=_landmark_batch(landmarks,cfg.training.batch_size)
            if stage=="representation":
                inp,sup=store.batch(tasks,cfg.training.device)
                physical_batches.append(len(tasks))
                with autocast(cfg.training.device,cfg.training.precision):
                    loss,terms=representation_objective(model,inp,sup)
                backward(loss,terms)
            if stage in {"flow","joint"}:
                if cfg.multistage.edge_sampling=="all_edges":
                    selected=edges.all_edges()
                else:
                    selected=(edges.sample(cfg.training.batch_size) if cfg.training.batch_size==1
                              else edges.sample_batch(cfg.training.batch_size))
                for batch in _edge_batches(store,selected,cfg.training.batch_size):
                    inp,truth=(store.make_edge_task(batch[0],cfg.training.device) if len(batch)==1
                               else store.make_edge_batch(batch,cfg.training.device))
                    physical_batches.append(len(batch))
                    with autocast(cfg.training.device,cfg.training.precision):
                        loss,terms=edge_objective(model,inp,truth)
                    terms.update({"edge_sampling_probability":batch[0].probability,
                                  "edge_effective_weight":batch[0].effective_weight})
                    weight=sum(edge.effective_weight for edge in batch)/(1 if cfg.multistage.edge_sampling=="all_edges" else cfg.training.batch_size)
                    backward(loss,terms,weight)
            fraction=(step+1)/total
            b0,b1,_=cfg.multistage.flow_phase_fractions
            free=stage in {"readout","joint"} or (stage=="flow" and fraction>b0 and (step+1)%cfg.multistage.rollout_every==0)
            if free:
                hops=2 if stage=="flow" and fraction<=b0+b1 else 3
                force_origin=0 if stage=="flow" and hops==3 and rollout_updates==0 else None
                rollout_tasks=_landmark_batch(origins,cfg.training.batch_size,origin_stage=force_origin)
                inp,sup=store.batch(rollout_tasks,cfg.training.device)
                physical_batches.append(len(rollout_tasks))
                seed=cfg.training.seed+step*1009+micro*31
                marginal=(cfg.multistage.marginal_pcr_start+(cfg.multistage.marginal_pcr_end-cfg.multistage.marginal_pcr_start)*fraction
                          if stage in {"readout","joint"} else 0.)
                with autocast(cfg.training.device,cfg.training.precision):
                    loss,terms=free_rollout_objective(model,inp,sup,seed=seed,max_hops=hops,marginal_weight=marginal)
                backward(loss,terms)
                if terms["rollout_depth"]==3:
                    rollout_updates+=1
                with autocast(cfg.training.device,cfg.training.precision):
                    loss,terms=observed_update_objective(model,inp,sup,seed=seed+1)
                backward(loss,terms)
            if stage in {"readout","joint"}:
                tasks=_landmark_batch(landmarks,cfg.training.batch_size)
                inp,sup=store.batch(tasks,cfg.training.device)
                physical_batches.append(len(tasks))
                with autocast(cfg.training.device,cfg.training.precision):
                    pcr,assim,_=prefix_objective(model,inp,sup)
                    loss=cfg.loss.real_pcr*pcr+(cfg.multistage.assimilation_reconstruction*assim if stage=="joint" else 0)
                backward(loss,{"loss/real_prefix_pcr":float(pcr.detach()),"support/label":int(sup.label_mask.sum())})
        grad={"grad_norm/"+name:_gradient_norm(module) for name,module in (
              ("image",model.velocity.image),("semantic",model.velocity.blocks),("history",model.history),
              ("assim",model.assimilator),("pcr",model.pcr))}
        for i,bridge in enumerate(model.velocity.bridges):
            aggregate[f"bridge/{i}/image_gate"]=float(bridge.image_gate.detach())
            aggregate[f"bridge/{i}/state_gate"]=float(bridge.state_gate.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,cfg.training.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        model.update_target()
        getattr(model,stage+"_ready").fill_(True)
        done=step+1
        if done%cfg.training.log_every==0 or done==total:
            record={"stage":stage,"step":done,"loss":total_loss,"gradient_norm":float(norm),
                    "elapsed_seconds":time.monotonic()-start_time,"lr":lr*factor,"rollout_updates":rollout_updates,
                    "batch_size":cfg.training.batch_size,"accumulation":cfg.training.accumulation,
                    "physical_batch_min":min(physical_batches),"physical_batch_max":max(physical_batches),
                    "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if str(cfg.training.device).startswith("cuda") else 0,
                    **aggregate,**grad}
            _append(folder/"training.jsonl",record)
            print(f"{stage} step={done} loss={total_loss:.6f} grad_norm={float(norm):.6f}",flush=True)
        early=False
        if done%cfg.training.validation_every==0 or done==total:
            validation(done,done==total)
            early=cfg.multistage.early_stopping_patience>0 and stale>=cfg.multistage.early_stopping_patience
        completed=done==total or early
        if done%cfg.training.checkpoint_every==0 or completed or (stop_after is not None and done>=stop_after):
            save_checkpoint(path,snapshot(done,completed))
        if early or (stop_after is not None and done>=stop_after):
            break
    write_json(folder/"sampling_audit.json",edges.audit())
    return {"stage":stage,"steps":done,"completed":completed,"checkpoint":str(path)}


def load_trained(path,device="cpu"):
    payload=load_checkpoint(path)
    if payload.get("schema")!="responsewm_checkpoint_v2":
        raise ValueError("Expected v2 checkpoint")
    cfg=from_dict(payload["config"])
    contract=payload["metadata"]["data_contract"]
    model=MultistageResponseWorldModel(cfg,len(contract["clinical_features"]),len(contract["action_features"]))
    model.load_state_dict(payload["model"],strict=True)
    model.stage=payload["stage"]
    model.to(device).eval().requires_grad_(False)
    return model,payload

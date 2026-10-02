"""Run real-shape loss, full temporal backward and AdamW without saving weights."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from responsewm.config import load_config
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.training_v2 import build_model, _gradient_norm
from responsewm.losses_v2 import (representation_objective,edge_objective,free_rollout_objective,
                                 prefix_objective,observed_update_objective)
from responsewm.samplers import BalancedEdgeSampler
from responsewm.io import write_json,seed_all,autocast,read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--manifest",required=True)
    p.add_argument("--statistics",required=True);p.add_argument("--output",required=True)
    p.add_argument("--stages",nargs="+",default=["representation","flow","readout","joint"])
    args=p.parse_args()
    cfg=load_config(args.config)
    seed_all(cfg.training.seed,cfg.training.threads)
    store=PatientTrajectoryStore(args.manifest)
    stats=read_json(args.statistics)
    store.set_statistics(stats)
    patients=[i for i in store.by_split["train"] if len(store.patients[i]["visits"])==4]
    if len(patients)<cfg.training.batch_size:
        raise ValueError("Preflight requires a full batch of complete real trajectories")
    patients=patients[:cfg.training.batch_size]
    model=build_model(cfg,store).to(cfg.training.device)
    report={"engineering_only":True,"synthetic":False,"device":torch.cuda.get_device_name(),
            "pytorch":str(torch.__version__),"config":cfg.to_dict(),"manifest_digest":store.manifest_digest,
            "physical_batch_size":len(patients),"results":{}}
    edge_sampler=BalancedEdgeSampler(store,cfg.training.seed)
    for stage in args.stages:
        # A retains gradients through the longest real prefix; B/D roll all three future intervals.
        origin=3 if stage=="representation" else 0
        inp,sup=store.batch([(index,origin) for index in patients],cfg.training.device)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        if stage!="representation":
            model.freeze_representation()
        params=model.configure_stage(stage)
        model.zero_grad(set_to_none=True)
        optimizer=torch.optim.AdamW(params,lr=cfg.training.joint_lr if stage=="joint" else cfg.training.lr)
        optimizer.zero_grad(set_to_none=True)
        start=time.monotonic(); terms={}; total=0.
        def run(fn,*values,**kwargs):
            nonlocal total
            with autocast(cfg.training.device,cfg.training.precision):
                loss,metrics=fn(*values,**kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite preflight loss")
            if loss.requires_grad:
                loss.backward()
            total+=float(loss.detach());terms.update(metrics)
        if stage=="representation":
            run(representation_objective,model,inp,sup)
        if stage in {"flow","joint"}:
            for _ in range(3):
                batch=edge_sampler.sample_batch(cfg.training.batch_size)
                prefix,truth=store.make_edge_batch(batch,cfg.training.device)
                run(edge_objective,model,prefix,truth)
        if stage!="representation":
            run(free_rollout_objective,model,inp,sup,seed=cfg.training.seed,max_hops=3,
                marginal_weight=.1 if stage in {"readout","joint"} else 0.)
            if stage in {"flow","joint"}:
                run(observed_update_objective,model,inp,sup,seed=cfg.training.seed+1)
        grad={name:_gradient_norm(module) for name,module in (("image",model.velocity.image),
              ("semantic",model.velocity.blocks),("history",model.history),("assim",model.assimilator),("pcr",model.pcr))}
        norm=torch.nn.utils.clip_grad_norm_(params,cfg.training.grad_clip,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        report["results"][stage]={"loss":total,"gradient_norm":float(norm),"module_gradients":grad,
            "physical_batch_size":len(inp.observed),"observed_shape":list(inp.observed.shape),
            "seconds":time.monotonic()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"terms":terms,"optimizer_updated":True}
        write_json(args.output,report)
        print(stage,report["results"][stage],flush=True)
        del optimizer
    print("Full-size multistage preflight complete",flush=True)


if __name__=="__main__":
    main()

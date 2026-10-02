"""Versioned multistage research and online inference commands."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import time
from .config import load_config
from .data_v2 import PatientTrajectoryStore
from .io import write_json
from .training import STAGES
from .training_v2 import train_stage,validate,load_trained


def main(argv):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    audit=sub.add_parser("audit-v2")
    audit.add_argument("--manifest",required=True); audit.add_argument("--output",required=True)
    audit.add_argument("--scan-arrays",action="store_true"); audit.add_argument("--allow-synthetic",action="store_true")
    train=sub.add_parser("train-v2")
    train.add_argument("--manifest",required=True); train.add_argument("--config",required=True)
    train.add_argument("--output",required=True); train.add_argument("--stage",choices=(*STAGES,"all"),default="all")
    train.add_argument("--resume",action="store_true"); train.add_argument("--warm-start")
    train.add_argument("--stop-after",type=int)
    smoke=sub.add_parser("smoke-v2")
    smoke.add_argument("--config",default=str(Path(__file__).resolve().parents[2]/"configs/multistage_smoke_v2.yaml"))
    smoke.add_argument("--output",required=True)
    init=sub.add_parser("initialize-v2")
    init.add_argument("--checkpoint",required=True); init.add_argument("--request",required=True)
    init.add_argument("--output",required=True); init.add_argument("--device",default="cpu")
    fc=sub.add_parser("forecast-v2")
    fc.add_argument("--state",required=True); fc.add_argument("--output",required=True)
    fc.add_argument("--output-stages",nargs="+",type=int); fc.add_argument("--codec")
    fc.add_argument("--state-output-dir")
    ob=sub.add_parser("observe-v2")
    ob.add_argument("--state",required=True); ob.add_argument("--observation",required=True)
    ob.add_argument("--output",required=True); ob.add_argument("--device",default="cpu")
    qr=sub.add_parser("query-pcr-v2")
    qr.add_argument("--state",required=True); qr.add_argument("--output",required=True)
    qr.add_argument("--mode",choices=("observed_landmark","forecasted_state","marginal_current"),default="observed_landmark")
    for command in (fc,qr):
        command.add_argument("--device",default="cpu"); command.add_argument("--samples",type=int,default=8)
        command.add_argument("--steps",type=int,default=20); command.add_argument("--seed",type=int,default=0)
    ev=sub.add_parser("evaluate-v2")
    ev.add_argument("--checkpoint",required=True); ev.add_argument("--manifest",required=True)
    ev.add_argument("--output",required=True); ev.add_argument("--device",default="cpu")
    ev.add_argument("--split",choices=("val","test"),default="val"); ev.add_argument("--bootstrap",type=int,default=1000)
    args=p.parse_args(argv)
    if args.command=="audit-v2":
        report=PatientTrajectoryStore(args.manifest,args.allow_synthetic).audit(args.scan_arrays)
        write_json(args.output,report)
    elif args.command in {"train-v2","smoke-v2"}:
        cfg=load_config(args.config)
        root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
        if args.command=="smoke-v2":
            from .synthetic_v2 import make_synthetic_v2
            if not cfg.training.allow_synthetic:
                raise ValueError("Smoke requires an explicitly synthetic configuration")
            manifest=make_synthetic_v2(root/"synthetic_data")
            stages=STAGES
        else:
            manifest=args.manifest
            stages=STAGES if args.stage=="all" else (args.stage,)
        store=PatientTrajectoryStore(manifest,cfg.training.allow_synthetic)
        reports=[]
        with (root/"controller.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                for stage in stages:
                    write_json(root/"controller_status.json",{"status":"running","stage":stage,"updated_unix":time.time()})
                    report=train_stage(store,cfg,root,stage,resume=getattr(args,"resume",False) and (root/stage/"last.pt").exists(),
                        warm_start=getattr(args,"warm_start",None) if stage=="representation" else None,
                        stop_after=getattr(args,"stop_after",None))
                    reports.append(report)
                    if not report["completed"]:
                        break
                write_json(root/"controller_status.json",{"status":"completed" if reports[-1]["completed"] else "paused",
                           "stages":reports,"updated_unix":time.time()})
            except Exception as exc:
                write_json(root/"controller_status.json",{"status":"failed","stage":stage,"error":str(exc),"updated_unix":time.time()})
                raise
        report={"stages":reports,"synthetic":store.manifest.get("synthetic",False)}
    elif args.command=="evaluate-v2":
        model,payload=load_trained(args.checkpoint,args.device)
        store=PatientTrajectoryStore(args.manifest,payload["metadata"]["synthetic"])
        if store.manifest_digest!=payload["manifest_digest"]:
            raise ValueError("Evaluation manifest differs from checkpoint")
        store.set_statistics(payload["metadata"]["statistics"])
        if not store.by_split[args.split]:
            raise ValueError("Requested split is absent")
        import copy
        evaluation_cfg=copy.deepcopy(model.cfg)
        evaluation_cfg.training.validation_cases=len(store.by_split[args.split])
        report=validate(model,store,evaluation_cfg,model.stage,split=args.split,bootstrap=args.bootstrap)
        from .evaluation_v2 import prequential_evaluation
        report["prequential"]=prequential_evaluation(model,store,evaluation_cfg,args.split,args.bootstrap)
        write_json(args.output,report)
    else:
        from . import inference_v2
        arguments=vars(args).copy(); arguments.pop("command")
        name={"initialize-v2":"initialize","forecast-v2":"forecast","observe-v2":"observe","query-pcr-v2":"query"}[args.command]
        report=getattr(inference_v2,name)(**arguments)
    print(json.dumps(report,ensure_ascii=False,indent=2))

"""Executable research workflow. Run `python joint.py --help`."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import platform
import torch
from .config import load_config
from .data import ManifestStore
from .io import write_json,read_json,seed_all
from .training import train_stage,evaluate,STAGES
from .synthetic import make_synthetic
from .inference import predict,encode_images
from .adapters import convert_v2


def parser():
    p=argparse.ArgumentParser(description="Image-grounded stochastic SymmFlow + longitudinal pCR")
    commands=p.add_subparsers(dest="command",required=True)
    smoke=commands.add_parser("smoke",help="Synthetic four-stage end-to-end engineering test")
    smoke.add_argument("--output",required=True)
    smoke.add_argument("--config",default=str(Path(__file__).resolve().parents[2]/"configs"/"smoke.yaml"))
    audit=commands.add_parser("audit")
    audit.add_argument("--manifest",required=True); audit.add_argument("--output",required=True)
    audit.add_argument("--scan-arrays",action="store_true"); audit.add_argument("--allow-synthetic",action="store_true")
    train=commands.add_parser("train")
    train.add_argument("--manifest",required=True); train.add_argument("--config",required=True)
    train.add_argument("--output",required=True); train.add_argument("--stage",choices=(*STAGES,"all"),default="all")
    train.add_argument("--resume",action="store_true"); train.add_argument("--init-v2")
    train.add_argument("--stop-after",type=int,help="Stop at a saved optimizer-step boundary (for controlled interruption testing)")
    pred=commands.add_parser("predict")
    pred.add_argument("--checkpoint",required=True); pred.add_argument("--request",required=True); pred.add_argument("--output",required=True)
    pred.add_argument("--device",default="cpu"); pred.add_argument("--samples",type=int); pred.add_argument("--steps",type=int)
    pred.add_argument("--seed",type=int,default=0); pred.add_argument("--codec")
    ev=commands.add_parser("evaluate")
    ev.add_argument("--checkpoint",required=True); ev.add_argument("--manifest",required=True); ev.add_argument("--output",required=True)
    ev.add_argument("--split",choices=("train","val","test"),default="test"); ev.add_argument("--device",default="cpu")
    ev.add_argument("--samples",type=int,default=8); ev.add_argument("--steps",type=int)
    ev.add_argument("--bootstrap",type=int,default=1000); ev.add_argument("--allow-synthetic",action="store_true")
    enc=commands.add_parser("encode")
    enc.add_argument("--images",required=True); enc.add_argument("--codec",required=True); enc.add_argument("--output",required=True)
    enc.add_argument("--device",default="cpu"); enc.add_argument("--normalization-record",required=True)
    cv=commands.add_parser("convert-v2")
    cv.add_argument("--manifest",required=True); cv.add_argument("--landmarks",required=True); cv.add_argument("--output",required=True)
    return p


def main(argv=None):
    import sys
    arguments=sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in {"train-v3", "smoke-v3", "evaluate-v3"}:
        from .cli_v3 import main as main_v3
        return main_v3(arguments)
    if arguments and arguments[0] in {"audit-v2","train-v2","smoke-v2","initialize-v2","forecast-v2","observe-v2","query-pcr-v2","evaluate-v2"}:
        from .cli_v2 import main as main_v2
        return main_v2(arguments)
    args=parser().parse_args(argv)
    if args.command == "smoke":
        out=Path(args.output).resolve()
        if (out/"run").exists():
            raise FileExistsError("Smoke output already contains a run; choose a new directory")
        cfg=load_config(args.config)
        if not cfg.training.allow_synthetic or cfg.training.device != "cpu":
            raise ValueError("Smoke requires explicit synthetic CPU configuration")
        manifest=make_synthetic(out/"data")
        store=ManifestStore(manifest,allow_synthetic=True)
        results=[]
        for stage in STAGES:
            results.append(train_stage(store,cfg,out/"run",stage))
        checkpoint=out/"run"/"joint"/"best.pt"
        evaluate(checkpoint,manifest,out/"synthetic_evaluation.json",device="cpu",samples=2,steps=2,
                 bootstrap=0,allow_synthetic=True)
        prediction=predict(checkpoint,out/"data"/"request.json",out/"synthetic_prediction.npz",samples=2,steps=2)
        report={"engineering_only":True,"clinical_validation":False,"python":platform.python_version(),
                "pytorch":torch.__version__,"cuda_available":torch.cuda.is_available(),"stages":results,
                "prediction_shapes":prediction["shape"]}
        write_json(out/"smoke_report.json",report)
        print(json.dumps(report,indent=2))
    elif args.command == "audit":
        report=ManifestStore(args.manifest,args.allow_synthetic).audit(args.scan_arrays)
        write_json(args.output,report); print(json.dumps(report,indent=2))
    elif args.command == "train":
        cfg=load_config(args.config)
        store=ManifestStore(args.manifest,cfg.training.allow_synthetic)
        stages=STAGES if args.stage == "all" else (args.stage,)
        if args.resume and args.stage == "all":
            raise ValueError("Resume one explicit stage at a time")
        for stage in stages:
            report=train_stage(store,cfg,args.output,stage,resume=args.resume,
                               init_v2=args.init_v2 if stage == "representation" else None,stop_after=args.stop_after)
            print(json.dumps(report))
            if not report["completed"]:
                break
    elif args.command == "predict":
        report=predict(args.checkpoint,args.request,args.output,device=args.device,samples=args.samples,
                       steps=args.steps,seed=args.seed,codec_path=args.codec)
        print(json.dumps(report,indent=2))
    elif args.command == "evaluate":
        report=evaluate(args.checkpoint,args.manifest,args.output,split=args.split,device=args.device,
                        samples=args.samples,steps=args.steps,bootstrap=args.bootstrap,allow_synthetic=args.allow_synthetic)
        print(json.dumps({k:v for k,v in report.items() if k != "rows"},indent=2))
    elif args.command == "encode":
        encode_images(args.images,args.codec,args.output,device=args.device,normalization_record=args.normalization_record)
    elif args.command == "convert-v2":
        print(json.dumps(convert_v2(args.manifest,args.landmarks,args.output),indent=2))

if __name__ == "__main__":
    main()

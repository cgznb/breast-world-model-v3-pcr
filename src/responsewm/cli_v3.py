"""Deployment protocol, fixed cohorts, and conservative controlled experiments."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import time

from .config import load_config
from .data_v2 import PatientTrajectoryStore
from .io import digest, write_json, read_json
from .training import STAGES
from .training_v3 import initialize_run, train_stage, load_trained
from .evaluation_v3 import validate
from .runtime_v3 import stage_gpu_lease


def evaluate_selected(store, cfg, root, stage):
    """Evaluate the selected artifact, independent of the last optimizer step."""
    folder = Path(root)/stage
    checkpoint = folder/"best.pt"
    target = folder/"evaluation_selected.json"
    fingerprint = digest(checkpoint)
    if target.exists() and read_json(target).get("checkpoint_sha256") == fingerprint:
        return
    model, payload = load_trained(checkpoint, cfg.training.device)
    report = validate(model, store, cfg, stage, full=True)
    report.update(checkpoint_sha256=fingerprint, selected_step=payload["step"])
    if cfg.protocol.imaging_manifest and cfg.protocol.image_eval_cases:
        from .imaging_evaluation_v3 import evaluate_imaging
        report["decoded_imaging"] = evaluate_imaging(model, store, cfg, folder/"imaging_selected")
    write_json(target, report)
    del model


def run_stages(store, cfg, root, stages, *, resume=False, stop_after=None, selected_evaluation=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for stage in stages:
                write_json(root/"controller_status.json", {"status":"running", "stage":stage,"updated_unix":time.time()})
                with stage_gpu_lease(stage, root, cfg.training.device):
                    result = train_stage(store, cfg, root, stage,
                        resume=resume and (root/stage/"last.pt").exists(), stop_after=stop_after)
                    reports.append(result)
                    if not result["completed"]:
                        break
                    if selected_evaluation and stage != "representation":
                        write_json(root/"controller_status.json", {"status":"evaluating", "stage":stage,"updated_unix":time.time()})
                        evaluate_selected(store, cfg, root, stage)
            write_json(root/"controller_status.json", {"status":"completed" if reports[-1]["completed"] else "paused",
                "stages":reports,"updated_unix":time.time()})
        except BaseException as exc:
            write_json(root/"controller_status.json", {"status":"failed", "stage":stage,"error":str(exc),"updated_unix":time.time()})
            raise
    return reports


def inherit_flow(source, target, cfg, store):
    """An explicit fork copies only frozen A/B weights; no C optimizer is reused."""
    source, target = Path(source), Path(target)
    initialize_run(store, cfg, target)
    target.joinpath("flow").mkdir(exist_ok=True)
    inherited = {"source_run":str(source.resolve()), "reason":"Matched B checkpoint for prespecified C ablation", "files":{}}
    for name in ("best.pt", "last.pt"):
        original, copied = source/"flow"/name, target/"flow"/name
        checksum = digest(original)
        inherited["files"][name] = checksum
        if copied.exists():
            if digest(copied) != checksum:
                raise ValueError("Inherited B checkpoint changed")
        else:
            shutil.copy2(original, copied)
    record = target/"inheritance.json"
    if record.exists() and read_json(record) != inherited:
        raise ValueError("Ablation lineage changed")
    write_json(record, inherited)


def run_ablations(store, cfg, root):
    """One matched seed diagnoses LR, update scope, and residual regularization."""
    results = {}
    specifications = {
        "A_lr_1e4": {"representation_lr":1e-4},
        "C_output_only": {"readout_scope":"output"},
        "C_all_blocks": {"readout_scope":"all"},
        "C_lr_1e4": {"readout_lr":1e-4},
        "C_no_residual_penalty": {"residual_penalty":0.0},
    }
    ablations = Path(root).parent/"ablations"
    ablations.mkdir(exist_ok=True)
    write_json(ablations/"prespecified_plan.json", {
        "reference":str(Path(root).resolve()), "specifications":specifications,
        "purpose":"One-factor diagnostics on a shared seed; primary deployment model remains prespecified last_block",
        "no_independent_test":not bool(store.by_split["test"])})
    for name, changes in specifications.items():
        alternative = copy.deepcopy(cfg)
        for key, value in changes.items():
            setattr(alternative.protocol, key, value)
        folder = ablations/name
        stages = ("representation",) if name.startswith("A_") else ("readout",)
        if stages == ("readout",):
            inherit_flow(root, folder, alternative, store)
        write_json(folder/"configuration.json", alternative.to_dict())
        reports = run_stages(store, alternative, folder, stages, resume=True)
        results[name] = reports
        write_json(ablations/"results.json", results)
        if not reports[-1]["completed"]:
            return False
    return True


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-v3")
    train.add_argument("--config", required=True); train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True); train.add_argument("--stage", choices=(*STAGES,"all"), default="all")
    train.add_argument("--resume", action="store_true"); train.add_argument("--stop-after", type=int)
    train.add_argument("--run-ablations", action="store_true")
    smoke = sub.add_parser("smoke-v3")
    smoke.add_argument("--config", default=str(Path(__file__).resolve().parents[2]/"configs/multistage_smoke_v3.yaml"))
    smoke.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-v3")
    evaluate.add_argument("--checkpoint", required=True); evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True); evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--split", choices=("val","test"), default="val")
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == "evaluate-v3":
        model, payload = load_trained(args.checkpoint, args.device)
        store = PatientTrajectoryStore(args.manifest, payload["metadata"]["synthetic"])
        if store.manifest_digest != payload["manifest_digest"]:
            raise ValueError("Evaluation manifest differs from checkpoint")
        store.set_statistics(payload["metadata"]["statistics"])
        cfg = copy.deepcopy(model.cfg); cfg.training.device = args.device
        report = validate(model, store, cfg, model.stage, split=args.split, bootstrap=args.bootstrap, full=True)
        if cfg.protocol.imaging_manifest and cfg.protocol.image_eval_cases:
            from .imaging_evaluation_v3 import evaluate_imaging
            report["decoded_imaging"] = evaluate_imaging(model, store, cfg, Path(args.output).with_suffix(""), split=args.split)
        write_json(args.output, report)
        print(json.dumps({k:v for k,v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
        return
    cfg = load_config(args.config)
    if cfg.schema != "responsewm_v3":
        raise ValueError("V3 commands require responsewm_v3 configuration")
    root = Path(args.output).resolve()
    if args.command == "smoke-v3":
        from .synthetic_v2 import make_synthetic_v2
        if not cfg.training.allow_synthetic or cfg.training.device != "cpu":
            raise ValueError("Smoke requires explicit synthetic CPU configuration")
        manifest = make_synthetic_v2(root/"synthetic_data")
        stages = STAGES
    else:
        manifest = args.manifest
        stages = STAGES if args.stage == "all" else (args.stage,)
    store = PatientTrajectoryStore(manifest, cfg.training.allow_synthetic)
    reports = run_stages(store, cfg, root, stages, resume=getattr(args,"resume",False),
                        stop_after=getattr(args,"stop_after",None))
    if getattr(args, "run_ablations", False) and reports[-1]["completed"]:
        if stages != STAGES:
            raise ValueError("Ablations follow a completed four-stage primary run")
        write_json(root/"ablation_status.json", {"completed":False,"updated_unix":time.time()})
        completed = run_ablations(store, cfg, root)
        write_json(root/"ablation_status.json", {"completed":completed,"updated_unix":time.time()})
    print(json.dumps({"stages":reports,"synthetic":store.manifest.get("synthetic",False)}, ensure_ascii=False, indent=2))

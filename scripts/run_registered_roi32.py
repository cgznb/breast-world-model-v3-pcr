"""Locked, detachable A-to-B execution with resumable checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from symm_observation.utils import file_identity, read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--stage", choices=["A", "B", "all"], default="all")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    root, data = Path(args.output).resolve(), Path(args.data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (root/"run.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This run already has an active controller")
    if (root/"launch.json").exists() and not args.resume:
        raise FileExistsError("Use --resume for an existing run")
    if args.detach:
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--config", str(Path(args.config).resolve()), "--data", str(data),
                   "--output", str(root), "--stage", args.stage, "--lock-fd", str(lock.fileno())]
        for key in ("resume", "wait_for_idle"):
            if getattr(args, key):
                command.append("--"+key.replace("_", "-"))
        if args.stop_after is not None:
            command.extend(["--stop-after", str(args.stop_after)])
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
               "CUDA_VISIBLE_DEVICES": "0", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
        with (root/"controller.log").open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        print(json.dumps({"pid": child.pid, "output": str(root), "log": str(root/"controller.log")}))
        return
    from symm_observation.config import load_config, stage_budget
    from symm_observation.training import train
    import torch
    cfg = load_config(args.config)
    binding = {"config": json.loads(json.dumps(cfg.to_dict())),
               "data": [file_identity(data/name) for name in ("visits.json", "pairs.json")],
               "sources": [file_identity(p) for p in sorted((REPO/"src/symm_observation").glob("*.py"))]
                          + [file_identity(__file__)]}
    binding_path = root/"runtime_binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise ValueError("Source/config/manifest changed; use a separate run")
    write_json(binding_path, binding)
    stopped = {"value": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.update(value=True))
    launch = {"pid": os.getpid(), "started_utc": now(), "config": str(Path(args.config).resolve()),
              "data": str(data), "output": str(root), "resume": args.resume,
              "torch": str(torch.__version__), "budgets": {s: stage_budget(cfg, s) for s in ("A", "B")}}
    write_json(root/"launch.json", launch)
    write_json(root/"resolved_config.json", cfg.to_dict())
    results = []
    try:
        if args.wait_for_idle:
            while not stopped["value"]:
                pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                if not pids:
                    break
                write_json(root/"progress.json", {"pid": os.getpid(), "status": "waiting_for_gpu", "updated_utc": now()})
                time.sleep(15)
        for stage in (("A", "B") if args.stage == "all" else (args.stage,)):
            if stopped["value"]:
                break
            if shutil.disk_usage(root).free < 10*1024**3:
                raise RuntimeError("Less than 10 GiB remains for checkpoints")
            for identity in binding["sources"] + binding["data"]:
                if file_identity(identity["path"]) != identity:
                    raise ValueError("A bound runtime asset changed")
            write_json(root/"progress.json", {"pid": os.getpid(), "status": "running", "stage": stage,
                                               "completed_stages": results, "updated_utc": now()})
            print(json.dumps({"event": "stage_started", "stage": stage, "utc": now()}), flush=True)
            result = train(stage, cfg, data/("visits.json" if stage == "A" else "pairs.json"), root,
                           observation_checkpoint=root/"A/best.pt" if stage == "B" else None,
                           resume=args.resume and (root/stage/"last.pt").exists(), stop_after=args.stop_after)
            results.append(result)
            print(json.dumps({"event": "stage_finished", **result}), flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not result["complete"] or result.get("stop_requested"):
                break
        expected = 2 if args.stage == "all" else 1
        status = "complete" if len(results) == expected and all(r["complete"] for r in results) else "stopped"
        result = {"pid": os.getpid(), "status": status, "stages": results, "updated_utc": now()}
        write_json(root/"progress.json", result)
        if status == "complete":
            write_json(root/"COMPLETE.json", result)
    except BaseException as exc:
        write_json(root/"progress.json", {"pid": os.getpid(), "status": "failed", "error": str(exc), "updated_utc": now()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

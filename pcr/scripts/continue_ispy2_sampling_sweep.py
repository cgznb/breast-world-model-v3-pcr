"""Resume the matched Euler sweep, then publish verified comparison artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ispy2_biflow_symmflow_steps_20260911"
STOP = threading.Event()


def record(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                                         **values), indent=2) + "\n")
    temporary.replace(path)


def passed(path, key, value):
    return path.exists() and json.loads(path.read_text()).get(key) == value


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def wait_for_gpu(index, memory_gib):
    last_update = 0
    while not STOP.is_set():
        output = subprocess.check_output([
            "nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        free = float(output.strip()) / 1024
        if free >= memory_gib + .5:
            return
        if time.monotonic() - last_update > 30:
            print(f"GPU {index}: waiting for {memory_gib + .5:.1f} GiB; {free:.1f} GiB free", flush=True)
            last_update = time.monotonic()
        STOP.wait(5)
    raise InterruptedError("Sweep interrupted while waiting for GPU")


def run_stage(model, steps, stage, gpu, memory_gib, *, script="ispy2_sampling_sweep.py", extra=()):
    if STOP.is_set():
        raise InterruptedError("Sweep interrupted")
    if stage in ("generate", "extract") and not (model == "symmflow" and steps == 20):
        wait_for_gpu(gpu, memory_gib)
    free = shutil.disk_usage(OUT).free / 2**30
    if free < (14 if stage == "generate" else 5):
        raise RuntimeError(f"Insufficient disk space before {model} Euler-{steps} {stage}: {free:.1f} GiB")
    command = [sys.executable, "-u", str(ROOT / "scripts" / script)]
    if script == "ispy2_sampling_sweep.py":
        command += [stage, "--model", model, "--steps", str(steps), "--device", "cuda:0",
                    "--memory-gib", str(memory_gib), "--workers", "2"]
    command += list(extra)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false")
    log_path = OUT / "logs" / f"{model}_euler{steps}_{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = OUT / f"{model}_status.json"
    print(f"Starting {model} Euler-{steps}: {stage}", flush=True)
    with log_path.open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        record(state_path, state="running", model=model, steps=steps, stage=stage,
               pid=child.pid, log=str(log_path), command=command)
        try:
            while child.poll() is None:
                if STOP.wait(1):
                    raise InterruptedError("Sweep interrupted")
            if child.returncode:
                raise RuntimeError(f"{model} Euler-{steps} {stage} exited {child.returncode}; {log_path}")
        except BaseException:
            stop_child(child)
            record(state_path, state="failed", model=model, steps=steps, stage=stage, log=str(log_path))
            raise
    record(state_path, state="stage_completed", model=model, steps=steps, stage=stage, log=str(log_path))


def wait_existing_biflow(pid):
    if not pid:
        return
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.exists():
        return
    args = command_path.read_bytes().decode().split("\0")
    if (not any(Path(arg).name == "ispy2_sampling_sweep.py" for arg in args)
        or "generate" not in args or args[args.index("--model") + 1] != "biflow"
        or args[args.index("--steps") + 1] != "20"):
        raise ValueError("Existing PID is not the expected BiFlow Euler-20 generation")
    start = Path(f"/proc/{pid}/stat").read_text().split()[21]
    record(OUT / "biflow_status.json", state="waiting_existing_generation", model="biflow", steps=20, pid=pid)
    while not STOP.wait(2):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            if stat[21] != start or stat[2] == "Z":
                return
        except FileNotFoundError:
            return
    raise InterruptedError("Sweep interrupted while waiting for existing generation")


def setting(model, steps, gpu, memory_gib, existing_pid=0):
    try:
        if model == "biflow" and steps == 20:
            wait_existing_biflow(existing_pid)
        root = OUT / model / f"euler_{steps}"
        if passed(root / "complete.json", "status", "completed"):
            return
        # An interrupted prune has already passed the full decoded-image audit.
        if passed(root / "audit.json", "status", "passed"):
            run_stage(model, steps, "prune", gpu, memory_gib)
            return
        if not passed(root / "generation.json", "complete", True):
            run_stage(model, steps, "generate", gpu, memory_gib)
        if not passed(root / "extraction.json", "complete", True):
            run_stage(model, steps, "extract", gpu, memory_gib)
        if not (root / "image_metrics/by_target_stage.csv").exists():
            run_stage(model, steps, "images", gpu, memory_gib)
        if not passed(root / "pcr/audit.json", "real_frozen_replay", "passed"):
            run_stage(model, steps, "pcr", gpu, memory_gib)
        run_stage(model, steps, "audit", gpu, memory_gib)
        run_stage(model, steps, "prune", gpu, memory_gib)
        record(OUT / f"{model}_status.json", state="setting_completed", model=model, steps=steps)
    except BaseException:
        STOP.set()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-biflow-pid", type=int, default=0)
    parser.add_argument("--biflow-gpu", type=int, default=0)
    parser.add_argument("--symmflow-gpu", type=int, default=1)
    args = parser.parse_args()
    if args.biflow_gpu == args.symmflow_gpu:
        parser.error("Concurrent model queues require separate GPUs")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: STOP.set())
        completed = []
        try:
            for steps in (20, 2, 10, 50):
                record(OUT / "status.json", state="running", steps=steps, completed_steps=completed,
                       pid=os.getpid(), gpu_by_model={"biflow": args.biflow_gpu, "symmflow": args.symmflow_gpu})
                with ThreadPoolExecutor(max_workers=2) as pool:
                    tasks = [pool.submit(setting, model, steps, gpu, cap, args.wait_biflow_pid)
                             for model, gpu, cap in (("biflow", args.biflow_gpu, 12),
                                                     ("symmflow", args.symmflow_gpu, 8))]
                    for task in tasks:
                        task.result()
                completed.append(steps)
                extra = ["--steps", *[str(value) for value in sorted(completed)]]
                run_stage("report", steps, "report", args.biflow_gpu, 0,
                          script="report_ispy2_sampling_sweep.py", extra=extra)
            record(OUT / "status.json", state="completed", completed_steps=completed,
                   report=str(OUT / "report.md"), gallery=str(OUT / "index.html"))
        except BaseException as error:
            record(OUT / "status.json", state="failed", completed_steps=completed, error=str(error))
            raise


if __name__ == "__main__":
    main()

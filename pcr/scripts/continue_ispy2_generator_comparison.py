"""Resume SymmFlow generation and frozen pCR evaluation against existing BiFlow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ispy2_generator_comparison_20260911"
REFERENCE = OUT / "symmflow_reference"
REMOTE = "qingyuan:/path/to/local/ispy2-symmflow3d/outputs/mewm_all_pairs_5090/comparison_export_20260911"
SSH = "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ControlPath=/root/.ssh/cm-ispy2-comparison-host"
CHUNK = 41943040
SIZE = 332689803
MODELS = ["biflow", "symmflow"]
STATUS = OUT / "continuation_status.json"


def status(stage, state, **details):
    value = {"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(),
             "stage": stage, "status": state, "models": MODELS, **details}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, STATUS)


def stop_child(child):
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def wait_for_existing_transfers(pids):
    active = list(pids)
    while active:
        remaining = []
        for pid in active:
            path = Path(f"/proc/{pid}/cmdline")
            try:
                command = path.read_bytes().decode().split("\0")
            except FileNotFoundError:
                continue
            if (command and Path(command[0]).name == "rsync"
                and any(value.startswith(REMOTE + "/inference.part") for value in command)):
                remaining.append(pid)
        active = remaining
        if active:
            transferred = 0
            for index in range(8):
                name = f"inference.part{index:02d}"
                candidates = [REFERENCE / name, *REFERENCE.glob(f".{name}.*")]
                transferred += max((path.stat().st_size for path in candidates if path.exists()), default=0)
            status("transfer", "running", transfer_pids=active, transferred_bytes=transferred,
                   expected_bytes=SIZE, reusing_existing_transfers=True)
            time.sleep(10)


def transfer_parts():
    active = {}
    logs = {}
    try:
        for index in range(8):
            part = REFERENCE / f"inference.part{index:02d}"
            expected = CHUNK if index < 7 else SIZE - 7 * CHUNK
            if part.exists() and part.stat().st_size == expected:
                continue
            logs[index] = (OUT / f"symmflow_transfer_part{index:02d}.log").open("a")
            command = ["rsync", "-az", "--partial", "--append-verify", "--timeout=180",
                       "--stats", "-e", SSH, f"{REMOTE}/{part.name}", str(REFERENCE) + "/"]
            active[index] = subprocess.Popen(command, cwd=ROOT, stdout=logs[index],
                                             stderr=subprocess.STDOUT)
        while active:
            status("transfer", "running", active_parts=sorted(active),
                   transfer_pids=[child.pid for child in active.values()])
            for index, child in list(active.items()):
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"transfer part {index} exited with code {code}")
                    del active[index]
            if active:
                time.sleep(10)
    finally:
        for child in active.values():
            stop_child(child)
        for output in logs.values():
            output.close()


def wait_for_gpu():
    while True:
        probe = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True,
                               text=True, timeout=20)
        memory, utilization = map(int, probe.stdout.strip().split(","))
        if memory < 1024 and utilization < 10:
            return
        status("waiting_for_gpu1", "running", gpu_memory_mib=memory, gpu_utilization=utilization)
        time.sleep(30)


def run_stage(name, script, arguments):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with (OUT / f"continuation_{name}.log").open("a") as output:
        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / script), *arguments],
                                 cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT)
        status(name, "running", child_pid=child.pid)
        try:
            code = child.wait()
        finally:
            stop_child(child)
        if code:
            raise RuntimeError(f"{name} exited with code {code}; inspect continuation_{name}.log")


def complete(path):
    return path.exists() and json.loads(path.read_text()).get("complete") is True


def run(transfer_pids):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for_existing_transfers(transfer_pids)
            transfer_parts()
            run_stage("assemble", "assemble_symmflow_parts.py", [])
            status("verify_transfer", "running")
            # Compare file contents without requesting timestamp/owner preservation.
            check = subprocess.run(["rsync", "-cni", "--out-format=%i", "--timeout=180", "-e", SSH,
                                    f"{REMOTE}/inference.pt", str(REFERENCE / "inference.pt")],
                                   check=True, capture_output=True, text=True, timeout=300)
            if check.stdout.strip():
                raise ValueError("Read-only full-file comparison found a checkpoint difference")
            verified = {"status": "passed", "bytes": SIZE, "tensor_assembly": "passed",
                        "read_only_remote_content_comparison": "passed",
                        "updated_utc": datetime.now(timezone.utc).isoformat()}
            (OUT / "symmflow_transfer_verification.json").write_text(json.dumps(verified, indent=2) + "\n")
            if not complete(OUT / "symmflow/direct_t0/generation_status.json"):
                wait_for_gpu()
                run_stage("generate_smoke", "compare_ispy2_generators.py", ["generate", "--models", "symmflow", "--limit", "1"])
                wait_for_gpu()
                run_stage("generate", "compare_ispy2_generators.py", ["generate", "--models", "symmflow"])
            wait_for_gpu()
            if not complete(OUT / "symmflow/direct_t0/embeddings/extraction_status.json"):
                run_stage("extract_smoke", "compare_ispy2_generators.py", ["extract", "--models", "symmflow", "--limit", "1"])
                wait_for_gpu()
                run_stage("extract", "compare_ispy2_generators.py", ["extract", "--models", "symmflow"])
            run_stage("pcr", "compare_ispy2_generators.py", ["pcr", "--models", *MODELS, "--device", "cpu"])
            run_stage("images", "report_ispy2_generators.py", ["images", "--models", *MODELS, "--workers", "2"])
            run_stage("figures", "report_ispy2_generators.py", ["figures", "--models", *MODELS, "--scope", *MODELS])
            run_stage("uncertainty", "report_ispy2_generators.py", ["uncertainty"])
            run_stage("report", "report_ispy2_generators.py", ["report", "--models", *MODELS, "--scope", *MODELS])
            status("available_models_complete", "completed", report=str(OUT / "report.md"))
        except BaseException as error:
            prior = json.loads(STATUS.read_text()) if STATUS.exists() else {}
            status(prior.get("stage", "starting"), "failed", error_type=type(error).__name__,
                   error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--wait-transfer-pids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.detach:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "continuation_launcher.log").open("a") as output:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--wait-transfer-pids", *map(str, args.wait_transfer_pids)], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({"pid": child.pid, "status_file": str(STATUS)}))
    else:
        run(args.wait_transfer_pids)


if __name__ == "__main__":
    main()

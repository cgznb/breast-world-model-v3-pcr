"""Wait for existing work, measure GPU batches, verify A/B, then train unattended."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from symm_observation.utils import file_identity, read_json, write_json, load_checkpoint
from symm_observation.config import load_config, from_dict, stage_budget


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return None if value[0] == "Z" else {"pid": pid, "start_ticks": value[19]}
    except FileNotFoundError:
        return None


def select_profile(reports):
    safe = [r for r in reports if r.get("passed")
            and r["peak_reserved_mib"] <= .93*r["device_total_mib"]]
    if not safe:
        raise RuntimeError("No batch passed the 93-percent allocator memory limit")
    best_rate = max(r["examples_per_second"] for r in safe)
    efficient = [r for r in safe if r["examples_per_second"] >= .95*best_rate]
    return max(efficient, key=lambda r: (r["peak_allocated_mib"], r["batch"]))


def verify_smoke(root):
    import torch
    a = load_checkpoint(root/"A/best.pt")
    b = load_checkpoint(root/"B/last.pt")
    assert a["training_run_completed"] and b["training_run_completed"]
    assert a["metadata"]["A_contains_longitudinal_predictor"] is False
    assert not any("future_predictor" in k or "clinical" in k or "pcr" in k for k in a["model"])
    compared = 0
    for key, value in a["model"].items():
        if key.startswith("target_encoder."):
            bk = "observation_encoder."+key.removeprefix("target_encoder.")
            assert torch.equal(value, b["model"][bk])
            assert torch.equal(value, b["teacher"][bk])
            compared += 1
    assert compared > 0
    for stage in ("A", "B"):
        assert read_json(root/stage/"status.json")["complete"]
        for line in (root/stage/"metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert all(not isinstance(v, float) or __import__("math").isfinite(v) for v in row.values())
    report = {"passed": True, "A_to_B_frozen_teacher_tensors": compared,
              "A_contains_clinical_or_future_predictor_or_pcr": False,
              "A_steps": read_json(root/"A/status.json")["step"],
              "B_steps": read_json(root/"B/status.json")["step"],
              "complete_stage_checkpoints": True}
    write_json(root/"verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    root, data = Path(args.output).resolve(), Path(args.data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (root/"pipeline.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This pipeline is already active")
    if (root/"pipeline_launch.json").exists() and not args.resume:
        raise FileExistsError("Existing pipeline: use --resume")
    if args.detach:
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--config", str(Path(args.config).resolve()), "--data", str(data),
                   "--output", str(root), "--lock-fd", str(lock.fileno())]
        if args.wait_for_pid:
            command.extend(["--wait-for-pid", str(args.wait_for_pid)])
        if args.resume:
            command.append("--resume")
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0"}
        with (root/"pipeline.log").open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        print(json.dumps({"pid": child.pid, "output": str(root), "log": str(root/"pipeline.log")}))
        return
    cfg = load_config(args.config)
    bindings = {"config": json.loads(json.dumps(cfg.to_dict())),
                "assets": [file_identity(p) for p in sorted((REPO/"src/symm_observation").glob("*.py"))]
                         + [file_identity(REPO/"scripts"/name) for name in
                            ("profile_registered_roi32.py", "run_registered_roi32.py", "launch_registered_roi32.py")]
                         + [file_identity(data/name) for name in ("visits.json", "pairs.json", "statistics.json")]}
    bound = root/"pipeline_binding.json"
    if bound.exists() and read_json(bound) != bindings:
        raise ValueError("Bound configuration or code changed; use an isolated pipeline")
    write_json(bound, bindings)
    stopped, child = {"value": False}, None

    def stop(signum, _frame):
        stopped["value"] = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)

    def status(state, **details):
        value = {"pid": os.getpid(), "status": state, "updated_utc": now(), **details}
        write_json(root/"pipeline_status.json", value)
        return value

    def verify_binding():
        for identity in bindings["assets"]:
            if file_identity(identity["path"]) != identity:
                raise ValueError("A bound source or data manifest changed")

    def run(command, log_path):
        nonlocal child
        if stopped["value"]:
            raise InterruptedError("Pipeline stopped")
        verify_binding()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            if stopped["value"]:
                child.terminate()
            code = child.wait()
        child = None
        if stopped["value"]:
            raise InterruptedError("Pipeline stopped")
        if code:
            raise RuntimeError(f"Child exited {code}; inspect {log_path}")

    blocker = process_identity(args.wait_for_pid) if args.wait_for_pid else None
    write_json(root/"pipeline_launch.json", {"pid": os.getpid(), "started_utc": now(), "blocker": blocker})
    try:
        while not stopped["value"]:
            blocked = blocker is not None and process_identity(blocker["pid"]) == blocker
            gpu_pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            if not blocked and not gpu_pids:
                break
            status("waiting_for_gpu", existing_queue=blocker, gpu_pids=gpu_pids.splitlines())
            time.sleep(10)
        if stopped["value"]:
            raise InterruptedError("Pipeline stopped before GPU work")
        selected = {}
        preflight = root/"preflight"
        for stage, candidates in (("A", (16, 32, 64, 128, 160, 200, 250, 256, 320, 400, 500, 640)),
                                  ("B", (16, 32, 64, 80, 100, 128, 160, 200))):
            reports = []
            for batch in candidates:
                status("profiling", stage=stage, batch=batch)
                output = preflight/f"profile_{stage}_{batch}.json"
                if not output.exists():
                    run([sys.executable, "-u", "-B", str(REPO/"scripts/profile_registered_roi32.py"),
                         "--config", str(Path(args.config).resolve()), "--data", str(data),
                         "--stage", stage, "--batch", str(batch), "--output", str(output)],
                        output.with_suffix(".log"))
                report = read_json(output)
                reports.append(report)
                if not report.get("passed") or report["peak_reserved_mib"] > .93*report["device_total_mib"]:
                    break
            selected[stage] = select_profile(reports)
        raw = cfg.to_dict()
        raw["training"]["stage_batches"] = {s: selected[s]["batch"] for s in ("A", "B")}
        cfg = from_dict(raw)
        config_path = root/"selected_config.json"
        if config_path.exists():
            if read_json(config_path) != json.loads(json.dumps(cfg.to_dict())):
                raise ValueError("Selected training configuration changed")
        else:
            write_json(config_path, cfg.to_dict())
        write_json(preflight/"selection.json", {"selected": selected, "allocator_limit_fraction": .93,
                   "throughput_floor_fraction": .95, "budgets": {s: stage_budget(cfg, s) for s in ("A", "B")}})
        smoke_config = cfg.to_dict()
        smoke_config["training"].update(reference_batch_size=0, stage_a_steps=3, stage_b_steps=3,
                                       validate_every=3, checkpoint_every=1, validation_cases=2, log_every=1,
                                       warmup_steps=1)
        smoke_config["sampling"].update(steps=2, samples=2)
        smoke_path, smoke_root = preflight/"smoke_config.json", preflight/"smoke"
        write_json(smoke_path, smoke_config)
        if not (smoke_root/"verification.json").exists():
            status("smoke_training", batches=cfg.training.stage_batches)
            command = [sys.executable, "-u", "-B", str(REPO/"scripts/run_registered_roi32.py"),
                       "--config", str(smoke_path), "--data", str(data), "--output", str(smoke_root)]
            if (smoke_root/"launch.json").exists():
                command.append("--resume")
            run(command, preflight/"smoke.log")
            verify_smoke(smoke_root)
        status("training", batches=cfg.training.stage_batches)
        command = [sys.executable, "-u", "-B", str(REPO/"scripts/run_registered_roi32.py"),
                   "--config", str(config_path), "--data", str(data), "--output", str(root)]
        if (root/"launch.json").exists():
            command.append("--resume")
        run(command, root/"controller.log")
        result = read_json(root/"progress.json")
        status(result["status"], stages=result["stages"])
    except InterruptedError as exc:
        status("stopped", reason=str(exc))
    except BaseException as exc:
        status("failed", error=str(exc))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

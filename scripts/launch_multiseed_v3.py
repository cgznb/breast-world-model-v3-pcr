"""Launch an isolated, resumable queue of measured physical-batch training runs."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import yaml


STAGES = ("representation", "flow", "readout", "joint")
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def process_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(record):
    if not record.get("pid"):
        return False
    identity = process_identity(record["pid"])
    return identity is not None and identity == record.get("process_start_ticks")


def acquire_lock(path):
    handle = Path(path).open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Another launcher or queue worker holds {path}") from None
    return handle


def controller_completed(run, require_ablations=False):
    path = Path(run) / "controller_status.json"
    if not path.exists():
        return False
    report = read_json(path)
    ablation_status = Path(run)/"ablation_status.json"
    if require_ablations and not ablation_status.exists():
        return False
    if ablation_status.exists() and not read_json(ablation_status).get("completed"):
        return False
    stages = report.get("stages", [])
    return (report.get("status") == "completed" and len(stages) == len(STAGES)
            and {stage.get("stage") for stage in stages} == set(STAGES)
            and all(stage.get("completed") is True for stage in stages))


def reject_live_processes(output):
    for receipt in output.glob("launch.*.json"):
        if process_alive(read_json(receipt)):
            raise RuntimeError(f"Queue process from {receipt.name} is still alive")
    status_path = output / "queue_status.json"
    if status_path.exists():
        status = read_json(status_path)
        for record in [status, *status.get("jobs", [])]:
            if process_alive(record):
                raise RuntimeError(f"Existing queue/training process {record['pid']} is still alive")


def validate_preflight(config, manifest, preflight):
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    from responsewm.config import from_dict

    batch_size=config.get("training", {}).get("batch_size")
    stage_batches=config.get("protocol", {}).get("stage_batch_sizes", {})
    if not isinstance(batch_size,int) or batch_size < 1:
        raise ValueError("This queue requires a positive training.batch_size")
    if preflight.get("manifest_digest") != digest(manifest):
        raise ValueError("Preflight manifest digest differs from the requested training data")
    if preflight.get("physical_batch_size") != batch_size:
        raise ValueError("Preflight must execute configured physical batch size")
    if preflight.get("config", {}).get("training", {}).get("batch_size") != batch_size:
        raise ValueError("Preflight configuration must use configured training.batch_size")
    expected = json.loads(json.dumps(from_dict(config).to_dict()))
    actual = copy.deepcopy(preflight["config"])
    expected["training"].pop("seed", None)
    actual["training"].pop("seed", None)
    if actual != expected:
        raise ValueError("Preflight configuration differs from the training template beyond training.seed")
    results = preflight.get("results", {})
    if set(results) != set(STAGES):
        raise ValueError("Completed A/B/C/D preflight results are required")
    for name, result in results.items():
        if result.get("optimizer_updated") is not True or result.get("physical_batch_size") != stage_batches.get(name,batch_size):
            raise ValueError(f"Preflight {name} did not finish an optimizer update with configured physical batch")
        for key in ("loss", "gradient_norm"):
            if not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]):
                raise ValueError(f"Preflight {name} has missing/nonfinite {key}")


def prepare_plan(args, output):
    template_path, manifest, preflight_path = (Path(value).resolve()
                                             for value in (args.config, args.manifest, args.preflight))
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError("Configuration template must be a YAML mapping")
    validate_preflight(template, manifest, read_json(preflight_path))
    if args.workers < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Workers must be positive and seeds must be nonempty and distinct")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    plan = {"schema": "responsewm_multiseed_queue_v3", "source_root": str(SOURCE_ROOT),
            "python": sys.executable, "output": str(output),
            "template": str(template_path), "template_sha256": digest(template_path),
            "manifest": str(manifest), "manifest_digest": digest(manifest),
            "preflight": str(preflight_path), "preflight_sha256": digest(preflight_path),
            "physical_batch_size": template["training"]["batch_size"], "workers": args.workers, "seeds": args.seeds, "jobs": []}
    generated = []
    for seed in args.seeds:
        config = copy.deepcopy(template)
        config["training"]["seed"] = seed
        content = yaml.safe_dump(config, sort_keys=False)
        folder = output / f"seed_{seed}"
        path, run = folder / "config.yaml", folder / "run"
        plan["jobs"].append({"seed": seed, "config": str(path), "run": str(run),
                             "config_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})
        generated.append((path, content, run))
    plan_path = output / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Existing queue plan differs; use a new experiment output directory")
    if (output / "controller_status.json").exists():
        raise ValueError("Queue output must be a new experiment root, not an existing training run")
    for path, content, run in generated:
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing seed configuration differs: {path}")
        if not plan_path.exists() and run.exists() and any(run.iterdir()):
            raise ValueError(f"Refusing to adopt existing training files without this queue plan: {run}")
    for path, content, run in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        run.mkdir(exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    if not plan_path.exists():
        write_json(plan_path, plan)
    return plan_path, plan


def validate_plan_files(plan):
    if plan.get("schema") != "responsewm_multiseed_queue_v3" or plan.get("source_root") != str(SOURCE_ROOT):
        raise ValueError("Queue plan does not belong to this source snapshot")
    for path, expected in [(plan["template"], plan["template_sha256"]),
                           (plan["manifest"], plan["manifest_digest"]),
                           (plan["preflight"], plan["preflight_sha256"]),
                           *((job["config"], job["config_sha256"]) for job in plan["jobs"])]:
        if digest(path) != expected:
            raise ValueError(f"Queue input changed after the plan was created: {path}")


def run_worker(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    output = Path(plan["output"])
    with acquire_lock(output / "queue.lock"):
        validate_plan_files(plan)
        previous = output / "queue_status.json"
        if previous.exists():
            for job in read_json(previous).get("jobs", []):
                if process_alive(job):
                    raise RuntimeError(f"Prior training process {job['pid']} remains alive")
        status = {"schema": plan["schema"], "plan_sha256": digest(plan_path), "status": "running",
                  "pid": os.getpid(), "process_start_ticks": process_identity(os.getpid()),
                  "workers": plan["workers"], "started_unix": time.time(), "jobs": []}
        for job in plan["jobs"]:
            status["jobs"].append({**job, "status": "completed" if controller_completed(job["run"], job["seed"] == plan["seeds"][0]) else "pending"})
        running = {}
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True

        previous_handlers = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}

        def persist():
            status["updated_unix"] = time.time()
            write_json(output / "queue_status.json", status)

        persist()
        try:
            while True:
                for seed, (process, job) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    completed = code == 0 and controller_completed(job["run"], job["seed"] == plan["seeds"][0])
                    job.update(status="completed" if completed else "failed", returncode=code,
                               ended_unix=time.time())
                    if not completed:
                        job["error"] = f"Training exited {code}; four completed controller stages are required"
                    del running[seed]
                    persist()
                if stopping:
                    for process, job in running.values():
                        if not job.get("termination_sent"):
                            process.terminate()
                            job["termination_sent"] = True
                    if not running:
                        break
                else:
                    for job in status["jobs"]:
                        if len(running) >= plan["workers"]:
                            break
                        if job["status"] != "pending":
                            continue
                        validate_plan_files(plan)
                        command = [plan["python"], "-u", str(SOURCE_ROOT / "joint.py"), "train-v3",
                                   "--stage", "all", "--resume", "--config", job["config"],
                                   "--manifest", plan["manifest"], "--output", job["run"]]
                        if job["seed"] == plan["seeds"][0]:
                            command.append("--run-ablations")
                        job["command"] = command
                        job["resuming_stages"] = [stage for stage in STAGES
                                                  if (Path(job["run"]) / stage / "last.pt").exists()]
                        try:
                            with (Path(job["run"]) / "console.log").open("ab", buffering=0) as log:
                                process = subprocess.Popen(command, cwd=SOURCE_ROOT, stdin=subprocess.DEVNULL,
                                                           stdout=log, stderr=subprocess.STDOUT,
                                                           env={**os.environ,"RESPONSEWM_GPU_LEASE_DIR":str(output/"gpu_leases")})
                        except OSError as exc:
                            job.update(status="failed", error=str(exc), ended_unix=time.time())
                        else:
                            job.update(status="running", pid=process.pid,
                                       process_start_ticks=process_identity(process.pid), started_unix=time.time())
                            running[job["seed"]] = (process, job)
                        persist()
                    if not running and all(job["status"] != "pending" for job in status["jobs"]):
                        break
                time.sleep(1)
            complete = all(job["status"] == "completed" for job in status["jobs"])
            status.update(status="completed" if complete else ("interrupted" if stopping else "failed"),
                          ended_unix=time.time())
            persist()
            return 0 if complete else 1
        except BaseException as exc:
            for process, _ in running.values():
                if process.poll() is None:
                    process.terminate()
            for process, job in running.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                job.update(status="failed", returncode=process.returncode, ended_unix=time.time())
            status.update(status="failed", error=str(exc), ended_unix=time.time())
            persist()
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Measured-batch YAML template; only the seed changes per run")
    parser.add_argument("--manifest")
    parser.add_argument("--output", help="Independent new multi-seed experiment root")
    parser.add_argument("--preflight", help="Completed measured physical-batch A/B/C/D preflight JSON")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-plan", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        return run_worker(args.worker_plan)
    for name in ("config", "manifest", "output", "preflight", "seeds"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with acquire_lock(output / "launch.lock"):
        with acquire_lock(output / "queue.lock"):
            reject_live_processes(output)
            plan_path, plan = prepare_plan(args, output)
            receipt_path = output / f"launch.{time.time_ns()}.{uuid.uuid4().hex[:8]}.json"
            command = [plan["python"], "-u", str(Path(__file__).resolve()), "--worker-plan", str(plan_path)]
            receipt = {"status": "launching", "command": command, "plan_sha256": digest(plan_path),
                       "started_unix": time.time(), "log": str(output / "queue.log")}
            write_json(receipt_path, receipt)
        try:
            with (output / "queue.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(command, cwd=SOURCE_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            receipt.update(status="failed", error=str(exc))
            write_json(receipt_path, receipt)
            raise
        receipt.update(status="launched", pid=process.pid, process_group=process.pid,
                       process_start_ticks=process_identity(process.pid))
        write_json(receipt_path, receipt)
    print(json.dumps({**receipt, "receipt": str(receipt_path), "plan": str(plan_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

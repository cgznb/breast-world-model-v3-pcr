"""Launch the nested five-seed, development-only gated pCR pilot."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def progress(cfg, stage, **fields):
    from src.first_post_pcr_data import now, write_json
    row = {"updated_utc": now(), "pid": os.getpid(), "branch": cfg["branch"],
           "gpu": cfg["gpu"], "stage": stage, **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", row)
    print(json.dumps(row), flush=True)


def run_tasks(cfg, planned, stage):
    from src import first_post_optimization as opt
    from src.single_phase_gated_training import run_task, task_complete
    pause = Path(cfg["output_dir"]) / "PAUSE_REQUESTED"
    if pause.exists():
        raise InterruptedError("Pause requested")
    pending = [t for t in planned if not task_complete(opt.task_path(cfg["output_dir"], t), t)]
    completed = initial = len(planned) - len(pending)
    progress(cfg, stage, completed=completed, total=len(planned))
    started, reported = time.perf_counter(), 0.0
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=cfg["jobs"], mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(pending), {}

        def submit():
            task = next(iterator, None)
            if task is not None:
                future = pool.submit(run_task, task, cfg["output_dir"], str(opt.source_directory(cfg, "physical_roi")))
                running[future] = task

        for _ in range(min(cfg["jobs"], len(pending))):
            submit()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                try:
                    future.result()
                except BaseException:
                    for other in running:
                        other.cancel()
                    raise
                completed += 1
                if not pause.exists():
                    submit()
            elapsed = time.perf_counter() - started
            rate = (completed - initial) / max(elapsed, 1e-6)
            if elapsed - reported >= 2 or not running:
                progress(cfg, stage, completed=completed, total=len(planned), fits_per_minute=rate * 60,
                         remaining_seconds=(len(planned) - completed) / rate if rate else None)
                reported = elapsed
    if completed != len(planned):
        raise InterruptedError("Paused after active fits; completed fits are resumable")


def smoke(cfg):
    import numpy as np
    import pandas as pd
    import torch
    from src import first_post_optimization as opt
    from src.first_post_pcr_data import read_json, write_json
    from src.single_phase_gated_models import build_model, predict
    from src.single_phase_gated_training import attach_t0, effective_config, prepare, run_task, task_complete, windows
    from src.single_phase_temporal_models import window_split
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    cfg["default_candidate"].update(epochs=2, patience=2, scheduler_t_max=2)
    folds = prepare(cfg)
    raw = opt.load_development(cfg["output_dir"], str(opt.source_directory(cfg, "physical_roi")))
    missing = {p for p, m in zip(raw["pids"], raw["masks"][:, 0]) if m <= 0}
    fold, inner = next((f, i) for f in folds for i in f["inner_folds"] if missing.issubset(i["train_ids"]))
    specs = [("baseline", 1), ("baseline", 4), ("wide64", 1), ("wide64", 4),
             ("gated64", 2), ("gated64", 3), ("gated64", 4), ("shared64", 4)]
    completed = []
    for stage, part in (("inner", inner), ("outer", {**fold, "inner": -1})):
        for candidate, depth in specs:
            task = {"stage": stage, "source": "physical_roi", "candidate": candidate, "depth": depth,
                "outer": fold["fold"], "inner": part["inner"], "seed": 42,
                "train_ids": part["train_ids"], "val_ids": part["val_ids"],
                "effective": effective_config(cfg, candidate, depth)}
            if stage == "outer":
                task["fixed_epochs"] = 2
            task = attach_t0(task, cfg["output_dir"])
            run_task(task, cfg["output_dir"], str(opt.source_directory(cfg, "physical_roi")))
            completed.append(task)
            progress(cfg, "smoke_fitting", completed=len(completed), total=16)
    maximum = 0.0
    for task in completed:
        path = opt.task_path(cfg["output_dir"], task)
        if not task_complete(path, task):
            raise ValueError("Smoke task did not validate")
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        if checkpoint["clinical_prior_train_count"] - checkpoint["neural_train_count"] != len(missing):
            raise ValueError("No-T0 neural-loss cohort changed")
        model = build_model(task["effective"], task["depth"]).to("cuda:0").eval()
        model.load_state_dict(checkpoint["model_state"])
        recorded = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        for depth in windows(task):
            for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
                split = window_split(raw, ids, task["depth"], task["effective"])
                prior = opt._prior_from_state(split["clinical"], checkpoint["clinical_prior"])
                values = predict(model, opt.tensor_split(split, prior, "cuda:0"), depth)
                expected = recorded[(recorded.depth == depth) & (recorded.role == role)].set_index("patient_id").loc[split["pids"]]
                maximum = max(maximum, float(np.abs(values["probability"] - expected.probability.to_numpy()).max()))
        if task["effective"].get("architecture") == "gated":
            base = torch.load(task["t0_reference"]["path"], map_location="cpu", weights_only=False)
            if any(not torch.equal(checkpoint["model_state"]["t0." + k], v) for k, v in base["model_state"].items()):
                raise ValueError("T0 reference changed during gated fitting")
            if not checkpoint["model_state"]["head.weight"].abs().sum() > 0:
                raise ValueError("Temporal correction head did not update")
    if maximum > 1e-7:
        raise ValueError("Smoke checkpoint prediction replay failed")
    result = {"complete": True, "verified_fits": len(completed), "maximum_prediction_error": maximum,
              "missing_t0_patients": len(missing), "holdout_loaded": False}
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", result)
    progress(cfg, "smoke_complete", **result)


def workflow(cfg):
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_gated_reporting import report_development
    from src.single_phase_gated_training import prepare, select_models, tasks
    progress(cfg, "preparing")
    folds = prepare(cfg)
    run_tasks(cfg, tasks(cfg, folds, "inner"), "inner_capacity_controls")
    run_tasks(cfg, tasks(cfg, folds, "inner", gated=True), "inner_gated_models")
    selection = select_models(cfg, folds)
    run_tasks(cfg, tasks(cfg, folds, "outer", ranking=selection["ranking"]), "outer_capacity_controls")
    run_tasks(cfg, tasks(cfg, folds, "outer", gated=True, ranking=selection["ranking"]), "outer_gated_models")
    progress(cfg, "reporting_development")
    result = report_development(cfg, selection)
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", {"complete": True, "completed_utc": now(),
        "formal_seeds": cfg["formal_seeds"], "selected_model_references": result["selected_model_references"],
        "distinct_outer_models": result["distinct_outer_models"], "holdout_evaluated": False,
        "generated_evaluated": False, "automatic_thirty_seed_extension": False})
    progress(cfg, "complete", models=result["distinct_outer_models"], holdout_evaluated=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_gated_pcr_v4.yaml")
    parser.add_argument("--branch", choices=["registered_dce0", "first_post"], required=True)
    parser.add_argument("--stage", choices=["prepare", "smoke", "run"], default="run")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    master = yaml.safe_load((ROOT / args.config).read_text())
    gpu = master["branches"][args.branch]["gpu"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_gated_training import load_config, prepare
    cfg = load_config(ROOT / args.config, args.branch)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (output / "workflow.lock").open("a")
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This v4 branch already has an active controller")
        if args.detach:
            if args.stage != "run":
                raise SystemExit("Only the formal pilot can detach")
            with (output / "workflow.log").open("a") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", cfg["config_path"],
                    "--branch", args.branch, "--lock-fd", str(lock.fileno())], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(lock.fileno(),), env=os.environ.copy())
            write_json(output / "launch.json", {"pid": process.pid, "created_utc": now(), "gpu": gpu,
                "branch": args.branch, "config": cfg["config_path"], "log": str(output / "workflow.log")})
            print(f"Launched PID {process.pid}; log: {output / 'workflow.log'}", flush=True)
            return

        def request_pause(signum, _frame):
            write_json(output / "PAUSE_REQUESTED", {"signal": signum, "created_utc": now()})

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, request_pause)
        try:
            if (output / "PAUSE_REQUESTED").exists():
                raise InterruptedError("Remove PAUSE_REQUESTED before resuming")
            if args.stage == "prepare":
                prepare(cfg)
                progress(cfg, "prepared")
            elif args.stage == "smoke":
                smoke(cfg)
            else:
                workflow(cfg)
        except BaseException as error:
            progress(cfg, "paused" if isinstance(error, InterruptedError) else "failed",
                     error_type=type(error).__name__, error=str(error)[-1000:])
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

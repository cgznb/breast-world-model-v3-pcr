"""Detached, resumable nested refits for both repeated-single-phase pCR cohorts."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import multiprocessing as mp
import os
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
    row = {"updated_utc": now(), "pid": os.getpid(), "gpu": cfg["gpu"],
           "branch": cfg["branch"], "stage": stage, **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", row)
    print(json.dumps(row), flush=True)


def run_tasks(cfg, tasks, stage):
    from src import first_post_optimization as opt
    from src.single_phase_repeat_retraining import run_task
    remaining = [t for t in tasks if not opt.task_complete(opt.task_path(cfg["output_dir"], t), t)]
    completed = initial = len(tasks) - len(remaining)
    started = time.perf_counter()
    progress(cfg, stage, completed=completed, total=len(tasks))
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=int(cfg["jobs"]), mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(remaining), {}

        def submit_next():
            task = next(iterator, None)
            if task is not None:
                future = pool.submit(run_task, task, cfg["output_dir"], str(opt.source_directory(cfg, task["source"])))
                running[future] = task

        for _ in range(min(len(remaining), int(cfg["jobs"]))):
            submit_next()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                try:
                    future.result()
                except BaseException:
                    for pending in running:
                        pending.cancel()
                    raise
                completed += 1
                if not (Path(cfg["output_dir"]).parent / "PAUSE_REQUESTED").exists():
                    submit_next()
            elapsed = time.perf_counter() - started
            rate = (completed - initial) / max(elapsed, 1e-6)
            progress(cfg, stage, completed=completed, total=len(tasks), fits_per_minute=rate * 60,
                     remaining_seconds=(len(tasks) - completed) / rate if rate else None)
        if completed != len(tasks):
            raise InterruptedError("Pause requested; completed fits can be resumed")


def smoke(cfg):
    import numpy as np
    import pandas as pd
    import torch
    from src import first_post_optimization as opt
    from src.first_post_pcr_data import read_json, write_json
    from src.single_phase_repeat_retraining import prepare
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    cfg["jobs"] = 2
    folds = prepare(cfg)
    raw = opt.load_development(cfg["output_dir"], str(opt.source_directory(cfg, "physical_roi")))
    missing = {p for p, m in zip(raw["pids"], raw["masks"][:, 0]) if m <= 0}
    candidates = opt.inner_tasks(cfg, folds, cfg["feature_sources"])
    tasks = []
    specs = [("compact16", d) for d in cfg["depths"]]
    specs += [(c, 4) for c in cfg["candidates"] if c != "compact16"]
    for candidate, depth in specs:
        task = copy.deepcopy(next(t for t in candidates if t["candidate"] == candidate and t["depth"] == depth
                                  and missing.issubset(t["train_ids"])))
        task["effective"].update(epochs=2, patience=2, scheduler_t_max=2)
        tasks.append(task)
    run_tasks(cfg, tasks, "smoke_inner")
    outer_tasks = []
    for task in tasks:
        refit = copy.deepcopy(task)
        outer = next(f for f in folds if f["fold"] == task["outer"])
        refit.update(stage="outer", inner=-1, fixed_epochs=2,
                     train_ids=outer["train_ids"], val_ids=outer["val_ids"])
        outer_tasks.append(refit)
    run_tasks(cfg, outer_tasks, "smoke_outer")
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    maximum_error = 0.0
    for task in tasks + outer_tasks:
        path = opt.task_path(cfg["output_dir"], task)
        if not opt.task_complete(path, task):
            raise ValueError("Smoke artifact validation failed")
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        model = opt.TDN({"downstream": task["effective"]}).to("cuda:0")
        model.load_state_dict(checkpoint["model_state"])
        saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            split = opt.select_patients(raw, ids, task["depth"])
            prior = opt._prior_from_state(split["clinical"], checkpoint["clinical_prior"])
            probability, _ = opt.predict(model, opt.tensor_split(split, prior, "cuda:0"))
            expected = saved[saved.role == role].set_index("patient_id").loc[split["pids"]].probability.to_numpy()
            error = float(np.max(np.abs(probability - expected)))
            maximum_error = max(maximum_error, error)
            if error > 1e-7:
                raise ValueError("Smoke prediction replay failed")
        summary = read_json(path / "COMPLETE.json")
        if summary["clinical_prior_train_count"] - summary["neural_train_count"] != len(missing):
            raise ValueError("Smoke missing-T0 loss policy failed")
    result = {"complete": True, "verified_fits": len(tasks + outer_tasks), "max_prediction_error": maximum_error,
              "missing_t0_train_patients": len(missing)}
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", result)
    progress(cfg, "smoke_complete", **result)


def workflow(cfg):
    from src import first_post_optimization as opt
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_repeat_retraining import evaluate_real, freeze_models, prepare, report
    progress(cfg, "preparing")
    folds = prepare(cfg)
    run_tasks(cfg, opt.inner_tasks(cfg, folds, cfg["feature_sources"]), "inner_search")
    selected = opt.select_models(cfg, folds, cfg["feature_sources"], "regularized")
    run_tasks(cfg, opt.formal_tasks(cfg, folds, selected), "outer_refits")
    progress(cfg, "reporting_development")
    report(cfg)
    refs = freeze_models(cfg)
    write_json(Path(cfg["output_dir"]) / "TRAINING_COMPLETE.json",
               {"complete": True, "models": len(refs), "completed_utc": now(), "holdout_evaluated": False})
    progress(cfg, "evaluating_frozen_real_holdout", models=len(refs))
    evaluate_real(cfg, refs)
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json",
               {"complete": True, "models": len(refs), "completed_utc": now(),
                "holdout_evaluated": True, "generated_evaluated": False})
    progress(cfg, "complete", models=len(refs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_repeat_pcr_v2.yaml")
    parser.add_argument("--branch", choices=["both", "first_post", "registered_dce0"], default="both")
    parser.add_argument("--stage", choices=["prepare", "smoke", "run"], default="run")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    master = yaml.safe_load(config_path.read_text())
    output = ROOT / master["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(master["gpu"]), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_repeat_retraining import load_config, prepare
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (output / "workflow.lock").open("a")
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A controller already holds this retraining lock")
        if args.detach:
            if args.stage != "run":
                raise SystemExit("Only the formal workflow can detach")
            with (output / "workflow.log").open("a") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                    "--branch", args.branch, "--stage", "run", "--lock-fd", str(lock.fileno())],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(lock.fileno(),), env=os.environ.copy())
            write_json(output / "launch.json", {"pid": process.pid, "created_utc": now(), "gpu": master["gpu"],
                       "config": str(config_path), "branches": args.branch, "log": str(output / "workflow.log")})
            print(f"Launched PID {process.pid}; log: {output / 'workflow.log'}", flush=True)
            return
        if (output / "PAUSE_REQUESTED").exists():
            raise SystemExit("PAUSE_REQUESTED exists; remove the request before resuming")
        branches = list(master["branches"]) if args.branch == "both" else [args.branch]
        for branch in branches:
            cfg = load_config(config_path, branch)
            try:
                if args.stage == "prepare":
                    prepare(cfg)
                    progress(cfg, "prepared")
                elif args.stage == "smoke":
                    smoke(cfg)
                else:
                    workflow(cfg)
            except BaseException as error:
                progress(cfg, "paused" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                         error_type=type(error).__name__, error=str(error)[-1000:])
                traceback.print_exc()
                raise
        if args.stage == "run" and args.branch == "both":
            write_json(output / "COMPLETE.json", {"complete": True, "branches": branches, "completed_utc": now()})


if __name__ == "__main__":
    main()

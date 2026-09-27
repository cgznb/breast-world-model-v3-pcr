"""Run one isolated thirty-seed temporal pCR branch with an exclusive controller."""

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
    from src.single_phase_temporal_training import run_task
    pending = [t for t in tasks if not opt.task_complete(opt.task_path(cfg["output_dir"], t), t)]
    completed = initial = len(tasks) - len(pending)
    started = time.perf_counter()
    progress(cfg, stage, completed=completed, total=len(tasks))
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=int(cfg["jobs"]), mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(pending), {}

        def submit_next():
            task = next(iterator, None)
            if task is not None:
                future = pool.submit(run_task, task, cfg["output_dir"], str(opt.source_directory(cfg, task["source"])))
                running[future] = task

        for _ in range(min(len(pending), cfg["jobs"])):
            submit_next()
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
                if not (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
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
    from src.first_post_pcr_data import write_json
    from src.single_phase_temporal_models import build_model, window_split
    from src.single_phase_temporal_training import prepare
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    cfg["jobs"] = 2
    folds = prepare(cfg)
    raw = opt.load_development(cfg["output_dir"], str(opt.source_directory(cfg, "physical_roi")))
    missing = {p for p, m in zip(raw["pids"], raw["masks"][:, 0]) if m <= 0}
    candidates = opt.inner_tasks(cfg, folds, cfg["feature_sources"])
    specs = [("delta32", d) for d in cfg["depths"]]
    specs += [(c, 4) for c in cfg["candidates"] if c != "delta32"]
    tasks = []
    for candidate, depth in specs:
        task = copy.deepcopy(next(t for t in candidates if t["candidate"] == candidate and t["depth"] == depth
                                  and missing.issubset(t["train_ids"])))
        task["effective"].update(epochs=2, patience=2, scheduler_t_max=2)
        tasks.append(task)
    run_tasks(cfg, tasks, "smoke_inner")
    refits = []
    for task in tasks:
        refit = copy.deepcopy(task)
        outer = next(f for f in folds if f["fold"] == task["outer"])
        refit.update(stage="outer", inner=-1, fixed_epochs=2,
                     train_ids=outer["train_ids"], val_ids=outer["val_ids"])
        refits.append(refit)
    run_tasks(cfg, refits, "smoke_outer")
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    maximum = 0.0
    for task in tasks + refits:
        path = opt.task_path(cfg["output_dir"], task)
        if not opt.task_complete(path, task):
            raise ValueError("GPU smoke artifact validation failed")
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        model = build_model(task["effective"], task["depth"]).to("cuda:0")
        model.load_state_dict(checkpoint["model_state"])
        saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            split = window_split(raw, ids, task["depth"], task["effective"])
            prior = opt._prior_from_state(split["clinical"], checkpoint["clinical_prior"])
            probability, _ = opt.predict(model, opt.tensor_split(split, prior, "cuda:0"))
            expected = saved[saved.role == role].set_index("patient_id").loc[split["pids"]].probability.to_numpy()
            maximum = max(maximum, float(np.abs(probability - expected).max()))
        if checkpoint["clinical_prior_train_count"] - checkpoint["neural_train_count"] != len(missing):
            raise ValueError("No-T0 neural loss exclusion changed")
    if maximum > 1e-7:
        raise ValueError("GPU checkpoint/transform prediction replay failed")
    result = {"complete": True, "verified_fits": len(tasks + refits),
              "maximum_prediction_error": maximum, "missing_t0_patients": len(missing)}
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", result)
    progress(cfg, "smoke_complete", **result)


def workflow(cfg):
    from src import first_post_optimization as opt
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_temporal_reporting import diagnose, evaluate_real, freeze_models, report_development, write_report
    from src.single_phase_temporal_training import prepare
    progress(cfg, "preparing")
    folds = prepare(cfg)
    diagnose(cfg)
    run_tasks(cfg, opt.inner_tasks(cfg, folds, cfg["feature_sources"]), "inner_search")
    selection = opt.select_models(cfg, folds, cfg["feature_sources"], "temporal")
    run_tasks(cfg, opt.formal_tasks(cfg, folds, selection), "outer_refits")
    progress(cfg, "reporting_development")
    report_development(cfg)
    references = freeze_models(cfg)
    write_json(Path(cfg["output_dir"]) / "TRAINING_COMPLETE.json", {"complete": True,
               "models": len(references), "completed_utc": now(), "holdout_evaluated": False})
    progress(cfg, "evaluating_frozen_real_holdout", models=len(references))
    evaluate_real(cfg, references)
    write_report(cfg)
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", {"complete": True,
        "models": len(references), "formal_seeds": cfg["formal_seeds"], "completed_utc": now(),
        "holdout_evaluated": True, "generated_evaluated": False})
    progress(cfg, "complete", models=len(references))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_temporal_pcr_v3.yaml")
    parser.add_argument("--branch", required=True, choices=["registered_dce0", "first_post"])
    parser.add_argument("--stage", default="run", choices=["prepare", "diagnose", "smoke", "run"])
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    master = yaml.safe_load(config_path.read_text())
    gpu = master["branches"][args.branch]["gpu"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_temporal_training import load_config, prepare
    cfg = load_config(config_path, args.branch)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (output / "workflow.lock").open("a")
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This branch already has an active controller")
        if args.detach:
            if args.stage != "run":
                raise SystemExit("Only formal training can detach")
            with (output / "workflow.log").open("a") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                    "--branch", args.branch, "--lock-fd", str(lock.fileno())], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                    pass_fds=(lock.fileno(),), env=os.environ.copy())
            write_json(output / "launch.json", {"pid": process.pid, "created_utc": now(), "gpu": gpu,
                       "config": str(config_path), "branch": args.branch, "log": str(output / "workflow.log")})
            print(f"Launched PID {process.pid}; log: {output / 'workflow.log'}", flush=True)
            return
        try:
            if (output / "PAUSE_REQUESTED").exists():
                raise InterruptedError("Remove PAUSE_REQUESTED before resuming")
            if args.stage == "prepare":
                prepare(cfg)
                progress(cfg, "prepared")
            elif args.stage == "diagnose":
                from src.single_phase_temporal_reporting import diagnose
                print(json.dumps(diagnose(cfg)), flush=True)
            elif args.stage == "smoke":
                smoke(cfg)
            else:
                workflow(cfg)
        except BaseException as error:
            progress(cfg, "paused" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error_type=type(error).__name__, error=str(error)[-1000:])
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

"""Resume-safe GPU1 optimization: cached heads, ROI extraction, joint nested selection."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.first_post_optimization import (
    formal_tasks, inner_tasks, load_study, prepare_study, run_task,
    select_models, source_directory, task_complete, task_path,
)
from src.first_post_pcr_data import now, read_json, write_json


def progress(cfg, stage, **fields):
    row = {"updated_utc": now(), "pid": os.getpid(), "gpu": cfg["gpu"], "stage": stage, **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", row)
    print(__import__("json").dumps(row), flush=True)


def run_tasks(cfg, tasks, stage):
    remaining = [t for t in tasks if not task_complete(task_path(cfg["output_dir"], t), t)]
    initial = len(tasks) - len(remaining)
    completed, started = initial, time.perf_counter()
    progress(cfg, stage, completed=completed, total=len(tasks))
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=int(cfg["jobs"]), mp_context=mp.get_context("spawn")) as pool:
        iterator = iter(remaining)
        running = {}

        def submit_next():
            task = next(iterator, None)
            if task is None:
                return
            future = pool.submit(run_task, task, cfg["output_dir"], str(source_directory(cfg, task["source"])))
            running[future] = task

        for _ in range(min(len(remaining), int(cfg["jobs"]) * 2)):
            submit_next()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                try:
                    future.result()
                except Exception:
                    for pending in running:
                        pending.cancel()
                    raise
                completed += 1
                submit_next()
            elapsed = time.perf_counter() - started
            rate = (completed - initial) / max(elapsed, 1e-6)
            progress(cfg, stage, completed=completed, total=len(tasks),
                     fits_per_minute=rate * 60,
                     remaining_seconds=(len(tasks) - completed) / rate if rate else None)
            if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                for pending in running:
                    pending.cancel()
                raise InterruptedError("Pause requested; completed fits can be resumed")


def report_stage(cfg, name):
    from scripts.report_first_post_optimization import report
    progress(cfg, f"reporting_{name}")
    report(cfg, name)
    write_json(Path(cfg["output_dir"]) / f"{name.upper()}_COMPLETE.json",
               {"complete": True, "completed_utc": now(), "holdout_evaluated": False})


def smoke(cfg):
    import copy
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    smoke_cfg["feature_sources"] = ["physical_roi"]
    smoke_cfg["jobs"] = 2
    folds = prepare_study(smoke_cfg)
    all_tasks = inner_tasks(smoke_cfg, folds, ["physical_roi"])
    tasks = []
    for depth in cfg["depths"]:
        task = copy.deepcopy(next(t for t in all_tasks if t["depth"] == depth and t["candidate"] == "compact32"))
        task["effective"].update(epochs=2, patience=2, scheduler_t_max=2)
        tasks.append(task)
    run_tasks(smoke_cfg, tasks, "smoke_inner")
    refits = []
    for task in tasks:
        refit = copy.deepcopy(task)
        refit.update(stage="outer", inner=-1, fixed_epochs=2)
        refits.append(refit)
    run_tasks(smoke_cfg, refits, "smoke_fixed_epoch")
    for task in tasks + refits:
        if not task_complete(task_path(smoke_cfg["output_dir"], task), task, verify_weights=True):
            raise ValueError("GPU smoke completion replay failed")
    from src.first_post_roi_optimization import extract_resized
    extract_resized(smoke_cfg, lambda stage, **fields: progress(smoke_cfg, stage, **fields), limit=2)
    progress(smoke_cfg, "smoke_complete", verified_fits=len(tasks) + len(refits), verified_roi_visits=2)


def workflow(cfg):
    folds = prepare_study(cfg)
    run_tasks(cfg, inner_tasks(cfg, folds, ["physical_roi"]), "physical_inner_search")
    selected = select_models(cfg, folds, ["physical_roi"], "physical_roi")
    run_tasks(cfg, formal_tasks(cfg, folds, selected), "physical_outer_refits")
    report_stage(cfg, "physical_roi")
    if "resized_roi" in cfg["feature_sources"]:
        from src.first_post_roi_optimization import extract_resized
        extract_resized(cfg, lambda stage, **fields: progress(cfg, stage, **fields))
        prepare_study(cfg)
        run_tasks(cfg, inner_tasks(cfg, folds, ["resized_roi"]), "resized_inner_search")
        selected = select_models(cfg, folds, cfg["feature_sources"], "joint_geometry")
        run_tasks(cfg, formal_tasks(cfg, folds, selected), "joint_outer_refits")
        report_stage(cfg, "joint_geometry")
    progress(cfg, "complete", holdout_evaluated=False)
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", {"complete": True, "completed_utc": now(),
               "holdout_evaluated": False, "reports": cfg["feature_sources"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/first_post_pcr_optimization_v2.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--detach", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_study(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(cfg["gpu"]), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if args.detach:
        config_path = str(Path(args.config).resolve())
        with (output / "workflow.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", config_path, "--run"],
                                       cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
        write_json(output / "launch.json", {"pid": process.pid, "created_utc": now(), "gpu": cfg["gpu"],
                   "config": config_path, "log": str(output / "workflow.log")})
        print(f"Launched PID {process.pid}; log: {output / 'workflow.log'}", flush=True)
        return
    lock_path = output / ("smoke.lock" if args.smoke else "workflow.lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An optimization controller already holds the run lock")
        try:
            if args.smoke:
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

"""Prepare, verify and train one repeated single-phase Pillar pCR branch."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def train_registered(cfg):
    from scripts.run_first_post_pcr import fold_task, training_tasks
    from src.single_phase_repeat_pcr import progress, recipe_configs
    from src.first_post_pcr_data import read_json, write_json, now
    output = Path(cfg["output_dir"])
    configs = recipe_configs(cfg)
    folds = read_json(output / "folds.json")
    task_cfg = {**cfg, "recipes": {name: {"max_tp": i + 1}
                                  for i, name in enumerate(("T0", "T0-T1", "T0-T2", "T0-T3"))}}
    tasks = training_tasks(task_cfg, configs, folds)
    records, started = [], time.perf_counter()
    progress(cfg, "training_pcr", completed=0, total=len(tasks), jobs=cfg["training_jobs"])
    with ProcessPoolExecutor(max_workers=cfg["training_jobs"], mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(tasks), set()

        def submit():
            task = next(iterator, None)
            if task is not None:
                running.add(pool.submit(fold_task, task))

        for _ in range(cfg["training_jobs"]):
            submit()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                running.remove(future)
                records.append(future.result())
                submit()
            elapsed = time.perf_counter() - started
            progress(cfg, "training_pcr", completed=len(records), total=len(tasks), jobs=cfg["training_jobs"],
                     remaining_seconds=elapsed / len(records) * (len(tasks) - len(records)) if records else None)
            if (output / "PAUSE_REQUESTED").exists():
                for future in running:
                    future.cancel()
                raise InterruptedError("Pause requested; active fits will drain")
    write_json(output / "TRAINING_COMPLETE.json", {"complete": True, "fits": len(tasks),
               "results": records, "holdout_used_for_fitting": False, "completed_utc": now()})


def workflow(cfg):
    from src.single_phase_repeat_pcr import (
        evaluate_real, extract, model_references, optimization_config, prepare, progress, smoke,
    )
    from src.first_post_pcr_data import now, write_json
    cohort = prepare(cfg)
    smoke(cfg, cohort)
    extract(cfg, cohort, "train")
    if cfg["branch"] == "registered_dce0":
        train_registered(cfg)
    else:
        from scripts.run_first_post_optimization import workflow as optimize
        from scripts.run_full978_anti_overfit import _atomic_text
        import yaml
        study = optimization_config(cfg)
        _atomic_text(Path(cfg["output_dir"]) / "configs/optimization.yaml", yaml.safe_dump(study, sort_keys=False))
        progress(cfg, "training_nested_pcr", progress_file=str(Path(study["output_dir"]) / "progress.json"))
        optimize(study)
        write_json(Path(cfg["output_dir"]) / "TRAINING_COMPLETE.json", {"complete": True,
                   "selection": "inner_validation_only", "holdout_used_for_fitting": False, "completed_utc": now()})
    refs = model_references(cfg)
    extract(cfg, cohort, "val")
    results = evaluate_real(cfg, cohort, refs)
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", {"complete": True, "completed_utc": now(),
               "branch": cfg["branch"], "input_mode": cfg["input_mode"], "real_holdout_results": results,
               "generated_evaluated": False})
    progress(cfg, "complete", real_holdout_results=results, generated_evaluated=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_repeat_pcr_v1.yaml")
    parser.add_argument("--branch", required=True, choices=("registered_dce0", "first_post"))
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "run", "detach"):
        modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    import yaml
    config_path = (ROOT / args.config).resolve()
    master = yaml.safe_load(config_path.read_text())
    gpu = int(master["branches"][args.branch]["gpu"])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    from src.single_phase_repeat_pcr import load_config, prepare, progress, smoke
    from src.first_post_pcr_data import now, write_json
    cfg = load_config(config_path, args.branch)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (output / "workflow.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                                        "--branch", args.branch, "--run"], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, env=os.environ.copy())
        write_json(output / "launch.json", {"pid": process.pid, "gpu": gpu, "created_utc": now(),
                   "config": str(config_path), "branch": args.branch, "log": str(output / "workflow.log")})
        print(f"Launched {args.branch}: PID {process.pid}, GPU {gpu}, {output / 'workflow.log'}")
        return
    with (output / "workflow.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This branch already has an active controller")
        try:
            if args.prepare:
                prepare(cfg)
                return
            used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                       "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
            if used > 512:
                raise RuntimeError(f"GPU {gpu} is occupied ({used} MiB); no new work launched")
            if args.smoke:
                smoke(cfg, prepare(cfg))
            else:
                workflow(cfg)
        except BaseException as error:
            progress(cfg, "paused" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error_type=type(error).__name__, error=str(error)[-1500:])
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

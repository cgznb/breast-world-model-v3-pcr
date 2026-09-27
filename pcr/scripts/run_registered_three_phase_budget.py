"""Train the declared V1/V4 300/50 experiment, then evaluate frozen models."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import first_post_optimization as opt
from src import registered_three_phase_budget as study
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import now, read_json, write_json


def train(cfg, tasks):
    remaining = [t for t in tasks if not study.task_complete(study.task_path(cfg["output_dir"], t), t)]
    completed = initial = len(tasks) - len(remaining)
    started, last = time.perf_counter(), 0.0
    shared.progress(cfg, "training", completed=completed, total=len(tasks))
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=cfg["jobs"], mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(remaining), {}

        def submit():
            task = next(iterator, None)
            if task is not None:
                future = pool.submit(study.run_task, task, cfg["output_dir"],
                                     str(opt.source_directory(cfg, "physical_roi")), cfg["device"])
                running[future] = task

        for _ in range(min(cfg["jobs"], len(remaining))):
            submit()
        while running:
            done, _ = wait(running, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                future.result()
                completed += 1
                submit()
            elapsed = time.perf_counter() - started
            if elapsed - last >= 15 or not running:
                last = elapsed
                rate = (completed - initial) / max(elapsed, 1e-6)
                shared.progress(cfg, "training", completed=completed, total=len(tasks),
                                remaining_seconds=(len(tasks) - completed) / rate if rate else None)


def workflow(cfg, training_only=False):
    output = Path(cfg["output_dir"])
    _, tasks, refs = study.prepare(cfg)
    if (output / "COMPLETE.json").exists():
        shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
        shared.verify_inventory(read_json(output / "COMPLETE.json")["result_inventory"])
        print("Completed experiment verified; no training or inference repeated.", flush=True)
        return
    if not (output / "TRAINING_COMPLETE.json").exists():
        train(cfg, tasks)
        audit = study.verify_training(cfg, tasks)
        study.freeze_models(cfg, tasks)
        shared.verify_inventory(read_json(output / "input_inventory.json"))
        write_json(output / "TRAINING_COMPLETE.json", {
            "completed_utc": now(), "models": len(tasks), "holdout_evaluated": False, "audit": audit})
    else:
        shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
    if training_only:
        shared.progress(cfg, "training_complete", models=len(tasks))
        return
    from src.registered_three_phase_budget_report import evaluate_and_report
    audit = evaluate_and_report(cfg, refs)
    shared.verify_inventory(read_json(output / "input_inventory.json"))
    shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
    write_json(output / "COMPLETE.json", {"passed": True, "completed_utc": now(),
               "schema": study.SCHEMA, **audit})
    shared.progress(cfg, "complete", models=len(tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_pcr_300_50.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    cfg = study.load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.smoke:
                print(study.smoke(cfg), flush=True)
            else:
                workflow(cfg, args.training_only)
        except BaseException as error:
            shared.progress(cfg, "failed", error_type=type(error).__name__, error=str(error)[-500:])
            raise


if __name__ == "__main__":
    main()

"""Run and resume the fixed four-window ROI32 pCR improvement protocol."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import first_post_optimization as opt
from src import registered_three_phase_optimization as study
from src.first_post_pcr_data import now, read_json, write_json


def run_tasks(cfg, tasks, stage):
    remaining = [t for t in tasks if not opt.task_complete(opt.task_path(cfg["output_dir"], t), t)]
    completed, initial = len(tasks) - len(remaining), len(tasks) - len(remaining)
    started = time.perf_counter()
    study.progress(cfg, stage, completed=completed, total=len(tasks))
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=cfg["jobs"], mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(remaining), {}

        def submit():
            task = next(iterator, None)
            if task is not None:
                future = pool.submit(study.run_task, task, cfg["output_dir"],
                                     str(opt.source_directory(cfg, task["source"])), cfg["device"])
                running[future] = task

        for _ in range(min(cfg["jobs"], len(remaining))):
            submit()
        last = 0.0
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
                study.progress(cfg, stage, completed=completed, total=len(tasks),
                               fits_per_minute=60 * rate,
                               remaining_seconds=(len(tasks) - completed) / rate if rate else None)


def workflow(cfg):
    output = Path(cfg["output_dir"])
    folds = study.prepare(cfg)
    if (output / "COMPLETE.json").exists():
        study.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
        print("Completed study verified; no training or evaluation repeated.", flush=True)
        return
    if (output / "FROZEN_MODELS.json").exists():
        import torch
        frozen = read_json(output / "FROZEN_MODELS.json")
        study.verify_inventory(frozen["artifacts"])
        tasks = [torch.load(p, map_location="cpu", weights_only=False)["task"]
                 for p in frozen["artifacts"] if p.endswith("/model.pt")]
    else:
        run_tasks(cfg, opt.inner_tasks(cfg, folds, cfg["feature_sources"]), "inner_selection")
        selection = opt.select_models(cfg, folds, cfg["feature_sources"], study.STUDY)
        # References are frozen before any outer fit is scored.
        run_tasks(cfg, opt.formal_tasks(cfg, folds, selection), "outer_fixed_epoch_refits")
        tasks = study.freeze_models(cfg, folds, selection)
    study.progress(cfg, "saved_weight_verification", models=len(tasks))
    audit = study.verify_fitted_predictions(cfg, tasks)
    write_json(output / "TRAINING_COMPLETE.json", {
        "completed_utc": now(), "models": len(tasks), "holdout_evaluated": False, "audit": audit,
    })
    from src.registered_three_phase_optimization_report import evaluate_and_report
    study.progress(cfg, "frozen_evaluation")
    evaluate_and_report(cfg)
    study.verify_inventory(read_json(output / "original_inventory.json"))
    study.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
    write_json(output / "COMPLETE.json", {"completed_utc": now(), "schema": study.SCHEMA,
               "passed": True, "training_verification": audit, "original_inputs_unchanged": True})
    study.progress(cfg, "complete", models=len(tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_pcr_optimization_v2.yaml")
    parser.add_argument("--smoke", choices=["cpu", "cuda:0"])
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    cfg = study.load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ("smoke.lock" if args.smoke else "workflow.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This study already has an active controller")
        if args.smoke:
            print(study.smoke(cfg, args.smoke), flush=True)
        else:
            try:
                workflow(cfg)
            except BaseException as error:
                study.progress(cfg, "failed", error_type=type(error).__name__, error=str(error)[-500:])
                raise


if __name__ == "__main__":
    main()

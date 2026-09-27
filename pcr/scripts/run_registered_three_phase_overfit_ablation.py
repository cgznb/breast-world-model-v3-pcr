"""Run a development-only, single-factor T3 ablation without touching old fits."""

from __future__ import annotations

import argparse
import copy
import fcntl
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_registered_three_phase_optimization import run_tasks
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as shared
from src import registered_three_phase_overfit_ablation as study
from src.first_post_pcr_data import now, read_json, write_json


def workflow(cfg, smoke=False):
    cfg = copy.deepcopy(cfg)
    if smoke:
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    output = Path(cfg["output_dir"])
    if (output / "COMPLETE.json").exists():
        shared.verify_inventory(read_json(output / "input_inventory.json"))
        shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
        print("Completed ablation verified; no fitting or inference repeated.", flush=True)
        return
    folds, tasks, refs = study.prepare(cfg)
    if smoke:
        tasks = [t for t in tasks if t["outer"] == 0 and t["seed"] == 42]
        checked = []
        for task in tasks:
            task = copy.deepcopy(task)
            task["fixed_epochs"] = 2
            task["effective"].update(epochs=2, scheduler_t_max=2)
            opt.run_task(task, output, opt.source_directory(cfg, "physical_roi"), "cpu")
            if not opt.run_task(task, output, opt.source_directory(cfg, "physical_roi"), "cpu")["skipped"]:
                raise ValueError("Completed smoke fit did not resume as a no-op")
            checked.append(task)
        audit = shared.verify_fitted_predictions(cfg, checked)
        write_json(output / "SMOKE_COMPLETE.json", {"passed": True, "completed_utc": now(), **audit})
        print(audit, flush=True)
        return
    run_tasks(cfg, tasks, "development_single_factor_fitting")
    shared.progress(cfg, "saved_weight_replay_and_development_report", models=len(refs))
    verification = study.report(cfg, folds, refs)
    write_json(output / "COMPLETE.json", verification)
    shared.progress(cfg, "complete", models=len(refs), new_fits=len(tasks), holdout_evaluated=False)
    print(output / "README.md", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_overfit_ablation_v4.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = study.load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ("smoke.lock" if args.smoke else "workflow.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This ablation already has an active controller")
        workflow(cfg, args.smoke)


if __name__ == "__main__":
    main()

"""Run a uniform-configuration five-fold study through all four input sources."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_registered_three_phase_optimization import run_tasks
from src import registered_three_phase_fixed_pcr as fixed
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import now, read_json, write_json


def workflow(cfg):
    output = Path(cfg["output_dir"])
    folds, tasks, refs = fixed.prepare(cfg)
    if (output / "COMPLETE.json").exists():
        shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
        shared.verify_inventory(read_json(output / "evaluation_input_inventory.json"))
        print("Completed fixed-configuration study verified; no fitting or inference repeated.", flush=True)
        return
    run_tasks(cfg, tasks, "uniform_fixed_epoch_training")
    fixed.freeze_models(cfg, folds, tasks, refs)
    shared.progress(cfg, "uniformity_and_saved_weight_verification", models=len(tasks))
    audit = shared.verify_fitted_predictions(cfg, tasks)
    write_json(output / "TRAINING_COMPLETE.json", {
        "completed_utc": now(), "models": len(tasks), "audit": audit,
        "all_folds_same_config": True, "all_fits_fixed_epochs": cfg["common_training"]["epochs"],
        "holdout_evaluated": False,
    })
    from src.registered_three_phase_fixed_report import evaluate_and_report
    shared.progress(cfg, "paired_real_generated_evaluation")
    verification = evaluate_and_report(cfg)
    shared.verify_inventory(read_json(output / "prior_inventory.json"))
    shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
    write_json(output / "COMPLETE.json", {"schema": fixed.SCHEMA, "passed": True,
               "completed_utc": now(), "training_audit": audit, "evaluation_audit": verification,
               "same_config_across_folds": True, "original_artifacts_preserved": True})
    shared.progress(cfg, "complete", models=len(tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_fixed_pcr_v3.yaml")
    parser.add_argument("--smoke", choices=["cpu", "cuda:0"])
    args = parser.parse_args()
    cfg = fixed.load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ("smoke.lock" if args.smoke else "workflow.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This fixed-configuration study already has an active controller")
        try:
            if args.smoke:
                print(fixed.smoke(cfg, args.smoke), flush=True)
            else:
                workflow(cfg)
        except BaseException as error:
            shared.progress(cfg, "failed", error_type=type(error).__name__, error=str(error)[-500:])
            raise


if __name__ == "__main__":
    main()

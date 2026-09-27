"""Run the explicitly requested 30-seed v4 extension in a separate output."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def workflow(cfg):
    from scripts.run_single_phase_gated_pcr import progress, run_tasks
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_gated_extension import additional_tasks, prepare_extension, report_extension, verify_parent_preserved
    progress(cfg, "preparing_extension")
    folds, selection = prepare_extension(cfg)
    for gated, stage in ((False, "additional_outer_capacity_controls"), (True, "additional_outer_gated_models")):
        run_tasks(cfg, additional_tasks(cfg, folds, selection, gated=gated), stage)
    progress(cfg, "reporting_thirty_seeds")
    result = report_extension(cfg, selection)
    verify_parent_preserved(cfg)
    write_json(Path(cfg["output_dir"]) / "COMPLETE.json", {"complete": True, "completed_utc": now(),
        "formal_seeds": cfg["formal_seeds"], "initial_seeds": cfg["initial_seeds"],
        "additional_seeds": cfg["additional_seeds"], "parent_preserved": True,
        "selected_model_references": result["selected_model_references"],
        "distinct_outer_models": result["distinct_outer_models"], "holdout_evaluated": False,
        "generated_evaluated": False, "selection_reused_without_retuning": True,
        "user_requested_thirty_seeds": True})
    progress(cfg, "complete", models=result["distinct_outer_models"], holdout_evaluated=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_gated_pcr_v4_30seeds.yaml")
    parser.add_argument("--branch", choices=["registered_dce0", "first_post"], required=True)
    parser.add_argument("--stage", choices=["prepare", "run"], default="run")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    master = yaml.safe_load((ROOT / args.config).read_text())
    parent = yaml.safe_load((ROOT / master["parent_config"]).read_text())
    gpu = parent["branches"][args.branch]["gpu"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from scripts.run_single_phase_gated_pcr import progress
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_gated_extension import load_config, prepare_extension
    cfg = load_config(ROOT / args.config, args.branch)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (output / "workflow.lock").open("a")
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This v4 extension branch already has an active controller")
        if args.detach:
            if args.stage != "run":
                raise SystemExit("Only formal extension training can detach")
            with (output / "workflow.log").open("a") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", cfg["config_path"],
                    "--branch", args.branch, "--lock-fd", str(lock.fileno())], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(lock.fileno(),), env=os.environ.copy())
            write_json(output / "launch.json", {"pid": process.pid, "created_utc": now(), "gpu": gpu,
                "branch": args.branch, "config": cfg["config_path"], "log": str(output / "workflow.log"),
                "initial_seeds": cfg["initial_seeds"], "additional_seeds": cfg["additional_seeds"]})
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
                prepare_extension(cfg)
                progress(cfg, "extension_prepared", initial_seeds=cfg["initial_seeds"], additional_seeds=cfg["additional_seeds"])
            else:
                workflow(cfg)
        except BaseException as error:
            progress(cfg, "paused" if isinstance(error, InterruptedError) else "failed",
                     error_type=type(error).__name__, error=str(error)[-1000:])
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

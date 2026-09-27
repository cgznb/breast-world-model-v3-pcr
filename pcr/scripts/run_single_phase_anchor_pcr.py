"""Run the clinical-anchored pCR repair study without disturbing existing jobs."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_tasks(cfg, tasks, stage):
    from src.single_phase_anchor_training import check_pause, run_task, task_complete
    from src.single_phase_anchor_workflow import progress
    pending = [task for task in tasks if not task_complete(cfg, task)]
    complete = len(tasks) - len(pending)
    progress(cfg, stage, completed=complete, total=len(tasks))
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=cfg["training_jobs"], mp_context=mp.get_context("spawn")) as pool:
        queue, running = iter(pending), set()

        def submit():
            check_pause(cfg)
            task = next(queue, None)
            if task is not None:
                running.add(pool.submit(run_task, cfg, task))

        for _ in range(min(cfg["training_jobs"], len(pending))):
            submit()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                running.remove(future)
                future.result()
                complete += 1
                submit()
            progress(cfg, stage, completed=complete, total=len(tasks))


def workflow(configs, mode):
    from src.first_post_pcr_data import identity, now, write_json
    from src.single_phase_anchor_workflow import (
        collect_oof, evaluate_holdout, formal_tasks, freeze, inner_tasks, prepare,
        progress, report, select_inner, smoke,
    )
    from src.single_phase_fmbcmri_data import unchanged_json
    files = [Path(__file__), Path(configs[0]["config_path"]),
             *sorted((ROOT / "src").glob("single_phase_anchor_*.py")),
             *[ROOT / "src" / name for name in (
                 "single_phase_response_data.py", "single_phase_response_models.py",
                 "single_phase_response_training.py", "single_phase_response_reporting.py",
                 "single_phase_temporal_models.py", "single_phase_fmbcmri_training.py",
                 "first_post_pcr_data.py", "data.py")]]
    unchanged_json(Path(configs[0]["study_dir"]) / "runtime_sources.json",
                   {str(path.relative_to(ROOT)): identity(path) for path in files})
    studies = [prepare(cfg) for cfg in configs]
    if mode == "prepare":
        return
    for cfg in configs:
        smoke(cfg)
    if mode == "smoke":
        return
    for cfg in configs:
        names = [name for name, arm in cfg["arms"].items() if "base" not in arm]
        run_tasks(cfg, inner_tasks(cfg, names), "training_baseline_inner")
        run_tasks(cfg, formal_tasks(cfg, names), "training_baseline_formal")
        names = [name for name, arm in cfg["arms"].items() if "base" in arm]
        run_tasks(cfg, inner_tasks(cfg, names), "training_followup_inner")
        run_tasks(cfg, formal_tasks(cfg, names), "training_followup_formal")
        choices = select_inner(cfg)
        frozen = freeze(cfg, choices)
        write_json(Path(cfg["output_dir"]) / "TRAINING_COMPLETE.json",
                   dict(complete=True, formal_models=len(frozen["models"]), inner_fits=len(inner_tasks(cfg)),
                        holdout_used=False))
    # All branch classifiers and choices are fixed before either holdout is loaded.
    summaries = []
    for cfg, study in zip(configs, studies):
        from src.first_post_pcr_data import read_json
        choices = read_json(Path(cfg["output_dir"]) / "selection.json")
        oof = collect_oof(cfg, choices)
        report(cfg, study, oof)
        holdout = evaluate_holdout(cfg, study)
        summaries.append(report(cfg, study, oof, holdout).assign(branch=cfg["branch"]))
        write_json(Path(cfg["output_dir"]) / "COMPLETE.json", dict(complete=True, completed_utc=now(), bootstrap=False))
        progress(cfg, "complete")
    import pandas as pd
    from scripts.run_full978_anti_overfit import _atomic_csv
    _atomic_csv(Path(configs[0]["study_dir"]) / "summary.csv", pd.concat(summaries, ignore_index=True))
    from src.single_phase_fmbcmri_data import BRANCHES
    if all((Path(configs[0]["study_dir"]) / branch / "COMPLETE.json").exists() for branch in BRANCHES):
        write_json(Path(configs[0]["study_dir"]) / "COMPLETE.json", dict(complete=True, completed_utc=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_anchor_pcr_v3.yaml")
    parser.add_argument("--branch", choices=("both", "registered_dce0", "unregistered_first_post"), default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    for option in ("prepare", "smoke", "run", "detach"):
        mode.add_argument(f"--{option}", action="store_true")
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_anchor_training import check_pause
    from src.single_phase_anchor_workflow import load_config
    from src.single_phase_fmbcmri_data import BRANCHES
    from src.single_phase_fmbcmri_training import deterministic_runtime
    deterministic_runtime()
    configs = [load_config(args.config, branch) for branch in (BRANCHES if args.branch == "both" else [args.branch])]
    root = Path(configs[0]["study_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "COMPLETE.json").exists():
        print(f"Already complete: {root}", flush=True)
        return
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (root / "workflow.lock").open("a")
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An anchored pCR controller already owns this output")
        if args.detach:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", configs[0]["config_path"],
                       "--branch", args.branch, "--run", "--lock-fd", str(lock.fileno())]
            if args.resume:
                command.append("--resume")
            with (root / "workflow.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(lock.fileno(),))
            write_json(root / "launch.json", dict(pid=process.pid, created_utc=now(), command=command,
                       device="cpu", gpu_allowed=0, existing_jobs_preserved=True))
            print(f"Launched anchored pCR controller {process.pid}: {root}", flush=True)
            return

        def pause(signum, frame):
            write_json(root / "PAUSE_REQUESTED", dict(signal=signum, updated_utc=now()))

        signal.signal(signal.SIGTERM, pause)
        signal.signal(signal.SIGINT, pause)
        if args.resume:
            (root / "PAUSE_REQUESTED").unlink(missing_ok=True)
        mode_name = "prepare" if args.prepare else "smoke" if args.smoke else "run"
        try:
            check_pause(configs[0])
            write_json(root / "controller_status.json", dict(stage="running", mode=mode_name, pid=os.getpid(), updated_utc=now()))
            workflow(configs, mode_name)
            write_json(root / "controller_status.json", dict(stage=f"{mode_name}_complete", pid=os.getpid(), updated_utc=now()))
        except BaseException as error:
            write_json(root / "controller_status.json", dict(stage="paused" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                       error_type=type(error).__name__, error=str(error), updated_utc=now()))
            raise


if __name__ == "__main__":
    main()

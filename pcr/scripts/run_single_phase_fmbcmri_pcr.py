"""Two independent frozen FM-BCMRI single-phase pCR studies, with recovery."""

from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_tasks(cfg, tasks, stage):
    from src.single_phase_fmbcmri_data import progress
    from src.single_phase_fmbcmri_training import run_task, task_complete, task_path
    remaining = [t for t in tasks if not task_complete(task_path(cfg["output_dir"], t), t)]
    initial = completed = len(tasks) - len(remaining)
    started = time.monotonic()
    progress(cfg, stage, completed=completed, total=len(tasks))
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=cfg["training_jobs"], mp_context=mp.get_context("spawn")) as pool:
        iterator, running = iter(remaining), set()

        def submit():
            task = next(iterator, None)
            if task is not None:
                running.add(pool.submit(run_task, task, cfg["output_dir"]))

        for _ in range(min(cfg["training_jobs"], len(remaining))):
            submit()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                running.remove(future)
                future.result()
                completed += 1
                if not (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                    submit()
            elapsed = time.monotonic() - started
            rate = (completed - initial) / max(elapsed, 1e-6)
            progress(cfg, stage, completed=completed, total=len(tasks),
                     remaining_seconds=(len(tasks) - completed) / rate if rate else None)
            if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                for future in running:
                    future.cancel()
                raise InterruptedError("Training tasks are pausing at saved epoch boundaries")


def smoke(cfg, cohort, device):
    import numpy as np
    import pandas as pd
    import torch
    from scripts.run_full978_anti_overfit import _load_split, _prior_from_state
    from scripts.run_full978_independent_cv import _canonical_split, _subset_split
    from src.first_post_optimization import predict, select_patients, tensor_split
    from src.first_post_pcr_data import read_json, write_json
    from src.single_phase_fmbcmri_data import DEPTHS, extract_real, progress
    from src.single_phase_fmbcmri_training import fit
    root = Path(cfg["output_dir"])
    result_path = root / "SMOKE_COMPLETE.json"
    if result_path.exists():
        return read_json(result_path)
    metadata = pd.read_csv(root / "development_metadata.csv", dtype={"pid": str})
    selected = []
    for label in (0, 1):
        selected.extend(sorted(metadata.loc[metadata.pCR == label, "pid"].tolist())[:4])
    extract_real(cfg, cohort, "train", device, selected)
    data = _load_split(root / "embeddings/real", root / "development_metadata.csv", selected)
    train_ids = selected[:3] + selected[4:7]
    val_ids = [selected[3], selected[7]]
    recipes = read_json(root / "recipes.json")
    records = []
    for window, depth in DEPTHS.items():
        effective = {**recipes[window], "epochs": 2, "patience": 2, "scheduler_t_max": 2}
        train, val = select_patients(data, train_ids, depth), select_patients(data, val_ids, depth)
        directory = root / "smoke" / window
        model, result = fit(train, val, effective, 142, depth, device, directory / "inner")
        initial, fixed = fit(train, None, effective, 142, depth, device, directory / "continuous", fixed_epochs=2)
        try:
            fit(train, None, effective, 142, depth, device, directory / "interrupted", fixed_epochs=2, pause_after=1)
        except InterruptedError:
            pass
        restored, recovered = fit(train, None, effective, 142, depth, device, directory / "interrupted", fixed_epochs=2)
        if not all(torch.equal(fixed["model_state"][k], v) for k, v in recovered["model_state"].items()):
            raise ValueError("Real development smoke recovery differs from continuous training")
        prior = _prior_from_state(val["clinical"], fixed["clinical_prior"])
        a = predict(initial, tensor_split(val, prior, device))[0]
        b = predict(restored, tensor_split(val, prior, device))[0]
        if not np.array_equal(a, b) or fixed["history"] != recovered["history"]:
            raise ValueError("Recovered learning curve or predictions differ")
        if fixed["updated_parameter_tensors"] == 0:
            raise ValueError("TDN did not update")
        records.append({"window": window, "exact_recovery": True, "prediction_error": float(np.max(np.abs(a - b))),
                        "parameters": fixed["parameters"], "selected_epochs": result["selected_epochs"],
                        "gradient_tensors": len(fixed["gradient_audit"]), "updated_tensors": fixed["updated_parameter_tensors"]})
        del model, initial, restored
    write_json(result_path, {"passed": True, "development_patients": len(selected), "windows": records,
               "formal_classifier_updates": 0, "holdout_read": False})
    progress(cfg, "smoke_complete", windows=4, exact_recovery=True)
    return read_json(result_path)


def workflow(cfg, cohort):
    from src.first_post_pcr_data import now, read_json, write_json
    from src.single_phase_fmbcmri_data import extract_real, progress, unchanged_json
    from src.single_phase_fmbcmri_training import formal_tasks, freeze_models, inner_tasks, select_epochs
    from src.single_phase_fmbcmri_generated import extract_generated
    from src.single_phase_fmbcmri_evaluation import evaluate
    root = Path(cfg["output_dir"])
    smoke(cfg, cohort, "cuda:0")
    extract_real(cfg, cohort, "train", "cuda:0")
    folds, recipes = read_json(root / "nested_folds.json"), read_json(root / "recipes.json")
    inner = inner_tasks(cfg, folds, recipes)
    run_tasks(cfg, inner, "training_inner_fits")
    selection = select_epochs(cfg, folds, recipes)
    outer = formal_tasks(cfg, folds, recipes, selection)
    run_tasks(cfg, outer, "training_formal_classifiers")
    refs = freeze_models(cfg, outer)
    unchanged_json(root / "TRAINING_COMPLETE.json", {"complete": True, "inner_fits": len(inner),
                   "formal_classifiers": len(outer), "holdout_used_for_selection": False})
    extract_real(cfg, cohort, "val", "cuda:0")
    extract_generated(cfg, cohort, "cuda:0")
    evaluate(cfg, cohort, refs, "cuda:0")
    write_json(root / "COMPLETE.json", {"complete": True, "completed_utc": now(), "formal_classifiers": len(refs),
               "sources": ["real", "symm", "bifm", "copy"], "statistics": "ten_seed_mean_sample_SD"})
    progress(cfg, "complete", formal_classifiers=len(refs), generated_evaluated=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_fmbcmri_pcr_v1.yaml")
    parser.add_argument("--branch", choices=("both", "registered_dce0", "unregistered_first_post"), default="both")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "smoke", "run", "detach"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    import yaml
    config_path = (ROOT / args.config).resolve()
    master = yaml.safe_load(config_path.read_text())
    gpu = args.gpu if args.gpu is not None else master["gpu"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    from src.first_post_pcr_data import now, write_json
    from src.single_phase_fmbcmri_data import BRANCHES, load_config, prepare, progress
    from src.single_phase_fmbcmri_training import deterministic_runtime
    deterministic_runtime()
    branches = BRANCHES if args.branch == "both" else (args.branch,)
    configs = [load_config(config_path, branch, gpu) for branch in branches]
    output = ROOT / master["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(config_path),
                   "--branch", args.branch, "--gpu", str(gpu), "--run"]
        if args.resume:
            command.append("--resume")
        with (output / "workflow.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
        write_json(output / "launch.json", {"pid": process.pid, "gpu": gpu, "created_utc": now(),
                   "branches": list(branches), "log": str(output / "workflow.log"), "command": command})
        print(f"Launched PID {process.pid}; GPU {gpu}; {output / 'workflow.log'}", flush=True)
        return
    with (output / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This FM-BCMRI study already has an active controller")
        def pause(signum, frame):
            for cfg in configs:
                write_json(Path(cfg["output_dir"]) / "PAUSE_REQUESTED", {"signal": signum, "updated_utc": now()})
        signal.signal(signal.SIGTERM, pause)
        signal.signal(signal.SIGINT, pause)
        active = configs[0]
        try:
            if args.resume:
                for cfg in configs:
                    (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").unlink(missing_ok=True)
            prepared = []
            for cfg in configs:
                active = cfg
                prepared.append(prepare(cfg))
            if args.prepare:
                return
            used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"], text=True).strip())
            if used > 512:
                raise RuntimeError(f"GPU {gpu} is occupied ({used} MiB); existing jobs remain running")
            for cfg, cohort in zip(configs, prepared, strict=True):
                active = cfg
                if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                    raise InterruptedError("Study pause marker is present; use --resume to continue")
                if args.smoke:
                    smoke(cfg, cohort, "cuda:0")
                else:
                    workflow(cfg, cohort)
        except BaseException as error:
            progress(active, "paused" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error_type=type(error).__name__, error=str(error)[-1200:])
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

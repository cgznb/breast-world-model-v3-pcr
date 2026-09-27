"""Prepare, smoke-test, train, and evaluate registered ROI32 three-phase pCR."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import SimpleITK as sitk
import torch

from src import registered_three_phase_pcr as study
from src.first_post_pcr_data import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/registered_three_phase_roi32_pcr_v1.yaml")
    parser.add_argument("--stage", choices=("prepare", "smoke", "extract", "train", "evaluate", "all"), default="all")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    cfg = study.load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        from scripts.run_first_post_pcr import check_idle_gpu
        if args.stage != "prepare":
            check_idle_gpu(cfg["gpu"])
        with (output / "workflow.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", cfg["config_path"], "--stage", args.stage]
        with (output / "workflow.log").open("a") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {"pid": process.pid, "gpu": cfg["gpu"], "stage": args.stage, "started_utc": now(),
                  "log": str(output / "workflow.log"), "command": command}
        write_json(output / "launch.json", launch)
        print(study.json.dumps(launch), flush=True)
        return
    if args.stage != "prepare" and os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["gpu"]):
        raise RuntimeError("Use --detach or set CUDA_VISIBLE_DEVICES to the configured GPU")
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            cohort = study.prepare(cfg)
            if args.stage in ("smoke", "all"):
                study.smoke(cfg, cohort)
            if args.stage in ("extract", "all"):
                study.extract(cfg, cohort, "train")
            if args.stage in ("train", "all"):
                study.train(cfg, cohort)
            if args.stage in ("evaluate", "all"):
                if not (output / "frozen_models.json").is_file():
                    raise ValueError("Freeze all classifiers before holdout feature extraction")
                study.extract(cfg, cohort, "val")
                study.evaluate(cfg, cohort)
            if args.stage == "all":
                write_json(output / "COMPLETE.json", {"schema": study.SCHEMA, "completed_utc": now(),
                           "fitted_models": 4 * 5 * len(cfg["seeds"]), "real_holdout_evaluated": True})
            study.progress(cfg, "complete" if args.stage == "all" else f"{args.stage}_complete")
        except BaseException as error:
            study.progress(cfg, "failed", failed_stage=args.stage, error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()

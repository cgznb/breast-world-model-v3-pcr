"""Run frozen pCR comparisons of latent rollout and real-predecessor generation."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import SimpleITK as sitk
import torch

from src import registered_three_phase_sequential_pcr as study
from src.first_post_pcr_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/registered_three_phase_sequential_pcr_v1.yaml")
    parser.add_argument("--stage", choices=("prepare", "generate", "encode", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if int(np.__version__.split(".")[0]) < 2:
        raise RuntimeError("Use /root/miniconda3/bin/python in the generator NumPy2 environment")
    cfg = study.load_config(args.config)
    if args.stage != "prepare" and os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["gpu"]):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the configured GPU")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            context = study.prepare(cfg)
            if args.stage in ("all", "generate"):
                study.generate(cfg, context)
            if args.stage in ("all", "encode"):
                study.encode(cfg, context)
            if args.stage in ("all", "evaluate"):
                write_json(output / "COMPLETE.json", study.evaluate(cfg, context))
            study.progress(cfg, "complete" if args.stage == "all" else args.stage + "_complete")
        except BaseException as error:
            study.progress(cfg, "failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()

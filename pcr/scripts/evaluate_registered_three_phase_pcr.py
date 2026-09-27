"""Evaluate generated ROI32 triplets using the frozen real-trained classifiers."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import SimpleITK as sitk
import numpy as np
import torch

from src import registered_three_phase_generated_pcr as evaluation
from src.first_post_pcr_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/registered_three_phase_generated_pcr_v1.yaml")
    parser.add_argument("--stage", choices=("prepare", "generate", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if int(np.__version__.split(".")[0]) < 2:
        raise RuntimeError("Use the generator environment: /root/miniconda3/bin/python (NumPy 2)")
    cfg = evaluation.load_config(args.config)
    evaluation.original.load_config(cfg["source_pcr_config"])
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
            source_cfg, world, baseline, manifest, cohort, routes, refs = evaluation.prepare(cfg)
            if args.stage in ("generate", "all"):
                evaluation.generate_features(cfg, source_cfg, world, baseline, manifest, routes)
            if args.stage in ("evaluate", "all"):
                result = evaluation.evaluate(cfg, source_cfg, cohort, routes, refs)
                write_json(output / "COMPLETE.json", result)
            evaluation.progress(cfg, "complete" if args.stage == "all" else f"{args.stage}_complete")
        except BaseException as error:
            evaluation.progress(cfg, "failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()

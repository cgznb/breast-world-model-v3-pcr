"""Exercise all production loss/backward paths on real full-resolution patients."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from responsewm.config import load_config
from responsewm.data import ManifestStore
from responsewm.io import autocast, seed_all, write_json
from responsewm.losses import stage_loss
from responsewm.training import STAGES, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.training.seed, cfg.training.threads)
    store = ManifestStore(args.manifest)
    store.fit_statistics()
    indices = [i for i in store.by_split["train"]
               if store.cases[i]["target"]["pcr"] is not None
               and all(v is not None for v in store.cases[i]["target"]["future"])]
    indices = indices[:cfg.training.batch_size]
    if len(indices) != cfg.training.batch_size:
        raise ValueError("Too few paired training patients for the configured batch size")
    report = {"real_data": True, "clinical_performance_evaluation": False,
              "latent_shape": store.manifest["latent_shape"], "batch_size": cfg.training.batch_size,
              "sampling": cfg.to_dict()["sampling"],
              "stages": []}
    for stage in STAGES:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, store).to(cfg.training.device)
        if stage != "representation":
            model.freeze_representation()
        params = model.configure_stage(stage)
        optimizer = torch.optim.AdamW(params, lr=cfg.training.joint_lr)
        inp, sup = store.batch(indices, cfg.training.device)
        start = time.monotonic()
        with autocast(cfg.training.device, cfg.training.precision):
            loss, terms = stage_loss(model, inp, sup, stage)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {stage} loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        row = {"stage": stage, "loss": float(loss.detach()), "gradient_norm": float(norm),
               "seconds": time.monotonic() - start,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
               "finite_backward_optimizer_step": True}
        report["stages"].append(row)
        write_json(args.output, report)
        print(row, flush=True)
        del model, params, optimizer, inp, sup, loss, terms, norm
    report["passed"] = True
    write_json(args.output, report)


if __name__ == "__main__":
    main()

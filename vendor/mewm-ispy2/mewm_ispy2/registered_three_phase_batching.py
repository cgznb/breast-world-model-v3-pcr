"""Measure larger three-phase batches without changing the sampled-pair budget."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import torch

from . import registered_roi32_fm_tuning as tuning
from . import registered_roi32_runtime as runtime
from .registered_roi32_data import file_identity, read_json, write_json


def settings(config):
    result = tuning.settings(config)
    if result is not None:
        budget = config["fm"]["batch_size"] * config["fm"]["max_updates"]
        if result["experiment_schema"] != config["schema"] or result["sample_budget_pairs"] != budget:
            raise ValueError("Three-phase batch selection changed the experiment or pair budget")
    return result


def choose_result(rows):
    eligible = [row for row in rows if row["status"] == "passed" and row["batch_size"] >= 16
                and math.isfinite(row["pairs_per_second"]) and row["pairs_per_second"] > 0]
    if not eligible:
        raise RuntimeError("No larger three-phase batch passed the measured GPU check")
    return max(eligible, key=lambda row: row["pairs_per_second"])


def update(trainer, batches, config, previous):
    # Count EMA warmup in reference samples, including shorter boundary batches.
    if trainer.ema is not None and trainer.ema.num_updates != previous:
        raise ValueError("Three-phase EMA and reference progress disagree")
    metrics, reference = tuning.update(trainer, batches, config, previous)
    if trainer.ema is not None:
        trainer.ema.num_updates = reference
    return metrics, reference


def measure(config, manifest, dataset, device, size, warmup, measured):
    from .registered_three_phase_training import bridge, build

    module = bridge(config)
    runtime.seed_all(config["seed"])
    model, optimizer, scheduler, ema, trainer = build(config, manifest["pairs"], device)
    base = config["fm"]["batch_size"]
    stream = runtime.TrainingBatches(dataset, batch_size=base, effective_batch=base, seed=config["seed"],
                                      workers=config["runtime"]["loader_workers"], balanced=True,
                                      collate_fn=module.collate_pairs)
    previous = 0
    try:
        for index in range(warmup + measured):
            if runtime.STOP_REQUESTED:
                raise InterruptedError("Paused during disposable three-phase batch measurement")
            if index == warmup:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
            batches = tuning.next_batches(stream, size, size, size)
            metrics, previous = update(trainer, batches, config, previous)
            if any(not math.isfinite(metrics[key]) for key in ("loss", "loss_x", "loss_y", "gradient_norm")):
                raise FloatingPointError("Non-finite three-phase batch measurement")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        return {"batch_size": size, "status": "passed", "updates": measured, "seconds": elapsed,
                "pairs_per_second": size * measured / elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
                "last_loss": metrics["loss"]}
    finally:
        stream.close()


def benchmark(config, baseline, manifest, device, batches=(4, 16, 32, 64), warmup=8, measured=30):
    from .registered_three_phase_latents import PhaseLatentPairs, contract_for

    chosen = settings(config)
    if chosen is not None:
        for key in ("benchmark", "codec", "latent_completion"):
            runtime.verify_contract_files(chosen[key])
        return chosen
    root = Path(config["output_dir"])
    if any((root / "fm" / name).exists() for name in ("last.pt", "best-loss.pt", "best-endpoint.pt")):
        raise ValueError("Existing three-phase checkpoints need an explicit batch migration")
    if device.type != "cuda" or warmup < 1 or measured < 1:
        raise ValueError("Batch measurement requires CUDA and positive warmup/measured counts")
    base = config["fm"]["batch_size"]
    if not batches or any(size < base or size % base for size in batches):
        raise ValueError("Measured batches must preserve complete reference sampler groups")
    dataset = PhaseLatentPairs(config, baseline, manifest, "train")
    report_path = root / "fm_tuning/benchmark.json"
    binding = {"latents": contract_for(config, baseline), "source": file_identity(__file__),
               "sample_budget_runtime": file_identity(tuning.__file__)}
    rows = []
    if report_path.exists():
        report = read_json(report_path)
        if (report["binding"] != binding or report["batches"] != list(batches)
                or report["warmup_updates"] != warmup or report["measured_updates"] != measured):
            raise ValueError("Partial three-phase batch measurement changed its protocol")
        rows = report["results"]
    for size in batches:
        if any(row["batch_size"] == size for row in rows):
            continue
        if runtime.STOP_REQUESTED:
            return None
        runtime.log_event(config, "fm_tuning", "measuring", batch_size=size)
        try:
            row = measure(config, manifest, dataset, device, size, warmup, measured)
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": size, "status": "out_of_memory"}
        except InterruptedError:
            return None
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        write_json(report_path, {"binding": binding, "batches": list(batches), "warmup_updates": warmup,
                                 "measured_updates": measured, "results": rows,
                                 "scope": "disposable fresh models; real training latents; formal training starts fresh"})
        runtime.log_event(config, "fm_tuning", "measured", **row)
    selected = choose_result(rows)["batch_size"]
    passed = [row["batch_size"] for row in rows if row["status"] == "passed"
              and row["batch_size"] < selected and selected % row["batch_size"] == 0]
    fallback = max(passed, default=base)
    result = {"schema": "registered_roi32_fm_batch_tuning_v1", "experiment_schema": config["schema"],
              "batch_size": selected, "fallback_batch_size": fallback, "reference_batch_size": base,
              "reference_max_updates": config["fm"]["max_updates"], "validation_batch_size": base,
              "sampler_batch_size": base, "sampler_effective_batch": base, "migration_step": 0,
              "target_optimizer_updates": tuning.remaining_updates(config, 0, selected),
              "sample_budget_pairs": base * config["fm"]["max_updates"],
              "policy": "fresh training; fixed pair budget, sample-based LR/EMA and reference validation groups",
              "ema_counter_units": "reference_updates",
              "benchmark": file_identity(report_path),
              "codec": file_identity(Path(baseline["output_dir"]) / "vq/best.pt"),
              "latent_completion": file_identity(root / "latents/COMPLETE.json")}
    write_json(tuning.tuning_path(config), result)
    runtime.log_event(config, "fm_tuning", "selected", batch_size=selected, fallback_batch_size=fallback,
                      target_updates=result["target_optimizer_updates"], sampled_pair_budget=result["sample_budget_pairs"])
    return settings(config)

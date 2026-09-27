"""Sample-budget preserving batch changes for the registered ROI32 flow."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .registered_roi32_data import read_json


def tuning_path(config):
    return Path(config["output_dir"]) / "fm" / "batch_tuning.json"


def settings(config):
    path = tuning_path(config)
    if not path.exists():
        return None
    result = read_json(path)
    base = config["fm"]["batch_size"]
    if result["schema"] != "registered_roi32_fm_batch_tuning_v1" or result["reference_batch_size"] != base:
        raise ValueError("FM batch tuning does not match the base experiment")
    for key in ("batch_size", "fallback_batch_size"):
        if result[key] < base or result[key] % base:
            raise ValueError("Tuned physical batches must be multiples of the reference batch")
    if result["batch_size"] % result["fallback_batch_size"]:
        raise ValueError("Tuned fallback must divide the effective batch")
    if result["reference_max_updates"] != config["fm"]["max_updates"]:
        raise ValueError("FM batch tuning changed the original sample budget")
    if base % result["sampler_batch_size"] or result["sampler_effective_batch"] != base:
        raise ValueError("Tuned training requires the original sampler grouping")
    return result


def merge_batches(batches):
    if len(batches) == 1:
        return batches[0]
    keys = set(batches[0]["conditions"])
    if any(set(batch["conditions"]) != keys for batch in batches):
        raise ValueError("Cannot combine different condition schemas")
    return {
        "earlier_latent": torch.cat([batch["earlier_latent"] for batch in batches]),
        "later_latent": torch.cat([batch["later_latent"] for batch in batches]),
        "records": [row for batch in batches for row in batch["records"]],
        "conditions": {key: [value for batch in batches for value in batch["conditions"][key]] for key in keys},
    }


def next_batches(stream, effective_batch, physical_batch, remaining_pairs):
    effective = min(effective_batch, remaining_pairs)
    physical = math.gcd(physical_batch, effective)
    if effective < 1 or physical % stream.batch_size:
        raise ValueError("Batch regrouping must preserve complete sampler batches")
    return [merge_batches([next(stream) for _ in range(physical // stream.batch_size)])
            for _ in range(effective // physical)]


def reference_step(state):
    return state.get("reference_step", state["step"])


def crossed(previous, current, interval):
    return current // interval > previous // interval


def pairs_to_next_gate(config, previous):
    boundaries = [config["fm"]["max_updates"]]
    for interval in (config["fm"]["validation_interval"], config["fm"]["endpoint_interval"],
                     config["runtime"]["checkpoint_interval"]):
        boundaries.append((previous // interval + 1) * interval)
    return (min(boundaries) - previous) * config["fm"]["batch_size"]


def remaining_updates(config, previous, effective_batch):
    count = 0
    while previous < config["fm"]["max_updates"]:
        pairs = pairs_to_next_gate(config, previous)
        count += math.ceil(pairs / effective_batch)
        previous += pairs // config["fm"]["batch_size"]
    return count


def update(trainer, batches, config, previous):
    pairs = sum(len(batch["later_latent"]) for batch in batches)
    base = config["fm"]["batch_size"]
    if pairs % base or len({len(batch["later_latent"]) for batch in batches}) != 1:
        raise ValueError("Accumulation needs equal physical batches and a whole reference update")
    advance = pairs // base
    trainer.gradient_accumulation = len(batches)
    trainer.micro_step += (-trainer.micro_step) % len(batches)
    if trainer.ema is not None:
        # Preserve the original EMA's sample horizon, including its warmup.
        decay = math.prod(min(config["fm"]["ema_decay"], (1 + i) / (10 + i))
                          for i in range(previous + 1, previous + advance + 1))
        next_count = trainer.ema.num_updates + 1
        if decay > (1 + next_count) / (10 + next_count) + 1e-12:
            raise ValueError("EMA warmup would override the sample-based decay")
        trainer.ema.decay = decay
    for batch in batches:
        metrics = trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
    if not metrics["optimizer_updated"]:
        raise RuntimeError("Regrouped batches did not finish an optimizer update")
    if trainer.scheduler is not None:
        for _ in range(advance - 1):
            trainer.scheduler.step()
    return metrics, previous + advance

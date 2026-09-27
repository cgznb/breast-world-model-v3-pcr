"""Complete A/B trainers, independent manifests, EMA and exact counter/RNG resume."""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import copy
import json
import math
import random
import signal
import time
import numpy as np
import torch
from .config import from_dict, stage_budget
from .conditioning import ConditionSchema
from .observation_encoder import ThreePhaseObservationEncoder
from .observation_ssl import ObservationPretrainer
from .observation_views import observation_batch
from .world_model import ThreePhaseWorldModel
from .data import VisitStore, PairStore, PatientSampler
from .utils import (autocast_context, seed_all, file_identity, write_json, save_checkpoint,
                    load_checkpoint, update_ema)

SCHEMA = "three_phase_observation_checkpoint_v3"


def rng_state():
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def validation_rng(device, seed):
    state = rng_state()
    try:
        torch.manual_seed(seed)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def lr_factor(step, total, warmup):
    warmup = min(warmup, max(1, total//5))
    if step < warmup:
        return (step+1)/max(1, warmup)
    progress = min(1., (step-warmup)/max(1, total-warmup))
    return .01+.99*.5*(1+math.cos(math.pi*progress))


def load_observation(checkpoint, device="cpu", *, require_completed=True):
    value = load_checkpoint(checkpoint)
    if value.get("schema") != SCHEMA or value.get("stage") != "A":
        raise ValueError("Expected an A-observation checkpoint, not old future-predictive v2")
    if require_completed and not value.get("training_run_completed", False):
        raise ValueError("A run is incomplete. Finish A before initializing B")
    cfg = from_dict(value["metadata"]["config"])
    encoder = ThreePhaseObservationEncoder(cfg.encoder, value["metadata"]["statistics"])
    state = {k.removeprefix("target_encoder."): v for k, v in value["model"].items() if k.startswith("target_encoder.")}
    encoder.load_state_dict(state, strict=True)
    encoder.eval().requires_grad_(False).to(device)
    return encoder, value["metadata"]


def _assert_supervision(store, cfg):
    a = cfg.observation
    rows = store.records("train")
    if not rows:
        raise ValueError("No stage-A training observations")
    if a.domain_enabled and any(not r.get("augmentations") for r in store.visits):
        raise ValueError("Domain mode requires image-backed augmentations for all A visits")
    for weight, field in ((a.kinetics_weight, "kinetic_path"), (a.segmentation_weight, "segmentation_path")):
        if weight and not any(r.get(field) for r in rows):
            raise ValueError(f"Enabled task {field} has no actual train supervision")
    if a.arcface_weight and not any(r.get("phenotype_label", -1) >= 0 for r in rows):
        raise ValueError("Enabled ArcFace has no current-visit labels")
    if a.dann_weight and len({r["domain_label"] for r in rows if r.get("domain_label", -1) >= 0}) < 2:
        raise ValueError("DANN needs at least two observed scanner/site domains")
    if (a.arcface_weight or a.dann_weight) and a.supervised_start_step >= cfg.training.stage_a_steps:
        raise ValueError("Supervised heads would never be trained; choose supervised_start_step < stage_a_steps")


def _patient_macro(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row["patient_id"]].append(float(row[key]))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def validation_records(rows, limit, balanced=False):
    if not balanced or limit is None:
        return rows if limit is None else rows[:limit]
    groups = defaultdict(list)
    for row in rows:
        groups[row["patient_id"]].append(row)
    selected = []
    for index, pid in enumerate(sorted(groups)):
        choices = sorted(groups[pid], key=lambda r: r["id"])
        selected.append(choices[index % len(choices)])
    return selected[:limit]


@torch.no_grad()
def evaluate_a(model, store, cfg, step):
    rows = validation_records(store.records("val"), cfg.training.validation_cases,
                              cfg.training.patient_balanced_validation)
    if not rows:
        return {"selection_score": None, "selection_metric": "no_validation_split", "records": []}
    was_training = model.training
    model.eval()
    results = []
    device = next(model.parameters()).device
    try:
        with validation_rng(device, cfg.training.seed+9929):
            for i, row in enumerate(rows):
                batch = observation_batch(store, [row], cfg, i, device, validation=True)
                with autocast_context(device, cfg.training.precision):
                    _, parts = model(batch, step)
                parts.update(patient_id=row["patient_id"], visit_id=row["visit_id"],
                             score=cfg.observation.local_weight*parts["local"]+cfg.observation.global_weight*parts["global"]+
                                   cfg.observation.reconstruction_weight*parts["reconstruction"])
                results.append(parts)
    finally:
        model.train(was_training)
    return {"selection_score": _patient_macro(results, "score"), "selection_metric": "same_visit_local_global_reconstruction",
            "split": "val", "independent_test": False, "records": results}


@torch.no_grad()
def evaluate_b(model, store, cfg, split="val", limit=None, seed=None):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = validation_records(store.records(split), limit, cfg.training.patient_balanced_validation)
    results = []
    try:
        with validation_rng(device, cfg.training.seed+2999 if seed is None else seed):
            for p in rows:
                # Generate first with source-only reads. Target is read only AFTER generation.
                observed = store.source(p)
                raw = observed["latent"][None].to(device)
                valid = observed["valid"].amin(0, keepdim=True)[None].to(device)
                source = model.observation_encoder.normalize(raw)
                samples = []
                for _ in range(cfg.sampling.samples):
                    with autocast_context(device, cfg.training.precision):
                        prediction = model.sample(source, [p["conditions"]], torch.randn_like(source), valid=valid)
                    samples.append(prediction.float())
                predictions = torch.stack(samples)
                target_data = store.read(p["target"])
                target = model.observation_encoder.normalize(target_data["latent"][None].to(device))
                if target.shape != source.shape:
                    raise ValueError("Evaluation source and target shapes differ")
                mask = target_data["valid"].amin(0, keepdim=True)[None].to(device).expand_as(target)
                denom = mask.sum().clamp_min(1.)
                ensemble_mean = predictions.mean(0)
                mae = float(((ensemble_mean-target).abs()*mask).sum()/denom)
                copy_mae = float(((source-target).abs()*mask).sum()/denom)
                dispersion = float((predictions.std(0, unbiased=False)*mask).sum()/denom)
                perphase = [float(((ensemble_mean[:, i*8:(i+1)*8]-target[:, i*8:(i+1)*8]).abs()*mask[:, i*8:(i+1)*8]).sum()/mask[:, i*8:(i+1)*8].sum().clamp_min(1.)) for i in range(3)]
                results.append({"patient_id": p["patient_id"], "pair_id": p["id"], "latent_mae": mae,
                                "copy_source_mae": copy_mae, "sample_std": dispersion, "phase_latent_mae": perphase})
    finally:
        model.train(was_training)
    return {"selection_score": _patient_macro(results, "latent_mae"),
            "selection_metric": "patient_macro_standardized_latent_ensemble_mean_mae",
            "split": split, "records": results,
            "copy_source_mae": _patient_macro(results, "copy_source_mae"),
            "sample_std": _patient_macro(results, "sample_std"),
            "independent_test": split == "test",
            "limitations": "Latent MAE/different samples are not clinical validity or uncertainty calibration"}


def train(stage, cfg, manifest, output, *, observation_checkpoint=None, resume=False, stop_after=None):
    if stage not in {"A", "B"}:
        raise ValueError("Only stages A (observation) and B (dynamics) exist in v3")
    cfg.validate()
    t = cfg.training
    budget = stage_budget(cfg, stage)
    seed_all(t.seed, t.cpu_threads, t.strict_determinism)
    device = torch.device(t.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use the CPU smoke configuration")
    folder = Path(output).resolve()/stage
    folder.mkdir(parents=True, exist_ok=True)
    last, best_path = folder/"last.pt", folder/"best.pt"
    if last.exists() and not resume:
        raise FileExistsError("Existing training run: use --resume or a fresh output directory")
    if resume and not last.exists():
        raise FileNotFoundError("--resume requires this stage's last.pt")
    old = load_checkpoint(last) if resume else None
    if stage == "A":
        store = VisitStore(manifest)
        _assert_supervision(store, cfg)
        statistics = old["metadata"]["statistics"] if old else store.fit_statistics()
        model = ObservationPretrainer(cfg, statistics).to(device)
        teacher = None
        meta = {"config": cfg.to_dict(), "statistics": statistics, "codec_id": store.codec_id,
                "patient_splits": store.patient_splits, "A_contains_longitudinal_predictor": False,
                "description": "Single-visit local/global/domain observation pretraining"}
        assets = store.assets(cfg.observation.domain_enabled)
        base_lr = t.lr_a
    else:
        if observation_checkpoint is None:
            raise ValueError("B requires --a-checkpoint from completed observation-only A")
        store = PairStore(manifest)
        encoder, a_meta = load_observation(observation_checkpoint)
        if json.dumps(a_meta["config"]["encoder"], sort_keys=True) != json.dumps(cfg.to_dict()["encoder"], sort_keys=True):
            raise ValueError("B encoder settings differ from the exported A encoder")
        if store.codec_id != a_meta["codec_id"]:
            raise ValueError("A and B use different VQ codecs")
        exposure = a_meta["patient_splits"]
        for pid, split in store.patient_splits.items():
            previous = exposure.get(pid)
            if split in {"val", "test"} and previous == "train":
                raise ValueError("B holdout patient was used for A training")
            if split == "test" and previous == "val":
                raise ValueError("B test patient was used for A checkpoint selection")
        schema = ConditionSchema.fit(store.records("train"))
        model = ThreePhaseWorldModel(cfg, encoder, schema).to(device)
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        meta = {"config": cfg.to_dict(), "statistics": a_meta["statistics"], "codec_id": store.codec_id,
                "condition_schema": schema.to_dict(), "A_patient_splits": exposure,
                "patient_splits": store.patient_splits, "a_checkpoint": file_identity(observation_checkpoint),
                "description": "Current-observation-conditioned SymmFlow; no A predictor in inference"}
        assets = store.assets()
        base_lr = t.lr_b
    total = budget["steps"]
    meta["budget"] = budget
    meta["data_provenance"] = store.manifest.get("provenance", {})
    if total < 1:
        raise ValueError("Configured stage has no optimizer steps")
    contract = {"schema": SCHEMA, "stage": stage, "config": cfg.to_dict(),
                "manifest": store.identity, "assets": assets, "a_checkpoint": meta.get("a_checkpoint")}
    if old and old["contract"] != contract:
        raise ValueError("Resume contract mismatch: config/data/normalization lineage/assets changed")
    if old and old["step"] >= total and old.get("training_run_completed"):
        # Preserve A best-file identity, which is bound by the B resume contract.
        return json.loads((folder/"status.json").read_text())
    if t.preload_latents:
        store.preload()
    write_json(folder/"contract.json", contract)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=base_lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total, budget["warmup_steps"]))
    step, best, directions = 0, None, {"forward": 0, "reverse": 0}
    if old:
        model.load_state_dict(old["model"], strict=True)
        if teacher is not None:
            teacher.load_state_dict(old["teacher"], strict=True)
        optimizer.load_state_dict(old["optimizer"]); scheduler.load_state_dict(old["scheduler"])
        step, best = old["step"], old["best_score"]
        directions = old.get("directions_trained", directions)
        restore_rng(old["rng"])
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    limit = min(total, step+stop_after) if stop_after is not None else total
    sampler = PatientSampler(store.records("train"), t.seed)
    stopping = {"value": False}
    handlers = {sig: signal.signal(sig, lambda *_: stopping.update(value=True)) for sig in (signal.SIGINT, signal.SIGTERM)}

    def save(path):
        save_checkpoint(path, {"schema": SCHEMA, "stage": stage, "step": step,
            "best_score": best, "training_run_completed": step == total, "contract": contract,
            "metadata": meta, "model": model.state_dict(), "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "directions_trained": directions})

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while step < limit and not stopping["value"]:
            start = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True)
            aggregate = defaultdict(float)
            reference_step = step * budget["effective_batch"] // budget["reference_batch"]
            for micro in range(t.accumulation):
                counter = step*t.accumulation+micro
                records = sampler.batch(counter, budget["batch"])
                batch = observation_batch(store, records, cfg, counter, device) if stage == "A" else store.batch(records, device)
                with autocast_context(device, t.precision):
                    loss, parts = model(batch, reference_step) if stage == "A" else model.loss(batch, reference_step)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss/t.accumulation).backward()
                for key, value in parts.items():
                    aggregate[key] += value/t.accumulation
                directions["forward"] += int(parts.get("forward_batches", 0))
                directions["reverse"] += int(parts.get("reverse_batches", 0))
            norm = torch.nn.utils.clip_grad_norm_(parameters, t.grad_clip, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            if stage == "A":
                reference_step = step * budget["effective_batch"] / budget["reference_batch"]
                aggregate["teacher_ema"] = model.update_target(reference_step,
                    exponent=budget["effective_batch"] / budget["reference_batch"])
            else:
                update_ema(teacher, model, budget["ema_decay"])
            aggregate.update(step=step, stage=stage, gradient_norm=float(norm),
                             lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter()-start,
                             sampled_examples=step*budget["effective_batch"], target_examples=budget["samples"])
            if device.type == "cuda":
                aggregate.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                                 peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
            with (folder/"metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(aggregate), allow_nan=False)+"\n")
            if step <= 2 or step % t.log_every == 0:
                print(json.dumps(dict(aggregate), allow_nan=False), flush=True)
                write_json(folder/"progress.json", dict(aggregate, configured_steps=total, status="running"))
            # No extra validation at an arbitrary partial stop: resume matches uninterrupted updates.
            if step % budget["validate_every"] == 0 or step == total:
                result = evaluate_a(model, store, cfg, reference_step) if stage == "A" else evaluate_b(teacher, store, cfg, limit=t.validation_cases)
                write_json(folder/f"validation_{step:07d}.json", result)
                score = result["selection_score"]
                if score is not None and (best is None or score < best):
                    best = score; save(best_path)
            if step == 1 or step % budget["checkpoint_every"] == 0 or step == limit or stopping["value"]:
                save(last)
        save(last)
        if step == total:
            if not best_path.exists():
                save(best_path)
            else:
                best_value = load_checkpoint(best_path)
                best_value["training_run_completed"] = True
                save_checkpoint(best_path, best_value)
    finally:
        for sig, original in handlers.items():
            signal.signal(sig, original)
    result = {"stage": stage, "step": step, "configured_steps": total, "complete": step == total,
              "best_score": best, "checkpoint": str(last), "directions_trained": directions,
              "device": str(device), "synthetic": bool(store.manifest.get("provenance", {}).get("synthetic")),
              "sampled_examples": step*budget["effective_batch"], "budget": budget,
              "stop_requested": stopping["value"]}
    write_json(folder/"status.json", result)
    write_json(folder/"progress.json", dict(result, status="complete" if step == total else "stopped"))
    return result


def load_world(path, device="cpu"):
    value = load_checkpoint(path)
    if value.get("schema") != SCHEMA or value.get("stage") != "B":
        raise ValueError("Generation requires a v3 stage-B checkpoint")
    meta = value["metadata"]
    cfg = from_dict(meta["config"])
    encoder = ThreePhaseObservationEncoder(cfg.encoder, meta["statistics"])
    model = ThreePhaseWorldModel(cfg, encoder, ConditionSchema(**meta["condition_schema"]))
    model.load_state_dict(value["teacher"] or value["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    meta = copy.deepcopy(meta)
    meta["directions_trained"] = value["directions_trained"]
    return model, meta

"""Original ROI32 optimizer, symmetric objective and recovery on three phases."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from . import registered_roi32_fm_tuning as tuning
from . import registered_three_phase_batching as batching
from .registered_roi32_data import file_identity, read_json, write_json
from .registered_roi32_fm import bridge
from .registered_roi32_smoke import assert_replay, snapshot
from .registered_three_phase_data import ThreePhaseCrops
from .registered_three_phase_latents import PhaseLatentPairs, fit_statistics
from .registered_three_phase_model import SharedPhaseCodec, ThreePhaseROI32Flow


def build(config, records, device):
    bridge(config)
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.engine import SymmFlowTrainer

    model = ThreePhaseROI32Flow(config, records).to(device)
    fm = config["fm"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=fm["learning_rate"], weight_decay=fm["weight_decay"])

    def multiplier(step):
        if step < fm["warmup_updates"]:
            return (step + 1) / max(1, fm["warmup_updates"])
        fraction = min(1.0, (step - fm["warmup_updates"]) / max(1, fm["max_updates"] - fm["warmup_updates"]))
        return 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    ema = ExponentialMovingAverage(model.velocity_model, decay=fm["ema_decay"])
    trainer = SymmFlowTrainer(model.velocity_model, model.condition_encoder, optimizer, model.objective,
                              device=device, precision="bf16" if device.type == "cuda" else "fp32",
                              gradient_clip_norm=config["runtime"]["gradient_clip"], scheduler=scheduler, ema=ema)
    return model, optimizer, scheduler, ema, trainer


def source_contract(config, baseline):
    from . import registered_three_phase_latents as latents
    from . import registered_three_phase_model as model

    source = Path(config["symm_repo"]) / "src/ispy2_symmflow"
    dependencies = [source / name for name in ("models/velocity.py", "models/conditioning.py", "flow/path.py",
                                               "flow/solver.py", "training/engine.py", "training/ema.py",
                                               "training/validation.py", "training/schema.py", "training/datasets.py")]
    extra = {"batch_tuning": file_identity(tuning.tuning_path(config))} if batching.settings(config) else {}
    return {"schema": config["schema"], "configuration": config,
            "inventory": file_identity(Path(config["output_dir"]) / "inventory.json"),
            "codec": file_identity(Path(baseline["output_dir"]) / "vq/best.pt"),
            "runtime_sources": [file_identity(path) for path in (__file__, model.__file__, latents.__file__,
                                                                  batching.__file__, tuning.__file__, *dependencies)], **extra}


def checkpoint_payload(contract, model, optimizer, scheduler, ema, trainer, state, stream=None):
    return {"contract": contract, "model": model.state_dict(), "model_description": model.description,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "ema": ema.state_dict(),
            "scaler": trainer.scaler.state_dict(), "micro_step": trainer.micro_step, "training_state": dict(state),
            "rng": runtime.rng_state(), "stream": stream.state_dict() if stream else None}


def restore(saved, model, optimizer, scheduler, ema, trainer, stream=None):
    if saved["model_description"] != model.description:
        raise ValueError("Three-phase FM architecture or condition schema changed")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    ema.load_state_dict(saved["ema"])
    trainer.scaler.load_state_dict(saved["scaler"])
    trainer.micro_step = saved["micro_step"]
    if stream is not None:
        stream.load_state_dict(saved["stream"])
    runtime.restore_rng(saved["rng"])
    return saved["training_state"]


def smoke(config, baseline, manifest, device, batch_size=1):
    module = bridge(config)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    runtime.seed_all(config["seed"] + 71)
    available = {row["visit_id"]: row for row in manifest["visits"] if all(row["phase_sources"])}
    selected = next(row for row in manifest["pairs"] if row["split"] == "train"
                    and row["earlier_visit_id"] in available and row["later_visit_id"] in available)
    visits = [available[selected[key]] for key in ("earlier_visit_id", "later_visit_id")]
    dataset = ThreePhaseCrops(baseline, manifest, records=visits)
    codec, _ = vq.load_frozen(baseline, device)
    codec = SharedPhaseCodec(codec)
    raw = {}
    with torch.no_grad(), runtime.autocast(device):
        for index in range(len(dataset)):
            item = dataset[index]
            raw[item["visit_id"]] = codec.encode(item["image"][None].to(device))[0].float().cpu().numpy()
    stats = fit_statistics(visits, raw.__getitem__)
    mean, std = (np.asarray(stats[key], dtype=np.float32).reshape(24, 1, 1, 1) for key in ("mean", "std"))
    values = {key: torch.from_numpy((value - mean) / std) for key, value in raw.items()}
    batch = module.collate_pairs([{"record": selected, "earlier_latent": values[selected["earlier_visit_id"]],
                                   "later_latent": values[selected["later_visit_id"]]} for _ in range(batch_size)])
    model, optimizer, scheduler, ema, trainer = build(config, manifest["pairs"], device)
    contract = source_contract(config, baseline)
    tuned = batching.settings(config)
    effective = tuned["batch_size"] if tuned else batch_size
    if effective % batch_size:
        raise ValueError("The physical smoke batch must divide its effective batch")

    def update(previous):
        if tuned:
            return batching.update(trainer, [batch] * (effective // batch_size), config, previous)
        return trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"]), previous + 1

    before = snapshot(model)
    first, reference = update(0)
    after = snapshot(model)
    if not any(not torch.equal(value, after[key]) for key, value in before.items()):
        raise AssertionError("The three-phase generator did not update")
    if any(not math.isfinite(first[key]) for key in ("loss", "loss_x", "loss_y", "gradient_norm")):
        raise FloatingPointError("Non-finite three-phase smoke update")
    del before, after
    root = Path(config["output_dir"]) / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recovery_", dir=root) as directory:
        path = Path(directory) / "state.pt"
        runtime.save_checkpoint(path, checkpoint_payload(contract, model, optimizer, scheduler, ema, trainer,
                                                         {"reference_step": reference}))
        _, expected_reference = update(reference)
        expected, expected_ema = snapshot(model), {key: value.clone() for key, value in ema.shadow.items()}
        saved = runtime.load_checkpoint(path, contract, device)
        state = restore(saved, model, optimizer, scheduler, ema, trainer)
        del saved
        _, recovered_reference = update(state["reference_step"])
        if recovered_reference != expected_reference:
            raise AssertionError("Three-phase reference progress changed after recovery")
        error = assert_replay(expected, snapshot(model))
        ema_error = assert_replay(expected_ema, ema.shadow)
        del expected, expected_ema
    model.eval()
    with torch.no_grad(), runtime.autocast(device), ema.average_parameters(model.velocity_model):
        source = batch["earlier_latent"].to(device)
        endpoint = model.sample(source, batch["conditions"], torch.zeros_like(source),
                                steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"])
        decoded = codec.decode(endpoint[:1], stats)
    if endpoint.shape != source.shape or decoded.shape != (1, 3, 32, 128, 128) or not torch.isfinite(decoded).all():
        raise AssertionError("Invalid three-phase decoded Heun endpoint")
    result = {"status": "passed", "contract": contract, "device": str(device), "batch_size": batch_size,
              "effective_batch": effective,
              "joint_shape": [batch_size, 48, 8, 32, 32], "decoded_shape": list(decoded.shape),
              "first_update": first, "recovery_max_error": error, "ema_recovery_max_error": ema_error,
              "heun_steps": config["fm"]["sampling_steps"], "heun_endpoint_finite": True,
              "limitation": "Fresh generator and two training visits for interface/recovery only; not forecast quality or production normalization."}
    if device.type == "cuda":
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
    write_json(root / ("fm-cpu.json" if device.type == "cpu" else "fm-gpu.json"), result)
    return result


def contract_for(config, baseline):
    return {**source_contract(config, baseline), "stage": "fm",
            "latent_statistics": file_identity(Path(config["output_dir"]) / "latents/statistics.json"),
            "latent_completion": file_identity(Path(config["output_dir"]) / "latents/COMPLETE.json")}


def train(config, baseline, manifest, device):
    module = bridge(config)
    from ispy2_symmflow.training.validation import validate_symmflow, validate_symmflow_endpoints

    root = Path(config["output_dir"]) / "fm"
    contract = contract_for(config, baseline)
    if runtime.stage_complete(config, "fm", contract):
        return True
    if device.type != "cuda":
        raise ValueError("Formal three-phase training requires the GPU smoke gate")
    audit = read_json(Path(config["output_dir"]) / "vq_audit/summary.json")
    if not audit["codebook_unchanged"] or audit["codec"] != contract["codec"]:
        raise ValueError("Frozen codec does not match the completed transfer audit")
    gate = read_json(Path(config["output_dir"]) / "smoke/fm-gpu.json")
    if gate["status"] != "passed" or gate["contract"] != source_contract(config, baseline):
        raise ValueError("Three-phase GPU smoke is missing or stale")
    batch_size = gate["batch_size"]
    tuned = batching.settings(config)
    effective = tuned["batch_size"] if tuned else config["fm"]["batch_size"]
    validation_batch = tuned["validation_batch_size"] if tuned else batch_size
    if effective % batch_size:
        raise ValueError("Physical batch must divide the effective batch")
    runtime.seed_all(config["seed"])
    training = PhaseLatentPairs(config, baseline, manifest, "train")
    validation = PhaseLatentPairs(config, baseline, manifest, "val")
    model, optimizer, scheduler, ema, trainer = build(config, manifest["pairs"], device)
    trainer.gradient_accumulation = effective // batch_size
    stream = runtime.TrainingBatches(training, batch_size=tuned["sampler_batch_size"] if tuned else batch_size,
                                      effective_batch=tuned["sampler_effective_batch"] if tuned else effective, seed=config["seed"],
                                      workers=config["runtime"]["loader_workers"], balanced=True, collate_fn=module.collate_pairs)
    state = {"step": 0, "best_loss": None, "best_endpoint": None}
    last, best_loss, best_endpoint = (root / name for name in ("last.pt", "best-loss.pt", "best-endpoint.pt"))
    if last.exists():
        saved = runtime.load_checkpoint(last, contract, device)
        state = restore(saved, model, optimizer, scheduler, ema, trainer, stream)
        del saved
    clock = runtime.UpdateClock(state["step"])
    target_updates = state["step"] + tuning.remaining_updates(config, tuning.reference_step(state), effective)

    def checkpoint(path):
        runtime.save_checkpoint(path, checkpoint_payload(contract, model, optimizer, scheduler, ema, trainer, state, stream))

    try:
        while tuning.reference_step(state) < config["fm"]["max_updates"]:
            previous = tuning.reference_step(state)
            if tuned:
                batches = tuning.next_batches(stream, effective, batch_size, tuning.pairs_to_next_gate(config, previous))
                metrics, reference = batching.update(trainer, batches, config, previous)
                state["reference_step"] = reference
            else:
                for _ in range(trainer.gradient_accumulation):
                    batch = next(stream)
                    metrics = trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
                reference = previous + 1
            if not metrics["optimizer_updated"] or any(not math.isfinite(metrics[key]) for key in ("loss", "loss_x", "loss_y", "gradient_norm")):
                raise FloatingPointError("Invalid three-phase optimizer update")
            state["step"] += 1
            step = state["step"]
            if step == 1 or step % config["runtime"]["log_interval"] == 0:
                runtime.log_event(config, "fm", "training", step=step, reference_step=reference,
                                  sampled_pairs=reference * config["fm"]["batch_size"], epoch=stream.epoch,
                                  batch_size=batch_size, effective_batch=effective, target_updates=target_updates,
                                  **metrics, **clock.metrics(step, target_updates))
                runtime.periodic_guard(config, contract)
            if tuning.crossed(previous, reference, config["fm"]["validation_interval"]):
                loader = runtime.evaluation_loader(validation, validation_batch, config["runtime"]["loader_workers"], module.collate_pairs)
                with runtime.fixed_rng(config["fm"]["validation_seed"]), ema.average_parameters(model.velocity_model):
                    metrics = validate_symmflow(model.velocity_model, model.condition_encoder, model.objective, loader,
                                               device=device, seed=config["fm"]["validation_seed"], precision="bf16")
                if state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
                    state["best_loss"] = metrics["loss"]
                    checkpoint(best_loss)
                runtime.log_event(config, "fm", "validation", step=step, reference_step=reference, **metrics)
            if tuning.crossed(previous, reference, config["fm"]["endpoint_interval"]):
                loader = runtime.evaluation_loader(validation, validation_batch, config["runtime"]["loader_workers"], module.collate_pairs)
                with runtime.fixed_rng(config["fm"]["validation_seed"]), ema.average_parameters(model.velocity_model):
                    metrics = validate_symmflow_endpoints(model.velocity_model, model.condition_encoder, loader, device=device,
                                                          seed=config["fm"]["validation_seed"], samples_per_pair=config["fm"]["endpoint_samples"],
                                                          steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"], precision="bf16")
                score = metrics["endpoint_candidate_mae"]
                if state["best_endpoint"] is None or score < state["best_endpoint"]:
                    state["best_endpoint"] = score
                    checkpoint(best_endpoint)
                runtime.log_event(config, "fm", "endpoint_validation", step=step, reference_step=reference, **metrics)
            if tuning.crossed(previous, reference, config["runtime"]["checkpoint_interval"]) or runtime.STOP_REQUESTED:
                checkpoint(last)
            if runtime.STOP_REQUESTED:
                return False
        checkpoint(last)
        if not best_endpoint.is_file():
            raise ValueError("No endpoint-selected three-phase checkpoint")
        runtime.stage_finished(config, "fm", contract, [last, best_loss, best_endpoint], updates=state["step"],
                               reference_updates=tuning.reference_step(state),
                               sampled_pairs=tuning.reference_step(state) * config["fm"]["batch_size"],
                               selection="fixed_noise_all_validation_pairs_phase_balanced_normalized_latent_mae")
        return True
    finally:
        stream.close()


def load_selected(config, baseline, manifest, device):
    contract = contract_for(config, baseline)
    if not runtime.stage_complete(config, "fm", contract):
        raise ValueError("Three-phase formal training is incomplete")
    saved = runtime.load_checkpoint(Path(config["output_dir"]) / "fm/best-endpoint.pt", contract, "cpu")
    model = ThreePhaseROI32Flow(config, manifest["pairs"]).to(device)
    model.load_state_dict(saved["model"])
    parameters = dict(model.velocity_model.named_parameters())
    with torch.no_grad():
        for name, value in saved["ema"]["shadow"].items():
            parameters[name].copy_(value.to(device))
    return model.eval().requires_grad_(False)

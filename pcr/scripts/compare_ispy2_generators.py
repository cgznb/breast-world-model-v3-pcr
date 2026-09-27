"""Compare frozen I-SPY2 generators under the existing full978 pCR policy."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BF = ROOT.parent / "MeWM-ISPY2-DCE0-BiFlowNet-RF"
sys.path.insert(0, str(BF))
OUT = ROOT / "results/ispy2_generator_comparison_20260911"
SCHEMA = "ispy2_four_generator_comparison_v1"
MODELS = ("biflow", "symmflow", "bridge025", "bridge050")
DEPTHS = ("T0", "T0-T1", "T0-T2", "T0-T3")
STEPS = 20
STRATEGIES = ("direct_t0", "adjacent_real", "rollout_generated_dce0_real_ser")

from scripts import generate_biflow_full978_trajectories as trajectories
from scripts.evaluate_generated_dce0_full978 import (
    _atomic_csv, _atomic_json, _atomic_torch_save, _cohort_inputs,
    _predict_checkpoint, _prefetched,
)
from scripts.evaluate_full978_frozen_mixed_biflow_trajectories import (
    _build_hybrid_volume, _canonical_split, _overlay_generated_test, _validate_checkpoint,
)
from src.biflow_trajectories import (
    build_trajectory_routes, index_registered_source_assets,
    shared_target_seed_schedule, compose_rollout_source_mri,
    previous_real_routes,
)
from src.data import load_all_splits
from src.metrics import compute_metrics, METRIC_KEYS
from src.mewm_data import create_registered_source


def context():
    config = trajectories._load_config(
        ROOT / "configs/mewm_ispy2_full978_locked102_biflow_trajectories.yaml"
    )
    ids, rows = trajectories._cohort_inputs(config)
    loaded, roi_cache = trajectories._load_world_inputs(config)
    text = trajectories._text_conditions(loaded, ids)
    routes = build_trajectory_routes(ids, rows, text)
    seeds = shared_target_seed_schedule(routes, base_seed=22026)
    routes["adjacent_real"] = previous_real_routes(routes["rollout_generated_dce0_real_ser"])
    assets = index_registered_source_assets(
        Path(config["cohort"]["selected_registered_manifest"])
        if Path(config["cohort"]["selected_registered_manifest"]).is_absolute()
        else ROOT / config["cohort"]["selected_registered_manifest"],
        [Path(p) for p in config["cohort"]["registered_source_manifests"]],
        patient_ids=ids,
    )
    norm = json.loads(Path(config["biflow"]["normalization_json"]).read_text())
    crops = json.loads(Path(config["biflow"]["crop_plans_json"]).read_text())
    loader = trajectories.RealSourceLoader(
        loaded=loaded, roi_cache=roi_cache, assets=assets, crop_plans=crops,
        channel_statistics=norm["channels"],
        target_spacing_xyz=config["biflow"]["strict_spacing_xyz"],
    )
    return config, ids, routes, seeds, loaded, roi_cache, loader


def model_root(model):
    return OUT / model


def decoded_root(model):
    if model == "biflow":
        return BF / "runs/ispy2_dce0_biflow_original_resume_no_early_stop_v2/generated_trajectories/full978_locked102_euler20_seed22026"
    return model_root(model)


def decoded_path(model, route):
    return decoded_root(model) / route.strategy / "decoded" / route.patient_id / f"T{route.target_stage}.pt"


def read_prediction(model, route, *, expected_seed=None):
    path = decoded_path(model, route)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    x = payload.get("prediction")
    if (
        payload.get("patient_id") != route.patient_id
        or payload.get("strategy") != route.strategy
        or payload.get("source_stage") != route.source_stage
        or payload.get("target_stage") != route.target_stage
        or payload.get("source_visit_id") != route.source_visit_id
        or payload.get("target_visit_id") != route.target_visit_id
        or payload.get("target_dce0_read") is not False
        or payload.get("solver_steps") != STEPS
        or (expected_seed is not None and payload.get("seed") != expected_seed)
        or ("source_dce0" in payload and payload["source_dce0"] != getattr(route, "source_dce0", "real"))
        or not isinstance(x, torch.Tensor)
        or x.shape != (1, 96, 256, 256)
        or x.dtype != torch.float16
        or not torch.isfinite(x).all()
    ):
        raise ValueError(f"Invalid decoded prediction: {path}")
    if model != "biflow" and (
        payload.get("model") != model or payload.get("schema") != SCHEMA
        or payload.get("candidate") != 0 or payload.get("solver") != "euler"
    ):
        raise ValueError("Generator identity or sampling contract differs")
    return x


def save_prediction(model, route, prediction, seed, step):
    x = prediction.detach().cpu().half().reshape(1, 96, 256, 256)
    if not torch.isfinite(x).all():
        raise ValueError("Generated image contains non-finite values")
    _atomic_torch_save(decoded_path(model, route), {
        "schema": SCHEMA, "model": model, "checkpoint_step": step,
        "patient_id": route.patient_id, "strategy": route.strategy,
        "source_stage": route.source_stage, "target_stage": route.target_stage,
        "source_visit_id": route.source_visit_id,
        "target_visit_id": route.target_visit_id,
        "target_dce0_read": False, "solver_steps": STEPS, "solver": "euler",
        "seed": seed, "candidate": 0, "prediction": x,
        "source_dce0": getattr(route, "source_dce0", "real"),
        "source_ser_used_by_generator": model != "symmflow",
    })


def snapshot_bridge(model):
    source = BF / f"runs/source_bridge_pilot_20260910/{model}/checkpoints/best.ckpt"
    destination = model_root(model) / "checkpoint.ckpt"
    if not destination.exists():
        if not source.exists():
            raise FileNotFoundError(f"{model} has no trained checkpoint yet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns
        ):
            temporary.unlink()
            raise RuntimeError("Training checkpoint changed while taking snapshot; retry")
        os.replace(temporary, destination)
    header = torch.load(destination, map_location="cpu", mmap=True, weights_only=False)
    contract = header["source_bridge_contract"]
    expected_scale = {"bridge025": 0.25, "bridge050": 0.50}[model]
    if (contract["experiment"]["family"] != "ispy2"
        or contract["experiment"]["noise_multiplier"] != expected_scale
        or contract["experiment"]["distribution"] != "source_gaussian"):
        raise ValueError("Bridge source checkpoint has the wrong experiment")
    step = int(header["global_step"])
    provenance_path = model_root(model) / "checkpoint_provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["optimizer_step"] != step:
            raise ValueError("Frozen bridge snapshot provenance differs")
        return header, contract
    training_result = source.parent.parent / "training_result.json"
    completed = training_result.exists() and json.loads(training_result.read_text()).get("status") == "completed"
    _atomic_json(provenance_path, {
        "model": model, "source_checkpoint": str(source),
        "snapshot": str(destination), "optimizer_step": step,
        "scheduled_steps": 40000, "intermediate_checkpoint": not completed,
        "training_complete_at_snapshot": completed,
        "selection": "minimum existing validation patient-macro latent MAE",
        "snapshot_bytes": destination.stat().st_size,
    })
    return header, contract


class BridgeGenerator:
    def __init__(self, model, config, device):
        from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config
        from mewm_ispy2.ispy2_biflow_workflow import _build_model
        from mewm_ispy2.ispy2_dce0_world_latents import ISPY2DCE0ContinuousLatentCache
        from mewm_ispy2.source_bridge import BridgeSystem, integrate
        header, contract = snapshot_bridge(model)
        base = load_ispy2_biflow_config(config["biflow"]["config"])
        self.latents = ISPY2DCE0ContinuousLatentCache(base.base.data.continuous_root)
        self.system = BridgeSystem(
            _build_model(base), family="ispy2", experiment_config=base,
            contract=contract, optimizer_factory=lambda _: None,
        )
        self.system.on_load_checkpoint(header)
        result = self.system.load_state_dict(header["state_dict"], strict=False)
        if result.unexpected_keys or any(
            not k.startswith("model.conditioner.text_tower.model.") for k in result.missing_keys
        ):
            raise ValueError("Bridge checkpoint restore is incomplete")
        self.step = int(header["global_step"])
        del header
        self.system.to(device).eval().requires_grad_(False)
        self.codec, self.decode = trajectories._load_vqgan(config, device)
        self.integrate = integrate
        self.device = device

    @torch.inference_mode()
    def predict(self, route, source, seed, generated_source=False):
        device = self.device
        if generated_source or route.source_visit_id not in self.latents.visit_ids:
            raw = self.codec.encode_continuous(source[:1][None].float().to(device))
            z = (2 * (raw - self.latents.codebook_min)
                 / (self.latents.codebook_max - self.latents.codebook_min) - 1)[:, None]
        else:
            z = self.latents.load(route.source_visit_id).to(device)[None, None]
        batch = {
            "source_mri": source[None].to(device),
            "clinical_text": [route.clinical_text], "treatment_text": [route.treatment_text],
            "delta_days": torch.tensor([route.delta_days], dtype=torch.float32, device=device),
            "target_stage": torch.tensor([route.target_stage], dtype=torch.long, device=device),
        }
        noise = torch.randn(z.shape, generator=torch.Generator(device=device).manual_seed(seed),
                            device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prepared = self.system.context(batch)
            endpoint = self.integrate(
                lambda state, t: self.system.velocity(state, t, prepared),
                z.float(), noise, steps=STEPS, **self.system.settings,
            )
            image = self.decode(self.codec, self.latents.denormalize(endpoint))
        return image[0, 0]


class SymmGenerator:
    def __init__(self, config, device, *, steps=STEPS):
        if type(steps) is not int or steps <= 0:
            raise ValueError("Sampling steps must be a positive integer")
        self.steps = steps
        sys.path.insert(0, "/path/to/local/ispy2-symmflow3d/src")
        from ispy2_symmflow.models import (
            ConditionSchema, StructuredConditionEncoder, build_velocity_model_from_config,
        )
        from ispy2_symmflow.models.mewm_vqgan import load_mewm_vqgan_codec
        from ispy2_symmflow.inference.sampler import SymmFlowSampler
        from ispy2_symmflow.training.datasets import pair_conditions
        from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config
        from mewm_ispy2.ispy2_dce0_world_latents import ISPY2DCE0ContinuousLatentCache
        base = load_ispy2_biflow_config(config["biflow"]["config"])
        path = OUT / "symmflow_reference/inference.pt"
        header = torch.load(path, map_location="cpu", weights_only=True)
        if (header.get("schema") != "symmflow_tensor_exact_inference_export_v1"
            or header.get("original_codec_binding_verified") is not True
            or header.get("original_statistics_binding_verified") is not True):
            raise ValueError("SymmFlow inference export contract differs")
        statistics = header["latent_statistics"]
        self.codec = load_mewm_vqgan_codec(
            base.base.data.vqgan_checkpoint,
            latent_mean=statistics["mean"], latent_std=statistics["std"],
        ).to(device).eval()
        self.velocity = build_velocity_model_from_config(header["velocity_config"])
        self.velocity.load_state_dict(header["velocity_state"], strict=True)
        self.velocity.to(device).eval().requires_grad_(False)
        self.encoder = StructuredConditionEncoder(ConditionSchema.from_dict(header["feature_schema"]))
        self.encoder.load_state_dict(header["condition_state"], strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.step = int(header["step"])
        if self.step != 100000 or header["sigma_min"] != 0:
            raise ValueError("SymmFlow checkpoint is not the completed clean-endpoint run")
        self.sampler = SymmFlowSampler(self.codec, self.velocity, sigma_min=0.0)
        self.latents = ISPY2DCE0ContinuousLatentCache(base.base.data.continuous_root)
        condition_path = OUT / "symmflow_reference/conditions.json"
        records = json.loads(condition_path.read_text())
        self.conditions = {r["pair_id"]: pair_conditions(r) for r in records}
        self.by_patient = {}
        for r in records:
            self.by_patient.setdefault(r["patient_id"], pair_conditions(r))
        self.device = device
        _atomic_json(model_root("symmflow") / "checkpoint_provenance.json", {
            "model": "symmflow", "source_checkpoint": str(path),
            "remote_source": "qingyuan:/path/to/local/ispy2-symmflow3d/outputs/mewm_all_pairs_5090/symmflow/symmflow_best.pt",
            "optimizer_step": self.step, "intermediate_checkpoint": False,
            "sampling_weights": "EMA velocity; checkpoint condition encoder",
            "selection": "minimum validation joint velocity loss",
        })
        del header

    @torch.inference_mode()
    def predict(self, route, source, seed, generated_source=False, *, return_latent=False):
        if generated_source or route.source_visit_id not in self.latents.visit_ids:
            z = self.codec.encode(source[:1][None].float().to(self.device), normalize=True)
        else:
            cached = self.latents.load(route.source_visit_id)
            raw = self.latents.denormalize(cached)[None].to(self.device)
            z = self.codec.normalize_latent(raw)
        pair_id = f"{route.patient_id}:T{route.source_stage}->T{route.target_stage}"
        cond = copy.deepcopy(self.conditions.get(pair_id, self.by_patient[route.patient_id]))
        cond.update(stage_i=f"T{route.source_stage}", stage_j=f"T{route.target_stage}",
                    delta_days=route.delta_days, interval_missing="no")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = self.encoder(cond, batch_size=1)
            batch = self.sampler.sample_forward_latent(
                z.float(), tokens, num_samples=1, seed=seed,
                steps=getattr(self, "steps", STEPS), solver="euler",
                **({"return_joint_state": True} if return_latent else {}),
            )
        if return_latent:
            return batch.samples[0, 0], batch.final_joint_states[0, :, :z.shape[1]].float().cpu()
        return batch.samples[0, 0]


def generate(model, strategies, limit, device):
    config, ids, routes, seeds, _, _, loader = context()
    if model == "biflow":
        for strategy in strategies:
            for route in routes[strategy]:
                read_prediction(model, route, expected_seed=seeds[route.target_key])
        return
    generator = (SymmGenerator(config, device) if model == "symmflow"
                 else BridgeGenerator(model, config, device))
    started = time.perf_counter()
    completed = 0
    for strategy in strategies:
        selected = routes[strategy]
        if limit:
            selected = selected[:limit]
        for route in selected:
            path = decoded_path(model, route)
            if path.exists():
                read_prediction(model, route, expected_seed=seeds[route.target_key])
                continue
            if strategy != "direct_t0" and route.source_stage == 0:
                direct = next(r for r in routes["direct_t0"] if r.target_key == route.target_key)
                image = read_prediction(model, direct, expected_seed=seeds[route.target_key])
            else:
                source, foreground, _ = loader.load(route.patient_id, route.source_stage)
                generated_source = route.source_dce0 == "generated"
                if generated_source:
                    prior = next(r for r in routes[strategy]
                                 if r.patient_id == route.patient_id and r.target_stage == route.source_stage)
                    source = compose_rollout_source_mri(
                        read_prediction(model, prior, expected_seed=seeds[prior.target_key]), source, foreground
                    )
                image = generator.predict(route, source, seeds[route.target_key], generated_source)
            save_prediction(model, route, image, seeds[route.target_key], generator.step)
            completed += 1
            print(f"{model} {strategy} {completed}: {route.patient_id} T{route.target_stage} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        valid = sum(decoded_path(model, r).exists() for r in routes[strategy])
        _atomic_json(model_root(model) / strategy / "generation_status.json", {
            "schema": SCHEMA, "model": model, "strategy": strategy,
            "expected": 296, "decoded_count": valid, "complete": valid == 296,
            "solver": "euler", "steps": STEPS, "candidate_count": 1,
            "checkpoint_step": generator.step,
            "source_policy": strategy,
            "source_ser_used_by_generator": model != "symmflow",
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        })


def extract(model, strategies, limit, device):
    from src.pillar import load_pillar, global_embedding
    from scripts.extract_pillar_mewm import validate_embedding
    config, _, routes, _, loaded, roi_cache, _ = context()
    base = yaml.safe_load((ROOT / "configs/mewm_ispy2_full978_locked102_table2.yaml").read_text())
    adapter, _, _, _, by_key = _cohort_inputs(base)
    source = create_registered_source(adapter)
    world = base["generated_dce0_test"]
    crops = json.loads(Path(world["crop_plans_json"]).read_text())
    stats = json.loads(Path(world["normalization_json"]).read_text())["channels"]["dce0"]
    local = {"trajectory": {"decoded_schema": trajectories.DECODED_SCHEMA if model == "biflow" else SCHEMA,
                             "solver_steps": STEPS}}
    pillar = load_pillar(device, revision=adapter.get("model_revision", "main"))
    for strategy in strategies:
        output = model_root(model) / strategy / "embeddings"
        tasks = []
        for route in routes[strategy][:limit or None]:
            destination = output / route.patient_id / f"{route.patient_id}_T{route.target_stage}.pt"
            if destination.exists():
                validate_embedding(destination)
                continue
            if strategy != "direct_t0" and route.source_stage == 0:
                direct_path = model_root(model) / "direct_t0/embeddings" / route.patient_id / destination.name
                if direct_path.exists():
                    direct = next(r for r in routes["direct_t0"] if r.target_key == route.target_key)
                    if not torch.equal(read_prediction(model, route), read_prediction(model, direct)):
                        raise ValueError("Shared T0-source candidates differ")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(direct_path, destination)
                    continue
            read_prediction(model, route)
            tasks.append((route.patient_id, route.target_stage, vars(route), decoded_path(model, route), destination))
        def build(task):
            return _build_hybrid_volume(
                task, config=local, base_config=base, by_key=by_key, source=source,
                loaded=loaded, roi_cache=roi_cache, crop_plans=crops,
                dce0_mean=float(stats["mean"]), dce0_std=float(stats["std"]),
                strict_spacing=tuple(world["strict_spacing_xyz"]),
            )
        started = time.perf_counter()
        for index, (task, volume) in enumerate(_prefetched(tasks, build, 2), 1):
            embedding = global_embedding(pillar, volume.to(device))
            if embedding.shape != (1152,) or not torch.isfinite(embedding).all():
                raise ValueError("Pillar embedding is invalid")
            _atomic_torch_save(task[-1], embedding)
            print(f"{model} {strategy} Pillar {index}/{len(tasks)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        count = len(list(output.glob("*/*.pt")))
        _atomic_json(output / "extraction_status.json", {
            "model": model, "strategy": strategy, "expected": 296,
            "embedding_count": count, "complete": count == 296,
            "future_pre": "generated_DCE0", "future_early_late": "real",
            "geometry_and_preprocessing": "existing full978 generated-DCE0 adapter",
        })


def checkpoint_path(depth, seed, fold):
    slug = depth.lower().replace("-", "_")
    if depth in ("T0", "T0-T1"):
        root = "mewm_ispy2_full978_locked102_all_depths_best200_fivefold"
        variant = "best200"
    elif depth == "T0-T2":
        root = "mewm_ispy2_full978_locked102_independent_cv_regularization_v2"
        variant = "layerwise_adamw"
    else:
        root = "mewm_ispy2_full978_locked102_t0_t3_layerwise_adamw_no_residual_10seed"
        variant = "layerwise_adamw"
    return ROOT / f"results/{root}/train/formal/{slug}/variant_{variant}/seed_{seed}/fold_{fold}/best.pt"


def pcr(models, strategies, device, output_dir=None, *, embedding_sources=None):
    base = yaml.safe_load((ROOT / "configs/mewm_ispy2_full978_locked102_table2.yaml").read_text())
    adapter, cohort, ids, _, _ = _cohort_inputs(base)
    real = load_all_splits(
        str(ROOT / adapter["embeddings_dir"]), str(cohort / "metadata_enriched.csv"),
        ids["train"], ids["val"], ids["test"],
    )["test"]
    canonical = ROOT / "results/mewm_ispy2_full978_locked102_final_hybrid_policy"
    prior = pd.read_csv(canonical / "all_test_predictions.csv", dtype={"patient_id": str})
    thresholds = pd.read_csv(canonical / "development_thresholds.csv").set_index("temporal_depth")["threshold"]
    sources = {("real", "real"): real}
    if embedding_sources is None:
        embedding_sources = {(model, strategy): model_root(model) / strategy / "embeddings"
                             for model in models if model != "biflow" for strategy in strategies}
        reuse_biflow = "biflow" in models
    else:
        reuse_biflow = False
    for key, directory in embedding_sources.items():
        sources[key] = _overlay_generated_test(real, Path(directory))
    output_rows = []
    fold_rows = []
    replay_rows = []
    for depth_index, depth in enumerate(DEPTHS, 1):
        current = {key: _canonical_split(split, depth_index) for key, split in sources.items()}
        for seed in range(42, 52):
            scores = {key: [] for key in sources}
            for fold in range(5):
                path = checkpoint_path(depth, seed, fold)
                header = torch.load(path, map_location="cpu", weights_only=False)
                _validate_checkpoint(header, path, {
                    "name": depth, "folds": 5,
                    "variant": "best200" if depth_index <= 2 else "layerwise_adamw",
                }, seed, fold)
                training_ids, validation_ids = set(header["train_ids"]), set(header["validation_ids"])
                if (training_ids & validation_ids or (training_ids | validation_ids) & set(ids["test"])
                    or training_ids | validation_ids != set(ids["train"]) | set(ids["val"])):
                    raise ValueError("pCR training/development/test patient boundary differs")
                for key, split in current.items():
                    prob = _predict_checkpoint(header, split, device)
                    scores[key].append(prob)
                    if key[0] != "real":
                        for pid, label, p in zip(real["pids"], real["labels"], prob):
                            fold_rows.append(dict(model=key[0], strategy=key[1], temporal_depth=depth,
                                                  seed=seed, fold=fold, patient_id=pid, label=int(label), probability=float(p)))
            for key, arrays in scores.items():
                mean = np.mean(np.stack(arrays), axis=0)
                if key[0] == "real":
                    ref = prior[(prior.temporal_depth == depth) & (prior.seed == seed)
                                & (prior.input_scheme == "real")].set_index("patient_id").loc[real["pids"]]
                    if not np.array_equal(ref.label.to_numpy(), real["labels"]):
                        raise ValueError("Real pCR cohort differs from frozen policy")
                    if not np.allclose(mean, ref.probability.to_numpy(), rtol=0, atol=1e-5):
                        raise ValueError(f"Frozen real pCR replay differs: {depth} seed {seed}")
                    replay_rows.append(dict(temporal_depth=depth, seed=seed,
                        maximum_probability_difference=float(np.max(np.abs(mean-ref.probability.to_numpy())))))
                    old_metrics = compute_metrics(real["labels"], ref.probability, threshold=float(thresholds[depth]))
                    new_metrics = compute_metrics(real["labels"], mean, threshold=float(thresholds[depth]))
                    if any(not np.isclose(old_metrics[k], new_metrics[k], atol=1e-12, rtol=0) for k in METRIC_KEYS):
                        raise ValueError("Real pCR metric replay changed")
                for pid, label, probability in zip(real["pids"], real["labels"], mean):
                    output_rows.append(dict(model=key[0], strategy=key[1], temporal_depth=depth,
                                            seed=seed, patient_id=pid, label=int(label), probability=float(probability)))
            print(f"pCR {depth} seed {seed}: frozen real-input replay passed", flush=True)
    if reuse_biflow:
        mapping = {"direct_t0": "direct_t0", "rollout_generated_dce0_real_ser": "rollout",
                   "adjacent_real": "adjacent_real"}
        for strategy in strategies:
            frame = prior[prior.input_scheme == mapping[strategy]]
            for row in frame.itertuples():
                label = "adjacent_real_historical" if strategy == "adjacent_real" else strategy
                output_rows.append(dict(model="biflow", strategy=label,
                                        temporal_depth=row.temporal_depth, seed=row.seed,
                                        patient_id=row.patient_id, label=row.label, probability=row.probability))
    predictions = pd.DataFrame(output_rows)
    metrics = []
    keys = ["model", "strategy", "temporal_depth", "seed"]
    for values, group in predictions.groupby(keys, sort=False):
        if len(group) != 102 or group.patient_id.nunique() != 102 or group.label.sum() != 32:
            raise ValueError("Incomplete pCR prediction group")
        for policy, cutoff in (("development_oof_bacc", float(thresholds[values[2]])), ("fixed_0.5", .5)):
            metrics.append(dict(zip(keys, values), threshold_policy=policy, threshold=cutoff,
                                **compute_metrics(group.label, group.probability, threshold=cutoff)))
    metric_frame = pd.DataFrame(metrics)
    summary = []
    groupkeys = ["model", "strategy", "temporal_depth", "threshold_policy"]
    for values, group in metric_frame.groupby(groupkeys, sort=False):
        row = dict(zip(groupkeys, values), n_seeds=len(group), patients=102, folds_per_seed=5)
        for metric in METRIC_KEYS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary.append(row)
    target = Path(output_dir) if output_dir is not None else OUT / "pcr"
    _atomic_csv(target / "predictions.csv", predictions)
    _atomic_csv(target / "fold_predictions.csv", pd.DataFrame(fold_rows))
    _atomic_csv(target / "metrics_per_seed.csv", metric_frame)
    _atomic_csv(target / "summary.csv", pd.DataFrame(summary))
    _atomic_csv(target / "real_replay_audit.csv", pd.DataFrame(replay_rows))
    _atomic_json(target / "audit.json", {"real_frozen_replay": "passed", "patients": 102,
                                        "positive_labels": 32, "models": models, "strategies": strategies,
                                        "effective_contiguous_visits_T0_T1_T2_T3":
                                            _canonical_split(real, 4)["masks"].sum(axis=0).astype(int).tolist(),
                                        "future_DCE0_only_replaced": True, "new_training": False,
                                        "biflow_adjacent_real_reference":
                                            ("historical Euler-20; seeds 2026/12026 and latest earlier Strict-A supplemental sources"
                                             if reuse_biflow else "not reused; supplied embeddings evaluated"),
                                        "embedding_sources": {f"{k[0]}:{k[1]}": str(v)
                                                              for k, v in embedding_sources.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "extract", "pcr"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["biflow", "symmflow"])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["direct_t0"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pcr-output", type=Path)
    parser.add_argument("--max-gpu-memory-gib", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.max_gpu_memory_gib is not None:
        if device.type != "cuda":
            parser.error("GPU memory limits require a CUDA device")
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = args.max_gpu_memory_gib * 2**30 / total
        if not 0 < fraction <= 1:
            parser.error("GPU memory limit must be positive and no larger than device memory")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    if args.stage == "pcr":
        pcr(args.models, args.strategies, device, args.pcr_output)
    else:
        if len(args.models) != 1:
            parser.error("generation/extraction require one model per process")
        (generate if args.stage == "generate" else extract)(args.models[0], args.strategies, args.limit, device)


if __name__ == "__main__":
    main()

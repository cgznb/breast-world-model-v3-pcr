"""Matched BiFlow/SymmFlow Euler-step sweep with bounded decoded-image storage."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import compare_ispy2_generators as base
from scripts import report_ispy2_generators as metrics
from scripts.extract_pillar_mewm import validate_embedding
from src.biflow_trajectories import validate_locked102_routes

ROOT = base.ROOT
OUT = ROOT / "results/ispy2_biflow_symmflow_steps_20260911"
MODELS = ("biflow", "symmflow")
STEPS = (2, 10, 20, 50)
POLICIES = base.STRATEGIES
ROLLOUT = "rollout_generated_dce0_real_ser"
SCHEMA = "ispy2_matched_euler_sweep_v1"
METRIC_COLUMNS = metrics.METRICS + metrics.CHANGE_METRICS


def setting_root(model, steps):
    return OUT / model / f"euler_{steps}"


def reused_setting(model, steps):
    return model == "symmflow" and steps == 20


def artifact(model, steps, route, kind):
    if reused_setting(model, steps) and kind in ("decoded", "embeddings"):
        root = base.model_root(model)
    else:
        root = setting_root(model, steps)
    if kind == "embeddings":
        return root / route.strategy / kind / route.patient_id / f"{route.patient_id}_T{route.target_stage}.pt"
    return root / route.strategy / kind / route.patient_id / f"T{route.target_stage}.pt"


def identity(model, steps, route, seed):
    return dict(schema=SCHEMA, model=model, solver="euler", solver_steps=steps,
                checkpoint_step=97104 if model == "biflow" else 100000,
                patient_id=route.patient_id, strategy=route.strategy,
                source_stage=route.source_stage, target_stage=route.target_stage,
                source_visit_id=route.source_visit_id, target_visit_id=route.target_visit_id,
                source_dce0=route.source_dce0, target_dce0_read=False, seed=seed, candidate=0)


def prediction(model, steps, route, seed):
    if reused_setting(model, steps):
        return base.read_prediction(model, route, expected_seed=seed)
    path = artifact(model, steps, route, "decoded")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = identity(model, steps, route, seed)
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Decoded identity or sampling contract differs: {path}")
    value = payload.get("prediction")
    if (not isinstance(value, torch.Tensor) or value.shape != (1, 96, 256, 256)
        or value.dtype != torch.float16 or not torch.isfinite(value).all()):
        raise ValueError(f"Invalid decoded image: {path}")
    return value


def read_latent(model, steps, route, seed):
    path = artifact(model, steps, route, "latents")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = identity(model, steps, route, seed)
    normalization = "codebook_minmax" if model == "biflow" else "symmflow_standardized"
    shape = (1, 1, 8, 24, 64, 64) if model == "biflow" else (1, 8, 24, 64, 64)
    dtype = torch.float16 if model == "biflow" else torch.float32
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Latent provenance differs: {path}")
    latent = payload.get("endpoint_latent")
    if (payload.get("normalization") != normalization or not isinstance(latent, torch.Tensor)
        or latent.shape != shape or latent.dtype != dtype or not torch.isfinite(latent).all()):
        raise ValueError(f"Invalid retained endpoint latent: {path}")
    return latent


def prepare():
    _, ids, routes, seeds, _, _, _ = base.context()
    validate_locked102_routes(ids, {p: routes[p] for p in ("direct_t0", ROLLOUT)})
    previous = json.loads((base.OUT / "figures/direct_t0/display_selection.json").read_text())["patients"]
    complete = sorted(pid for pid in ids if sum(r.patient_id == pid for r in routes["direct_t0"]) == 3)
    remaining = [pid for pid in complete if pid not in previous]
    selected = sorted(previous + np.random.default_rng(20260911).choice(remaining, 17, replace=False).tolist())
    document = dict(schema=SCHEMA, patients=102, targets=296, models=list(MODELS),
                    steps=list(STEPS), policies=list(POLICIES), solver="euler", candidates=1,
                    base_seed=22026, display_patients=selected, display_seed=20260911,
                    display_selection="previous three cases plus 17 seeded random complete-visit patients",
                    display_uses_outcomes_or_errors=False, display_slice=48,
                    display_window="real T0 foreground percentiles 0.5 to 99.5",
                    image_regions="fixed real T0 and target, common across models/steps/policies",
                    pcr_policy="previous frozen final hybrid policy, 200 checkpoints, seeds 42-51",
                    future_early_late="real", rollout_foreground="real previous visit",
                    biflow_source="DCE0 plus real SER", symmflow_source="DCE0, no SER",
                    decoded_retention="20 selected patients; other owned decoded intermediates pruned after verification",
                    latent_retention="all newly generated targets", biflow_batch_size=1,
                    biflow_euler20="fresh matched replay; historical outputs preserved",
                    symmflow_euler20="reuse verified existing images and embeddings")
    destination = OUT / "protocol.json"
    if destination.exists() and json.loads(destination.read_text()) != document:
        raise ValueError("Existing sweep protocol differs")
    base._atomic_json(destination, document)
    route_rows = [dict(model=model, steps=steps, **r.public_payload(seed=seeds[r.target_key]))
                  for model in MODELS for steps in STEPS for p in POLICIES for r in routes[p]]
    base._atomic_csv(OUT / "routes.csv", pd.DataFrame(route_rows))
    return document


class BiFlowGenerator:
    def __init__(self, config, device, steps):
        self.system, self.latents, self.integrate = base.trajectories._load_biflow_system(config, device)
        self.system.eval().requires_grad_(False)
        self.codec, self.decode = base.trajectories._load_vqgan(config, device)
        self.device, self.steps = device, steps

    @torch.inference_mode()
    def predict(self, route, source, seed):
        device = self.device
        noise = base.trajectories._noise_for_route(seed, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            endpoint = self.integrate(
                self.system.model, source_mri=source[None].to(device),
                clinical_text=[route.clinical_text], treatment_text=[route.treatment_text],
                delta_days=torch.tensor([route.delta_days], dtype=torch.float32, device=device),
                target_stage=torch.tensor([route.target_stage], dtype=torch.long, device=device),
                solver_steps=self.steps, noise=noise)
            # Preserve the original BiFlow saved-latent rounding before VQ decoding.
            latent = endpoint.cpu().half()
            image = self.decode(self.codec, self.latents.denormalize(latent.float().to(device)))
        return image[0, 0], latent


def generate(model, steps, device, limit=0):
    config, ids, routes, seeds, _, _, loader = base.context()
    indexed = {p: {r.target_key: r for r in routes[p]} for p in POLICIES}
    root = setting_root(model, steps)
    if (root / "complete.json").exists():
        return
    if reused_setting(model, steps):
        for p in POLICIES:
            for route in routes[p]:
                prediction(model, steps, route, seeds[route.target_key])
        base._atomic_json(root / "generation.json", dict(complete=True, reused=True, targets_per_policy=296))
        return
    generator = (BiFlowGenerator(config, device, steps) if model == "biflow" else
                 base.SymmGenerator(config, device, steps=steps))
    started = time.monotonic()
    done = 0
    for policy in POLICIES:
        for route in routes[policy][:limit or None]:
            path = artifact(model, steps, route, "decoded")
            seed = seeds[route.target_key]
            if path.exists():
                prediction(model, steps, route, seed)
                continue
            if policy != "direct_t0" and route.source_stage == 0:
                direct = indexed["direct_t0"][route.target_key]
                value = prediction(model, steps, direct, seed)
                original = torch.load(artifact(model, steps, direct, "latents"), weights_only=True)
                latent = original["endpoint_latent"]
            else:
                source, foreground, provenance = loader.load(route.patient_id, route.source_stage)
                if route.source_dce0 == "generated":
                    previous = indexed[policy][(route.patient_id, route.source_stage)]
                    source = base.compose_rollout_source_mri(
                        prediction(model, steps, previous, seeds[previous.target_key]), source, foreground)
                if model == "biflow":
                    value, latent = generator.predict(route, source, seed)
                else:
                    value, latent = generator.predict(route, source, seed,
                        generated_source=route.source_dce0 == "generated", return_latent=True)
                value = value.detach().cpu().half().reshape(1, 96, 256, 256)
            if not torch.isfinite(value).all() or not torch.isfinite(latent).all():
                raise ValueError("Nonfinite generator output")
            meta = identity(model, steps, route, seed)
            base._atomic_torch_save(artifact(model, steps, route, "latents"),
                dict(meta, endpoint_latent=latent.detach().cpu(),
                     normalization="codebook_minmax" if model == "biflow" else "symmflow_standardized"))
            base._atomic_torch_save(path, dict(meta, prediction=value))
            done += 1
            if done % 10 == 0:
                print(f"{model} Euler-{steps} {policy}: {done} outputs, {time.monotonic()-started:.1f}s", flush=True)
                base._atomic_json(root / "generation_progress.json", dict(
                    model=model, steps=steps, policy=policy, outputs_this_run=done,
                    elapsed_seconds=time.monotonic()-started))
    complete = all(artifact(model, steps, r, "decoded").exists() for p in POLICIES for r in routes[p])
    base._atomic_json(root / "generation.json", dict(complete=complete, model=model, steps=steps,
        targets_per_policy=296, elapsed_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))


def extract(model, steps, device, limit=0):
    from src.pillar import load_pillar, global_embedding
    root = setting_root(model, steps)
    if (root / "complete.json").exists():
        return
    if reused_setting(model, steps):
        _, _, routes, _, _, _, _ = base.context()
        for p in POLICIES:
            for route in routes[p]:
                validate_embedding(artifact(model, steps, route, "embeddings"))
        base._atomic_json(root / "extraction.json", dict(complete=True, reused=True, embeddings=888))
        return
    _, _, routes, seeds, loaded, roi_cache, _ = base.context()
    config = yaml.safe_load((ROOT / "configs/mewm_ispy2_full978_locked102_table2.yaml").read_text())
    adapter, _, _, _, by_key = base._cohort_inputs(config)
    source = base.create_registered_source(adapter)
    world = config["generated_dce0_test"]
    crops = json.loads(Path(world["crop_plans_json"]).read_text())
    stats = json.loads(Path(world["normalization_json"]).read_text())["channels"]["dce0"]
    local = {"trajectory": {"decoded_schema": SCHEMA, "solver_steps": steps}}
    pillar = load_pillar(device, revision=adapter.get("model_revision", "main"))
    for policy in POLICIES:
        tasks = []
        for route in routes[policy][:limit or None]:
            dest = artifact(model, steps, route, "embeddings")
            if dest.exists():
                validate_embedding(dest)
                continue
            image = prediction(model, steps, route, seeds[route.target_key])
            if policy != "direct_t0" and route.source_stage == 0:
                direct = next(r for r in routes["direct_t0"] if r.target_key == route.target_key)
                if not torch.equal(image, prediction(model, steps, direct, seeds[route.target_key])):
                    raise ValueError("Shared T0-source images differ")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(artifact(model, steps, direct, "embeddings"), dest)
            else:
                tasks.append((route.patient_id, route.target_stage, vars(route),
                              artifact(model, steps, route, "decoded"), dest))
        def build(task):
            return base._build_hybrid_volume(task, config=local, base_config=config,
                by_key=by_key, source=source, loaded=loaded, roi_cache=roi_cache, crop_plans=crops,
                dce0_mean=float(stats["mean"]), dce0_std=float(stats["std"]),
                strict_spacing=tuple(world["strict_spacing_xyz"]))
        for index, (task, volume) in enumerate(base._prefetched(tasks, build, 2), 1):
            embedding = global_embedding(pillar, volume.to(device))
            if embedding.shape != (1152,) or embedding.dtype != torch.float32 or not torch.isfinite(embedding).all():
                raise ValueError("Invalid Pillar embedding")
            base._atomic_torch_save(task[-1], embedding)
            if index % 10 == 0 or index == len(tasks):
                print(f"{model} Euler-{steps} {policy}: Pillar {index}/{len(tasks)}", flush=True)
    count = sum(artifact(model, steps, r, "embeddings").exists() for p in POLICIES for r in routes[p])
    base._atomic_json(root / "extraction.json", dict(complete=count == 888, embeddings=count))


def validate_metric_group(frame, model, steps, routes):
    direct = routes["direct_t0"]
    required = {"schema", "model", "steps", "strategy", "patient_id", "target_stage",
                "generation_source_stage", "metric_reference_stage", "region",
                "voxel_count", "ssim_center_count", *METRIC_COLUMNS}
    if (not required <= set(frame) or len(frame) != 9
        or set(frame.schema) != {SCHEMA} or set(frame.model) != {model}
        or set(frame.steps) != {steps} or set(frame.strategy) != set(POLICIES)
        or set(frame.patient_id) != {direct.patient_id}
        or set(frame.target_stage) != {direct.target_stage}
        or set(frame.metric_reference_stage) != {0}
        or frame.duplicated(["strategy", "region"]).any()):
        raise ValueError("Invalid image metric shard identity or sampling contract")
    for policy, group in frame.groupby("strategy"):
        if (set(group.region) != metrics.REGIONS
            or set(group.generation_source_stage) != {routes[policy].source_stage}):
            raise ValueError("Image metric source or regions differ")
    if not frame.groupby("region")[["voxel_count", "ssim_center_count"]].nunique().eq(1).all().all():
        raise ValueError("Image metric supports differ across policies")


def image_metrics(model, steps, workers):
    root = setting_root(model, steps)
    if not json.loads((root / "generation.json").read_text()).get("complete"):
        raise ValueError("Image metrics require completed generation")
    directory = root / "image_metrics"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".metrics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _image_metrics(model, steps, workers)


def _image_metrics(model, steps, workers):
    root = setting_root(model, steps)
    _, _, routes, seeds, loaded, roi, loader = base.context()
    indexed = {p: {r.target_key: r for r in routes[p]} for p in POLICIES}
    if reused_setting(model, steps):
        previous = pd.read_csv(base.OUT / "source_policies/image_metrics/endpoint_metrics.csv")
        endpoint = previous[previous.model.isin(POLICIES)].rename(columns={"model": "strategy"})
        endpoint = endpoint.assign(model=model, steps=steps, schema=SCHEMA)
    else:
        def one(direct):
            path = root / "image_metrics/shards" / direct.patient_id / f"T{direct.target_stage}.csv"
            group_routes = {p: indexed[p][direct.target_key] for p in POLICIES}
            if path.exists():
                frame = pd.read_csv(path)
                validate_metric_group(frame, model, steps, group_routes)
                return frame
            source, sf, _ = loader.load(direct.patient_id, 0)
            target, tf, _ = loader.load(direct.patient_id, direct.target_stage)
            masks = metrics.region_masks(sf.numpy(), tf.numpy(),
                metrics.lesion_mask(loaded, roi, direct.patient_id, 0),
                metrics.lesion_mask(loaded, roi, direct.patient_id, direct.target_stage))
            centers = metrics.comparison_ssim_centers(masks["common_foreground"])
            rows = []
            for policy in POLICIES:
                route = indexed[policy][direct.target_key]
                if policy != "direct_t0" and route.source_stage == 0:
                    rows.extend(dict(row, strategy=policy) for row in tuple(rows) if row["strategy"] == "direct_t0")
                    continue
                value = prediction(model, steps, route, seeds[route.target_key])
                scores = metrics.comparison_metrics(value, target[:1], masks, centers)
                for region, score in scores.items():
                    change = metrics.change_metrics(value[0].float(), source[0].float(), target[0].float(), masks[region])
                    rows.append(dict(schema=SCHEMA, model=model, steps=steps, strategy=policy, patient_id=route.patient_id,
                        target_stage=route.target_stage, generation_source_stage=route.source_stage,
                        metric_reference_stage=0, region=region, **score, **change))
            frame = pd.DataFrame(rows)
            validate_metric_group(frame, model, steps, group_routes)
            base._atomic_csv(path, frame)
            return frame
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = []
            for index, frame in enumerate(pool.map(one, routes["direct_t0"]), 1):
                frames.append(frame)
                if index % 20 == 0 or index == 296:
                    print(f"{model} Euler-{steps}: image metrics {index}/296", flush=True)
        endpoint = pd.concat(frames, ignore_index=True)
    groups = {key: group for key, group in endpoint.groupby(["patient_id", "target_stage"])}
    if set(groups) != set(indexed["direct_t0"]):
        raise ValueError("Image metric cohort differs")
    for key, group in groups.items():
        validate_metric_group(group, model, steps, {p: indexed[p][key] for p in POLICIES})
    base._atomic_csv(root / "image_metrics/endpoint_metrics.csv", endpoint)
    patient = endpoint.groupby(["model", "steps", "strategy", "region", "patient_id"])[list(METRIC_COLUMNS)].mean().reset_index()
    base._atomic_csv(root / "image_metrics/patient_metrics.csv", patient)
    for name, frame, grouping in (
        ("summary", patient, ["model", "steps", "strategy", "region"]),
        ("by_target_stage", endpoint, ["model", "steps", "strategy", "region", "target_stage"])):
        summary = frame.groupby(grouping)[list(METRIC_COLUMNS)].agg(["mean", "std", "count"])
        summary.columns = ["_".join(column) for column in summary.columns]
        base._atomic_csv(root / "image_metrics" / f"{name}.csv", summary.reset_index())


def evaluate_pcr(model, steps):
    directories = {(model, policy): (base.model_root(model) if reused_setting(model, steps) else
                    setting_root(model, steps)) / policy / "embeddings" for policy in POLICIES}
    base.pcr([model], list(POLICIES), torch.device("cpu"), setting_root(model, steps) / "pcr",
             embedding_sources=directories)


def audit(model, steps):
    root = setting_root(model, steps)
    _, ids, routes, seeds, _, _, _ = base.context()
    generated = json.loads((root / "generation.json").read_text())
    extracted = json.loads((root / "extraction.json").read_text())
    if not generated["complete"] or not extracted["complete"]:
        raise ValueError("Incomplete generation or extraction")
    shared = Counter()
    for policy in POLICIES:
        for route in routes[policy]:
            value = prediction(model, steps, route, seeds[route.target_key])
            if not reused_setting(model, steps):
                read_latent(model, steps, route, seeds[route.target_key])
            embedding = validate_embedding(artifact(model, steps, route, "embeddings"))
            if embedding.dtype != torch.float32:
                raise ValueError("Embedding dtype differs")
            if policy != "direct_t0" and route.source_stage == 0:
                direct = next(r for r in routes["direct_t0"] if r.target_key == route.target_key)
                if (not torch.equal(value, prediction(model, steps, direct, seeds[route.target_key]))
                    or not torch.equal(embedding, validate_embedding(artifact(model, steps, direct, "embeddings")))):
                    raise ValueError("Shared T0-source images or embeddings differ")
                shared[policy] += 1
    if dict(shared) != {"adjacent_real": 102, ROLLOUT: 102}:
        raise ValueError("Shared target inventory differs")
    endpoint = pd.read_csv(root / "image_metrics/endpoint_metrics.csv")
    indexed = {p: {r.target_key: r for r in routes[p]} for p in POLICIES}
    groups = {key: group for key, group in endpoint.groupby(["patient_id", "target_stage"])}
    if set(groups) != set(indexed["direct_t0"]):
        raise ValueError("Image metric target inventory differs")
    for key, group in groups.items():
        validate_metric_group(group, model, steps, {p: indexed[p][key] for p in POLICIES})
    pcr = pd.read_csv(root / "pcr/predictions.csv")
    expected_groups = {(m, p, depth, seed) for m, p in [("real", "real"), *[(model, p) for p in POLICIES]]
                       for depth in base.DEPTHS for seed in range(42, 52)}
    if (set(pcr.groupby(["model", "strategy", "temporal_depth", "seed"]).groups) != expected_groups
        or pcr.duplicated(["model", "strategy", "temporal_depth", "seed", "patient_id"]).any()
        or not np.isfinite(pcr.probability).all()):
        raise ValueError("pCR prediction group inventory differs")
    for _, group in pcr.groupby(["model", "strategy", "temporal_depth", "seed"]):
        if len(group) != 102 or set(group.patient_id) != set(ids) or group.label.sum() != 32:
            raise ValueError("pCR cohort changed")
    gen = pcr[pcr.model == model]
    for depth in ("T0", "T0-T1"):
        pivot = gen[gen.temporal_depth == depth].pivot(index=["patient_id", "seed"], columns="strategy", values="probability")
        if not pivot.eq(pivot.direct_t0, axis=0).all().all():
            raise ValueError("Early pCR outputs differ across policies")
    base._atomic_json(root / "audit.json", dict(status="passed", model=model, steps=steps,
        patients=102, targets_per_policy=296, shared_targets=dict(shared), image_metric_rows=len(endpoint),
        frozen_real_replay="passed", future_DCE0_only_replaced=True,
        source_foreground="real previous visit for rollout", source_ser=model == "biflow"))


def prune(model, steps):
    root = setting_root(model, steps)
    audit_record = json.loads((root / "audit.json").read_text())
    if (audit_record.get("status") != "passed" or audit_record.get("model") != model
        or audit_record.get("steps") != steps):
        raise ValueError("Pruning requires completed verification")
    document = json.loads((OUT / "protocol.json").read_text())
    retained = set(document["display_patients"])
    removal_paths = []
    if not reused_setting(model, steps):
        _, _, routes, seeds, _, _, _ = base.context()
        for policy in POLICIES:
            for route in routes[policy]:
                if route.patient_id in retained:
                    continue
                path = artifact(model, steps, route, "decoded")
                read_latent(model, steps, route, seeds[route.target_key])
                if path.exists():
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError("Refusing to prune an external decoded image")
                    removal_paths.append(path)
    for path in removal_paths:
        path.unlink()
    base._atomic_json(root / "complete.json", dict(status="completed", model=model, steps=steps,
        updated_utc=datetime.now(timezone.utc).isoformat(), decoded_intermediates_removed=len(removal_paths),
        all_embeddings_and_metrics_retained=True, selected_patients=20,
        all_new_endpoint_latents_retained=not reused_setting(model, steps)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "extract", "images", "pcr", "audit", "prune"))
    parser.add_argument("--model", choices=MODELS, default="biflow")
    parser.add_argument("--steps", type=int, choices=STEPS, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-gib", type=float, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4 if args.stage in ("generate", "extract", "pcr") else 2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare()
        return
    if args.stage == "images":
        image_metrics(args.model, args.steps, args.workers)
        return
    root = setting_root(args.model, args.steps)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".stage.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = torch.device(args.device)
    if args.stage in ("generate", "extract"):
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = args.memory_gib * 2**30 / total
        if not 0 < fraction <= 1:
            parser.error("Invalid CUDA memory cap")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        {"generate": generate, "extract": extract}[args.stage](args.model, args.steps, device, args.limit)
    else:
        {"pcr": evaluate_pcr, "audit": audit, "prune": prune}[args.stage](args.model, args.steps)


if __name__ == "__main__":
    main()

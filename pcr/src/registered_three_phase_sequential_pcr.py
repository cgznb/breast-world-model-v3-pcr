"""Matched latent rollout and real-predecessor forecasts with frozen pCR heads."""

from __future__ import annotations

import gc
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _canonical_split, _prediction_frame
from src import registered_three_phase_generated_pcr as direct
from src.data import EmbStore
from src.first_post_pcr_data import disk_gate, identity, now, public, read_json, repo_path, save_tensor, validate_embedding, write_json
from src.single_phase_repeat_pcr import pillar_identity, unchanged_json
from src.tdn import TDN

original = direct.original
SCHEMA = "registered_three_phase_sequential_pcr_v1"
MODES = ("rollout", "previous_real")
LATENT_SHAPE = (4, 24, 8, 32, 32)


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if (cfg["schema"] != SCHEMA or cfg["draws"] != 4 or cfg["sampling_steps"] != 25
            or cfg["solver"] != "heun" or cfg["rollout_state"] != "normalized_continuous_latent_without_reencoding"
            or cfg["support_policy"] != "fixed_real_T0_per_phase_foreground"
            or cfg["time_policy"] != "recorded_adjacent_followup_intervals"):
        raise ValueError("Unexpected sequential pCR protocol")
    for key in ("output_dir", "direct_config"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    cfg["config_path"] = str(path)
    return cfg


def progress(cfg, stage, **fields):
    record = {"schema": SCHEMA, "stage": stage, "pid": os.getpid(), "updated_utc": now(), **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", record)
    print(json.dumps(public(record), allow_nan=False), flush=True)


def adjacent_routes(direct_routes, manifest):
    pairs = {(r["patient_id"], r["earlier_stage"], r["later_stage"]): r
             for r in manifest["pairs"] if r["split"] == "val"}
    result, previous = [], {}
    for route in direct_routes:
        pid, tp = route["patient_id"], route["timepoint"]
        key = (pid, f"T{tp - 1}", f"T{tp}")
        if key not in pairs or (tp > 1 and (pid, tp - 1) not in previous):
            raise ValueError("Sequential evaluation requires a complete adjacent prefix")
        record = {k: pairs[key][k] for k in route["record"]}
        if (record["earlier_visit_id"] != f"{pid}:T{tp - 1}" or record["later_visit_id"] != f"{pid}:T{tp}"
                or record["baseline_clinical"] != route["record"]["baseline_clinical"]
                or record["treatment"] != route["record"]["treatment"]
                or record["delta_days"] + previous.get((pid, tp - 1), 0) != route["record"]["delta_days"]):
            raise ValueError("Adjacent conditions disagree with the direct comparison")
        previous[pid, tp] = route["record"]["delta_days"]
        result.append({**route, "record": record, "elapsed_days_from_T0": route["record"]["delta_days"]})
    return result


def prepare(cfg):
    old_cfg = direct.load_config(cfg["direct_config"])
    source_cfg = original.load_config(old_cfg["source_pcr_config"])
    from mewm_ispy2.registered_three_phase_data import load_config as load_world
    from mewm_ispy2.registered_roi32_runtime import verify_contract_files

    old_root, output = Path(old_cfg["output_dir"]), Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    if output == old_root or not read_json(old_root / "COMPLETE.json")["passed"]:
        raise ValueError("Requires a separate output and completed direct-T0 comparison")
    previous_protocol = read_json(old_root / "protocol.json")
    for section in ("inputs", "sources"):
        for path, expected in previous_protocol[section].items():
            if identity(path) != expected:
                raise ValueError("The completed direct-T0 inputs or sources changed")
    refs = previous_protocol["classifiers"]
    for ref in refs:
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Frozen classifier changed")
    pcr = Path(source_cfg["output_dir"])
    contract = read_json(pcr / "input_contract.json")
    for section in ("runtime_sources", "input_sources", "crop_reports"):
        for path, expected in contract[section].items():
            if identity(path) != expected:
                raise ValueError("Original pCR input contract changed")
    if pillar_identity() != contract["pillar_files"]:
        raise ValueError("Frozen Pillar files changed")
    world, baseline = load_world(source_cfg["world_config"])
    world_root = Path(world["output_dir"])
    for stage in ("fm", "latents"):
        marker = read_json(world_root / stage / "COMPLETE.json")
        if marker["status"] != "passed":
            raise ValueError("Incomplete generator stage")
        verify_contract_files(marker["contract"])
    manifest = read_json(world_root / "inventory.json")
    cohort = read_json(pcr / "cohort.json")
    direct_routes = read_json(old_root / "routes.json")["routes"]
    if any(not direct.case_complete(old_cfg, route) for route in direct_routes):
        raise ValueError("Completed direct-T0 features are missing")
    routes = adjacent_routes(direct_routes, manifest)
    counts = Counter(r["timepoint"] for r in routes)
    if dict(counts) != {1: 100, 2: 92, 3: 84} or len(refs) != 200:
        raise ValueError("Comparison cohort changed")
    files = [Path(__file__), repo_path("scripts/evaluate_registered_three_phase_sequential_pcr.py"),
             Path(direct.__file__), Path(cfg["config_path"])]
    inputs = [old_root / name for name in ("protocol.json", "routes.json", "COMPLETE.json",
              "evaluation/fold_predictions.csv", "evaluation/predictions.csv")]
    protocol = {"schema": SCHEMA, "config": cfg, "sources": {str(p): identity(p) for p in files},
                "inputs": {str(p): identity(p) for p in inputs}, "classifiers": refs,
                "runtime_versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
                "routes_per_arm": len(routes), "shared_T1_visits": counts[1],
                "new_future_visits_per_arm": counts[2] + counts[3], "patients": len(cohort["split"]["val"]),
                "full_prefix_patients": previous_protocol["full_prefix_patients"],
                "noise": "same_four_per_target_noise_draws_as_direct_T0; paired_across_modes",
                "rollout": "real_T0_then_same_draw_previous_generated_continuous_latent; no_reencoding_or_averaging",
                "previous_real": "each_target_generated_from_its_real_adjacent_predecessor",
                "pcr_sequences": "real_T0_plus_all_generated_future_visits_in_each_arm",
                "support": "identical_real_T0_foreground_for_every_mode_and_future_visit",
                "ensemble": "mean_pCR_probability_over_four_complete_draw_sequences",
                "timing_and_availability": "retrospective_recorded_adjacent_intervals_and_same_observed_prefix",
                "test_role": "historical_holdout_used_for_generator_validation",
                "selection": "no_model_threshold_noise_or_method_selection_on_holdout"}
    unchanged_json(output / "protocol.json", protocol)
    unchanged_json(output / "routes.json", {"routes": routes})
    progress(cfg, "prepared", patients=102, shared_T1=100, new_triplets_per_arm=176, new_volumes=1408)
    return {"old_cfg": old_cfg, "source_cfg": source_cfg, "world": world, "baseline": baseline,
            "manifest": manifest, "cohort": cohort, "routes": routes, "refs": refs}


def canonical_mode(mode, route):
    if mode not in MODES:
        raise ValueError("Unknown generation mode")
    return "shared_T1" if route["timepoint"] == 1 else mode


def case_path(cfg, mode, route):
    return Path(cfg["output_dir"]) / canonical_mode(mode, route) / "cases" / (route["key"] + ".json")


def case_record(cfg, mode, route):
    path = case_path(cfg, mode, route)
    if not path.is_file():
        return None
    record = read_json(path)
    if (record["route"] != route or record["mode"] != canonical_mode(mode, route)
            or record["protocol"] != identity(Path(cfg["output_dir"]) / "protocol.json")):
        raise ValueError("Sequential case protocol changed")
    for artifact in record["artifacts"]:
        if identity(artifact["path"]) != artifact["identity"]:
            raise ValueError("Sequential generated artifact changed")
    return record


def source_batch(mode, route, t0, previous_generated, read_real, draws):
    if mode not in MODES:
        raise ValueError("Unknown generation mode")
    if route["timepoint"] == 1:
        return t0.repeat(draws, 1, 1, 1, 1)
    if mode == "rollout":
        if previous_generated is None or previous_generated.shape != (draws, *t0.shape[1:]):
            raise ValueError("Rollout must preserve each previous generated draw")
        return previous_generated
    source = read_real(route["record"]["earlier_visit_id"])[None].to(t0.device)
    return source.repeat(draws, 1, 1, 1, 1)


def sampling_noise(source, seed):
    generator = torch.Generator(device=source.device).manual_seed(seed)
    return torch.cat([torch.randn(source[:1].shape, generator=generator, device=source.device)
                      for _ in range(len(source))])


@torch.inference_mode()
def generate(cfg, context):
    from mewm_ispy2 import registered_roi32_runtime as runtime
    from mewm_ispy2 import registered_roi32_vq as vq
    from mewm_ispy2.registered_roi32_data import save_npz, visit_filename
    from mewm_ispy2.registered_roi32_latents import latent_filename
    from mewm_ispy2.registered_roi32_fm import bridge
    from mewm_ispy2.registered_three_phase_data import ThreePhaseCrops
    from mewm_ispy2.registered_three_phase_latents import PhaseLatentPairs
    from mewm_ispy2.registered_three_phase_model import SharedPhaseCodec
    from mewm_ispy2.registered_three_phase_training import load_selected

    routes, world, baseline, manifest = (context[k] for k in ("routes", "world", "baseline", "manifest"))
    output, world_root = Path(cfg["output_dir"]), Path(world["output_dir"])
    tasks = [(mode, route) for route in routes for mode in (MODES[:1] if route["timepoint"] == 1 else MODES)]
    completed = sum(case_record(cfg, mode, route) is not None for mode, route in tasks)
    if completed == len(tasks):
        return
    device = torch.device("cuda:0")
    runtime.seed_all(cfg["generation_seed"])
    torch.use_deterministic_algorithms(True)
    model = load_selected(world, baseline, manifest, device)
    codec = SharedPhaseCodec(vq.load_frozen(baseline, device)[0])
    pairs = PhaseLatentPairs(world, baseline, manifest, "val")
    crops = ThreePhaseCrops(baseline, manifest, records=[v for v in manifest["visits"] if v["fold"] == "val" and v["visit"] == "T0"])
    by_patient = {r["patient_id"]: i for i, r in enumerate(crops.records)}
    module = bridge(world)
    data = Path(baseline["output_dir"]) / "data"
    denied = {s["path"] for row in manifest["visits"] for s in row["phase_sources"]}
    denied |= {row["model_mask_path"] for row in manifest["visits"] if row["model_mask_path"]}
    prefixes = [world_root / "latents/raw", data / "images", data / "patients",
                Path("/path/to/research/Datasets"), world_root / "source_supplement"]
    current_pid, t0, foreground, previous = None, None, None, None
    started, newly_completed = time.monotonic(), 0
    for mode, route in tasks:
        pid, tp = route["patient_id"], route["timepoint"]
        if pid != current_pid:
            current_pid, t0, foreground, previous = pid, None, None, None
        existing = case_record(cfg, mode, route)
        if existing is not None:
            if mode == "rollout":
                previous = torch.load(existing["latent_path"], map_location=device, weights_only=True)
            continue
        row = crops.records[by_patient[pid]]
        t0_id = f"{pid}:T0"
        allowed = [world_root / "latents/raw" / latent_filename(t0_id), data / "images" / visit_filename(t0_id),
                   data / "patients" / (pid + ".npz"), row["crop_report"]["path"], row["metadata_source"]["path"],
                   *[s["path"] for s in row["phase_sources"]]]
        if mode == "previous_real":
            allowed.append(world_root / "latents/raw" / latent_filename(route["record"]["earlier_visit_id"]))
        with direct.source_read_guard(allowed, denied, prefixes) as audit:
            if t0 is None:
                t0 = pairs.read(t0_id)[None].to(device)
                foreground = crops[by_patient[pid]]["valid"].numpy()
            source = source_batch(mode, route, t0, previous, pairs.read, cfg["draws"])
            noise = sampling_noise(source, route["noise_seed"])
            items = [{"record": route["record"], "earlier_latent": s, "later_latent": s} for s in source]
            conditions = module.collate_pairs(items)["conditions"]
            with runtime.autocast(device):
                latent = model.sample(source, conditions, noise, steps=cfg["sampling_steps"], solver=cfg["solver"])
                samples = codec.decode(latent, pairs.statistics).float().cpu().numpy()
        if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
            raise ValueError("Invalid generated rollout latent")
        if samples.shape != (cfg["draws"], *original.SHAPE) or not np.isfinite(samples).all():
            raise ValueError("Invalid generated triplets")
        mode_root = output / canonical_mode(mode, route)
        latent_path = mode_root / "latents" / (route["key"] + ".pt")
        save_tensor(latent_path, latent)
        replay_error = None
        if tp == 1:
            image_path = Path(context["old_cfg"]["output_dir"]) / "volumes" / (route["key"] + ".npz")
            with np.load(image_path, allow_pickle=False) as old:
                replay_error = float(np.abs(samples - old["samples"]).max())
                if replay_error != 0 or not np.array_equal(foreground, old["source_foreground"]):
                    raise ValueError("T0-T1 replay must exactly match the existing direct forecast before reuse")
        else:
            image_path = mode_root / "volumes" / (route["key"] + ".npz")
            save_npz(image_path, samples=samples, source_foreground=foreground)
        parent_path = None
        if mode == "rollout" and tp > 1:
            parent_route = next(r for r in routes if r["patient_id"] == pid and r["timepoint"] == tp - 1)
            parent_path = case_record(cfg, mode, parent_route)["latent_path"]
        opened_latents = [p for p in audit["opened"] if str(world_root / "latents/raw") + os.sep in p]
        record = {"schema": SCHEMA, "mode": canonical_mode(mode, route), "route": route,
                  "protocol": identity(output / "protocol.json"), "latent_path": str(latent_path),
                  "image_path": str(image_path), "parent_generated_latent": identity(parent_path) if parent_path else None,
                  "source_kind": "real_T0" if tp == 1 else ("previous_generated_same_draw" if mode == "rollout" else "real_previous_visit"),
                  "artifacts": [{"path": str(p), "identity": identity(p)} for p in (latent_path, image_path)],
                  "source_read_audit": {"passed": True, "real_latent_files_opened": opened_latents,
                                        "target_image_or_latent_reads": 0, "target_mask_used": False},
                  "shared_T1_volume_replay_max_error": replay_error, "completed_utc": now()}
        write_json(case_path(cfg, mode, route), record)
        if mode == "rollout":
            previous = latent
        completed += 1
        newly_completed += 1
        elapsed = time.monotonic() - started
        progress(cfg, "generating", completed_cases=completed, total_cases=len(tasks), mode=mode, target_stage=f"T{tp}",
                 elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / newly_completed * (len(tasks) - completed))
        disk_gate(cfg)
        del source, noise, samples, latent
    write_json(output / "GENERATION_COMPLETE.json", {"cases": len(tasks), "shared_T1_visits": 100,
               "new_generated_triplets": 352, "new_volumes": 1408, "passed": True, "completed_utc": now()})
    del model, codec, pairs, crops, previous, t0
    gc.collect()
    torch.cuda.empty_cache()


def feature_path(cfg, mode, route, draw):
    return Path(cfg["output_dir"]) / mode / "embeddings" / f"draw_{draw}" / route["patient_id"] / (route["key"] + ".pt")


def feature_complete(cfg, mode, route, draw):
    path = feature_path(cfg, mode, route, draw)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        return False
    receipt = read_json(path.with_suffix(".json"))
    if (receipt["protocol"] != identity(Path(cfg["output_dir"]) / "protocol.json")
            or receipt["embedding"] != identity(path) or receipt["case"] != identity(case_path(cfg, mode, route))):
        raise ValueError("Sequential feature binding changed")
    validate_embedding(path)
    return True


@torch.inference_mode()
def encode(cfg, context):
    from mewm_ispy2.registered_roi32_data import read_json as world_json

    routes = context["routes"]
    normalization = world_json(Path(context["baseline"]["output_dir"]) / "data/normalization.json")
    for route in routes:
        for mode in MODES:
            if case_record(cfg, mode, route) is None:
                raise ValueError("All generation must complete before feature extraction")
    tasks = [(mode, route) for route in routes for mode in MODES
             if any(not feature_complete(cfg, mode, route, draw) for draw in range(cfg["draws"]))]
    if not tasks:
        return
    model = load_frozen_pillar()
    protocol = identity(Path(cfg["output_dir"]) / "protocol.json")
    started, done = time.monotonic(), 0

    def preprocess(task):
        mode, route = task
        if route["timepoint"] == 1:
            return None
        record = case_record(cfg, mode, route)
        with np.load(record["image_path"], allow_pickle=False) as array:
            samples, foreground = array["samples"], array["source_foreground"]
        return [original.build_volume(sample, foreground, normalization) for sample in samples]

    with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as pool:
        queue, submitted = {}, 0
        for index, (mode, route) in enumerate(tasks):
            while submitted < min(len(tasks), index + cfg["preprocess_workers"]):
                queue[submitted] = pool.submit(preprocess, tasks[submitted])
                submitted += 1
            prepared = queue.pop(index).result()
            for draw in range(cfg["draws"]):
                path = feature_path(cfg, mode, route, draw)
                if feature_complete(cfg, mode, route, draw):
                    continue
                if route["timepoint"] == 1:
                    old_path = direct.feature_path(context["old_cfg"], route, draw)
                    vector = torch.load(old_path, map_location="cpu", weights_only=True)
                    geometry = {"reused_direct_T1_embedding": identity(old_path)}
                else:
                    volume, geometry = prepared[draw]
                    vector = pillar_forward(model, volume[None].to("cuda:0"))[0]
                save_tensor(path, vector)
                validate_embedding(path)
                write_json(path.with_suffix(".json"), {"protocol": protocol, "embedding": identity(path),
                           "case": identity(case_path(cfg, mode, route)), "geometry": geometry})
            done += 1
            elapsed = time.monotonic() - started
            if done == 1 or done % 5 == 0 or done == len(tasks):
                progress(cfg, "encoding", completed_cases=done, total_cases=len(tasks),
                         elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / done * (len(tasks) - done))
            del prepared, vector
            if route["timepoint"] != 1:
                del volume
    write_json(Path(cfg["output_dir"]) / "FEATURES_COMPLETE.json", {"passed": True, "vectors": 2208,
               "newly_encoded_vectors": 1408, "shared_T1_vectors_copied_into_each_arm": 800, "completed_utc": now()})
    del model
    gc.collect()
    torch.cuda.empty_cache()


def ensemble_predictions(folds, heldout, seeds):
    arms = []
    for source, group in folds.groupby("source"):
        frame = original.aggregate_predictions(group, heldout, seeds)
        frame["source"] = source
        arms.append(frame)
    individual = pd.concat(arms, ignore_index=True)
    keys = ["temporal_depth", "seed", "patient_id", "label"]
    for mode in ("direct", *MODES):
        draws = individual[individual.source.str.startswith(mode + "_draw_")]
        if not (draws.groupby(keys).size() == 4).all() or draws.empty:
            raise ValueError("Each sequence needs four distinct draw predictions")
        frame = draws.groupby(keys, as_index=False).probability.mean()
        frame["source"] = mode + "_mc4"
        arms.append(frame)
    return pd.concat(arms, ignore_index=True)


def paired_differences(cfg, predictions):
    comparisons = [("rollout_mc4", s) for s in ("direct_mc4", "real", "copy_T0")]
    comparisons += [("previous_real_mc4", s) for s in ("rollout_mc4", "direct_mc4", "real", "copy_T0")]
    rng, rows = np.random.default_rng(cfg["bootstrap_seed"]), []
    for depth, frame in predictions.groupby("temporal_depth"):
        ids = sorted(frame.patient_id.unique())
        labels = frame.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
        npos, nneg = int(labels.sum()), int((1 - labels).sum())
        wp = rng.multinomial(npos, np.full(npos, 1 / npos), size=cfg["bootstrap_samples"])
        wn = rng.multinomial(nneg, np.full(nneg, 1 / nneg), size=cfg["bootstrap_samples"])
        points, samples = {}, {}
        for arm in {s for pair in comparisons for s in pair}:
            matrix = frame[frame.source == arm].pivot(index="seed", columns="patient_id", values="probability")[ids].to_numpy()
            kernel = direct.mean_auc_kernel(labels, matrix)
            point = np.mean([roc_auc_score(labels, row) for row in matrix])
            if not np.isclose(point, kernel.mean(), atol=1e-12, rtol=0):
                raise ValueError("Paired bootstrap AUROC kernel mismatch")
            points[arm], samples[arm] = point, ((wp @ kernel) * wn).sum(axis=1) / (npos * nneg)
        for left, right in comparisons:
            lo, hi = np.quantile(samples[left] - samples[right], [.025, .975])
            rows.append({"depth": depth, "comparison": f"{left}_minus_{right}",
                         "auroc_difference": points[left] - points[right], "ci95_low": lo, "ci95_high": hi,
                         "patients": len(ids), "bootstrap_samples": cfg["bootstrap_samples"]})
    result = pd.DataFrame(rows)
    _atomic_csv(Path(cfg["output_dir"]) / "evaluation/paired_auroc_differences.csv", result)
    return result


@torch.inference_mode()
def evaluate(cfg, context):
    routes, source_cfg, cohort, refs = (context[k] for k in ("routes", "source_cfg", "cohort", "refs"))
    output, old_root, pcr = Path(cfg["output_dir"]), Path(context["old_cfg"]["output_dir"]), Path(source_cfg["output_dir"])
    if any(not feature_complete(cfg, mode, r, d) for r in routes for mode in MODES for d in range(cfg["draws"])):
        raise ValueError("Sequential feature extraction is incomplete")
    heldout = cohort["split"]["val"]
    real = _canonical_split(_load_split(pcr / "embeddings/real", pcr / "holdout_metadata.csv", heldout), 4)
    splits = {"real": real}
    for mode in MODES:
        for draw in range(cfg["draws"]):
            store = EmbStore(str(output / mode / "embeddings" / f"draw_{draw}"))
            splits[f"{mode}_draw_{draw}"] = direct.replaced_split(real, routes, mode, store)
    rows = []
    for index, ref in enumerate(refs):
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Classifier changed during evaluation")
        saved = torch.load(ref["path"], map_location="cpu", weights_only=False)
        if (set(saved["train_ids"]) | set(saved["validation_ids"])) & set(heldout):
            raise ValueError("Holdout entered classifier fitting")
        model = TDN({"downstream": saved["effective_config"]}).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["model_state"], strict=True)
        prior = _prior_from_state(real["clinical"], saved["clinical_prior"])
        for arm, raw in splits.items():
            frame = _prediction_frame(model, _canonical_split(raw, ref["max_tp"]), prior,
                                      {"name": ref["depth"], "max_tp": ref["max_tp"]}, ref["seed"], ref["fold"],
                                      "holdout", "cuda:0", 64)
            frame["source"] = arm
            rows.append(frame)
        del model
        if (index + 1) % 25 == 0:
            progress(cfg, "scoring", completed_models=index + 1, total_models=len(refs))
    new = pd.concat(rows, ignore_index=True)
    baseline = pd.read_csv(old_root / "evaluation/fold_predictions.csv", dtype={"probability": np.float32})
    keys = ["temporal_depth", "seed", "fold", "patient_id", "label"]
    comparison = new[new.source == "real"].merge(baseline[baseline.source == "real"], on=keys, validate="one_to_one")
    replay_error = float((comparison.probability_x - comparison.probability_y).abs().max())
    if len(comparison) != 20400 or replay_error > 2e-7:
        raise ValueError("Real-input frozen classifier replay changed")
    baseline["source"] = baseline.source.map(lambda s: "direct_" + s if s.startswith("draw_") else s)
    folds = pd.concat([baseline, new[new.source != "real"]], ignore_index=True)
    predictions = ensemble_predictions(folds, heldout, source_cfg["seeds"])
    if len(folds) != 285600 or len(predictions) != 69360:
        raise ValueError("Unexpected prediction coverage")
    t0 = predictions[predictions.temporal_depth == "T0"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    t0_error = float(t0.sub(t0.real, axis=0).abs().max().max())
    t1 = predictions[predictions.temporal_depth == "T0-T1"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    t1_error = max(float((t1[f"{mode}_{suffix}"] - t1[f"direct_{suffix}"]).abs().max())
                   for mode in MODES for suffix in ["mc4", *[f"draw_{d}" for d in range(cfg["draws"])]] )
    if t0_error > 1e-7 or t1_error > 1e-7:
        raise ValueError("Shared T0/T1 prediction controls failed")
    _atomic_csv(output / "evaluation/fold_predictions.csv", folds)
    _atomic_csv(output / "evaluation/predictions.csv", predictions)
    summary = direct.summarize(cfg, predictions, pd.read_csv(pcr / "evaluation/real/seed_metrics.csv"))
    differences = paired_differences(cfg, predictions)
    report = {"schema": SCHEMA, "passed": True, "patients": 102, "classifiers": 200,
              "fold_prediction_rows": len(folds), "patient_prediction_rows": len(predictions),
              "real_probability_replay_max_error": replay_error, "T0_probability_max_error": t0_error,
              "shared_T1_probability_max_error": t1_error, "optimizer_updates": 0, "completed_utc": now()}
    write_json(output / "evaluation/verification.json", report)
    selected = summary[summary.threshold_policy == "fixed_0.5"].set_index(["source", "depth"])
    lines = ["# Sequential Three-Phase ROI32 pCR", "",
             "Same 102 historical holdout patients and 200 frozen classifiers: five folds and ten seeds.",
             "Each classifier was fitted on real images from 764 development patients. No model was retrained.",
             "Real T0 plus ALL generated future visits is the pCR input in each generation arm.",
             "Direct: real T0 independently generates each target. Rollout: same-draw generated continuous latent feeds the next step, without re-encoding.",
             "Previous-real: real T0 generates T1, real T1 generates T2, and real T2 generates T3. This arm has additional observed follow-up information.",
             "Heun25, four matched noise draws per target, same fixed T0 foreground and physical-space Pillar preprocessing.",
             "MC4 averages four complete-sequence pCR probabilities, never MRI volumes, embeddings or intermediate states.",
             "Recorded follow-up intervals and observed prefix availability are retained. T0 and T1 controls reproduce exactly within reported numerical checks.", "",
             "| Window | Real | Copy T0 | Direct MC4 | Rollout MC4 | Previous-real MC4 |", "|---|---:|---:|---:|---:|---:|"]
    for depth in original.DEPTHS:
        values = [selected.loc[(arm, depth)] for arm in ("real", "copy_T0", "direct_mc4", "rollout_mc4", "previous_real_mc4")]
        lines.append("| " + depth + " | " + " | ".join(f"{v.auroc_mean:.5f} +/- {v.auroc_std:.5f}" for v in values) + " |")
    lines += ["", "Values are ten-seed mean AUROC +/- sample SD, not patient confidence intervals.", "",
              "| Window | Contrast | Difference | Paired patient 95% CI |", "|---|---|---:|---:|"]
    for row in differences.itertuples():
        if row.depth in ("T0-T2", "T0-T3") and row.comparison in (
                "rollout_mc4_minus_direct_mc4", "previous_real_mc4_minus_rollout_mc4", "previous_real_mc4_minus_real"):
            lines.append(f"| {row.depth} | {row.comparison} | {row.auroc_difference:.5f} | [{row.ci95_low:.5f}, {row.ci95_high:.5f}] |")
    lines += ["", "Intervals use 2000 stratified paired patient bootstraps conditional on fitted models/draws, without refitting or multiple-comparison correction.",
              "Thresholds are inherited from development OOF predictions. Holdout outcomes do not select methods, weights, noise or thresholds.",
              "This is retrospective known-time/availability evaluation. The historical holdout participated in generator validation and is not an untouched end-to-end test.",
              "Previous-real uses real earlier follow-up images internally, even though generated images replace every future pCR input slot; it is not baseline-only forecasting.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return report

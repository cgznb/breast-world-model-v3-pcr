"""Frozen pCR evaluation with complete generated triplets from real T0."""

from __future__ import annotations

import copy
import contextvars
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _canonical_split, _prediction_frame
from src import registered_three_phase_pcr as original
from src.data import EmbStore
from src.first_post_pcr_data import disk_gate, identity, now, public, read_json, repo_path, save_tensor, validate_embedding, write_json
from src.metrics import compute_metrics
from src.single_phase_repeat_pcr import unchanged_json
from src.tdn import TDN

SCHEMA = "registered_three_phase_generated_pcr_v1"
_READ_POLICY = contextvars.ContextVar("registered_pcr_source_read_policy", default=None)


def _audit_read(event, arguments):
    policy = _READ_POLICY.get()
    if policy is None or event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = str(Path(os.fsdecode(arguments[0])).resolve())
    if path in policy["allowed"]:
        policy["opened"].add(path)
    elif path in policy["denied"] or any(path.startswith(prefix) for prefix in policy["prefixes"]):
        raise RuntimeError("T0-only generation attempted to read an undeclared image, mask, or latent")


sys.addaudithook(_audit_read)


@contextmanager
def source_read_guard(allowed, denied, prefixes):
    policy = {"allowed": {str(Path(p).resolve()) for p in allowed},
              "denied": {str(Path(p).resolve()) for p in denied},
              "prefixes": [str(Path(p).resolve()) + os.sep for p in prefixes], "opened": set()}
    token = _READ_POLICY.set(policy)
    try:
        yield policy
    finally:
        _READ_POLICY.reset(token)


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if (cfg["schema"] != SCHEMA or cfg["draws"] != 4 or cfg["solver"] != "heun"
            or cfg["sampling_steps"] != 25 or cfg["source_policy"] != "direct_T0"
            or cfg["support_policy"] != "source_T0_per_phase_nonzero_foreground"
            or cfg["time_policy"] != "existing_recorded_followup_intervals"):
        raise ValueError("Unexpected frozen generated-pCR protocol")
    for key in ("output_dir", "source_pcr_config"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    cfg["config_path"] = str(path)
    return cfg


def progress(cfg, stage, **fields):
    value = {"schema": SCHEMA, "stage": stage, "pid": os.getpid(), "gpu": cfg["gpu"], "updated_utc": now(), **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", value)
    print(json.dumps(public(value), allow_nan=False), flush=True)


def routes_for(cohort, manifest, seed):
    heldout = set(cohort["split"]["val"])
    available = {pid: {v["timepoint"] for v in cohort["visits"] if v["patient_id"] == pid} for pid in heldout}
    pairs = [r for r in manifest["pairs"] if r["split"] == "val"]
    direct = {(r["patient_id"], int(r["later_stage"][1:])): (index, r)
              for index, r in enumerate(pairs) if r["earlier_stage"] == "T0"}
    routes, counts = [], [0] * 4
    for pid in sorted(heldout):
        length = 0
        while length in available[pid] and length < 4:
            counts[length] += 1
            length += 1
        if length == 0:
            raise ValueError("Every comparison patient requires real T0")
        for tp in range(1, length):
            if (pid, tp) not in direct:
                raise ValueError("Missing a direct-T0 pair for a retained future visit")
            index, record = direct[pid, tp]
            if record["earlier_visit_id"] != f"{pid}:T0" or record["later_visit_id"] != f"{pid}:T{tp}":
                raise ValueError("Inconsistent direct-T0 route")
            fields = ("pair_id", "patient_id", "split", "earlier_stage", "later_stage", "earlier_visit_id",
                      "later_visit_id", "delta_days", "interval_missing", "interval_source", "baseline_clinical", "treatment")
            routes.append({"patient_id": pid, "timepoint": tp, "key": f"{pid}_T{tp}",
                           "noise_seed": int(seed) + index, "record": {k: record[k] for k in fields}})
    return routes, dict(zip(original.DEPTHS, counts, strict=True))


def prepare(cfg):
    source_cfg = original.load_config(cfg["source_pcr_config"])
    from mewm_ispy2.registered_three_phase_data import load_config as load_world
    from mewm_ispy2.registered_roi32_runtime import verify_contract_files

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    pcr = Path(source_cfg["output_dir"])
    if not (pcr / "COMPLETE.json").is_file():
        raise ValueError("Real-input pCR fitting and evaluation must already be complete")
    contract = read_json(pcr / "input_contract.json")
    for section in ("runtime_sources", "input_sources", "crop_reports"):
        for path, recorded in contract[section].items():
            if identity(path) != recorded:
                raise ValueError("Frozen pCR inputs or source code changed")
    cohort = read_json(pcr / "cohort.json")
    world, baseline = load_world(source_cfg["world_config"])
    root = Path(world["output_dir"])
    manifest = read_json(root / "inventory.json")
    for stage in ("fm", "latents"):
        marker = read_json(root / stage / "COMPLETE.json")
        if marker["status"] != "passed":
            raise ValueError("Generator stage is incomplete")
        verify_contract_files(marker["contract"])
    routes, counts = routes_for(cohort, manifest, cfg["generation_seed"])
    refs = read_json(pcr / "frozen_models.json")["models"]
    if len(refs) != 200 or len(cohort["split"]["val"]) != 102:
        raise ValueError("Frozen classifier or holdout cohort size changed")
    for ref in refs:
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Frozen pCR checkpoint changed")
    inputs = [pcr / name for name in ("input_contract.json", "cohort.json", "frozen_models.json", "holdout_metadata.csv",
              "evaluation/real/predictions.csv", "evaluation/real/seed_metrics.csv")]
    inputs += [root / "inventory.json", root / "fm/best-endpoint.pt", root / "latents/statistics.json",
               Path(baseline["output_dir"]) / "vq/best.pt", Path(cfg["config_path"])]
    sources = [Path(__file__), repo_path("scripts/evaluate_registered_three_phase_pcr.py")]
    protocol = {"schema": SCHEMA, "config": cfg, "inputs": {str(p): identity(p) for p in inputs},
                "runtime_versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
                "sources": {str(p): identity(p) for p in sources}, "classifiers": refs,
                "holdout_patients": 102, "future_triplets": len(routes), "full_prefix_patients": counts,
                "phase_order": original.PHASES, "T0": "same_real_feature_in_every_arm",
                "generator_inputs": "T0_images_latents_foreground_and_baseline_clinical_treatment",
                "conditioning_time": "recorded_target_stage_and_elapsed_interval; retrospective_known_time",
                "missing_visits": "same_observed_contiguous_prefix_as_real_reference",
                "generated_ensemble": "average_pCR_probabilities_over_four_draws; never_average_images",
                "selection": "no_fitting_or_checkpoint_threshold_solver_noise_selection_on_holdout"}
    unchanged_json(output / "protocol.json", protocol)
    unchanged_json(output / "routes.json", {"routes": routes, "full_prefix_patients": counts})
    progress(cfg, "prepared", patients=102, future_triplets=len(routes), generated_volumes=len(routes) * cfg["draws"])
    return source_cfg, world, baseline, manifest, cohort, routes, refs


def feature_path(cfg, route, draw):
    return Path(cfg["output_dir"]) / "embeddings" / f"draw_{draw}" / route["patient_id"] / f"{route['key']}.pt"


def case_complete(cfg, route):
    root = Path(cfg["output_dir"])
    path = root / "cases" / (route["key"] + ".json")
    if not path.is_file():
        return False
    record = read_json(path)
    if record["route"] != route or record["protocol"] != identity(root / "protocol.json"):
        raise ValueError("A generated case belongs to a different protocol")
    for artifact in record["artifacts"]:
        if identity(artifact["path"]) != artifact["identity"]:
            raise ValueError("Generated case artifact changed")
    for draw in range(cfg["draws"]):
        validate_embedding(feature_path(cfg, route, draw))
    return True


@torch.inference_mode()
def generate_features(cfg, source_cfg, world, baseline, manifest, routes):
    from mewm_ispy2 import registered_roi32_runtime as runtime
    from mewm_ispy2 import registered_roi32_vq as vq
    from mewm_ispy2.registered_roi32_data import save_npz, visit_filename
    from mewm_ispy2.registered_roi32_latents import latent_filename
    from mewm_ispy2.registered_three_phase_data import ThreePhaseCrops
    from mewm_ispy2.registered_three_phase_latents import PhaseLatentPairs
    from mewm_ispy2.registered_three_phase_model import SharedPhaseCodec
    from mewm_ispy2.registered_three_phase_training import load_selected
    from mewm_ispy2.registered_roi32_fm import bridge

    output, world_root = Path(cfg["output_dir"]), Path(world["output_dir"])
    pending = [r for r in routes if not case_complete(cfg, r)]
    if not pending:
        return
    device = torch.device("cuda:0")
    runtime.seed_all(cfg["generation_seed"])
    torch.use_deterministic_algorithms(True)
    model = load_selected(world, baseline, manifest, device)
    codec, _ = vq.load_frozen(baseline, device)
    codec = SharedPhaseCodec(codec)
    pillar = load_frozen_pillar()
    if any(p.requires_grad for module in (model, codec, pillar) for p in module.parameters()):
        raise ValueError("Evaluation must freeze all networks")
    pairs = PhaseLatentPairs(world, baseline, manifest, "val")
    crops = ThreePhaseCrops(baseline, manifest, records=[v for v in manifest["visits"] if v["fold"] == "val" and v["visit"] == "T0"])
    indices = {row["patient_id"]: i for i, row in enumerate(crops.records)}
    module = bridge(world)
    baseline_root = Path(baseline["output_dir"]) / "data"
    denied = {source["path"] for row in manifest["visits"] for source in row["phase_sources"]}
    denied |= {row["model_mask_path"] for row in manifest["visits"]}
    prefixes = [world_root / "latents/raw", baseline_root / "images", baseline_root / "patients",
                Path("/path/to/research/Datasets"), world_root / "source_supplement"]
    completed, started = len(routes) - len(pending), time.monotonic()
    source_cache = {}
    for index, route in enumerate(pending):
        pid, source_id = route["patient_id"], route["record"]["earlier_visit_id"]
        row = crops.records[indices[pid]]
        allowed = [world_root / "latents/raw" / latent_filename(source_id),
                   baseline_root / "images" / visit_filename(source_id), baseline_root / "patients" / (pid + ".npz"),
                   row["metadata_source"]["path"], row["crop_report"]["path"], *[p["path"] for p in row["phase_sources"]]]
        with source_read_guard(allowed, denied, prefixes) as audit:
            if pid not in source_cache:
                source = pairs.read(source_id)[None].to(device)
                item = crops[indices[pid]]
                volume, _ = original.build_volume(item["image"].numpy(), item["valid"].numpy(), crops.baseline.normalization)
                replay = pillar_forward(pillar, volume[None].to(device))[0]
                saved_path = Path(source_cfg["output_dir"]) / "embeddings/real" / pid / f"{pid}_T0.pt"
                saved = torch.load(saved_path, map_location="cpu", weights_only=True)
                error = float((replay - saved).abs().max())
                if not torch.allclose(replay, saved, atol=1e-6, rtol=1e-6):
                    raise ValueError("Source preprocessing does not reproduce the trained Pillar features")
                source_cache = {pid: (source, item["valid"].numpy(), error)}
                del volume, replay, saved
            source, foreground, replay_error = source_cache[pid]
            generator = torch.Generator(device=device).manual_seed(route["noise_seed"])
            noise = torch.cat([torch.randn(source.shape, generator=generator, device=device) for _ in range(cfg["draws"])])
            items = [{"record": route["record"], "earlier_latent": source[0], "later_latent": source[0]}] * cfg["draws"]
            conditions = module.collate_pairs(items)["conditions"]
            with runtime.autocast(device):
                latent = model.sample(source.repeat(cfg["draws"], 1, 1, 1, 1), conditions, noise,
                                      steps=cfg["sampling_steps"], solver=cfg["solver"])
                decoded = codec.decode(latent, pairs.statistics).float().cpu().numpy()
            if decoded.shape != (cfg["draws"], *original.SHAPE) or not np.isfinite(decoded).all():
                raise ValueError("Invalid generated triplets")
            features, geometries = [], []
            for prediction in decoded:
                volume, geometry = original.build_volume(prediction, foreground, crops.baseline.normalization)
                features.append(pillar_forward(pillar, volume[None].to(device))[0])
                geometries.append(geometry)
                del volume
        array_path = output / "volumes" / (route["key"] + ".npz")
        save_npz(array_path, samples=decoded, source_foreground=foreground)
        artifacts = [{"path": str(array_path), "identity": identity(array_path)}]
        for draw, vector in enumerate(features):
            path = feature_path(cfg, route, draw)
            save_tensor(path, vector)
            validate_embedding(path)
            artifacts.append({"path": str(path), "identity": identity(path)})
        write_json(output / "cases" / (route["key"] + ".json"),
                   {"schema": SCHEMA, "route": route, "protocol": identity(output / "protocol.json"),
                    "artifacts": artifacts, "source_only_audit": {"passed": True, "source_files_opened": len(audit["opened"]),
                    "future_image_reads": 0, "future_latent_reads": 0, "future_masks_used": False},
                    "T0_pillar_replay_max_error": replay_error, "geometry": geometries, "completed_utc": now()})
        completed += 1
        elapsed = time.monotonic() - started
        progress(cfg, "generating_and_encoding", completed_triplets=completed, total_triplets=len(routes),
                 completed_volumes=completed * cfg["draws"], elapsed_seconds=elapsed,
                 estimated_remaining_seconds=elapsed / (index + 1) * (len(pending) - index - 1))
        disk_gate(cfg)
        del latent, decoded, features, noise
    write_json(output / "GENERATION_COMPLETE.json", {"schema": SCHEMA, "triplets": len(routes),
               "volumes": len(routes) * cfg["draws"], "future_image_reads": 0, "optimizer_updates": 0, "completed_utc": now()})
    del model, codec, pillar, pairs, crops, source_cache
    gc.collect()
    torch.cuda.empty_cache()


def replaced_split(real, routes, arm, store=None):
    result = copy.deepcopy(real)
    positions = {pid: i for i, pid in enumerate(real["pids"])}
    expected = {(pid, tp) for i, pid in enumerate(real["pids"]) for tp in range(1, 4) if real["masks"][i, tp]}
    supplied = {(r["patient_id"], r["timepoint"]) for r in routes}
    if supplied != expected or len(supplied) != len(routes):
        raise ValueError("Replacement routes must cover every retained future visit exactly once")
    for route in routes:
        pid, tp = route["patient_id"], route["timepoint"]
        index = positions[pid]
        if arm == "copy_T0":
            result["embs"][index, tp] = real["embs"][index, 0]
        else:
            if store is None:
                raise ValueError("Generated replacement requires a feature store")
            result["embs"][index, tp] = store.load(pid, tp, require_finite=True)
    for name in ("masks", "days", "clinical", "labels"):
        if not np.array_equal(real[name], result[name]):
            raise ValueError("Replacement changed comparison metadata")
    if not np.array_equal(real["embs"][:, 0], result["embs"][:, 0]):
        raise ValueError("Replacement changed the observed T0")
    return result


def summarize(cfg, predictions, reference_metrics):
    thresholds = reference_metrics[reference_metrics.threshold_policy == "development_oof"].set_index(["depth", "seed"]).threshold
    records = []
    for (source, depth, seed), group in predictions.groupby(["source", "temporal_depth", "seed"]):
        for policy, threshold in (("fixed_0.5", .5), ("development_oof", thresholds.loc[depth, seed])):
            values = compute_metrics(group.label, group.probability, threshold=threshold)
            predicted = group.probability.to_numpy() >= threshold
            target = group.label.to_numpy().astype(bool)
            denominator = int(predicted.sum()) + int(target.sum())
            records.append({"source": source, "depth": depth, "seed": int(seed), "threshold_policy": policy,
                            "threshold": float(threshold), **values,
                            "f1": 2 * int((predicted & target).sum()) / denominator if denominator else 0.,
                            "logloss": log_loss(group.label, group.probability, labels=[0, 1]),
                            "brier": brier_score_loss(group.label, group.probability)})
    metrics = pd.DataFrame(records)
    columns = ["auroc", "prauc", "acc", "sens", "spec", "prec", "npv", "bacc", "f1", "logloss", "brier"]
    summary = metrics.groupby(["source", "depth", "threshold_policy"])[columns].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    output = Path(cfg["output_dir"]) / "evaluation"
    _atomic_csv(output / "seed_metrics.csv", metrics)
    _atomic_csv(output / "summary.csv", summary)
    return summary


def mean_auc_kernel(labels, probability):
    """Average seed-specific Mann-Whitney kernels, retaining half credit for ties."""
    y, p = np.asarray(labels), np.asarray(probability)
    positive, negative = p[:, y == 1], p[:, y == 0]
    delta = positive[:, :, None] - negative[:, None, :]
    return ((delta > 0).astype(np.float64) + .5 * (delta == 0)).mean(axis=0)


def paired_bootstrap(cfg, predictions):
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    rows = []
    comparisons = [("generated_mc4", "real"), ("generated_mc4", "copy_T0"),
                   ("draw_0", "real"), ("draw_0", "copy_T0")]
    for depth, frame in predictions.groupby("temporal_depth"):
        ids = sorted(frame.patient_id.unique())
        y = frame.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
        npos, nneg = int(y.sum()), int((1 - y).sum())
        wp = rng.multinomial(npos, np.full(npos, 1 / npos), size=cfg["bootstrap_samples"])
        wn = rng.multinomial(nneg, np.full(nneg, 1 / nneg), size=cfg["bootstrap_samples"])
        samples, points = {}, {}
        for source in {s for pair in comparisons for s in pair}:
            matrix = frame[frame.source == source].pivot(index="seed", columns="patient_id", values="probability")[ids].to_numpy()
            kernel = mean_auc_kernel(y, matrix)
            point = float(np.mean([roc_auc_score(y, row) for row in matrix]))
            if not np.isclose(kernel.mean(), point, atol=1e-12, rtol=0):
                raise ValueError("AUROC kernel differs from independent sklearn scoring")
            points[source] = point
            samples[source] = np.sum((wp @ kernel) * wn, axis=1) / (npos * nneg)
        for left, right in comparisons:
            low, high = np.quantile(samples[left] - samples[right], [.025, .975])
            rows.append({"depth": depth, "comparison": f"{left}_minus_{right}", "auroc_difference": points[left] - points[right],
                         "ci95_low": low, "ci95_high": high, "patients": len(ids), "bootstrap_samples": cfg["bootstrap_samples"]})
    result = pd.DataFrame(rows)
    _atomic_csv(Path(cfg["output_dir"]) / "evaluation/paired_auroc_differences.csv", result)
    return result


@torch.inference_mode()
def evaluate(cfg, source_cfg, cohort, routes, refs):
    output, source = Path(cfg["output_dir"]), Path(source_cfg["output_dir"])
    if any(not case_complete(cfg, route) for route in routes):
        raise ValueError("Generated feature extraction is incomplete")
    heldout = cohort["split"]["val"]
    real = _canonical_split(_load_split(source / "embeddings/real", source / "holdout_metadata.csv", heldout), 4)
    splits = {"real": real, "copy_T0": replaced_split(real, routes, "copy_T0")}
    for draw in range(cfg["draws"]):
        arm = f"draw_{draw}"
        splits[arm] = replaced_split(real, routes, arm, EmbStore(str(output / "embeddings" / arm)))
    rows = []
    for index, ref in enumerate(refs):
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Frozen classifier changed")
        saved = torch.load(ref["path"], weights_only=False, map_location="cpu")
        if (set(saved["train_ids"]) | set(saved["validation_ids"])) & set(heldout):
            raise ValueError("Holdout patient entered classifier training or selection")
        model = TDN({"downstream": saved["effective_config"]}).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["model_state"], strict=True)
        prior = _prior_from_state(real["clinical"], saved["clinical_prior"])
        for arm, raw in splits.items():
            selected = _canonical_split(raw, ref["max_tp"])
            frame = _prediction_frame(model, selected, prior, {"name": ref["depth"], "max_tp": ref["max_tp"]},
                                      ref["seed"], ref["fold"], "holdout", "cuda:0", 64)
            frame["source"] = arm
            rows.append(frame)
        del model
        if (index + 1) % 25 == 0:
            progress(cfg, "evaluating_frozen_classifiers", completed_models=index + 1, total_models=len(refs))
    folds = pd.concat(rows, ignore_index=True)
    predictions = []
    for arm, frame in folds.groupby("source"):
        result = original.aggregate_predictions(frame, heldout, source_cfg["seeds"])
        result["source"] = arm
        predictions.append(result)
    predictions = pd.concat(predictions, ignore_index=True)
    keys = ["temporal_depth", "seed", "patient_id", "label"]
    ensemble = predictions[predictions.source.str.startswith("draw_")].groupby(keys, as_index=False).probability.mean()
    ensemble["source"] = "generated_mc4"
    predictions = pd.concat([predictions, ensemble], ignore_index=True)
    old = pd.read_csv(source / "evaluation/real/predictions.csv", dtype={"probability": np.float32})
    check = predictions[predictions.source == "real"].merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    replay = float((check.probability_new - check.probability_old).abs().max())
    if len(check) != 4080 or replay > 2e-7:
        raise ValueError("Frozen real-input probabilities did not replay")
    t0 = predictions[predictions.temporal_depth == "T0"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    t0_error = float((t0.sub(t0.real, axis=0)).abs().max().max())
    if t0_error > 1e-7:
        raise ValueError("A comparison branch changed T0 predictions")
    _atomic_csv(output / "evaluation/fold_predictions.csv", folds)
    _atomic_csv(output / "evaluation/predictions.csv", predictions)
    summary = summarize(cfg, predictions, pd.read_csv(source / "evaluation/real/seed_metrics.csv"))
    differences = paired_bootstrap(cfg, predictions)
    report = {"schema": SCHEMA, "passed": True, "patients": len(heldout), "classifiers": len(refs),
              "fold_prediction_rows": len(folds), "patient_prediction_rows": len(predictions),
              "real_replay_max_error": replay, "T0_probability_max_error": t0_error,
              "future_image_reads_during_generation": 0, "optimizer_updates": 0, "completed_utc": now()}
    write_json(output / "evaluation/verification.json", report)
    lines = ["# Generated Three-Phase ROI32 pCR", "",
             "102 historical internal holdout patients; 200 frozen classifiers, five folds and ten seeds.",
             "All future phases are generated directly from real T0. Real T0 remains unchanged in every branch.",
             "Heun25, four prespecified noise draws; MC4 averages pCR probabilities, not images or embeddings.",
             "Recorded followup times, baseline clinical/treatment inputs and the observed contiguous visit prefixes are retained.",
             "Thus this is retrospective forecasting conditional on known timing/availability, not a fully prospective baseline-only protocol.",
             "Source T0 foreground defines generated support. No future image, latent or mask is read during generation.", "",
             "| Window | Real AUROC | Copy T0 AUROC | Draw 0 AUROC | MC4 AUROC |", "|---|---:|---:|---:|---:|"]
    selected = summary[summary.threshold_policy == "fixed_0.5"].set_index(["source", "depth"])
    for depth in original.DEPTHS:
        values = [selected.loc[arm, depth] for arm in ("real", "copy_T0", "draw_0", "generated_mc4")]
        lines.append("| " + depth + " | " + " | ".join(f"{v.auroc_mean:.5f} +/- {v.auroc_std:.5f}" for v in values) + " |")
    lines += ["", "Seed SD is not a patient confidence interval. AUROC differences below use 2000 paired stratified patient bootstraps,",
              "conditional on the fitted models and generated draws, without refitting or multiple-comparison correction.", "",
              "| Window | MC4 minus real (95% CI) | MC4 minus copy (95% CI) |", "|---|---:|---:|"]
    for depth in original.DEPTHS:
        values = [differences[(differences.depth == depth) & (differences.comparison == name)].iloc[0]
                  for name in ("generated_mc4_minus_real", "generated_mc4_minus_copy_T0")]
        lines.append("| " + depth + " | " + " | ".join(f"{v.auroc_difference:.5f} [{v.ci95_low:.5f}, {v.ci95_high:.5f}]" for v in values) + " |")
    lines += ["", "Classification thresholds are inherited from the real development OOF predictions; holdout labels select nothing.",
              "The historical holdout previously participated in generator validation and is not an untouched end-to-end test set.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return report

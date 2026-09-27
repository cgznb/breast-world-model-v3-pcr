"""Real registered ROI32 triplets with the existing frozen-Pillar/TDN recipes."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import brier_score_loss, log_loss

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state, _set_seed
from scripts.run_full978_independent_cv import _canonical_split, _prediction_frame, _training_dir
from src.first_post_optimization import validate_partitions
from src.first_post_pcr_data import (
    PILLAR_SHAPE, disk_gate, identity, now, pillar_channel, public, read_json,
    repo_path, save_tensor, validate_embedding, write_json,
)
from src.metrics import compute_metrics
from src.single_phase_repeat_pcr import pillar_identity, recipe_configs, unchanged_csv, unchanged_json
from src.tdn import TDN

SCHEMA = "registered_three_phase_roi32_pcr_v1"
PHASES = ["pre_aqc0", "first_post_aqc1", "metadata_late"]
SHAPE = (3, 32, 128, 128)
SPACING = (2.0, 0.7032, 0.7032)
DEPTHS = {"T0": 1, "T0-T1": 2, "T0-T2": 3, "T0-T3": 4}


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if (cfg["schema"] != SCHEMA or cfg["phase_order"] != PHASES
            or tuple(cfg["shape_zyx"]) != SHAPE[1:]
            or tuple(cfg["spacing_zyx_mm"]) != SPACING
            or cfg["preprocessing"] != "physical_1mm_then_center_pad_crop"
            or cfg["encoder"] != "frozen_pillar_float32"):
        raise ValueError("Unsupported registered three-phase pCR input contract")
    for key in ("output_dir", "world_repo", "world_config", "source_cohort", "source_folds", "recipes_config"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    if cfg["world_repo"] not in sys.path:
        sys.path.insert(0, cfg["world_repo"])
    cfg["config_path"] = str(path)
    cfg["recipes"] = yaml.safe_load(Path(cfg["recipes_config"]).read_text())["recipes"]
    return cfg


def progress(cfg, stage, **fields):
    value = {"schema": SCHEMA, "stage": stage, "pid": os.getpid(), "gpu": cfg["gpu"],
             "updated_utc": now(), **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", value)
    print(json.dumps(public(value), allow_nan=False), flush=True)


def dataset(cfg):
    from mewm_ispy2.registered_three_phase_data import ThreePhaseCrops, load_config as world_config
    config, baseline = world_config(cfg["world_config"])
    manifest_path = Path(config["output_dir"]) / "inventory.json"
    manifest = read_json(manifest_path)
    if manifest["phase_order"] != PHASES or manifest["missing_sources"]:
        raise ValueError("Registered three-phase sources are incomplete")
    crops = ThreePhaseCrops(baseline, manifest)
    if crops.baseline.normalization != manifest["image_normalization"]:
        raise ValueError("Frozen generator intensity normalization changed")
    return crops, manifest, manifest_path, baseline


def intersect_cohort(manifest, metadata, original_development, original_holdout, original_folds):
    if metadata.pid.duplicated().any():
        raise ValueError("Duplicate patient metadata")
    roles = {role: {r["patient_id"] for r in manifest["visits"] if r["fold"] == role}
             for role in ("train", "val")}
    if (roles["train"] & roles["val"] or not roles["train"] <= set(original_development)
            or roles["val"] != set(original_holdout)):
        raise ValueError("Generator and original pCR patient boundaries disagree")
    selected = metadata[metadata.pid.isin(roles["train"] | roles["val"])].copy()
    if set(selected.pid) != roles["train"] | roles["val"] or not selected.pCR.isin([0, 1]).all():
        raise ValueError("Every selected patient needs an original binary pCR label")
    visits, seen, available = [], set(), {}
    for row in manifest["visits"]:
        pid, tp = row["patient_id"], int(row["visit"][1:])
        if row["fold"] not in roles or not 0 <= tp <= 3 or (pid, tp) in seen:
            raise ValueError("Invalid or repeated registered visit")
        if row["phase_indices"][:2] != [0, 1] or row["phase_indices"][2] <= 1:
            raise ValueError("Registered triplet phase order changed")
        seen.add((pid, tp))
        available.setdefault(pid, []).append(tp)
        visits.append({"visit_id": row["visit_id"], "patient_id": pid,
                       "canonical_patient_id": pid, "timepoint": tp, "visit": row["visit"],
                       "split": row["fold"]})
    if any(0 not in tps for tps in available.values()):
        raise ValueError("A fixed T0 ROI requires an available baseline visit")
    selected["registered_timepoints"] = selected.pid.map(
        lambda pid: ";".join(f"T{t}" for t in sorted(available[pid])))
    selected["n_registered_timepoints"] = selected.pid.map(lambda pid: len(available[pid]))
    folds = [{"fold": f["fold"], **{f"{role}_ids": sorted(set(f[f"{role}_ids"]) & roles["train"])
                                   for role in ("train", "val")}} for f in original_folds]
    split = {role: sorted(ids) for role, ids in roles.items()}
    validate_partitions(split["train"], split["val"], folds)
    for fold in folds:
        for role in ("train", "val"):
            if selected[selected.pid.isin(fold[f"{role}_ids"])].pCR.nunique() != 2:
                raise ValueError("Every training/validation fold must retain both outcome classes")
    cohort = {"schema": SCHEMA, "split": split, "visits": visits, "phase_order": PHASES,
              "image_shape_czyx": list(SHAPE), "spacing_zyx_mm": list(SPACING),
              "pillar_shape_chwd": list(PILLAR_SHAPE), "embedding_dim": 1152,
              "prefix_policy": "contiguous_T0_starting_prefix",
              "fold_policy": "intersection_with_original_registered_folds",
              "test_role": "historical_pcr_holdout_previously_used_for_generator_validation"}
    return cohort, selected, folds


def prepare(cfg):
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    crops, manifest, manifest_path, baseline = dataset(cfg)
    base, fold_root = Path(cfg["source_cohort"]), Path(cfg["source_folds"])
    metadata = pd.read_csv(base / "metadata_enriched.csv", dtype={"pid": str})
    original_development = sum([(base / "splits" / f"{r}_ids.txt").read_text().split() for r in ("train", "val")], [])
    original_holdout = (base / "splits/test_ids.txt").read_text().split()
    original_folds = [{"fold": i, **{f"{r}_ids": (fold_root / f"fold_{i}" / f"{r}_ids.txt").read_text().split()
                                    for r in ("train", "val")}} for i in range(5)]
    cohort, selected, folds = intersect_cohort(manifest, metadata, original_development, original_holdout, original_folds)
    if (len(cohort["split"]["train"]), len(cohort["split"]["val"]), len(cohort["visits"])) != (
            cfg["expected_development"], cfg["expected_holdout"], cfg["expected_visits"]):
        raise ValueError("Configured registered ROI32 cohort size changed")
    from mewm_ispy2.registered_roi32_data import verify_identity
    reports = {}
    for row in manifest["visits"]:
        for source in [row["metadata_source"], row["crop_report"], *row["phase_sources"]]:
            verify_identity(source)
        path = row["crop_report"]["path"]
        if path not in reports:
            report = read_json(path)
            for source in report["cache_identities"]:
                verify_identity(source)
            reports[path] = identity(path)
    sources = [Path(__file__), repo_path("scripts/run_registered_three_phase_pcr.py"),
               *[repo_path(p) for p in ("src/pillar.py", "src/tdn.py", "src/data.py", "src/temporal.py",
                 "src/first_post_pcr_data.py", "src/single_phase_repeat_pcr.py", "src/first_post_optimization.py",
                 "scripts/run_first_post_pcr.py", "scripts/run_full978_independent_cv.py",
                 "scripts/run_full978_anti_overfit.py")],
               *[Path(cfg["world_repo"]) / "mewm_ispy2" / p for p in
                 ("registered_three_phase_data.py", "registered_roi32_data.py")]]
    bindings = [Path(cfg["config_path"]), Path(cfg["world_config"]), Path(cfg["recipes_config"]), manifest_path,
                base / "metadata_enriched.csv", Path(baseline["output_dir"]) / "data/COMPLETE.json",
                Path(baseline["output_dir"]) / "data/normalization.json",
                *[repo_path(r["path"]) for r in cfg["recipes"].values()]]
    contract = {"schema": SCHEMA, "config": cfg, "runtime_sources": {str(p): identity(p) for p in sources},
                "input_sources": {str(p): identity(p) for p in bindings}, "crop_reports": reports,
                "pillar_files": pillar_identity(), "encoder_frozen": True, "encoder_precision": "float32",
                "spatial_policy": "generator_fixed_T0_roi_then_physical_1mm_center_pad_crop",
                "intensity_policy": "invert_frozen_generator_scale_restore_background_then_per_phase_p1_p99_minmax",
                "fitting": "real_development_images_only", "old_full_field_embeddings_reused": False}
    unchanged_json(output / "input_contract.json", contract)
    unchanged_json(output / "cohort.json", cohort)
    unchanged_json(output / "folds.json", folds)
    for role, name in (("train", "metadata_enriched.csv"), ("val", "holdout_metadata.csv")):
        unchanged_csv(output / name, selected[selected.pid.isin(cohort["split"][role])].reset_index(drop=True))
    recipe_configs(cfg)
    summary = {"development_patients": len(cohort["split"]["train"]), "holdout_patients": len(cohort["split"]["val"]),
               "development_positive": int(selected[selected.pid.isin(cohort["split"]["train"])].pCR.sum()),
               "visits": len(cohort["visits"]), "folds": [{"fold": f["fold"], "train": len(f["train_ids"]),
                                                             "validation": len(f["val_ids"])} for f in folds]}
    unchanged_json(output / "cohort_summary.json", summary)
    progress(cfg, "prepared", **summary)
    del crops
    return cohort


def build_volume(images, valid, normalization):
    images, valid = np.asarray(images, np.float32), np.asarray(valid, bool)
    if images.shape != SHAPE or valid.shape != SHAPE or not np.isfinite(images).all():
        raise ValueError("Expected three finite normalized ROI32 phases and matching foreground")
    mean, std = float(normalization["mean"]), float(normalization["std"])
    if not np.isfinite([mean, std]).all() or std <= 0 or not valid.reshape(3, -1).any(axis=1).all():
        raise ValueError("Invalid frozen intensity mapping or empty phase")
    raw = images * std + mean
    raw[~valid] = 0
    volume = torch.stack([pillar_channel(phase, SPACING) for phase in raw])
    if tuple(volume.shape) != PILLAR_SHAPE or not torch.isfinite(volume).all():
        raise ValueError("Invalid physical-space Pillar input")
    return volume, {"shape_chwd": list(volume.shape), "source_shape_czyx": list(SHAPE),
                    "spacing_zyx_mm": list(SPACING), "min": float(volume.min()), "max": float(volume.max()),
                    "nonzero_fraction": [float(torch.count_nonzero(v)) / v.numel() for v in volume]}


def feature_path(cfg, visit):
    pid = visit["canonical_patient_id"]
    return Path(cfg["output_dir"]) / "embeddings/real" / pid / f"{pid}_T{visit['timepoint']}.pt"


def feature_complete(cfg, visit):
    path = feature_path(cfg, visit)
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return False
    value = read_json(receipt)
    if (value["contract"] != identity(Path(cfg["output_dir"]) / "input_contract.json")
            or value["embedding"] != identity(path) or value["visit_id"] != visit["visit_id"]):
        raise ValueError("Stored ROI32 feature or its input contract changed")
    validate_embedding(path)
    return True


def extract(cfg, cohort, role, patient_ids=None):
    from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
    selected = [v for v in cohort["visits"] if v["split"] == role
                and (patient_ids is None or v["patient_id"] in patient_ids)]
    pending = [v for v in selected if not feature_complete(cfg, v)]
    if not pending:
        return
    progress(cfg, "extracting_real_features", role=role, completed=len(selected) - len(pending), total=len(selected))
    crops, _, _, _ = dataset(cfg)
    indices = {r["visit_id"]: i for i, r in enumerate(crops.records)}
    model = load_frozen_pillar()
    started, completed = time.perf_counter(), len(selected) - len(pending)
    contract = identity(Path(cfg["output_dir"]) / "input_contract.json")

    def preprocess(visit):
        item = crops[indices[visit["visit_id"]]]
        return build_volume(item["image"].numpy(), item["valid"].numpy(), crops.baseline.normalization)

    with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as pool:
        queue = {}
        submitted = 0
        for index, visit in enumerate(pending):
            while submitted < min(len(pending), index + cfg["preprocess_workers"]):
                queue[submitted] = pool.submit(preprocess, pending[submitted])
                submitted += 1
            volume, audit = queue.pop(index).result()
            with torch.inference_mode():
                vector = pillar_forward(model, volume[None].to("cuda:0"))[0]
                if index == 0:
                    replay = pillar_forward(model, volume[None].to("cuda:0"))[0]
                    if not torch.allclose(vector, replay, atol=1e-6, rtol=1e-6):
                        raise ValueError("Frozen Pillar feature replay differs")
                    audit["encoder_replay_max_error"] = float((vector - replay).abs().max())
                    audit["encoder_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            path = feature_path(cfg, visit)
            save_tensor(path, vector)
            validate_embedding(path)
            write_json(path.with_suffix(".json"), {"schema": SCHEMA, "visit_id": visit["visit_id"],
                       "contract": contract, "embedding": identity(path), "geometry": audit})
            completed += 1
            if index == 0 or completed % 10 == 0 or index + 1 == len(pending):
                elapsed = time.perf_counter() - started
                progress(cfg, "extracting_real_features", role=role, completed=completed, total=len(selected),
                         newly_extracted=index + 1, elapsed_seconds=elapsed,
                         estimated_remaining_seconds=elapsed / (index + 1) * (len(pending) - index - 1))
                disk_gate(cfg)
            del volume, vector
    del model, crops
    gc.collect()
    torch.cuda.empty_cache()


def smoke(cfg, cohort):
    from scripts.run_first_post_pcr import fold_task
    output = Path(cfg["output_dir"])
    if (output / "SMOKE_COMPLETE.json").exists():
        return read_json(output / "SMOKE_COMPLETE.json")
    metadata = pd.read_csv(output / "metadata_enriched.csv", dtype={"pid": str})
    classes = [sorted(metadata[metadata.pCR == label].pid)[:4] for label in (0, 1)]
    train_ids = sorted(classes[0][:2] + classes[1][:2])
    val_ids = sorted(classes[0][2:] + classes[1][2:])
    extract(cfg, cohort, "train", set(train_ids + val_ids))
    configs, results = recipe_configs(cfg), []
    for depth, count in DEPTHS.items():
        task = (configs[depth], str(output / "smoke"), "formal", {"name": depth, "max_tp": count},
                "prespecified", 2026, {"fold": 0, "train_ids": train_ids, "val_ids": val_ids}, "cuda:0", True, False)
        progress(cfg, "smoke_training", depth=depth)
        result = fold_task(task)
        run = _training_dir(output / "smoke", "formal", depth, "prespecified", task[5], 0)
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        _set_seed(task[5] * 1000 + count)
        initial = TDN({"downstream": saved["effective_config"]}).state_dict()
        changed = any(not torch.equal(v, initial[k]) for k, v in saved["model_state"].items())
        split = _canonical_split(_load_split(output / "embeddings/real", output / "metadata_enriched.csv", val_ids), count)
        prior = _prior_from_state(split["clinical"], saved["clinical_prior"])
        model = TDN({"downstream": saved["effective_config"]}).to("cuda:0")
        model.load_state_dict(saved["model_state"], strict=True)
        replay = _prediction_frame(model, split, prior, task[3], task[5], 0, "oof_val", "cuda:0", 64)
        original = pd.read_csv(run / "val_predictions.csv")
        error = float(np.max(np.abs(replay.probability.to_numpy() - original.probability.to_numpy())))
        if not changed or error > 1e-6 or len(pd.read_csv(run / "history.csv")) != 2:
            raise ValueError("Real pCR smoke did not update weights or replay predictions")
        results.append({**result, "weights_updated": changed, "checkpoint_replay_max_error": error})
        del model
    report = {"schema": SCHEMA, "passed": True, "development_patients": 8, "holdout_used": False,
              "four_recipes": results, "completed_utc": now()}
    write_json(output / "SMOKE_COMPLETE.json", report)
    torch.cuda.empty_cache()
    return report


def train(cfg, cohort):
    from scripts.run_first_post_pcr import fold_task, training_tasks
    output = Path(cfg["output_dir"])
    if not read_json(output / "SMOKE_COMPLETE.json")["passed"]:
        raise ValueError("Real-data smoke must pass before formal training")
    for visit in cohort["visits"]:
        if visit["split"] == "train" and not feature_complete(cfg, visit):
            raise ValueError("Development feature extraction is incomplete")
    write_json(output / "DEVELOPMENT_FEATURES_COMPLETE.json", {"schema": SCHEMA, "completed_utc": now(),
               "visits": sum(v["split"] == "train" for v in cohort["visits"])})
    folds = read_json(output / "folds.json")
    validate_partitions(cohort["split"]["train"], cohort["split"]["val"], folds)
    tasks = training_tasks(cfg, recipe_configs(cfg), folds)
    progress(cfg, "training_pcr", completed_fits=0, total_fits=len(tasks), jobs=cfg["training_jobs"])
    results, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg["training_jobs"], mp_context=mp.get_context("spawn")) as pool:
        pending = [pool.submit(fold_task, task) for task in tasks]
        for future in as_completed(pending):
            results.append(future.result())
            elapsed = time.perf_counter() - started
            progress(cfg, "training_pcr", completed_fits=len(results), total_fits=len(tasks), latest=results[-1],
                     estimated_remaining_seconds=elapsed / len(results) * (len(tasks) - len(results)))
            disk_gate(cfg)
    write_json(output / "TRAINING_COMPLETE.json", {"schema": SCHEMA, "fits": len(results), "results": results,
               "fit_images": "real_only", "holdout_used_for_fitting": False, "completed_utc": now()})
    refs = []
    for depth, count in DEPTHS.items():
        for seed in cfg["seeds"]:
            for fold in range(5):
                path = _training_dir(output / "tdn", "formal", depth, "prespecified", seed, fold) / "best.pt"
                refs.append({"depth": depth, "max_tp": count, "seed": seed, "fold": fold,
                             "path": str(path), "identity": identity(path)})
    unchanged_json(output / "frozen_models.json", {"schema": SCHEMA, "models": refs, "holdout_used_for_selection": False})


def aggregate_predictions(folds, heldout, seeds):
    expected = pd.MultiIndex.from_product([list(DEPTHS), seeds, heldout], names=["temporal_depth", "seed", "patient_id"])
    counts = folds.groupby(["temporal_depth", "seed", "patient_id"]).agg(n=("fold", "size"),
                                                                          unique=("fold", "nunique"), labels=("label", "nunique"))
    if (set(counts.index) != set(expected) or not (counts["n"] == 5).all()
            or not (counts["unique"] == 5).all() or not (counts["labels"] == 1).all()):
        raise ValueError("Each heldout patient/window/seed needs exactly five distinct fold predictions")
    return folds.groupby(["temporal_depth", "seed", "patient_id", "label"], as_index=False).probability.mean()


def evaluate(cfg, cohort):
    from scripts.evaluate_first_post_pcr import oof_threshold
    output = Path(cfg["output_dir"])
    refs = read_json(output / "frozen_models.json")["models"]
    heldout = cohort["split"]["val"]
    raw = _load_split(output / "embeddings/real", output / "holdout_metadata.csv", heldout)
    rows, oof = [], []
    for index, ref in enumerate(refs):
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Frozen pCR checkpoint changed")
        saved = torch.load(ref["path"], weights_only=False, map_location="cpu")
        if (set(saved["train_ids"]) | set(saved["validation_ids"])) & set(heldout):
            raise ValueError("Holdout patient entered fitting or checkpoint selection")
        split = _canonical_split(raw, ref["max_tp"])
        prior = _prior_from_state(split["clinical"], saved["clinical_prior"])
        model = TDN({"downstream": saved["effective_config"]}).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["model_state"], strict=True)
        rows.append(_prediction_frame(model, split, prior, {"name": ref["depth"], "max_tp": ref["max_tp"]},
                                      ref["seed"], ref["fold"], "holdout", "cuda:0", 64))
        oof.append(pd.read_csv(Path(ref["path"]).parent / "val_predictions.csv"))
        del model
        if (index + 1) % 25 == 0:
            progress(cfg, "evaluating_real_holdout", completed_models=index + 1, total_models=len(refs))
    folds, development = pd.concat(rows, ignore_index=True), pd.concat(oof, ignore_index=True)
    predictions = aggregate_predictions(folds, heldout, cfg["seeds"])
    metrics = []
    for (depth, seed), frame in predictions.groupby(["temporal_depth", "seed"]):
        train = development[(development.temporal_depth == depth) & (development.seed == seed)]
        if set(train.patient_id) != set(cohort["split"]["train"]) or train.patient_id.duplicated().any():
            raise ValueError("Development OOF predictions do not cover each patient once")
        threshold = oof_threshold(train.label, train.probability)
        for policy, value in (("fixed_0.5", 0.5), ("development_oof", threshold)):
            metric = compute_metrics(frame.label, frame.probability, threshold=value)
            denom = metric["prec"] + metric["sens"]
            metric["f1"] = 2 * metric["prec"] * metric["sens"] / denom if denom > 0 else 0.0
            metrics.append({"depth": depth, "seed": int(seed), "threshold_policy": policy, "threshold": value,
                            **metric, "logloss": log_loss(frame.label, frame.probability, labels=[0, 1]),
                            "brier": brier_score_loss(frame.label, frame.probability)})
    metrics = pd.DataFrame(metrics)
    columns = ["auroc", "prauc", "acc", "sens", "spec", "prec", "f1", "logloss", "brier"]
    summary = metrics.groupby(["depth", "threshold_policy"])[columns].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    for name, frame in (("fold_predictions", folds), ("predictions", predictions), ("development_oof", development),
                        ("seed_metrics", metrics), ("summary", summary)):
        _atomic_csv(output / "evaluation/real" / f"{name}.csv", frame)
    write_json(output / "evaluation/real/COMPLETE.json", {"schema": SCHEMA, "completed_utc": now(),
               "patients": len(heldout), "models": len(refs), "optimizer_updates": 0,
               "aggregation": "five_fold_probability_mean_then_mean_and_sample_sd_of_ten_seed_metrics",
               "test_role": cohort["test_role"]})
    lines = ["# Registered ROI32 three-phase pCR", "",
             "Frozen Pillar, 1152-D visit features, original four independent TDN recipes and clinical priors.",
             "Real precontrast / first-post / metadata-late triplets on the generator's fixed T0 patient ROI.",
             f"Development: {len(cohort['split']['train'])}; historical internal holdout: {len(heldout)}.",
             "The holdout previously participated in generator validation; it is not an untouched end-to-end test.",
             "Inherited five-fold membership, ten seeds. Classifier fitting and selection use real development data only.", "",
             "| Window | AUROC mean | Seed SD | Log loss mean |", "|---|---:|---:|---:|"]
    for depth in DEPTHS:
        row = summary[(summary.depth == depth) & (summary.threshold_policy == "fixed_0.5")].iloc[0]
        lines.append(f"| {depth} | {row.auroc_mean:.5f} | {row.auroc_std:.5f} | {row.logloss_mean:.5f} |")
    lines += ["", "Generated-future and source-copy comparisons are separate follow-up evaluations with these frozen classifiers.",
              "ROI geometry, selected phases, and the development cohort differ from the historical full-field baseline.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return summary.to_dict("records")

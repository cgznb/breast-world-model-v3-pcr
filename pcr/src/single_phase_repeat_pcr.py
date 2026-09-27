"""Single-phase inputs for the existing frozen Pillar and pCR training recipes."""

from __future__ import annotations

import copy
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _canonical_split, _effective_config, _training_dir
from src.data import load_ids
from src.first_post_optimization import fit_tdn, metric_values, predict, tensor_split, validate_partitions
from src.first_post_pcr_data import (
    IMAGE_SHAPE, PILLAR_SHAPE, disk_gate, identity, now, pillar_channel, public,
    read_json, repo_path, resample_native_roi, save_tensor, validate_embedding,
    world_imports, write_json,
)
from src.mewm_data import RegisteredManifestSource, patient_key
from src.tdn import TDN

SCHEMA = "single_phase_repeat_pillar_pcr_v1"
DEPTHS = {"T0": 1, "T0-T1": 2, "T0-T2": 3, "T0-T3": 4}


def load_config(path, branch):
    path = repo_path(path).resolve()
    master = yaml.safe_load(path.read_text())
    if master.get("schema") != SCHEMA or branch not in master["branches"]:
        raise ValueError("Unsupported single-phase run or branch")
    cfg = {k: copy.deepcopy(v) for k, v in master.items() if k != "branches"}
    cfg.update(master["branches"][branch])
    cfg.update(branch=branch, config_path=str(path))
    cfg["output_dir"] = str(repo_path(master["output_dir"]) / branch)
    for key in ("source_config", "source_cohort", "recipes_config", "source_folds", "optimization_reference"):
        if key in cfg:
            cfg[key] = str(repo_path(cfg[key]).resolve())
    if cfg["embedding_dim"] != 1152 or cfg["input_mode"] not in (
            "registered_dce0_repeat", "native_first_post_repeat"):
        raise ValueError("Single-phase Pillar input contract changed")
    return cfg


def progress(cfg, stage, **fields):
    value = {"schema": SCHEMA, "updated_utc": now(), "pid": os.getpid(),
             "branch": cfg["branch"], "gpu": cfg["gpu"], "stage": stage, **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", value)
    print(__import__("json").dumps(public(value)), flush=True)


def unchanged_json(path, value):
    if Path(path).exists():
        if read_json(path) != public(value):
            raise ValueError(f"Frozen run artifact changed: {path}")
    else:
        write_json(path, value)


def unchanged_csv(path, frame):
    if Path(path).exists():
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype={"pid": str}),
                                      frame.reset_index(drop=True), check_dtype=False)
    else:
        _atomic_csv(path, frame)


def pillar_identity():
    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE) / "models--YalaLab--Pillar0-BreastMRI"
    snapshot = cache / "snapshots" / (cache / "refs/main").read_text().strip()
    files = {p.name: identity(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    if not files:
        raise FileNotFoundError("Cached Pillar weights are absent")
    return files


def recipe_configs(cfg):
    original = yaml.safe_load(Path(cfg["recipes_config"]).read_text())
    output = Path(cfg["output_dir"])
    paths = {}
    for depth, count in DEPTHS.items():
        spec = original["recipes"][depth]
        source = yaml.safe_load(repo_path(spec["path"]).read_text())
        effective = _effective_config(source, spec["variant"], depth)
        value = {"data": {"metadata_csv": str(output / "metadata_enriched.csv"),
                           "train_embeddings_dir": str(output / "embeddings/real"),
                           "embedding_dim": 1152},
                 "downstream": effective, "variants": {"prespecified": {}},
                 "independent_cv": {"temporal_depths": [{"name": depth, "max_tp": count}]}}
        path = output / "configs" / f"{depth.lower().replace('-', '_')}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and yaml.safe_load(path.read_text()) != value:
            raise ValueError("Frozen per-window recipe changed")
        if not path.exists():
            _atomic_text(path, yaml.safe_dump(value, sort_keys=False))
        paths[depth] = str(path)
    return paths


def _registered_cohort(cfg):
    base = Path(cfg["source_cohort"])
    metadata = pd.read_csv(base / "metadata_enriched.csv", dtype={"pid": str})
    development = sorted(load_ids(base / "splits/train_ids.txt") + load_ids(base / "splits/val_ids.txt"))
    heldout = load_ids(base / "splits/test_ids.txt")
    folds = [{"fold": fold,
              **{f"{role}_ids": load_ids(Path(cfg["source_folds"]) / f"fold_{fold}" / f"{role}_ids.txt")
                 for role in ("train", "val")}} for fold in range(5)]
    source_cfg = yaml.safe_load(Path(cfg["source_config"]).read_text())
    source = RegisteredManifestSource(repo_path(source_cfg["mewm_adapter"]["registered_manifest_csv"]),
                                      input_mode="registered_dce0_repeat")
    visits = []
    held = set(heldout)
    for row in metadata.to_dict("records"):
        for token in str(row["registered_timepoints"]).split(";"):
            tp = int(token.strip()[1:])
            visit = source.visits[(patient_key(row["pid"]), tp)]
            path = visit.dce_paths[0]
            if not path.endswith("_dce_aqc_0.nii.gz"):
                raise ValueError("Registered DCE0 path is not acquisition zero")
            visits.append({"canonical_patient_id": row["pid"], "timepoint": tp,
                           "split": "val" if row["pid"] in held else "train",
                           "input_path": path, "input_identity": identity(path),
                           "meta_path": visit.meta_path, "meta_identity": identity(visit.meta_path),
                           "shape_zyx": list(visit.shape_zyx), "spacing_zyx": list(visit.spacing_zyx)})
    return metadata, development, heldout, folds, visits


def _first_post_cohort(cfg):
    base = Path(cfg["source_cohort"])
    original = read_json(base / "cohort.json")
    metadata = pd.read_csv(base / "metadata_enriched.csv", dtype={"pid": str})
    folds = read_json(base / "folds.json")
    if isinstance(folds, dict):
        folds = list(folds.values())
    keys = ("canonical_patient_id", "timepoint", "split", "shape_zyx", "crop_geometry",
            "image_orientation_patient", "image_position_patient_first", "pixel_spacing_yx_mm",
            "slice_spacing_mm", "native_first_post", "native_first_post_identity", "image_path", "crop_identity")
    visits = [{k: v[k] for k in keys} for v in original["visits"]]
    for visit in visits:
        if (identity(visit["native_first_post"]) != visit["native_first_post_identity"]
                or identity(visit["image_path"]) != visit["crop_identity"]):
            raise ValueError("First-post source or original VQ crop changed")
    return metadata, original["split"]["train"], original["split"]["val"], folds, visits


def prepare(cfg):
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    data = _registered_cohort(cfg) if cfg["branch"] == "registered_dce0" else _first_post_cohort(cfg)
    metadata, development, heldout, folds, visits = data
    validate_partitions(development, heldout, folds)
    if (len(development), len(heldout), len(visits)) != (
            cfg["expected_development"], cfg["expected_holdout"], cfg["expected_visits"]):
        raise ValueError("The original cohort size changed")
    if metadata.pid.duplicated().any() or set(metadata.pid) != set(development) | set(heldout):
        raise ValueError("Metadata does not match the original patient cohort")
    keys = [(v["canonical_patient_id"], v["timepoint"]) for v in visits]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated patient visit")
    for visit in visits:
        allowed = development if visit["split"] == "train" else heldout
        if visit["canonical_patient_id"] not in allowed:
            raise ValueError("Image visit crosses the patient split")
    phase = "dce0" if cfg["branch"] == "registered_dce0" else "first_post"
    cohort = {"schema": SCHEMA, "phase_order": [phase] * 3,
              "input_mode": cfg["input_mode"], "pillar_shape_chwd": list(PILLAR_SHAPE),
              "split": {"train": development, "val": heldout}, "visits": visits,
              "prefix_policy": "contiguous_T0_starting_prefix; missing_T0_excluded_from_TDN_loss",
              "test_role": "historical_internal_pCR_holdout_previously_used_for_upstream_validation"}
    runtime = ["src/single_phase_repeat_pcr.py", "scripts/run_single_phase_repeat_pcr.py",
               "src/tdn.py", "src/pillar.py", "src/data.py", "src/temporal.py", "src/mewm_data.py",
               "src/first_post_pcr_data.py", "src/first_post_optimization.py",
               "scripts/run_first_post_pcr.py", "scripts/run_first_post_optimization.py",
               "scripts/report_first_post_optimization.py", "scripts/run_full978_independent_cv.py",
               "scripts/run_full978_anti_overfit.py"]
    source_files = [cfg["config_path"], cfg["source_config"], cfg["recipes_config"],
                    str(Path(cfg["source_cohort"]) / "metadata_enriched.csv")]
    recipes = yaml.safe_load(Path(cfg["recipes_config"]).read_text())["recipes"]
    source_files.extend(str(repo_path(v["path"])) for v in recipes.values())
    if "optimization_reference" in cfg:
        source_files.append(cfg["optimization_reference"])
    contract = {"schema": SCHEMA, "config": cfg, "pillar_model": "YalaLab/Pillar0-BreastMRI",
                "pillar_files": pillar_identity(), "frozen_encoder": True, "precision": "float32",
                "runtime_sources": {p: identity(repo_path(p)) for p in runtime},
                "configuration_sources": {p: identity(p) for p in source_files},
                "preprocessing": "same_original_spatial_and_intensity_policy; repeat_one_channel_three_times",
                "fit_images": "real_single_phase_only", "old_three_phase_embeddings_reused": False}
    unchanged_json(output / "input_contract.json", contract)
    unchanged_json(output / "cohort.json", cohort)
    unchanged_json(output / "folds.json", folds)
    # Training readers receive a file containing development patients only.
    unchanged_csv(output / "metadata_enriched.csv", metadata[metadata.pid.isin(development)].reset_index(drop=True))
    unchanged_csv(output / "holdout_metadata.csv", metadata[metadata.pid.isin(heldout)].reset_index(drop=True))
    recipe_configs(cfg)
    progress(cfg, "prepared", development_patients=len(development), holdout_patients=len(heldout), visits=len(visits))
    return cohort


def build_volume(cfg, visit):
    if cfg["branch"] == "registered_dce0":
        if (identity(visit["input_path"]) != visit["input_identity"]
                or identity(visit["meta_path"]) != visit["meta_identity"]):
            raise ValueError("Registered source changed after preparation")
        roi = np.asarray(nib.load(visit["input_path"]).dataobj, dtype=np.float32)
        if tuple(roi.shape) != tuple(visit["shape_zyx"]) or not np.isfinite(roi).all():
            raise ValueError("Invalid registered DCE0 volume")
        spacing = visit["spacing_zyx"]
        crop_error = None
    else:
        source_cfg = yaml.safe_load(Path(cfg["source_config"]).read_text())
        world_imports(source_cfg)
        if (identity(visit["native_first_post"]) != visit["native_first_post_identity"]
                or identity(visit["image_path"]) != visit["crop_identity"]):
            raise ValueError("First-post source changed after preparation")
        roi = resample_native_roi(visit["native_first_post"], visit)
        cached = np.load(visit["image_path"], allow_pickle=False).astype(np.float32)
        normalized = np.zeros_like(roi)
        foreground = roi != 0
        normalized[foreground] = (roi[foreground] - 142.555409) / 283.804038
        if cached.shape != IMAGE_SHAPE or not np.allclose(cached, normalized, rtol=0.001, atol=0.0001):
            raise ValueError("First-post ROI differs from the original VQ crop")
        crop_error = float(np.abs(cached - normalized).max())
        spacing = visit["crop_geometry"]["spacing_xyz_mm"][::-1]
    channel = pillar_channel(roi, spacing)
    volume = torch.stack([channel, channel, channel])
    if tuple(volume.shape) != PILLAR_SHAPE or not torch.isfinite(volume).all():
        raise ValueError("Invalid repeated single-phase Pillar input")
    audit = {"input_mode": cfg["input_mode"], "shape_chwd": list(volume.shape),
             "identical_channels": bool(torch.equal(volume[0], volume[1]) and torch.equal(volume[1], volume[2])),
             "vq_crop_max_absolute_difference": crop_error,
             "source_identity": visit.get("input_identity", visit.get("native_first_post_identity"))}
    return volume, audit


def embedding_path(cfg, visit):
    pid = visit["canonical_patient_id"]
    return Path(cfg["output_dir"]) / "embeddings/real" / pid / f"{pid}_T{visit['timepoint']}.pt"


def feature_complete(cfg, visit):
    path = embedding_path(cfg, visit)
    audit = Path(cfg["output_dir"]) / "feature_audits" / f"{path.stem}.json"
    if not path.is_file() or not audit.is_file():
        return False
    value = read_json(audit)
    expected = visit.get("input_identity", visit.get("native_first_post_identity"))
    if (value.get("input_mode") != cfg["input_mode"] or not value.get("identical_channels")
            or value.get("source_identity") != expected or value.get("embedding_identity") != identity(path)):
        raise ValueError("Existing feature belongs to a different or changed input")
    validate_embedding(path)
    return True


def extract(cfg, cohort, role, patient_ids=None):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    visits = [v for v in cohort["visits"] if v["split"] == role
              and (patient_ids is None or v["canonical_patient_id"] in patient_ids)]
    visits.sort(key=lambda v: (v["canonical_patient_id"], v["timepoint"]))
    pending = [v for v in visits if not feature_complete(cfg, v)]
    completed, initial, started = len(visits) - len(pending), len(visits) - len(pending), time.perf_counter()
    stage = "extracting_development" if role == "train" else "extracting_frozen_holdout"
    progress(cfg, stage, completed=completed, total=len(visits), smoke_subset=patient_ids is not None)
    if pending:
        model = load_frozen_pillar()
        with ThreadPoolExecutor(max_workers=int(cfg["preprocess_workers"])) as pool:
            queue, submitted = {}, 0
            for index, visit in enumerate(pending):
                while submitted < min(len(pending), index + int(cfg["preprocess_workers"])):
                    queue[submitted] = pool.submit(build_volume, cfg, pending[submitted])
                    submitted += 1
                volume, audit = queue.pop(index).result()
                vector = pillar_forward(model, volume[None].to("cuda:0"))[0].clone()
                path = embedding_path(cfg, visit)
                save_tensor(path, vector)
                audit.update(embedding_identity=identity(path), frozen_encoder=True, created_utc=now())
                write_json(Path(cfg["output_dir"]) / "feature_audits" / f"{path.stem}.json", audit)
                completed += 1
                elapsed = time.perf_counter() - started
                if completed == len(visits) or completed % 10 == 0 or index == 0:
                    progress(cfg, stage, completed=completed, total=len(visits),
                             smoke_subset=patient_ids is not None,
                             remaining_seconds=elapsed / (completed - initial) * (len(visits) - completed))
                    disk_gate(cfg)
                del volume, vector
                if (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").exists():
                    raise InterruptedError("Pause requested after a completed image")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if patient_ids is None:
        marker = "REAL_FEATURES_COMPLETE.json" if role == "train" else "HOLDOUT_FEATURES_COMPLETE.json"
        unchanged_json(Path(cfg["output_dir"]) / marker,
                       {"schema": SCHEMA, "complete": True, "visits": len(visits), "role": role,
                        "input_mode": cfg["input_mode"]})


def optimization_config(cfg):
    result = yaml.safe_load(Path(cfg["optimization_reference"]).read_text())
    result.update(output_dir=str(Path(cfg["output_dir"]) / "optimization"),
                  baseline_run=cfg["output_dir"], input_config=cfg["config_path"], gpu=cfg["gpu"],
                  jobs=cfg["training_jobs"], feature_sources=["physical_roi"],
                  formal_seeds=cfg["seeds"],
                  candidates={k: result["candidates"][k] for k in ("baseline", "unbounded_l2")})
    result.pop("preprocessing", None)
    for key in ("inner_folds", "inner_fold_seed", "tuning_seeds", "auroc_tolerance", "bootstrap_samples"):
        result[key] = cfg[key]
    result["single_phase_input_mode"] = cfg["input_mode"]
    return result


def smoke(cfg, cohort):
    output = Path(cfg["output_dir"])
    if (output / "SMOKE_COMPLETE.json").exists():
        return
    folds = read_json(output / "folds.json")
    metadata = pd.read_csv(output / "metadata_enriched.csv", dtype={"pid": str}).set_index("pid")
    available_t0 = {v["canonical_patient_id"] for v in cohort["visits"] if v["timepoint"] == 0}
    chosen = {}
    for role in ("train", "val"):
        ids = [pid for pid in folds[0][f"{role}_ids"] if pid in available_t0]
        chosen[role] = []
        for label in (0, 1):
            chosen[role].extend([pid for pid in ids if int(metadata.loc[pid, "pCR"]) == label][:2])
        if len(chosen[role]) != 4:
            raise ValueError("Smoke needs both labels in disjoint development subsets")
    extract(cfg, cohort, "train", set(chosen["train"] + chosen["val"]))
    recipes = recipe_configs(cfg)
    records = []
    for depth, count in DEPTHS.items():
        train = _canonical_split(_load_split(output / "embeddings/real", output / "metadata_enriched.csv", chosen["train"]), count)
        val = _canonical_split(_load_split(output / "embeddings/real", output / "metadata_enriched.csv", chosen["val"]), count)
        candidates = {"baseline": yaml.safe_load(Path(recipes[depth]).read_text())["downstream"]}
        if cfg["branch"] == "first_post":
            from src.first_post_optimization import effective_config
            candidates["unbounded_l2"] = effective_config(optimization_config(cfg), "unbounded_l2", count)
        for name, effective in candidates.items():
            effective = {**effective, "epochs": 2, "patience": 2, "scheduler_t_max": 2,
                         "epoch_selection": effective.get("epoch_selection", "auroc")}
            seed = 7720 + count
            torch.manual_seed(seed)
            initial = TDN({"downstream": effective}).state_dict()
            model, state = fit_tdn(train, val, effective, seed, count, "cuda:0")
            if not any(not torch.equal(initial[k], v) for k, v in state["model_state"].items()):
                raise RuntimeError("Smoke optimizer did not update any model weight")
            logits = _prior_from_state(val["clinical"], state["clinical_prior"])
            before, _ = predict(model, tensor_split(val, logits, "cuda:0"))
            path = output / "smoke" / depth / name / "model.pt"
            save_tensor(path, {**state, "effective_config": effective, "smoke_only": True})
            restored = torch.load(path, weights_only=False, map_location="cpu")
            model.load_state_dict(restored["model_state"], strict=True)
            after, _ = predict(model, tensor_split(val, logits, "cuda:0"))
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-7)
            if not np.isfinite(pd.DataFrame(state["history"]).select_dtypes(include="number").to_numpy()).all():
                raise FloatingPointError("Nonfinite smoke history")
            records.append({"depth": depth, "candidate": name, "epochs": state["epochs_trained"],
                            "weights_updated": True, "reload_max_error": float(np.max(np.abs(before - after)))})
            del model
    write_json(output / "SMOKE_COMPLETE.json", {"schema": SCHEMA, "complete": True, "records": records,
               "development_patients": 8, "holdout_used": False, "completed_utc": now()})
    progress(cfg, "smoke_complete", fits=len(records), development_patients=8)


def model_references(cfg):
    output = Path(cfg["output_dir"])
    refs = []
    if cfg["branch"] == "registered_dce0":
        for depth, count in DEPTHS.items():
            for seed in cfg["seeds"]:
                for fold in range(5):
                    path = _training_dir(output / "tdn", "formal", depth, "prespecified", seed, fold) / "best.pt"
                    refs.append({"depth": depth, "max_tp": count, "seed": seed, "fold": fold,
                                 "path": str(path), "identity": identity(path)})
    else:
        models = read_json(output / "optimization/selection/physical_roi_outer_models.json")
        for row in models:
            if row["arm"] == "selected":
                path = Path(row["path"]) / "model.pt"
                depth = next(name for name, count in DEPTHS.items() if count == row["depth"])
                refs.append({"depth": depth, "max_tp": row["depth"], "seed": row["seed"], "fold": row["outer"],
                             "path": str(path), "identity": identity(path)})
    unchanged_json(output / "frozen_models.json", {"models": refs, "holdout_used_for_selection": False})
    return refs


def evaluate_real(cfg, cohort, refs):
    output = Path(cfg["output_dir"])
    heldout = cohort["split"]["val"]
    raw = _load_split(output / "embeddings/real", output / "holdout_metadata.csv", heldout)
    fold_rows = []
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    for index, ref in enumerate(refs):
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("Frozen pCR checkpoint changed")
        saved = torch.load(ref["path"], weights_only=False, map_location="cpu")
        task = saved.get("task", {})
        trained = set(task.get("train_ids", saved.get("train_ids", [])))
        validated = set(task.get("val_ids", saved.get("validation_ids", [])))
        if not trained or (trained | validated) & set(heldout):
            raise ValueError("Holdout patient entered classifier fitting or selection")
        effective = task.get("effective", saved.get("effective_config"))
        split = _canonical_split(raw, ref["max_tp"])
        prior = _prior_from_state(split["clinical"], saved["clinical_prior"])
        model = TDN({"downstream": effective}).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["model_state"], strict=True)
        probability, _ = predict(model, tensor_split(split, prior, "cuda:0"))
        fold_rows.append(pd.DataFrame({"patient_id": heldout, "label": split["labels"].astype(int),
                                       "probability": probability.astype(np.float64), "depth": ref["depth"],
                                       "seed": ref["seed"], "fold": ref["fold"]}))
        del model
        if (index + 1) % 25 == 0:
            progress(cfg, "evaluating_frozen_real_holdout", completed=index + 1, total=len(refs))
    folds = pd.concat(fold_rows, ignore_index=True)
    if not (folds.groupby(["depth", "seed", "patient_id"]).size() == 5).all():
        raise ValueError("Every holdout prediction must average exactly five folds")
    predictions = folds.groupby(["depth", "seed", "patient_id", "label"], as_index=False).probability.mean()
    records = []
    for (depth, seed), frame in predictions.groupby(["depth", "seed"]):
        if set(frame.patient_id) != set(heldout) or len(frame) != len(heldout):
            raise ValueError("Real holdout coverage changed")
        records.append({"depth": depth, "seed": int(seed), "patients": len(heldout),
                        **metric_values(frame.label.to_numpy(), frame.probability.to_numpy())})
    metrics = pd.DataFrame(records)
    summary = metrics.groupby("depth")[["auroc", "logloss", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reindex(DEPTHS).reset_index()
    for name, frame in (("fold_predictions", folds), ("predictions", predictions),
                        ("seed_metrics", metrics), ("summary", summary)):
        _atomic_csv(output / "evaluation/real" / f"{name}.csv", frame)
    write_json(output / "evaluation/real/COMPLETE.json", {"schema": SCHEMA, "complete": True,
               "patients": len(heldout), "models": len(refs), "optimizer_updates": 0,
               "aggregation": "five_fold_probability_mean_then_mean_seed_metric",
               "test_role": cohort["test_role"], "completed_utc": now()})
    lines = [f"# {cfg['branch']}: repeated single-phase pCR", "",
             f"Input: {cohort['phase_order']}; frozen Pillar, 1152D embeddings, four T0-starting windows.",
             f"Development: {len(cohort['split']['train'])}; historical internal holdout: {len(heldout)}.",
             "All fitting and model selection use real development inputs. Holdout inference uses fixed weights.",
             "The holdout was previously used for upstream validation and is not an untouched end-to-end test.",
             "Five fold probabilities are averaged within each seed; reported metrics are means over ten seeds.", "",
             "| Window | AUROC mean | Seed SD | Log loss mean |", "|---|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['depth']} | {row['auroc_mean']:.5f} | {row['auroc_std']:.5f} | {row['logloss_mean']:.5f} |")
    lines.extend(["", "Generated-image evaluation is not included in this real-input training run.",
                  "Source preprocessing and patient availability follow the original cohort. Single-phase repetition does not restore enhancement dynamics.", ""])
    _atomic_text(output / "README.md", "\n".join(lines))
    return summary.to_dict("records")

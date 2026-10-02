"""Fixed-cohort decoded MRI checks in the codec's original intensity coordinates.

The masks are model-derived proxies, never expert tumour annotations. Future
images and masks are read only after the complete source-only forecast finishes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import csv
import json
import shutil

import numpy as np
import torch

from .io import autocast, digest, read_json, write_json
from .legacy.codec import load_codec
from .losses_v2 import interval_plan
from .rollout import NoiseLedger

SCHEMA = "responsewm_imaging_bundle_v3"


def select_complete_patients(patients, split="val", count=8):
    """Selection uses identifiers and visit availability, never outcome labels."""
    if count < 1:
        raise ValueError("Imaging cohort must contain at least one patient")
    eligible = [p for p in patients if p["split"] == split
                and {v["stage"] for v in p["visits"]} == {0, 1, 2, 3}
                and all(v["available_at"] == v["stage"] for v in p["visits"])]
    return sorted(eligible, key=lambda p: p["patient_key"])[:count]


def _source_path(data_root, declared):
    """Resolve the audited local mirror without accidentally using stale roots."""
    p = Path(declared)
    mirrored = Path(data_root) / p.parent.name / p.name
    if mirrored.exists():
        return mirrored
    p = p if p.is_absolute() else Path(data_root) / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _pseudo_valid(mask, metadata):
    if not np.any(mask):
        return False, "empty_pseudo_mask"
    measurement = metadata.get("measurement", {})
    if measurement.get("largest_clipped") or measurement.get("all_clipped"):
        return False, "pseudo_mask_clipped_by_crop"
    retention = measurement.get("retained_fraction")
    if retention is not None and retention < .95:
        return False, "less_than_95_percent_registered_mask_retained"
    return True, None


def prepare_bundle(trajectory_manifest, data_root, output, *, count=8, split="val"):
    """Materialize only a prespecified complete-case subset as portable NPZs."""
    data_root, output = Path(data_root), Path(output)
    trajectories = read_json(trajectory_manifest)
    images = read_json(data_root / "image_manifest.json")
    masks = read_json(data_root / "mask_manifest.json")
    old = read_json(data_root / "manifest.json")
    if images.get("status") != "passed" or masks.get("status") != "passed":
        raise ValueError("Source image/mask audits did not pass")
    if images["phase_order"] != trajectories["phase_order"]:
        raise ValueError("Image and trajectory phase order differ")
    if images["image_normalization"] != old["image_normalization"]:
        raise ValueError("Image and latent source intensity coordinates differ")
    codec_source = data_root / "codec.pt"
    codec_identity = "sha256:" + digest(codec_source)
    if codec_identity != trajectories["vq_identity"]:
        raise ValueError("Codec identity differs from trajectory data")
    selected = select_complete_patients(trajectories["patients"], split, count)
    if len(selected) != count:
        raise ValueError("Insufficient complete patients for the prespecified cohort")
    image_index = {v["view_id"]: v for v in images["views"]}
    mask_index = {v["view_id"]: v for v in masks["views"]}
    latent_index = {v["id"]: v for v in old["views"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "views").mkdir(exist_ok=True)
    entries = []
    for patient in selected:
        pid = patient["patient_key"]
        source_grid, source_roi = None, None
        for visit in patient["visits"]:
            stage = visit["stage"]
            key = f"{pid}:T{stage}"
            image_meta, mask_meta, latent_meta = image_index[key], mask_index.get(key), latent_index[key]
            if any(v["split"] != split or v["patient_id"] != pid for v in (image_meta, latent_meta)):
                raise ValueError("Image/latent split or patient mismatch")
            image_path = _source_path(data_root, image_meta["path"])
            with np.load(image_path, allow_pickle=False) as source:
                arrays = {k: np.asarray(source[k]) for k in ("image", "latent", "roi", "support")}
            image, latent, roi, support = (arrays[k] for k in ("image", "latent", "roi", "support"))
            if image.shape != tuple(images["arrays"]["image"]["shape"]) or latent.shape != tuple(trajectories["latent_shape"]):
                raise ValueError("Unexpected registered image or latent geometry")
            if not np.isfinite(image).all() or not np.isfinite(latent).all():
                raise ValueError("Nonfinite source image/latent")
            if roi.dtype != np.bool_ or support.dtype != np.bool_ or roi.shape != (1, *image.shape[1:]) or support.shape != image.shape:
                raise ValueError("Invalid source ROI/support mask")
            raw_path = Path(visit["latent"])
            raw_path = raw_path if raw_path.is_absolute() else Path(trajectory_manifest).resolve().parent / raw_path
            raw = np.load(raw_path, allow_pickle=False)
            if not np.array_equal(latent.astype(np.float32), raw.astype(np.float32)):
                raise ValueError("Bundled latent differs from the training trajectory")
            if not latent_meta.get("source_available_grid") or latent_meta["geometry"]["shape_zyx"] != list(image.shape[1:]):
                raise ValueError("Source-available shared geometry is not verified")
            grid = (latent_meta["grid_id"], latent_meta["geometry"])
            if source_grid is None:
                source_grid, source_roi = grid, roi
            elif grid != source_grid or not np.array_equal(roi, source_roi):
                raise ValueError("Within-patient fixed T0 grid or localization changed")
            pseudo = np.zeros(image.shape[1:], dtype=bool)
            pseudo_ok, pseudo_reason = False, "pseudo_mask_unavailable"
            if mask_meta:
                if mask_meta["split"] != split or mask_meta["patient_id"] != pid or mask_meta["label_kind"] != "mapped_model_pseudo_mask":
                    raise ValueError("Unsupported mask identity/provenance")
                mask_path = _source_path(data_root, mask_meta["path"])
                with np.load(mask_path, allow_pickle=False) as data:
                    pseudo = np.asarray(data["mask"])
                if pseudo.dtype != np.bool_ or pseudo.shape != image.shape[1:]:
                    raise ValueError("Invalid pseudo mask")
                pseudo_ok, pseudo_reason = _pseudo_valid(pseudo, mask_meta)
            rel = f"views/{pid}_T{stage}.npz"
            np.savez_compressed(output / rel, image=image, latent=latent, fixed_roi=roi,
                                support=support, pseudo_roi=pseudo)
            entries.append({"patient_key": pid, "stage": stage, "split": split, "file": rel,
                            "sha256": digest(output / rel), "shape": list(image.shape),
                            "grid_id": latent_meta["grid_id"], "geometry": latent_meta["geometry"],
                            "source_sha256": digest(image_path), "latent_sha256": digest(raw_path),
                            "support_valid": bool(support.any()), "fixed_roi_valid": bool((roi & support).any()),
                            "pseudo_roi_valid": pseudo_ok, "pseudo_roi_exclusion": pseudo_reason,
                            "pseudo_measurement": None if mask_meta is None else mask_meta.get("measurement")})
    shutil.copy2(codec_source, output / "codec.pt")
    manifest = {"schema": SCHEMA, "split": split, "patient_keys": [p["patient_key"] for p in selected],
                "selection": "first_identifiers_sorted_complete_T0_T1_T2_T3_no_outcome_selection",
                "requested_patients": count, "phase_order": images["phase_order"], "vq_identity": codec_identity,
                "codec_path": "codec.pt", "image_normalization": images["image_normalization"],
                "metric_coordinates": "original_fixed_training_foreground_zscore_no_per_image_renormalization",
                "roi_provenance": {"fixed_roi": "fixed_T0_model_predicted_localization",
                                   "pseudo_roi": "mapped_model_pseudo_mask_not_expert_ground_truth",
                                   "support": "acquired_support_per_phase",
                                   "pseudo_retention_threshold": .95},
                "source_audit": {n: digest(data_root / n) for n in ("image_manifest.json", "mask_manifest.json", "manifest.json")},
                "trajectory_manifest_sha256": digest(trajectory_manifest), "visits": entries}
    write_json(output / "manifest.json", manifest)
    summary = {"manifest": str(output / "manifest.json"), "patients": len(selected), "visits": len(entries),
               "patient_keys": manifest["patient_keys"], "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
               "pseudo_valid_visits": sum(v["pseudo_roi_valid"] for v in entries), "roi_provenance": manifest["roi_provenance"]}
    write_json(output / "audit.json", summary)
    return summary


def image_errors(prediction, target, mask=None):
    """Absolute intensity errors, with missing/empty masks explicitly excluded."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("MRI shape mismatch or nonfinite intensity")
    if mask is None:
        mask = np.ones(target.shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise ValueError("Metric masks must be boolean")
    mask = np.broadcast_to(mask, target.shape)
    if not mask.any():
        return None
    error = prediction.astype(np.float64)[mask] - target.astype(np.float64)[mask]
    mse = float(np.mean(error * error))
    return {"mae": float(np.mean(np.abs(error))), "mse": mse, "rmse": float(np.sqrt(mse)),
            "bias": float(np.mean(error)), "voxels": int(mask.sum())}


def _regions(arrays, entry):
    regions = {"whole_crop": np.ones(arrays["image"].shape, dtype=bool)}
    if entry["support_valid"]:
        regions["acquired_support"] = arrays["support"]
    if entry["fixed_roi_valid"]:
        regions["fixed_T0_proxy_ROI"] = arrays["fixed_roi"] & arrays["support"]
    if entry["pseudo_roi_valid"]:
        regions["current_pseudo_ROI"] = arrays["pseudo_roi"][None] & arrays["support"]
    return regions


def _rows(prediction, target, regions, *, pid, stage, method, reference, sample=None):
    rows = []
    for region, mask in regions.items():
        for phase in ("all", 0, 1, 2):
            pred = prediction if phase == "all" else prediction[phase:phase+1]
            truth = target if phase == "all" else target[phase:phase+1]
            masked = mask if phase == "all" else np.broadcast_to(mask, target.shape)[phase:phase+1]
            errors = image_errors(pred, truth, masked)
            if errors is not None:
                rows.append({"patient_key": pid, "stage": stage, "method": method, "reference": reference,
                             "region": region, "phase": str(phase), "sample": sample, **errors})
    return rows


def _summarize(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[k] for k in ("stage", "method", "reference", "region", "phase"))
        grouped[key][row["patient_key"]].append(row)
    result = []
    for key, patients in sorted(grouped.items()):
        result.append({**dict(zip(("stage", "method", "reference", "region", "phase"), key)),
                       "patients": len(patients), "weighting": "sample_mean_within_patient_then_patient_mean",
                       **{metric: float(np.mean([np.mean([r[metric] for r in values]) for values in patients.values()]))
                          for metric in ("mae", "mse", "rmse", "bias")}})
    return result


def _persistence_comparisons(summaries):
    keys = ("stage", "reference", "region", "phase")
    baseline = {tuple(r[k] for k in keys): r for r in summaries if r["method"] == "decoded_T0_persistence"}
    result = []
    for row in summaries:
        if row["method"] not in {"generated_sample", "generated_ensemble_image_mean"}:
            continue
        ref = baseline[tuple(row[k] for k in keys)]
        result.append({**{k: row[k] for k in keys}, "method": row["method"], "patients": row["patients"],
                       "mae_difference_from_persistence": row["mae"] - ref["mae"],
                       "mse_difference_from_persistence": row["mse"] - ref["mse"],
                       "mae_fractional_improvement": 1 - row["mae"] / ref["mae"] if ref["mae"] > 0 else None})
    return result


def _future_token_probe(model, belief, trace, diseases):
    """Readout-only sensitivity: replace predicted disease, keep real memory."""
    events = tuple(replace(e, disease=diseases[e.stage]) if e.origin == "predicted" else e
                   for e in trace.final_state.model_state_log)
    final = replace(trace.final_state, model_state_log=events)
    altered = replace(trace, internal_full_trace=trace.internal_full_trace[:-1] + (final,))
    return float(model.query_pcr(belief, mode="marginal_current", future_trace=altered).probability[0])


def _figure(path, pid, actual, decoded, persistence, generated_mean, generated_sample, roi):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    # Same source-selected slice and same fixed intensity scale for every image.
    counts = roi.reshape(-1, *roi.shape[-3:]).sum((0, 2, 3))
    section = int(counts.argmax()) if counts.any() else actual.shape[1] // 2
    panels = [actual, decoded, persistence, generated_mean, generated_sample]
    titles = ["Actual MRI", "Codec reconstruction", "Decoded T0 persistence", "Generated image mean", "Generated sample 0"]
    fig, axes = plt.subplots(3, 5, figsize=(14, 8), squeeze=False)
    for phase in range(3):
        for col, (value, title) in enumerate(zip(panels, titles)):
            im = axes[phase, col].imshow(value[phase, section], cmap="gray", vmin=-1, vmax=4)
            axes[phase, col].axis("off")
            if phase == 0:
                axes[phase, col].set_title(title, fontsize=9)
            if col == 0:
                axes[phase, col].text(-.1, .5, f"phase {phase}", transform=axes[phase, col].transAxes,
                                      rotation=90, va="center")
    fig.suptitle(f"{pid}; fixed T0 proxy ROI slice {section}; fixed training-z-score display [-1,4]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_imaging(model, store, cfg, output, split="val"):
    """Decode full T0 forecasts for the immutable imaging cohort; save artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(cfg.protocol.imaging_manifest).resolve()
    bundle = read_json(manifest_path)
    if bundle.get("schema") != SCHEMA or bundle["split"] != split:
        raise ValueError("Imaging bundle schema/split mismatch")
    selected = select_complete_patients(store.patients, split, cfg.protocol.image_eval_cases)
    if [p["patient_key"] for p in selected] != bundle["patient_keys"]:
        raise ValueError("Fixed imaging cohort does not match this store/protocol")
    codec_path = Path(cfg.protocol.codec_path) if cfg.protocol.codec_path else manifest_path.parent / bundle["codec_path"]
    codec_path = codec_path if codec_path.is_absolute() else manifest_path.parent / codec_path
    if "sha256:" + digest(codec_path) != bundle["vq_identity"] or bundle["vq_identity"] != store.manifest["vq_identity"]:
        raise ValueError("Imaging codec identity mismatch")
    if bundle["phase_order"] != store.manifest["phase_order"]:
        raise ValueError("Imaging phase order mismatch")
    device = next(model.parameters()).device
    rng_devices = [device.index or 0] if device.type == "cuda" else []
    # Loading constructs modules before loading weights and consumes CPU RNG.
    # Evaluation must not change subsequent training/noise sampling sequences.
    with torch.random.fork_rng(devices=rng_devices):
        decoder = load_codec(codec_path, device)
    precision = cfg.training.precision if device.type != "cpu" else "fp32"
    index = {p["patient_key"]: i for i, p in enumerate(store.patients)}
    visits = {(v["patient_key"], v["stage"]): v for v in bundle["visits"]}
    rows, pcr_probes, figure_paths, exclusions = [], [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(cfg.protocol.validation_seed)
            for patient in selected:
                pid = patient["patient_key"]
                # make_prefix is input-only: future arrays/labels are not read.
                inp = store.make_prefix(index[pid], 0, device=device)
                with autocast(device, precision):
                    belief = model.initialize(inp)
                    trace = model.forecast(belief, samples=cfg.training.validate_samples,
                                           steps=cfg.sampling.inference_steps,
                                           noise_ledger=NoiseLedger(cfg.protocol.validation_seed), plan=interval_plan(inp))
                if trace.stage_ids != (1, 2, 3):
                    raise ValueError("Imaging evaluation requires complete free T0 rollout")
                mean, std = model.encoder.latent_mean, model.encoder.latent_std
                raw = trace.latent[0].float() * std[0, None] + mean[0, None]
                baseline_raw = inp.observed[:, 0].float() * std + mean
                persistence = decoder.decode(baseline_raw)[0].float().cpu().numpy()
                roundtrip_states, reencoded_states = {}, {}
                for stage in range(4):
                    entry = visits[(pid, stage)]
                    asset = (manifest_path.parent / entry["file"]).resolve()
                    if manifest_path.parent not in asset.parents or digest(asset) != entry["sha256"]:
                        raise ValueError("Imaging asset path/hash mismatch")
                    with np.load(asset, allow_pickle=False) as source:
                        arrays = {k: np.asarray(source[k]) for k in source.files}
                    if not entry["pseudo_roi_valid"]:
                        exclusions.append({"patient_key": pid, "stage": stage, "region": "current_pseudo_ROI",
                                           "reason": entry["pseudo_roi_exclusion"]})
                    target = arrays["image"]
                    real_raw = torch.as_tensor(arrays["latent"].astype(np.float32), device=device)[None]
                    real_decoded = decoder.decode(real_raw)[0].float().cpu().numpy()
                    regions = _regions(arrays, entry)
                    rows += _rows(real_decoded, target, regions, pid=pid, stage=stage, method="codec_reconstruction",
                                  reference="actual_MRI")
                    if stage == 0:
                        source_roi = arrays["fixed_roi"]
                        continue
                    decoded_samples, roundtrip = [], []
                    for k in range(raw.shape[0]):
                        generated = decoder.decode(raw[k:k+1, stage-1])
                        decoded_samples.append(generated[0].float().cpu().numpy())
                        encoded_again = decoder.encode(generated)
                        with autocast(device, precision):
                            roundtrip.append(model.target_encoder((encoded_again - mean) / std).disease)
                    generated_mean = np.mean(decoded_samples, axis=0)
                    roundtrip_states[stage] = torch.stack(roundtrip, 1)
                    reencoded_states[stage] = trace.image_states[stage-1]
                    for reference, truth in (("actual_MRI", target), ("decoded_real_latent", real_decoded)):
                        rows += _rows(persistence, truth, regions, pid=pid, stage=stage,
                                      method="decoded_T0_persistence", reference=reference)
                        rows += _rows(generated_mean, truth, regions, pid=pid, stage=stage,
                                      method="generated_ensemble_image_mean", reference=reference)
                        for k, value in enumerate(decoded_samples):
                            rows += _rows(value, truth, regions, pid=pid, stage=stage,
                                          method="generated_sample", reference=reference, sample=k)
                    name = f"{pid}_T{stage}.png"
                    _figure(output / name, f"{pid} T{stage}", target, real_decoded, persistence,
                            generated_mean, decoded_samples[0], source_roi)
                    figure_paths.append(name)
                with autocast(device, precision):
                    clinical_probability = model.pcr.prior(belief.clinical, belief.clinical_mask).float().sigmoid()
                    pcr_probes.append({"patient_key": pid, "clinical_only_probability": float(clinical_probability[0]),
                                       "observed_probability": float(model.query_pcr(belief).probability[0]),
                                       "joint_probability": float(trace.probability[0]),
                                       "latent_reencoded_probability": _future_token_probe(model, belief, trace, reencoded_states),
                                       "decoded_MRI_roundtrip_probability": _future_token_probe(model, belief, trace, roundtrip_states),
                                       "label": patient["target"]["pcr"]})
    finally:
        model.train(was_training)
        del decoder
    summaries = _summarize(rows)
    result = {"schema": "responsewm_imaging_evaluation_v3", "split": split, "patients": len(selected),
              "patient_keys": bundle["patient_keys"], "selection": bundle["selection"],
              "bundle_sha256": digest(manifest_path), "vq_identity": bundle["vq_identity"],
              "origin_stage": 0, "generated_stages": [1, 2, 3], "samples": cfg.training.validate_samples,
              "steps": cfg.sampling.inference_steps, "seed": cfg.protocol.validation_seed,
              "metric_coordinates": bundle["metric_coordinates"], "image_normalization": bundle["image_normalization"],
              "roi_provenance": bundle["roi_provenance"], "summaries": summaries,
              "persistence_comparisons": _persistence_comparisons(summaries),
              "roi_exclusions": exclusions, "pcr_probes": pcr_probes,
              "pcr_probe_interpretation": "readout_only_future_disease_token_replacement_on_same_trajectories_and_same_real_source_memory; not_dynamic_rerollout",
              "figures": figure_paths,
              "limitations": ["Eight prespecified complete development-validation patients, not an independent test.",
                              "Model-derived localization/pseudo masks are not expert tumour truth.",
                              "Fixed crop may omit disease outside the field of view.",
                              "Ensemble image mean is average of decoded samples, not decode of average latent.",
                              "Codec reconstruction is a reference floor, not a mathematical lower bound on forecasting error."]}
    write_json(output / "imaging_evaluation.json", result)
    with (output / "imaging_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return result

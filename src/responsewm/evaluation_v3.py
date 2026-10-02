"""Fixed-cohort deployment evaluation with source-only, shared-noise forecasts.

The reencoded comparison changes only the PCR evidence tokens on the SAME
continuous VQ-latent trajectory. It does not rerun dynamics, decode MRI, or
replace a prediction with a real future MRI. Decoded MRI evaluation is separate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math

import torch

from .io import autocast
from .losses_v2 import interval_plan
from .metrics import classification_metrics, paired_bootstrap_delta, patient_bootstrap
from .rollout import NoiseLedger

PROBABILITY_KEYS = {
    "clinical_only": "clinical_only_probability",
    "real_mri_history": "observed_probability",
    "generated_future_joint": "marginal_probability",
    "reencoded_generated_latent": "reencoded_marginal_probability",
}


def fixed_cohorts(store, split="val"):
    """Prespecify cohorts from availability, never labels or model predictions."""
    indices = sorted(store.by_split[split], key=lambda i: store.patients[i]["patient_key"])
    if not indices:
        raise ValueError(f"Empty fixed evaluation split: {split}")
    complete = [i for i in indices if {v["stage"] for v in store.patients[i]["visits"]
                                     if v["available_at"] <= v["stage"]} == {0, 1, 2, 3}]

    def describe(ids):
        keys = [store.patients[i]["patient_key"] for i in ids]
        identity = json.dumps({"manifest": store.manifest_digest, "split": split,
                               "patient_keys": keys}, sort_keys=True, separators=(",", ":"))
        return {"patient_indices": ids, "patient_keys": keys, "patients": len(ids),
                "sha256": hashlib.sha256(identity.encode()).hexdigest()}

    return {"primary_t0": describe(indices), "common_complete_t0_t3": describe(complete)}


def _batches(store, indices, origin, count, *, representation=False):
    """Group complete observed histories; retain incomplete trajectories at T0."""
    groups = defaultdict(list)
    for i in indices:
        patient = store.patients[i]
        prefix = tuple(v["stage"] for v in patient["visits"]
                       if v["stage"] <= origin and v["available_at"] <= origin)
        key = (prefix,)
        if representation:
            future = tuple(any(v["stage"] == s for v in patient["visits"])
                           for s in range(origin + 1, 4))
            key += (future, patient["target"]["pcr"] is not None)
        groups[key].append(i)
    for group in groups.values():
        for start in range(0, len(group), count):
            yield [(i, origin) for i in group[start:start + count]]


def reencoded_trace_readout(model, trace):
    """Change predicted disease tokens only; preserve trajectory and source memory."""
    if tuple(trace.stage_ids) != tuple(range(trace.origin_stage + 1, 4)):
        raise ValueError("Shared-trajectory readout requires a complete remaining future")
    images = dict(zip(trace.stage_ids, trace.image_states))
    changed_states = []
    for state in trace.all_states:
        events = tuple(replace(event, disease=images[event.stage])
                       if event.origin == "predicted" else event
                       for event in state.model_state_log)
        changed_states.append(replace(state, model_state_log=events))
    changed = replace(trace, internal_full_trace=tuple(changed_states), pcr_marginal=None)
    return model.query_pcr(trace.source_state, mode="marginal_current", future_trace=changed)


def _image_scores(samples, truth, persistence):
    """Each output is a per-patient scalar in standardized continuous latent space."""
    samples, truth, persistence = samples.float(), truth.float(), persistence.float()
    error = samples - truth[:, None]
    dimensions = tuple(range(2, error.ndim))
    mean_error = samples.mean(1) - truth
    spatial = tuple(range(1, truth.ndim))
    distances = error.square().mean(dimensions).sqrt().mean(1)
    spread = torch.zeros_like(distances)
    # Sum pair distances without materializing a [B,K,K,C,D,H,W] tensor.
    for a in range(samples.shape[1]):
        for b in range(a + 1, samples.shape[1]):
            spread += (samples[:, a] - samples[:, b]).square().mean(spatial).sqrt()
    spread = spread * (2.0 / samples.shape[1] ** 2)
    return {
        "sample_mae": error.abs().mean(dimensions).mean(1),
        "ensemble_mean_mae": mean_error.abs().mean(spatial),
        "ensemble_mean_mse": mean_error.square().mean(spatial),
        "persistence_mae": (persistence - truth).abs().mean(spatial),
        "persistence_mse": (persistence - truth).square().mean(spatial),
        "energy_score": distances - 0.5 * spread,
        "sample_spread_rms": spread,
    }


def _classification_group(rows, bootstrap, seed):
    labelled = [r for r in rows if r["label"] is not None]
    result = {"patients": len(rows), "labelled_patients": len(labelled), "modalities": {}}
    if not labelled:
        return result
    labels = [r["label"] for r in labelled]
    keys = [r["patient_key"] for r in labelled]
    available = []
    for name, field in PROBABILITY_KEYS.items():
        if not all(field in r for r in labelled):
            continue
        available.append(name)
        probabilities = [r[field] for r in labelled]
        result["modalities"][name] = classification_metrics(labels, probabilities)
        if bootstrap:
            result["modalities"][name]["patient_bootstrap"] = patient_bootstrap(
                labels, probabilities, keys, bootstrap, seed)
    if bootstrap:
        result["paired_comparisons"] = {}
        for a, b in (("real_mri_history", "clinical_only"),
                     ("generated_future_joint", "real_mri_history"),
                     ("generated_future_joint", "clinical_only"),
                     ("reencoded_generated_latent", "generated_future_joint")):
            if a in available and b in available:
                result["paired_comparisons"][a + "_minus_" + b] = paired_bootstrap_delta(
                    labels, [r[PROBABILITY_KEYS[a]] for r in labelled],
                    [r[PROBABILITY_KEYS[b]] for r in labelled], keys, bootstrap, seed)
    return result


@torch.no_grad()
def validate(model, store, cfg, stage, *, split="val", bootstrap=0, full=False):
    """Evaluate all T0 patients; full=True adds T1/T2-origin forecast comparisons.

    Routine validation includes real-history predictions at T1/T2/T3 on the
    fixed complete-history cohort. Primary selection always uses all T0 patients.
    Future MRI files are opened for scoring only after the source-only forecast.
    Representation objectives may use real future MRI as representation targets.
    """
    if stage not in {"representation", "flow", "readout", "joint"}:
        raise ValueError("Unknown training stage")
    if cfg.network.readout_source != "joint":
        raise ValueError("V3 fixed deployment protocol requires joint rollout dynamics")
    if bootstrap < 0:
        raise ValueError("Bootstrap repetitions must be nonnegative")
    device = next(model.parameters()).device
    was_training = model.training
    lineage_versions = dict(getattr(model, "_lineage_versions", {}))
    cohorts = fixed_cohorts(store, split)
    indices = cohorts["primary_t0"]["patient_indices"]
    complete = cohorts["common_complete_t0_t3"]["patient_indices"]
    complete_keys = set(cohorts["common_complete_t0_t3"]["patient_keys"])
    seed = cfg.protocol.validation_seed
    batch_size = cfg.protocol.validation_batch_size
    devices = [device.index or 0] if device.type == "cuda" else []
    rows, representation_values, grounding = [], [], []
    generation_by_stage = defaultdict(list)
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            for origin, population in ((0, indices), (1, complete), (2, complete), (3, complete)):
                for tasks in _batches(store, population, origin, batch_size,
                                      representation=stage == "representation" and origin == 0):
                    # Input-only construction makes accidental future-truth use testable.
                    inp = store.batch(tasks, device, supervised=False)
                    with autocast(device, cfg.training.precision):
                        belief = model.initialize(inp)
                        observed = model.query_pcr(belief)
                        clinical = model.pcr.prior(belief.clinical, belief.clinical_mask).float().sigmoid()
                        trace = reencoded = None
                        if stage != "representation" and origin < 3 and (origin == 0 or full):
                            trace = model.forecast(
                                belief, samples=cfg.training.validate_samples,
                                steps=cfg.sampling.inference_steps,
                                noise_ledger=NoiseLedger(seed, "v3-fixed-validation"), plan=interval_plan(inp))
                            reencoded = reencoded_trace_readout(model, trace)
                    batch_rows = []
                    for j, (index, _) in enumerate(tasks):
                        patient = store.patients[index]
                        row = {"patient_key": patient["patient_key"], "origin_stage": origin,
                               "observed_stage_ids": list(belief.observed_stage_ids[j]),
                               "label": patient["target"]["pcr"],
                               "in_common_complete_cohort": patient["patient_key"] in complete_keys,
                               "clinical_only_probability": float(clinical[j]),
                               "observed_probability": float(observed.probability[j])}
                        if trace is not None:
                            row.update(marginal_probability=float(trace.probability[j]),
                                       reencoded_marginal_probability=float(reencoded.probability[j]))
                            for k, target in enumerate(trace.stage_ids):
                                row[f"forecasted_state_probability_T{target}"] = float(trace.stage_probs[j, k])
                        batch_rows.append(row)
                    # Ground truth is deliberately loaded after all prediction calls.
                    if origin == 0 and (trace is not None or stage == "representation"):
                        _, sup = store.batch(tasks, device, supervised=True)
                        if trace is not None:
                            ground = (trace.state.float() - trace.image_state.float()).square().flatten(1).mean(1)
                            grounding.extend(ground.tolist())
                            for k, target in enumerate(trace.output_stages):
                                valid = sup.future_mask[:, target - origin - 1]
                                if not bool(valid.any()):
                                    continue
                                values = _image_scores(trace.latent[:, :, k], sup.future[:, target - origin - 1],
                                                       belief.anchor_latent[:, 0])
                                for j in valid.nonzero(as_tuple=True)[0].tolist():
                                    scores = {name: float(value[j]) for name, value in values.items()}
                                    batch_rows[j][f"latent_scores_T{target}"] = scores
                                    batch_rows[j][f"latent_MAE_T{target}"] = scores["sample_mae"]
                                    generation_by_stage[target].append({"patient_key": batch_rows[j]["patient_key"], **scores})
                        else:
                            from .losses_v3 import representation_objective
                            with autocast(device, cfg.training.precision):
                                _, terms = representation_objective(model, inp, sup)
                            representation_values.extend([terms["loss/reconstruction"] + terms["loss/masked_jepa"]] * len(tasks))
                    rows.extend(batch_rows)
    finally:
        model.train(was_training)
        if hasattr(model, "_lineage_versions"):
            model._lineage_versions = lineage_versions
    rows.sort(key=lambda row: (row["origin_stage"], row["patient_key"]))
    primary_rows = [r for r in rows if r["origin_stage"] == 0]
    primary = _classification_group(primary_rows, bootstrap, seed)
    secondary = {f"T{origin}": _classification_group(
        [r for r in rows if r["origin_stage"] == origin and r["in_common_complete_cohort"]], bootstrap, seed)
        for origin in range(4)}
    generation = {}
    for target, values in sorted(generation_by_stage.items()):
        names = [name for name in values[0] if name != "patient_key"]
        scores = {name: sum(v[name] for v in values) / len(values) for name in names}
        scores.update(patients=len(values), patient_keys=[v["patient_key"] for v in values])
        scores["ensemble_mse_to_persistence_ratio"] = scores["ensemble_mean_mse"] / max(scores["persistence_mse"], 1e-12)
        generation[f"T{target}"] = scores
    generation_score = (sum(v["ensemble_mse_to_persistence_ratio"] for v in generation.values()) / len(generation)
                        if generation else None)
    t0_marginal = primary["modalities"].get("generated_future_joint", {}).get("nll")
    observed_nll = primary["modalities"].get("real_mri_history", {}).get("nll")
    mean = lambda values: sum(values) / len(values) if values else None
    patient_nll = defaultdict(list)
    by_prefix = defaultdict(list)
    for row in rows:
        by_prefix["{" + ",".join(map(str, row["observed_stage_ids"])) + "}"].append(row)
        if row["label"] is not None:
            p = min(max(row["observed_probability"], 1e-12), 1 - 1e-12)
            patient_nll[row["patient_key"]].append(-math.log(p if row["label"] else 1 - p))
    legacy_groups = {}
    for key, group in by_prefix.items():
        scored = _classification_group(group, 0, seed)
        legacy_groups[key] = {"patients": len(group), "cohort": "primary_t0" if group[0]["origin_stage"] == 0 else "common_complete_t0_t3"}
        for old_name, new_name in (("observed", "real_mri_history"), ("marginal", "generated_future_joint")):
            if new_name in scored["modalities"]:
                legacy_groups[key][old_name] = scored["modalities"][new_name]
    if stage == "representation":
        if observed_nll is None:
            raise ValueError("Representation selection needs labelled T0 validation patients")
        selection = mean(representation_values) + cfg.multistage.representation_pcr_weight * observed_nll
    elif stage == "flow":
        selection = generation_score
    else:
        selection = t0_marginal
    if selection is None or not math.isfinite(selection):
        raise ValueError("Fixed cohort cannot support the prespecified selection metric")
    return {
        "selection_score": selection, "generation_score": generation_score,
        "t0_marginal_nll": t0_marginal, "representation_reconstruction_plus_jepa": mean(representation_values),
        "real_prefix_patient_mean_nll": mean([mean(values) for values in patient_nll.values()]),
        "generation_grounding": mean(grounding), "generation_by_target": generation,
        "cohorts": cohorts, "primary_t0": primary, "secondary_common_complete": secondary,
        "by_observed_prefix": legacy_groups, "edge_objectives": {}, "edge_support": {},
        "comparisons": {"design": "same_generated_latent_trajectory_readout_only",
                        "primary": "generated_future_joint", "noise_seed": seed,
                        "noise_trajectory_id": "v3-fixed-validation",
                        "samples": cfg.training.validate_samples, "inference_steps": cfg.sampling.inference_steps,
                        "secondary_future_forecasts": full and stage != "representation",
                        "reencoded_input": "teacher_encoder_of_generated_continuous_VQ_latent",
                        "reencoded_is_decoded_image_roundtrip": False,
                        "generation_space": "train_standardized_continuous_VQ_latent",
                        "generation_selection": "equal_stage_mean_ensemble_MSE_divided_by_persistence_MSE",
                        "energy_estimator": "empirical_energy_with_all_K_squared_pairs",
                        "source_only_inputs": True, "future_truth_loaded_after_prediction": True},
        "rows": rows, "split": split, "patients": len(indices), "manifest_digest": store.manifest_digest,
        "synthetic": store.manifest.get("synthetic", False),
    }

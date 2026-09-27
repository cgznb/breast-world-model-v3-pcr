"""Frozen four-source evaluation and prediction-derived ten-seed reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _canonical_split
from src.data import _vec_from_tensor
from src.first_post_optimization import predict, select_patients, tensor_split
from src.first_post_pcr_data import identity, now, read_json, repo_path, write_json
from src.single_phase_fmbcmri_data import DEPTHS, SCHEMA, progress
from src.single_phase_fmbcmri_training import METRICS, deterministic_runtime, load_development, metrics
from src.tdn import TDN

SOURCES = ("real", "symm", "bifm", "copy")


def overlay(real, directory):
    result = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v
              for k, v in real.items()}
    for index, pid in enumerate(real["pids"]):
        for stage in range(1, 4):
            path = Path(directory) / pid / f"{pid}_T{stage}.pt"
            if not real["masks"][index, stage]:
                if path.exists():
                    raise ValueError("Generated feature invents an unavailable visit")
                continue
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value.shape != (768,):
                raise ValueError("Generated feature is not 768-dimensional")
            result["embs"][index, stage] = _vec_from_tensor(value, require_finite=True)
    if not np.array_equal(result["embs"][:, 0], real["embs"][:, 0]):
        raise ValueError("A generated overlay changed real T0")
    return result


def average_folds(frame, expected_ids, seeds):
    keys = ["source", "window", "seed", "patient_id", "label", "complete_window", "available_prefix"]
    if frame.duplicated(["source", "window", "seed", "patient_id", "outer"]).any():
        raise ValueError("A fold prediction repeats a patient")
    for _, rows in frame.groupby(["source", "window", "seed"], sort=False):
        if set(rows.patient_id) != set(expected_ids):
            raise ValueError("Fixed evaluation patient set changed")
        if not rows.groupby("patient_id").outer.agg(lambda x: set(x) == set(range(5))).all():
            raise ValueError("Every patient needs five distinct fold predictions per seed")
        if (rows.groupby("patient_id")[["label", "complete_window", "available_prefix"]].nunique() != 1).any().any():
            raise ValueError("Patient metadata differs between fold predictions")
    expected_groups = {(s, w, seed) for s in SOURCES for w in DEPTHS for seed in seeds}
    if set(frame.groupby(["source", "window", "seed"]).groups) != expected_groups:
        raise ValueError("Missing source, window or seed predictions")
    result = frame.groupby(keys, as_index=False, sort=False).probability.mean()
    baseline = result[result.window == "T0"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    if not np.array_equal(baseline.to_numpy(), np.repeat(baseline[["real"]].to_numpy(), len(SOURCES), axis=1)):
        raise ValueError("Four-source T0 probabilities are not identical")
    return result


def metric_tables(predictions, population):
    records = []
    for (source, window, seed), frame in predictions.groupby(["source", "window", "seed"], sort=False):
        for subset, rows in (("all_prefixes", frame), ("complete_window", frame[frame.complete_window])):
            records.append({"population": population, "subset": subset, "source": source, "window": window,
                            "seed": seed, "patients": len(rows), **metrics(rows.label, rows.probability)})
    per_seed = pd.DataFrame(records)
    summary = []
    fields = ["population", "subset", "source", "window"]
    for keys, frame in per_seed.groupby(fields, sort=False):
        row = dict(zip(fields, keys))
        if frame.patients.nunique() != 1 or frame.seed.nunique() != len(frame):
            raise ValueError("Metric seed groups have different patients or duplicate seeds")
        row.update(patients=int(frame.patients.iloc[0]), seeds=len(frame))
        for key in METRICS:
            row[key + "_mean"], row[key + "_std"] = float(frame[key].mean()), float(frame[key].std(ddof=1))
        summary.append(row)
    return per_seed, pd.DataFrame(summary)


def legacy_predictions(cfg):
    if cfg["phase"] == "first_post":
        path = Path(cfg["source_cohort"]) / "evaluation/predictions.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str}).rename(columns={"temporal_depth": "window"})
        frame["legacy_file"] = str(path)
        return frame
    paths = {
        "bifm": repo_path("results/ispy2_biflow_symmflow_steps_20260911/biflow/euler_20/pcr/predictions.csv"),
        "symm": repo_path("results/ispy2_generator_comparison_20260911/source_policies/pcr/predictions.csv"),
    }
    pieces = []
    for source, path in paths.items():
        frame = pd.read_csv(path, dtype={"patient_id": str}).rename(columns={"temporal_depth": "window"})
        if source == "bifm":
            real = frame[frame.model == "real"].copy()
            real["source"], real["legacy_file"] = "real", str(path)
            pieces.append(real)
        rows = frame[(frame.model == ("biflow" if source == "bifm" else "symmflow"))
                     & (frame.strategy == "adjacent_real")].copy()
        rows["source"], rows["legacy_file"] = source, str(path)
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def compare_legacy(cfg, predictions):
    old = legacy_predictions(cfg)
    keys = ["source", "window", "seed", "patient_id"]
    if old.duplicated(keys).any():
        raise ValueError("Legacy prediction rows are not unique")
    paired = predictions.merge(old[keys + ["label", "probability", "legacy_file"]], on=keys,
                               suffixes=("_new", "_old"), validate="one_to_one")
    if not (paired.label_new == paired.label_old).all():
        raise ValueError("Legacy/new patient labels differ")
    if len(paired) != len(old):
        raise ValueError("Legacy prediction patients or seeds are not fully matched")
    records = []
    for (source, window, seed), frame in paired.groupby(["source", "window", "seed"], sort=False):
        for subset, rows in (("all_prefixes", frame), ("complete_window", frame[frame.complete_window])):
            new, legacy = metrics(rows.label_new, rows.probability_new), metrics(rows.label_old, rows.probability_old)
            records.append({"source": source, "window": window, "seed": seed, "subset": subset, "patients": len(rows),
                            **{f"new_{k}": v for k, v in new.items()}, **{f"old_{k}": v for k, v in legacy.items()},
                            **{f"delta_{k}": new[k] - legacy[k] for k in METRICS}})
    per_seed = pd.DataFrame(records)
    summary = []
    for keys, rows in per_seed.groupby(["source", "window", "subset"], sort=False):
        row = dict(zip(("source", "window", "subset"), keys))
        row.update(patients=int(rows.patients.iloc[0]), seeds=len(rows))
        for prefix in ("new", "old", "delta"):
            for key in METRICS:
                row[f"{prefix}_{key}_mean"] = float(rows[f"{prefix}_{key}"].mean())
                row[f"{prefix}_{key}_std"] = float(rows[f"{prefix}_{key}"].std(ddof=1))
        summary.append(row)
    destination = Path(cfg["output_dir"]) / "evaluation"
    _atomic_csv(destination / "legacy_paired_predictions.csv", paired)
    _atomic_csv(destination / "legacy_metrics_per_seed.csv", per_seed)
    _atomic_csv(destination / "legacy_comparison.csv", pd.DataFrame(summary))
    return pd.DataFrame(summary)


def report(cfg, oof, predictions):
    destination = Path(cfg["output_dir"]) / "evaluation"
    oof_seed, oof_summary = metric_tables(oof, "development_oof")
    holdout_seed, holdout_summary = metric_tables(predictions, "fixed_holdout")
    summary = pd.concat([oof_summary, holdout_summary], ignore_index=True)
    _atomic_csv(destination / "metrics_per_seed.csv", pd.concat([oof_seed, holdout_seed], ignore_index=True))
    _atomic_csv(destination / "summary.csv", summary)
    comparison = compare_legacy(cfg, predictions)
    lines = ["# Single-phase FM-BCMRI pCR", "", f"Branch: {cfg['branch']}; phase: {cfg['phase']}.", "",
             "Frozen single-channel FM-BCMRI CLS768; complete TDN trained from scratch. Four windows, five outer folds, seeds 42-51.",
             "Three inner folds with seeds 42/43 select the minimum-validation-log-loss epoch. Each outer refit uses the ceiling of the six-epoch median.",
             "The table reports ten-seed mean and sample SD. Holdout probabilities average five folds within each seed; OOF uses only the one held-out fold per patient.",
             "AUPRC is average precision. No bootstrap, confidence intervals or hyperparameter search.", "",
             "## Interpretation limits", "",
             "Both fixed evaluation cohorts previously participated in upstream model selection; these are internal retrospective evaluations, not independent tests.",
             "Registered DCE0 uses fixed T0-localized ROIs, including 110 supplemental T0 crop plans. Native first-post uses each visit's existing annotated/predicted-mask-localized ROI; future visit localization is not available prospectively.",
             "World normalization is frozen separately by branch. The selected single ROI is trilinearly resized to 48 cubed, then volume z-scored for both real and generated images. CLS features are L2-normalized by the unchanged downstream reader.",
             "All four inputs retain real T0 and the original contiguous T0 prefix and time encoding. Patients without T0 remain in their original partition and clinical prior, but do not enter neural training loss.",
             "Generated inputs are archived single-candidate Euler20 previous-real forecasts: registered Symm100000/BiFM97104; native Symm80000 EMA/BiFM10000 ordinary. BiFM archives also used real source SER upstream; this study reads only their saved DCE0 prediction.",
             "Development OOF reports real-only classifier generalization. Archived four-source forecasts exist on the fixed evaluation patients, so no generated-development OOF is asserted.",
             "Legacy Pillar uses 1152 features and three-phase preprocessing; its generated inputs retained real other phases. Encoder, phase count and preprocessing all change here. Registered early-window legacy models also use a different selected architecture/protocol; this is not an isolated phase ablation.",
             "Registered legacy previous-real copy has no corresponding saved single-phase feature policy and is omitted from legacy pairing; the new copy baseline is fully evaluated.", "",
             "## Metrics", "", "| Cohort | Subset | Source | Window | N | AUROC (SD) | AUPRC (SD) | Log loss (SD) | Brier (SD) |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        values = " | ".join(f"{getattr(row, k + '_mean'):.4f} ({getattr(row, k + '_std'):.4f})" for k in METRICS)
        lines.append(f"| {row.population} | {row.subset} | {row.source} | {row.window} | {row.patients} | {values} |")
    lines += ["", "## Matched legacy comparison", "", "| Source | Window | N | New AUROC | Old Pillar AUROC | Mean difference |",
              "|---|---|---:|---:|---:|---:|"]
    for row in comparison[comparison.subset == "all_prefixes"].itertuples():
        lines.append(f"| {row.source} | {row.window} | {row.patients} | {row.new_auroc_mean:.4f} | {row.old_auroc_mean:.4f} | {row.delta_auroc_mean:+.4f} |")
    lines += ["", "Predictions, per-seed metrics and complete-window/legacy details are saved alongside this report. SD describes training-seed variability; it is not a patient-sampling interval."]
    _atomic_text(destination / "report.md", "\n".join(lines) + "\n")
    return summary


def evaluate(cfg, cohort, references, device):
    deterministic_runtime()
    root = Path(cfg["output_dir"])
    holdout = cohort["split"]["val"]
    real = _load_split(root / "embeddings/real", root / "holdout_metadata.csv", holdout)
    inputs = {"real": real, **{source: overlay(real, root / "embeddings" / source) for source in SOURCES[1:]}}
    canonical = {(window, source): _canonical_split(split, depth)
                 for window, depth in DEPTHS.items() for source, split in inputs.items()}
    development = load_development(str(root))
    oof, held, replay_records = [], [], []
    for index, ref in enumerate(references):
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("A frozen classifier changed before evaluation")
        checkpoint = torch.load(ref["path"], map_location="cpu", weights_only=False)
        task = checkpoint["task"]
        if (set(task["train_ids"]) | set(task["val_ids"])) & set(holdout):
            raise ValueError("Holdout patient entered classifier fitting")
        model = TDN({"downstream": checkpoint["effective"]}).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        split = select_patients(development, task["val_ids"], DEPTHS[ref["window"]])
        prior = _prior_from_state(split["clinical"], checkpoint["clinical_prior"])
        probability, _ = predict(model, tensor_split(split, prior, device))
        saved = pd.read_csv(Path(ref["path"]).parent / "predictions.csv", dtype={"patient_id": str})
        saved = saved[saved.role == "validation"].set_index("patient_id").loc[split["pids"]]
        error = float(np.abs(probability - saved.probability.to_numpy()).max())
        if error > 1e-6 or not np.array_equal(split["labels"], saved.label):
            raise ValueError("Frozen OOF checkpoint prediction replay failed")
        fields = {k: ref[k] for k in ("window", "seed", "outer")}
        oof.append(pd.DataFrame({**fields, "source": "real", "patient_id": split["pids"],
                   "label": split["labels"].astype(int), "probability": probability.astype(np.float64),
                   "complete_window": split["masks"].sum(1) == DEPTHS[ref["window"]],
                   "available_prefix": split["masks"].sum(1).astype(int)}))
        replay_records.append({**fields, "max_error": error})
        for source in SOURCES:
            values = canonical[ref["window"], source]
            prior = _prior_from_state(values["clinical"], checkpoint["clinical_prior"])
            probability, _ = predict(model, tensor_split(values, prior, device))
            held.append(pd.DataFrame({**fields, "source": source, "patient_id": holdout,
                        "label": values["labels"].astype(int), "probability": probability.astype(np.float64),
                        "complete_window": values["masks"].sum(1) == DEPTHS[ref["window"]],
                        "available_prefix": values["masks"].sum(1).astype(int)}))
        del model
        if index % 10 == 0:
            progress(cfg, "evaluating_frozen_classifiers", completed=index + 1, total=len(references))
        if (root / "PAUSE_REQUESTED").exists():
            raise InterruptedError("Paused during frozen evaluation; no classifier refitting is needed")
    oof, folds = pd.concat(oof, ignore_index=True), pd.concat(held, ignore_index=True)
    for _, rows in oof.groupby(["window", "seed"]):
        if len(rows) != len(cohort["split"]["train"]) or set(rows.patient_id) != set(cohort["split"]["train"]) or rows.patient_id.duplicated().any():
            raise ValueError("OOF must cover each development patient exactly once per seed")
    destination = root / "evaluation"
    _atomic_csv(destination / "oof_predictions.csv", oof)
    _atomic_csv(destination / "holdout_fold_predictions.csv", folds)
    # Read the actual serialized predictions before computing the report.
    saved_folds = pd.read_csv(destination / "holdout_fold_predictions.csv", dtype={"patient_id": str})
    means = average_folds(saved_folds, holdout, cfg["seeds"])
    _atomic_csv(destination / "holdout_predictions.csv", means)
    summary = report(cfg, pd.read_csv(destination / "oof_predictions.csv", dtype={"patient_id": str}),
                     pd.read_csv(destination / "holdout_predictions.csv", dtype={"patient_id": str}))
    write_json(destination / "verification.json", {"passed": True, "checkpoint_replays": replay_records,
               "max_checkpoint_prediction_error": max(r["max_error"] for r in replay_records),
               "four_source_t0_equal": True, "metrics_computed_from_saved_predictions": True,
               "holdout_patients": len(holdout), "oof_patients": len(cohort["split"]["train"]),
               "formal_classifiers": len(references), "completed_utc": now()})
    return summary

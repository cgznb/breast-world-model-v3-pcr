"""Seed-level OOF and four-source holdout evaluation, without patient resampling."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src.first_post_pcr_data import identity, read_json, write_json
from src.single_phase_fmbcmri_data import load_encoder
from src.single_phase_fmbcmri_training import METRICS, metrics
from src.single_phase_response_data import check_pause, load_data, progress
from src.single_phase_response_models import ResponseModel
from src.single_phase_response_training import (
    clinical_predict, formal_tasks, predict, prediction_frame, task_complete, task_path,
)

WINDOWS = {1: "T0", 2: "T0-T1", 3: "T0-T2", 4: "T0-T3"}


def validate_prediction_frame(frame, ids, seeds):
    keys = ["arm", "source", "population", "seed", "prefix"]
    for key, part in frame.groupby(keys):
        if set(part.patient_id) != set(ids[key[2]]) or part.patient_id.duplicated().any():
            raise ValueError("Predictions do not cover each patient exactly once")
    for key, part in frame.groupby(["arm", "source", "population", "prefix"]):
        if set(part.seed) != set(seeds):
            raise ValueError("An arm is missing a formal seed")
    if not frame.probability.between(0, 1).all() or not np.isfinite(frame.probability).all():
        raise ValueError("Invalid probabilities")


def metric_rows(frame):
    rows = []
    for keys, group in frame.groupby(["arm", "population", "source", "prefix", "seed"]):
        for name, part in (("all_prefix", group), ("complete_window", group[group.complete]),
                           ("complete_four_visits", group[group.available_prefix == 4])):
            if not len(part):
                continue
            rows.append(dict(zip(("arm", "population", "source", "prefix", "seed"), keys),
                             window=WINDOWS[keys[3]], subset=name, patients=len(part),
                             **metrics(part.label, part.probability)))
    return pd.DataFrame(rows)


def summarize(rows):
    groups = ["arm", "population", "source", "prefix", "window", "subset", "patients"]
    output = []
    for key, part in rows.groupby(groups):
        record = dict(zip(groups, key), seeds=part.seed.nunique())
        for metric in METRICS:
            record[f"{metric}_mean"], record[f"{metric}_std"] = float(part[metric].mean()), float(part[metric].std(ddof=1))
        output.append(record)
    return pd.DataFrame(output)


def collect_oof(cfg, arm_names):
    tasks = formal_tasks(cfg, arm_names)
    frames = []
    for task in tasks:
        if not task_complete(cfg, task):
            raise ValueError("OOF reporting requires completed fixed refits")
        frame = pd.read_csv(task_path(cfg, task) / "predictions.csv", dtype={"patient_id": str})
        for seed in cfg["seeds"] if cfg["arms"][task["arm"]]["model"] == "clinical" else [task["seed"]]:
            frames.append(frame.assign(arm=task["arm"], seed=seed, source="real", population="oof", fold=task["fold"]))
    return pd.concat(frames, ignore_index=True)


def freeze_models(cfg):
    tasks = formal_tasks(cfg)
    references = []
    for task in tasks:
        if not task_complete(cfg, task):
            raise ValueError("Cannot freeze an incomplete study")
        path = task_path(cfg, task) / "model.pt"
        references.append(dict(task=task, path=str(path), identity=identity(path)))
    write_json(Path(cfg["output_dir"]) / "FROZEN_MODELS.json", dict(models=references, formal_models=len(references),
               neural_models=sum(cfg["arms"][r["task"]["arm"]]["model"] != "clinical" for r in references),
               one_model_predicts_four_windows=True, holdout_used_for_selection=False))
    return references


def evaluate_holdout(cfg, study):
    root = Path(cfg["output_dir"])
    refs = read_json(root / "FROZEN_MODELS.json")["models"]
    data_cache, frames, encoder, parity = {}, [], None, []
    for number, ref in enumerate(refs):
        task, arm = ref["task"], cfg["arms"][ref["task"]["arm"]]
        if identity(ref["path"]) != ref["identity"]:
            raise ValueError("A frozen classifier changed")
        checkpoint = torch.load(ref["path"], map_location="cpu", weights_only=False)
        device = "cuda:0" if arm.get("lora") else "cpu"
        if arm["model"] != "clinical":
            if arm.get("lora") and encoder is None:
                encoder = load_encoder(cfg, "cpu")
            model = ResponseModel(arm, cfg["training"], checkpoint["prevalence"], encoder, cfg["adaptation"]).to(device)
            model.load_portable_state(checkpoint["state"])
        source_t0 = []
        for source in ("real", "symm", "bifm", "copy"):
            key = (arm["roi"], source)
            if key not in data_cache:
                data_cache[key] = load_data(cfg, study, arm["roi"], "val", source)
            data = data_cache[key]
            if arm["model"] == "clinical":
                probability = clinical_predict(data, checkpoint["clinical"])
            else:
                batch = cfg["adaptation"]["batch_size"] if arm.get("lora") else cfg["training"]["batch_size"]
                probability = predict(model, data, checkpoint["clinical"], device, batch)[0]
            source_t0.append(probability[:, 0])
            part = prediction_frame(data, probability, "holdout")
            for seed in cfg["seeds"] if arm["model"] == "clinical" else [task["seed"]]:
                frames.append(part.assign(arm=task["arm"], seed=seed, source=source, population="holdout", fold=task["fold"]))
        error = max(float(np.max(np.abs(source_t0[0] - value))) for value in source_t0[1:])
        if error > 1e-6:
            raise ValueError("Four-source T0 predictions are not invariant")
        parity.append(error)
        if arm["model"] != "clinical":
            del model
        if number % 10 == 0:
            progress(cfg, "evaluating_fixed_holdout", completed=number + 1, total=len(refs))
        check_pause(cfg)
    all_folds = pd.concat(frames, ignore_index=True)
    groups = ["arm", "source", "population", "seed", "prefix", "patient_id", "label", "complete", "available_prefix"]
    counts = all_folds.groupby(groups, dropna=False).fold.nunique()
    if not (counts == 5).all():
        raise ValueError("Each seed prediction requires five distinct folds")
    averaged = all_folds.groupby(groups, as_index=False, dropna=False).probability.mean()
    _atomic_csv(root / "evaluation/holdout_fold_predictions.csv", all_folds)
    write_json(root / "evaluation/t0_invariance.json", dict(passed=True, models=len(refs), maximum_error=max(parity)))
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return averaged


def legacy_comparison(cfg, predictions, output):
    rows = []
    for population, filename in (("oof", "oof_predictions.csv"), ("holdout", "holdout_predictions.csv")):
        previous = pd.read_csv(Path(cfg["baseline_run"]) / "evaluation" / filename, dtype={"patient_id": str})
        previous["prefix"] = previous.window.map({v: k for k, v in WINDOWS.items()})
        for arm, new in predictions[predictions.population == population].groupby("arm"):
            pair = new.merge(previous, on=["source", "prefix", "seed", "patient_id"], suffixes=("_new", "_old"), validate="one_to_one")
            if len(pair) != len(new) or not np.array_equal(pair.label_new, pair.label_old):
                raise ValueError("Legacy comparison has different patients or labels")
            for (source, prefix, seed), group in pair.groupby(["source", "prefix", "seed"]):
                a, b = metrics(group.label_new, group.probability_new), metrics(group.label_old, group.probability_old)
                rows.append(dict(arm=arm, population=population, source=source, prefix=prefix, seed=seed, patients=len(group),
                                 **{f"delta_{m}": a[m] - b[m] if a[m] is not None and b[m] is not None else None for m in METRICS}))
    _atomic_csv(output / "comparison_to_v1_per_seed.csv", pd.DataFrame(rows))


def report(cfg, study, *, include_holdout=False, arm_names=None):
    names = list(cfg["arms"]) if arm_names is None else arm_names
    root = Path(cfg["output_dir"]) / ("evaluation" if include_holdout else "development_review")
    frame = collect_oof(cfg, names)
    if include_holdout:
        frame = pd.concat([frame, evaluate_holdout(cfg, study)], ignore_index=True)
    validate_prediction_frame(frame, {"oof": study["split"]["train"], "holdout": study["split"]["val"]}, cfg["seeds"])
    _atomic_csv(root / "predictions.csv", frame)
    saved = pd.read_csv(root / "predictions.csv", dtype={"patient_id": str})
    rows, saved_rows = metric_rows(frame), metric_rows(saved)
    error = float(np.nanmax(np.abs(rows[list(METRICS)].to_numpy() - saved_rows[list(METRICS)].to_numpy())))
    if error > 1e-12:
        raise ValueError("Saved predictions do not reproduce metrics")
    summary = summarize(saved_rows)
    _atomic_csv(root / "metrics_per_seed.csv", saved_rows)
    _atomic_csv(root / "summary.csv", summary)
    legacy_comparison(cfg, saved, root)
    write_json(root / "verification.json", dict(passed=True, metric_replay_max_error=error,
               holdout_included=include_holdout, bootstrap_samples=0, seed_count=len(cfg["seeds"])))
    lines = [f"# Single-phase response study: {cfg['branch']}", "",
             "One shared model produces four causal prefix predictions. All neural fits use ordinary BCE; epoch selection uses patient-averaged prefix validation log loss.", "",
             "Five fixed outer folds and ten formal seeds (42-51). Three inner folds and two seeds select the ceiling of the median of six best epochs. Each seed averages the five holdout probabilities before scoring. Reported SD is variability over seeds, not patient-sampling uncertainty. No bootstrap or bootstrap intervals.", "",
             "Clinical-only is a deterministic logistic baseline fitted once per outer fold, repeated identically across the ten seed rows. MRI-only arms do not consume clinical inputs. The primary crop is context; tumor is a prespecified ablation. No crop or arm is selected using the fixed holdout.", "",
             "The primary I-SPY2 clinical release agrees with both inherited conflict labels. The ancillary release disagrees in two development cases. Their original labels remain; the sensitivity arm excludes these cases only from fitting and fitted preprocessing. This does not resolve the source disagreement.", "",
             "Real input uses a physical cube from the full single-phase MRI, with side length set from T0, followed by frozen world intensity normalization, physical trilinear sampling to 48-cube and ROI z-score. Registered localization is fixed at T0. Native observed visits use that visit's localization and acquisition axes, with the same T0 physical side; there is no registration in this branch.", "",
             "Generated extraction receives only the observed source localization, a T0-derived physical side and archived forecast pixels. The native archive is interpreted on the observed source canvas. Its models were trained on retrospectively localized visit grids, so this cannot establish a prospectively leakage-free forecast. Generated support outside the archived canvas is unavailable, whereas full real-image support may be present.", "",
             "The historical fixed holdout was previously used for upstream model selection. This is an internal comparison, not a new independent external test. Previous studies also inspected these results when motivating this protocol.", "",
             "## Real MRI AUROC (Mean +/- Sample SD)", "",
             "| Population | Arm | T0 | T0-T1 | T0-T2 | T0-T3 |", "|---|---|---:|---:|---:|---:|"]
    selected = summary[(summary.source == "real") & (summary.subset == "all_prefix")]
    for (population, arm), part in selected.groupby(["population", "arm"]):
        part = part.set_index("prefix")
        lines.append(f"| {population} | {arm} | " + " | ".join(f"{part.loc[t, 'auroc_mean']:.4f} +/- {part.loc[t, 'auroc_std']:.4f}" for t in range(1, 5)) + " |")
    lines.extend(["", "Detailed per-seed AUROC, AUPRC, log loss and Brier scores are in `metrics_per_seed.csv`; it includes all-prefix, complete-window and the same four-visit-complete populations.",
                  "`comparison_to_v1_per_seed.csv` compares identical patients and labels to frozen FM-BCMRI v1. ROI, temporal architecture and training objective all changed, so this comparison alone does not isolate one mechanism. The v1 report retains its matched three-phase Pillar comparison."])
    if not include_holdout:
        lines.extend(["", "Status: development review only. The adapter comparison and fixed holdout evaluation may still be running. These development results are not used to alter the prespecified experiment."])
    _atomic_text(root / "report.md", "\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    populations = sorted(selected.population.unique())
    fig, axes = plt.subplots(1, len(populations), figsize=(8 * len(populations), 5), squeeze=False)
    for ax, population in zip(axes[0], populations):
        for arm, part in selected[selected.population == population].groupby("arm"):
            part = part.sort_values("prefix")
            ax.errorbar(part.prefix, part.auroc_mean, yerr=part.auroc_std, marker="o", capsize=2, label=arm)
        ax.set_xticks(range(1, 5), WINDOWS.values())
        ax.set_ylabel("AUROC; error bars: seed SD")
        ax.set_title(population)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "real_prefix_comparison.png", dpi=170)
    fig.savefig(root / "real_prefix_comparison.pdf")
    plt.close(fig)
    return summary

"""Aggregate nested outer predictions without reading the historical holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src.first_post_optimization import DEPTH_NAMES, SCHEMA, load_study, metric_values
from src.first_post_pcr_data import now, read_json, write_json


def paired_bootstrap(labels, selected, baseline, samples, seed=2026):
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    draws = []
    for _ in range(samples):
        index = np.concatenate([rng.choice(positive, len(positive), replace=True),
                                rng.choice(negative, len(negative), replace=True)])
        first = metric_values(labels[index], selected[index])
        second = metric_values(labels[index], baseline[index])
        draws.append([first[k] - second[k] for k in ("auroc", "logloss", "brier")])
    intervals = np.percentile(draws, [2.5, 97.5], axis=0)
    return {k: {"lower": float(intervals[0, i]), "upper": float(intervals[1, i])}
            for i, k in enumerate(("auroc", "logloss", "brier"))}


def report(cfg, name):
    output = Path(cfg["output_dir"])
    references = read_json(output / "selection" / f"{name}_outer_models.json")
    destination = output / "reports" / name
    destination.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    folds = read_json(output / "nested_folds.json")
    expected_ids = set(folds[0]["train_ids"] + folds[0]["val_ids"])
    for reference in references:
        run = Path(reference["path"])
        summary = read_json(run / "COMPLETE.json")
        frame = pd.read_csv(run / "predictions.csv", dtype={"patient_id": str})
        val = frame.loc[frame.role == "validation"].copy()
        val["arm"], val["depth"], val["seed"], val["outer"] = (
            reference["arm"], reference["depth"], reference["seed"], reference["outer"])
        predictions.append(val)
        rows.append({**{k: reference[k] for k in ("arm", "depth", "seed", "outer")},
                     "source": summary["task"]["source"], "candidate": summary["task"]["candidate"],
                     "epochs": summary["selected_epochs"], "gap": summary["auroc_gap"],
                     **{f"train_{k}": v for k, v in summary["train"].items()},
                     **{f"validation_{k}": v for k, v in summary["validation"].items()}})
    all_predictions = pd.concat(predictions, ignore_index=True)
    per_fit = pd.DataFrame(rows)
    metrics = []
    for (arm, depth, seed), frame in all_predictions.groupby(["arm", "depth", "seed"]):
        if len(frame) != len(expected_ids) or set(frame.patient_id) != expected_ids or not frame.patient_id.is_unique:
            raise ValueError("Outer OOF predictions do not cover development exactly once")
        fit = per_fit[(per_fit.arm == arm) & (per_fit.depth == depth) & (per_fit.seed == seed)]
        metrics.append({"arm": arm, "depth": depth, "seed": seed,
                        **metric_values(frame.label.to_numpy(), frame.probability.to_numpy()),
                        "train_auroc": fit.train_auroc.mean(), "fold_validation_auroc": fit.validation_auroc.mean(),
                        "gap": fit.gap.mean(), "epochs": fit.epochs.mean()})
    metric_frame = pd.DataFrame(metrics)
    aggregate = metric_frame.groupby(["depth", "arm"])[
        ["auroc", "logloss", "brier", "train_auroc", "fold_validation_auroc", "gap", "epochs"]].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    comparisons = {}
    for depth in cfg["depths"]:
        frames = {}
        for arm in ("selected", "baseline"):
            frame = all_predictions[(all_predictions.depth == depth) & (all_predictions.arm == arm)]
            frames[arm] = frame.groupby("patient_id").agg(label=("label", "first"), probability=("probability", "mean"))
        a, b = frames["selected"], frames["baseline"].reindex(frames["selected"].index)
        if not np.array_equal(a.label.to_numpy(), b.label.to_numpy()):
            raise ValueError("Paired labels changed")
        first, second = metric_values(a.label.to_numpy(), a.probability.to_numpy()), metric_values(b.label.to_numpy(), b.probability.to_numpy())
        comparisons[DEPTH_NAMES[depth]] = {
            "seed_ensemble_selected": first, "seed_ensemble_baseline": second,
            "difference": {k: first[k] - second[k] for k in first},
            "patient_bootstrap_95pct": paired_bootstrap(a.label.to_numpy(), a.probability.to_numpy(),
                                                         b.probability.to_numpy(), cfg["bootstrap_samples"])}
    _atomic_csv(destination / "outer_oof_predictions.csv", all_predictions)
    _atomic_csv(destination / "fit_metrics.csv", per_fit)
    _atomic_csv(destination / "seed_metrics.csv", metric_frame)
    _atomic_csv(destination / "summary.csv", aggregate)
    payload = {"schema": SCHEMA, "name": name, "created_utc": now(), "development_patients": len(expected_ids),
               "holdout_evaluated": False, "selection_nested_inside_outer_training": True,
               "summary": aggregate.to_dict("records"), "comparisons": comparisons,
               "uncertainty_note": "Patient resampling of fixed nested OOF predictions; overlapping training sets and full refit uncertainty are not modeled. No multiple-comparison adjustment."}
    write_json(destination / "summary.json", payload)
    lines = [f"# Native first-post pCR: {name}", "",
             "Development-only nested 5-fold evaluation; 96 historical holdout patients were not evaluated.",
             "Epochs and candidates were chosen using only inner validation patients, followed by fixed-epoch outer refits.",
             "Baseline here uses the original network/training settings with inner-selected duration; it is not the old selection-conditioned OOF result.", "",
             "| Input | Model | OOF AUROC | Log loss | Train/fold-validation AUROC gap |", "|---|---|---:|---:|---:|"]
    for row in aggregate.to_dict("records"):
        lines.append(f"| {DEPTH_NAMES[row['depth']]} | {row['arm']} | {row['auroc_mean']:.5f} | {row['logloss_mean']:.5f} | {row['gap_mean']:.5f} |")
    lines.extend(["", "Candidate selection permits at most 0.005 inner mean AUROC below the best candidate, then minimizes log loss.",
                  "A smaller gap alone is not evidence of improvement. Seed variation is not a patient confidence interval.",
                  "The resized-ROI comparison changes physical scale within the same tumor crop; it does not test registration or acquisition-phase changes.", ""])
    _atomic_text(destination / "README.md", "\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric, label in zip(axes, ("auroc", "logloss", "gap"), ("Nested OOF AUROC", "Nested OOF log loss", "Train - validation AUROC")):
        for arm, color in (("baseline", "#b25a42"), ("selected", "#21847b")):
            part = aggregate[aggregate.arm == arm].sort_values("depth")
            axis.errorbar(part.depth, part[f"{metric}_mean"], yerr=part[f"{metric}_std"].fillna(0), marker="o", capsize=3, label=arm, color=color)
        axis.set_xticks(cfg["depths"], [DEPTH_NAMES[d] for d in cfg["depths"]])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(destination / "comparison.png", dpi=180)
    fig.savefig(destination / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(len(cfg["depths"]), 2, figsize=(11, 3 * len(cfg["depths"])), squeeze=False, constrained_layout=True)
    for index, depth in enumerate(cfg["depths"]):
        for arm, color in (("baseline", "#b25a42"), ("selected", "#21847b")):
            ref = next(r for r in references if r["depth"] == depth and r["arm"] == arm and r["outer"] == 0 and r["seed"] == cfg["formal_seeds"][0])
            record = read_json(Path(ref["path"]) / "COMPLETE.json")
            task = record["task"]
            history_path = output / "fits/inner" / task["source"] / DEPTH_NAMES[depth] / task["candidate"] / "outer_0/inner_0" / f"seed_{cfg['tuning_seeds'][0]}" / "history.csv"
            if task["candidate"] == "clinical_only":
                continue
            history = pd.read_csv(history_path)
            for axis, metric in zip(axes[index], ("auroc", "logloss")):
                axis.plot(history.epoch, history[f"train_{metric}"], color=color, linestyle="--", label=f"{arm} train")
                axis.plot(history.epoch, history[f"validation_{metric}"], color=color, label=f"{arm} inner validation")
                axis.set_ylabel(f"{DEPTH_NAMES[depth]} {metric}")
                axis.set_xlabel("Epoch")
                axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(destination / "representative_inner_curves.png", dpi=160)
    plt.close(fig)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/first_post_pcr_optimization_v2.yaml")
    parser.add_argument("--name", default="physical_roi")
    args = parser.parse_args()
    report(load_study(args.config), args.name)

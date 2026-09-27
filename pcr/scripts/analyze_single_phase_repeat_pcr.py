"""Recompute aggregate v1 fitting gaps and descriptive seed rankings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src.first_post_optimization import DEPTH_NAMES, metric_values
from src.first_post_pcr_data import now, read_json, repo_path, write_json


def rank_seeds(frame):
    rows = []
    for depth, group in frame.groupby("depth", sort=True):
        best = group.auroc.max()
        rows.append({"depth": depth, "seeds": sorted(group.loc[np.isclose(group.auroc, best, atol=1e-12, rtol=0), "seed"].astype(int).tolist()),
                     "auroc": float(best)})
    means = frame.groupby("seed").auroc.mean()
    best = means.max()
    return {"windows": rows, "four_window_mean": {
        "seeds": means.index[np.isclose(means, best, atol=1e-12, rtol=0)].astype(int).tolist(), "auroc": float(best)}}


def analyze(baseline, output):
    baseline, output = Path(baseline), Path(output)
    payload = {"created_utc": now(), "source": str(baseline), "branches": {},
        "interpretation": "Seed rankings are descriptive; historical holdout rankings must not choose a seed for performance reporting.",
        "registered_oof_limit": "Fold validation selected checkpoints, so registered v1 OOF is selection-conditioned.",
        "first_post_oof_limit": "v1 candidates and durations were selected using inner validation before outer refits."}
    lines = ["# Repeated single-phase pCR v1 diagnosis", "",
             "Gaps use mean train AUC minus mean fold-validation AUC over the same 50 selected fits per window.",
             "Pooled OOF and historical holdout AUC are separate quantities.", ""]
    for branch in ("registered_dce0", "first_post"):
        root = baseline / branch
        if branch == "registered_dce0":
            fits, predictions = [], []
            for path in sorted((root / "tdn/train/formal").glob("**/summary.json")):
                summary = read_json(path)
                fields = {"depth": summary["temporal_depth"], "seed": summary["seed"], "outer": summary["fold"]}
                record = {**fields, "epochs": summary["selection"]["best_epoch_zero_based"] + 1}
                for role, filename in (("train", "train_predictions.csv"), ("validation", "val_predictions.csv")):
                    frame = pd.read_csv(path.parent / filename, dtype={"patient_id": str})
                    record.update({f"{role}_{k}": v for k, v in metric_values(frame.label, frame.probability).items()})
                    if role == "validation":
                        predictions.append(frame.assign(**fields))
                record["gap"] = record["train_auroc"] - record["validation_auroc"]
                fits.append(record)
            fits, predictions = pd.DataFrame(fits), pd.concat(predictions, ignore_index=True)
            dev_ids = set(read_json(root / "cohort.json")["split"]["train"])
            rows = []
            for (depth, seed), frame in predictions.groupby(["depth", "seed"]):
                if len(frame) != len(dev_ids) or not frame.patient_id.is_unique or set(frame.patient_id) != dev_ids:
                    raise ValueError("Registered OOF coverage changed")
                rows.append({"depth": depth, "seed": int(seed), **metric_values(frame.label, frame.probability)})
            development = pd.DataFrame(rows)
        else:
            directory = root / "optimization/reports/physical_roi"
            fits, development = pd.read_csv(directory / "fit_metrics.csv"), pd.read_csv(directory / "seed_metrics.csv")
            fits = fits[fits.arm == "selected"].copy()
            development = development[development.arm == "selected"].copy()
            fits["depth"] = fits.depth.map(DEPTH_NAMES)
            development["depth"] = development.depth.map(DEPTH_NAMES)
        if len(fits) != 200 or len(development) != 40:
            raise ValueError("Expected 200 selected fits and 40 seed/window metrics per branch")
        summary = fits.groupby("depth")[["train_auroc", "validation_auroc", "gap", "train_logloss", "validation_logloss", "epochs"]].mean().reset_index()
        holdout = pd.read_csv(root / "evaluation/real/seed_metrics.csv")
        if len(holdout) != 40:
            raise ValueError("Expected 40 historical holdout seed/window metrics")
        result = {"fits": len(fits), "overfit": summary.to_dict("records"),
                  "development_oof_ranking": rank_seeds(development), "historical_holdout_ranking": rank_seeds(holdout)}
        payload["branches"][branch] = result
        for name, frame in (("fit_metrics", fits), ("overfit", summary), ("development_seed_metrics", development),
                            ("historical_holdout_seed_metrics", holdout)):
            _atomic_csv(output / branch / f"{name}.csv", frame)
        lines += [f"## {branch}", "", "| Window | Train AUC | Fold validation AUC | Gap |", "|---|---:|---:|---:|"]
        for row in summary.to_dict("records"):
            lines.append(f"| {row['depth']} | {row['train_auroc']:.5f} | {row['validation_auroc']:.5f} | {row['gap']:.5f} |")
        lines += ["", "| Split | Window | Highest observed seed(s) | AUC |", "|---|---|---|---:|"]
        for name in ("development_oof", "historical_holdout"):
            ranking = result[f"{name}_ranking"]
            for row in ranking["windows"] + [{"depth": "four-window mean", **ranking["four_window_mean"]}]:
                lines.append(f"| {name} | {row['depth']} | {', '.join(map(str, row['seeds']))} | {row['auroc']:.5f} |")
        lines.append("")
    lines += [payload["registered_oof_limit"], payload["first_post_oof_limit"], payload["interpretation"], ""]
    write_json(output / "summary.json", payload)
    _atomic_text(output / "README.md", "\n".join(lines))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/single_phase_repeat_pcr_v1_20260915")
    parser.add_argument("--output", default="results/single_phase_repeat_pcr_v2_20260915/diagnosis_v1")
    args = parser.parse_args()
    result = analyze(repo_path(args.baseline), repo_path(args.output))
    print({branch: data["development_oof_ranking"]["four_window_mean"] for branch, data in result["branches"].items()})


if __name__ == "__main__":
    main()

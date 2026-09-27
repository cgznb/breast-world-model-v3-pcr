"""Independently replay development metrics and exported single-factor comparisons."""

import argparse
import os
import sys
from pathlib import Path
from statistics import mean, stdev

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import registered_three_phase_optimization as shared
from src import registered_three_phase_overfit_ablation as study
from src.first_post_pcr_data import now, read_json, write_json


def auc(y, p):
    y = np.asarray(y, dtype=int)
    positive = int(y.sum())
    ranks = rankdata(np.asarray(p), method="average")
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * (len(y) - positive)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_overfit_ablation_v4.yaml")
    args = parser.parse_args()
    cfg = study.load_config(args.config)
    output = Path(cfg["output_dir"])
    if not read_json(output / "COMPLETE.json")["passed"]:
        raise ValueError("Experiment did not complete successfully")
    shared.verify_inventory(read_json(output / "input_inventory.json"))
    shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
    refs = read_json(output / "model_references.json")
    folds = read_json(output / "nested_folds.json")
    _, expected_refs = study.build_tasks(cfg, folds)
    assert refs == expected_refs
    frame = pd.read_csv(output / "fold_metrics.csv", float_precision="round_trip").set_index(["arm", "seed", "outer"])
    summary = pd.read_csv(output / "per_seed_fivefold.csv", float_precision="round_trip").set_index(["arm", "seed"])
    comparisons = pd.read_csv(output / "paired_comparisons.csv", float_precision="round_trip").set_index(["arm", "seed"])
    errors, replay = [], []
    for ref in refs:
        path = Path(ref["path"])
        history = pd.read_csv(path / "history.csv")
        assert history.epoch.tolist() == list(range(1, 41))
        assert not any(c.startswith("validation_") for c in history)
        predictions = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
        row = {"arm": ref["arm"], "seed": ref["seed"], "outer": ref["outer"]}
        for role, ids in (("train", ref["task"]["train_ids"]), ("validation", ref["task"]["val_ids"])):
            part = predictions[predictions.role == role]
            assert part.patient_id.is_unique and set(part.patient_id) == set(ids)
            y, p = part.label.to_numpy(), part.probability.to_numpy()
            clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
            metrics = {"auroc": auc(y, p), "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
                       "brier": float(np.mean((p - y) ** 2))}
            saved = frame.loc[(ref["arm"], ref["seed"], ref["outer"])]
            for key, value in metrics.items():
                row[f"{role}_{key}"] = value
                errors.append(abs(value - saved[f"{role}_{key}"]))
        row["auroc_gap"] = row["train_auroc"] - row["validation_auroc"]
        replay.append(row)
    recomputed = pd.DataFrame(replay)
    for key, part in recomputed.groupby(["arm", "seed"]):
        saved = summary.loc[key]
        for metric in ("train_auroc", "validation_auroc", "validation_logloss", "validation_brier", "auroc_gap"):
            errors.extend([abs(mean(part[metric]) - saved[f"{metric}_mean"]),
                           abs(stdev(part[metric]) - saved[f"{metric}_fold_sd"])])
        for r in part.itertuples():
            errors.append(abs(r.validation_auroc - saved[f"F{r.outer + 1}_auroc"]))
        if key[0] != "reference":
            baseline = summary.loc[("reference", key[1])]
            for metric in ("validation_auroc", "validation_logloss", "validation_brier", "auroc_gap"):
                errors.append(abs(saved[f"{metric}_mean"] - baseline[f"{metric}_mean"]
                                  - comparisons.loc[key, f"delta_{metric}"]))
    assert max(errors) < 1e-12
    sheets = {"Per Seed Five Folds": "per_seed_fivefold", "Individual Folds": "fold_metrics",
              "Paired Comparisons": "paired_comparisons", "Original Windows": "original_window_context"}
    for name, filename in sheets.items():
        actual = pd.read_excel(output / "development_ablation.xlsx", sheet_name=name)
        expected = pd.read_csv(output / f"{filename}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)
    audit = {"passed": True, "completed_utc": now(), "models": len(refs),
             "independent_metric_and_aggregate_checks": len(errors), "max_metric_error": max(errors),
             "excel_sheets_verified": len(sheets), "all_recipes_match_declared_single_factor_changes": True,
             "original_and_new_weight_identities_unchanged": True, "new_holdout_inference": False}
    write_json(output / "independent_verification.json", audit)
    print(audit)


if __name__ == "__main__":
    main()

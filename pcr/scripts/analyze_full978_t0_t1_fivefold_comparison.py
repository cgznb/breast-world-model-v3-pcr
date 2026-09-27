"""Compare compact T0-T1 all-development refits with five-fold ensembles."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_json, _atomic_text
from src.metrics import METRIC_KEYS, compute_metrics


ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "results/mewm_ispy2_full978_locked102_t0_t1_optimized_refit"
FIVEFOLD = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_t0_t1_fixed_epoch12_fivefold_evaluation"
)
OUTPUT = FIVEFOLD / "analysis"
SEEDS = tuple(range(42, 52))
SOURCES = ("real", "direct_t0", "rollout_generated_dce0_real_ser")
POLICIES = ("fixed_0.5", "development_oof_bacc")


def _prediction_path(strategy, source, seed):
    if strategy == "single_all_development_refit":
        return SINGLE / f"evaluation/{source}/seed_{seed}/test_predictions.csv"
    if strategy == "fivefold_probability_ensemble":
        return FIVEFOLD / f"evaluation/{source}/t0_t1/seed_{seed}/test_predictions.csv"
    raise ValueError(f"unknown strategy: {strategy}")


def _load_predictions(strategy, source, seed, reference_ids=None):
    frame = pd.read_csv(
        _prediction_path(strategy, source, seed), dtype={"patient_id": str}
    )
    expected_split = (
        "test_single_refit"
        if strategy == "single_all_development_refit"
        else "test_fold_ensemble"
    )
    if (
        len(frame) != 102
        or frame["patient_id"].duplicated().any()
        or int(frame["label"].sum()) != 32
        or set(frame["seed"]) != {seed}
        or set(frame["source"]) != {source}
        or set(frame["split"]) != {expected_split}
        or not np.isfinite(
            frame[["probability", "residual_logit", "clinical_prior_probability"]]
        ).all().all()
    ):
        raise RuntimeError(f"invalid prediction contract: strategy={strategy} source={source} seed={seed}")
    ids = tuple(frame["patient_id"])
    if reference_ids is not None and ids != reference_ids:
        raise RuntimeError("single-refit and five-fold test membership/order differ")
    if strategy == "fivefold_probability_ensemble":
        fold_path = _prediction_path(strategy, source, seed).with_name(
            "fold_predictions.csv"
        )
        folds = pd.read_csv(fold_path, dtype={"patient_id": str})
        if (
            len(folds) != 510
            or set(folds["fold"]) != set(range(5))
            or not folds.groupby("patient_id")["fold"].nunique().eq(5).all()
        ):
            raise RuntimeError("five-fold component predictions are incomplete")
        averaged = (
            folds.groupby("patient_id", sort=False)[
                ["probability", "residual_logit", "clinical_prior_probability"]
            ]
            .mean()
            .reindex(frame["patient_id"])
        )
        np.testing.assert_allclose(
            averaged.to_numpy(dtype=float),
            frame[
                ["probability", "residual_logit", "clinical_prior_probability"]
            ].to_numpy(dtype=float),
            atol=1e-7,
            rtol=0,
        )
    return frame, ids


def _evaluate(threshold):
    rows = []
    all_seed_rows = []
    predictions = {}
    strategies = (
        ("single_all_development_refit", 1),
        ("fivefold_probability_ensemble", 5),
    )
    reference_ids = None
    for strategy, models_per_seed in strategies:
        for source in SOURCES:
            frames = []
            for seed in SEEDS:
                frame, ids = _load_predictions(
                    strategy, source, seed, reference_ids=reference_ids
                )
                if reference_ids is None:
                    reference_ids = ids
                frames.append(frame)
                for policy, cutoff in (
                    ("fixed_0.5", 0.5),
                    ("development_oof_bacc", threshold),
                ):
                    rows.append(
                        {
                            "strategy": strategy,
                            "source": source,
                            "seed": seed,
                            "threshold_policy": policy,
                            "threshold": cutoff,
                            "models_per_seed": models_per_seed,
                            "brier": float(
                                np.mean((frame["probability"] - frame["label"]) ** 2)
                            ),
                            **compute_metrics(
                                frame["label"], frame["probability"], threshold=cutoff
                            ),
                        }
                    )
            combined = pd.concat(frames, ignore_index=True)
            predictions[(strategy, source)] = combined.sort_values(
                ["seed", "patient_id"]
            ).reset_index(drop=True)
            grand = (
                combined.groupby(["patient_id", "label"], as_index=False, sort=False)[
                    "probability"
                ].mean()
            )
            for policy, cutoff in (
                ("fixed_0.5", 0.5),
                ("development_oof_bacc", threshold),
            ):
                all_seed_rows.append(
                    {
                        "strategy": strategy,
                        "source": source,
                        "threshold_policy": policy,
                        "threshold": cutoff,
                        "models_ensembled": models_per_seed * len(SEEDS),
                        "brier": float(
                            np.mean((grand["probability"] - grand["label"]) ** 2)
                        ),
                        **compute_metrics(
                            grand["label"], grand["probability"], threshold=cutoff
                        ),
                    }
                )
    for strategy in ("single_all_development_refit", "fivefold_probability_ensemble"):
        direct = predictions[(strategy, "direct_t0")]
        rollout = predictions[(strategy, "rollout_generated_dce0_real_ser")]
        if not np.array_equal(
            direct["probability"].to_numpy(), rollout["probability"].to_numpy()
        ):
            raise RuntimeError(f"T0-T1 Direct-T0 and rollout differ for {strategy}")
    return pd.DataFrame(rows), pd.DataFrame(all_seed_rows)


def _summarize(metrics):
    rows = []
    keys = [
        "strategy",
        "source",
        "threshold_policy",
        "threshold",
        "models_per_seed",
    ]
    for values, frame in metrics.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = len(frame)
        for metric in (*METRIC_KEYS, "brier"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _compare(summary):
    rows = []
    for source in SOURCES:
        for policy in POLICIES:
            selected = summary[
                (summary["source"] == source)
                & (summary["threshold_policy"] == policy)
            ].set_index("strategy")
            single = selected.loc["single_all_development_refit"]
            folded = selected.loc["fivefold_probability_ensemble"]
            row = {
                "source": source,
                "threshold_policy": policy,
                "threshold": float(folded["threshold"]),
                "single_models_per_seed": 1,
                "fivefold_models_per_seed": 5,
            }
            for metric in (*METRIC_KEYS, "brier"):
                row[f"single_{metric}_mean"] = float(single[f"{metric}_mean"])
                row[f"single_{metric}_std"] = float(single[f"{metric}_std"])
                row[f"fivefold_{metric}_mean"] = float(folded[f"{metric}_mean"])
                row[f"fivefold_{metric}_std"] = float(folded[f"{metric}_std"])
                row[f"fivefold_minus_single_{metric}"] = float(
                    folded[f"{metric}_mean"] - single[f"{metric}_mean"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    threshold_payload = json.loads(
        (SINGLE / "analysis/development_threshold.json").read_text()
    )
    if (
        threshold_payload.get("source")
        != "ten_seed_mean_five_fold_oof_predictions"
        or threshold_payload.get("test_data_loaded_for_threshold_selection") is not False
    ):
        raise RuntimeError("development-only threshold contract changed")
    threshold = float(threshold_payload["threshold"])
    metrics, all_seed = _evaluate(threshold)
    summary = _summarize(metrics)
    comparison = _compare(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(OUTPUT / "metrics_per_seed.csv", metrics)
    _atomic_csv(OUTPUT / "summary.csv", summary)
    _atomic_csv(OUTPUT / "single_vs_fivefold.csv", comparison)
    _atomic_csv(OUTPUT / "all_seed_ensemble_metrics.csv", all_seed)

    real = comparison[
        (comparison["source"] == "real")
        & (comparison["threshold_policy"] == "development_oof_bacc")
    ].iloc[0]
    direct = comparison[
        (comparison["source"] == "direct_t0")
        & (comparison["threshold_policy"] == "development_oof_bacc")
    ].iloc[0]
    report = f"""# Compact T0-T1 Single Refit versus Five-Fold Ensemble

The architecture, loss, optimizer, regularization, seeds, and fixed 12 epochs are
identical. A single-refit model sees all 876 development patients; each fold model
sees four fifths, and five fold probabilities are averaged per seed.

The comparison was not used to select the configuration. The locked102 cohort has
already been inspected, so these are descriptive results.

## Ten-seed mean +/- sample standard deviation

- Real single AUROC/PR-AUC: {real['single_auroc_mean']:.5f} +/- {real['single_auroc_std']:.5f} / {real['single_prauc_mean']:.5f} +/- {real['single_prauc_std']:.5f}.
- Real five-fold AUROC/PR-AUC: {real['fivefold_auroc_mean']:.5f} +/- {real['fivefold_auroc_std']:.5f} / {real['fivefold_prauc_mean']:.5f} +/- {real['fivefold_prauc_std']:.5f}.
- Direct-T0 single AUROC/PR-AUC: {direct['single_auroc_mean']:.5f} +/- {direct['single_auroc_std']:.5f} / {direct['single_prauc_mean']:.5f} +/- {direct['single_prauc_std']:.5f}.
- Direct-T0 five-fold AUROC/PR-AUC: {direct['fivefold_auroc_mean']:.5f} +/- {direct['fivefold_auroc_std']:.5f} / {direct['fivefold_prauc_mean']:.5f} +/- {direct['fivefold_prauc_std']:.5f}.
- Development-OOF threshold: {threshold:.9f}.

Direct-T0 and rollout are identical at T0-T1 for both strategies.
"""
    _atomic_text(OUTPUT / "report.md", report)
    _atomic_json(
        OUTPUT / "ANALYSIS_COMPLETE.json",
        {
            "schema": "pillar_full978_t0_t1_fivefold_comparison_v1",
            "seeds": list(SEEDS),
            "sources": list(SOURCES),
            "threshold": threshold,
            "test_used_for_model_or_threshold_selection": False,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()

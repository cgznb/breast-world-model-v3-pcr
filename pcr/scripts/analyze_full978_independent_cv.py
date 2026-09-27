"""Recompute and audit the completed independent-depth CV experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_dual_strategy_qc_sensitivity import (
    ANALYSIS_STATUS,
    COHORTS,
    excluded_patient_ids,
)
from src.metrics import METRIC_KEYS, compute_metrics


DEFAULT_NEW_ROOT = ROOT / "results/mewm_ispy2_full978_locked102_independent_cv"
DEFAULT_OLD_ROOT = ROOT / "results/mewm_ispy2_full978_locked102_dual_strategy"
SCHEMA = "pillar_full978_independent_depth_cv_analysis_v1"
DEPTHS = (("T0-T2", 3), ("T0-T3", 4))
SEEDS = tuple(range(42, 52))
SOURCES = ("real", "generated")


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_ids(path):
    values = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate patient IDs in {path}")
    return values


def _assert_close(saved, recomputed, keys, columns, label):
    merged = saved.merge(recomputed, on=keys, suffixes=("_saved", "_recomputed"))
    if len(merged) != len(saved) or len(merged) != len(recomputed):
        raise ValueError(f"{label} row-key mismatch")
    for column in columns:
        left = merged[f"{column}_saved"].to_numpy(dtype=float)
        right = merged[f"{column}_recomputed"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True):
            raise ValueError(f"{label} mismatch for {column}")


def _validate_prediction_contract(
    frame, patient_ids, *, seed, fold, split, depth, max_tp, label
):
    required = {
        "patient_id", "label", "probability", "seed", "fold", "split",
        "temporal_depth", "max_tp",
    }
    if required - set(frame):
        raise ValueError(f"{label} missing prediction columns")
    if (
        frame["patient_id"].tolist() != patient_ids
        or frame["patient_id"].duplicated().any()
        or len(frame) != len(patient_ids)
        or set(frame["seed"].astype(int)) != {int(seed)}
        or set(frame["fold"].astype(int)) != {int(fold)}
        or set(frame["split"]) != {split}
        or set(frame["temporal_depth"]) != {depth}
        or set(frame["max_tp"].astype(int)) != {int(max_tp)}
    ):
        raise ValueError(f"{label} prediction contract mismatch")
    if not np.isfinite(frame["probability"]).all():
        raise ValueError(f"{label} has non-finite probabilities")
    if not frame["probability"].between(0, 1).all():
        raise ValueError(f"{label} has probabilities outside [0,1]")


def _fold_contracts(new_root, pool_ids):
    contracts = []
    for fold in range(5):
        fold_dir = new_root / "folds" / f"fold_{fold}"
        train_ids = _read_ids(fold_dir / "train_ids.txt")
        val_ids = _read_ids(fold_dir / "val_ids.txt")
        if (
            set(train_ids) & set(val_ids)
            or set(train_ids) | set(val_ids) != set(pool_ids)
        ):
            raise ValueError(f"fold {fold} is not an exact development partition")
        contracts.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})
    held_out = [pid for contract in contracts for pid in contract["val_ids"]]
    if len(held_out) != len(pool_ids) or set(held_out) != set(pool_ids):
        raise ValueError("fold validation sets do not cover the pool exactly once")
    return contracts


def _validate_checkpoint_grid(new_root, selection, contracts):
    expected = []
    tune_variants = (
        "cv_current", "prefix_dropout", "ultra_compact_adamw",
        "ultra_compact_prefix",
    )
    for stage, seeds, variants in (
        ("tune", (2026, 2027), tune_variants),
        ("formal", SEEDS, None),
    ):
        for depth, max_tp in DEPTHS:
            depth_variants = variants or (selection["depths"][depth]["variant"],)
            for variant in depth_variants:
                for seed in seeds:
                    for contract in contracts:
                        fold = contract["fold"]
                        path = (
                            new_root / "train" / stage
                            / depth.lower().replace("-", "_")
                            / f"variant_{variant}" / f"seed_{seed}"
                            / f"fold_{fold}" / "best.pt"
                        )
                        expected.append(path)
                        checkpoint = torch.load(
                            path, map_location="cpu", weights_only=False
                        )
                        if (
                            checkpoint.get("schema")
                            != "pillar_full978_independent_depth_cv_v1"
                            or checkpoint.get("stage") != stage
                            or checkpoint.get("temporal_depth") != depth
                            or int(checkpoint.get("max_tp", -1)) != max_tp
                            or checkpoint.get("variant") != variant
                            or int(checkpoint.get("seed", -1)) != int(seed)
                            or int(checkpoint.get("fold", -1)) != fold
                            or checkpoint.get("train_ids") != contract["train_ids"]
                            or checkpoint.get("validation_ids") != contract["val_ids"]
                            or checkpoint.get("test_data_loaded_during_training") is not False
                            or checkpoint.get("test_embeddings_or_labels_loaded_during_training") is not False
                            or checkpoint.get("clinical_prior", {}).get("fitted_on")
                            != "fold_train_only"
                            or not checkpoint.get("model_state")
                        ):
                            raise ValueError(f"checkpoint contract mismatch: {path}")
    discovered = list((new_root / "train/tune").glob("**/best.pt"))
    discovered += list((new_root / "train/formal").glob("**/best.pt"))
    if set(discovered) != set(expected):
        raise ValueError("checkpoint path grid mismatch")
    return len(expected)


def _validate_individual_test_artifacts(new_root, aggregate, test_ids):
    checked = 0
    compare_columns = [
        "label", "probability", "residual_logit", "clinical_prior_probability",
        "seed", "fold", "max_tp",
    ]
    for source in SOURCES:
        for depth, max_tp in DEPTHS:
            slug = depth.lower().replace("-", "_")
            for seed in SEEDS:
                run_dir = new_root / "evaluation" / source / slug / f"seed_{seed}"
                ensemble = pd.read_csv(
                    run_dir / "test_predictions.csv", dtype={"patient_id": str}
                )
                _validate_prediction_contract(
                    ensemble, test_ids, seed=seed, fold=-1,
                    split="test_fold_ensemble", depth=depth, max_tp=max_tp,
                    label=f"{source} {depth} seed {seed} ensemble",
                )
                selected = aggregate[
                    (aggregate["source"] == source)
                    & (aggregate["temporal_depth"] == depth)
                    & (aggregate["seed"].astype(int) == seed)
                ]
                _assert_close(
                    selected, ensemble,
                    ["patient_id", "source", "temporal_depth", "split"],
                    compare_columns, f"{source} {depth} seed {seed} aggregate",
                )
                folds = pd.read_csv(
                    run_dir / "fold_predictions.csv", dtype={"patient_id": str}
                )
                if (
                    len(folds) != 5 * len(test_ids)
                    or set(folds["fold"].astype(int)) != set(range(5))
                ):
                    raise ValueError(f"{source} {depth} seed {seed} fold count mismatch")
                for fold in range(5):
                    frame = folds[folds["fold"].astype(int) == fold]
                    _validate_prediction_contract(
                        frame, test_ids, seed=seed, fold=fold, split="test_fold",
                        depth=depth, max_tp=max_tp,
                        label=f"{source} {depth} seed {seed} fold {fold}",
                    )
                means = folds.groupby("patient_id", sort=False)["probability"].mean()
                if not np.allclose(
                    means.loc[test_ids].to_numpy(),
                    ensemble["probability"].to_numpy(),
                    rtol=1e-7, atol=1e-7,
                ):
                    raise ValueError(f"{source} {depth} seed {seed} ensemble mismatch")
                sentinel = json.loads(
                    (run_dir / "EVALUATION_COMPLETE.json").read_text()
                )
                if (
                    sentinel.get("complete") is not True
                    or sentinel.get("test_ids") != test_ids
                    or int(sentinel.get("folds_ensembled", -1)) != 5
                    or sentinel.get("test_used_for_selection") is not False
                ):
                    raise ValueError(f"{source} {depth} seed {seed} sentinel mismatch")
                checked += 1
    return checked


def _recompute_test(new_root, test_ids):
    all_predictions = []
    all_metrics = []
    all_summaries = []
    expected_ids = set(test_ids)
    for source in SOURCES:
        source_dir = new_root / "evaluation" / source
        predictions = pd.read_csv(
            source_dir / "all_test_predictions.csv", dtype={"patient_id": str}
        )
        if len(predictions) != len(SEEDS) * len(DEPTHS) * len(test_ids):
            raise ValueError(f"{source} aggregate prediction count mismatch")
        required = {
            "patient_id", "label", "probability", "seed", "temporal_depth",
            "max_tp", "source", "split",
        }
        if required - set(predictions):
            raise ValueError(f"{source} predictions are missing required columns")
        if set(predictions["source"]) != {source}:
            raise ValueError(f"{source} source column mismatch")
        if set(predictions["split"]) != {"test_fold_ensemble"}:
            raise ValueError(f"{source} split column mismatch")
        if not np.isfinite(predictions["probability"]).all():
            raise ValueError(f"{source} has non-finite probabilities")
        if not predictions["probability"].between(0, 1).all():
            raise ValueError(f"{source} has probabilities outside [0,1]")

        rows = []
        for depth, max_tp in DEPTHS:
            for seed in SEEDS:
                frame = predictions[
                    (predictions["temporal_depth"] == depth)
                    & (predictions["seed"].astype(int) == seed)
                ]
                if (
                    frame["patient_id"].tolist() != test_ids
                    or set(frame["patient_id"]) != expected_ids
                    or len(frame) != len(test_ids)
                    or frame["patient_id"].duplicated().any()
                    or set(frame["max_tp"].astype(int)) != {max_tp}
                ):
                    raise ValueError(f"{source} {depth} seed {seed} test coverage mismatch")
                rows.append(
                    {
                        "source": source,
                        "temporal_depth": depth,
                        "max_tp": max_tp,
                        "seed": seed,
                        **compute_metrics(frame["label"], frame["probability"], threshold=0.5),
                    }
                )
        recomputed = pd.DataFrame(rows)
        saved = pd.read_csv(source_dir / "metrics_per_seed.csv")
        _assert_close(
            saved,
            recomputed,
            ["source", "temporal_depth", "max_tp", "seed"],
            METRIC_KEYS,
            f"{source} per-seed metrics",
        )
        summary_rows = []
        for depth, max_tp in DEPTHS:
            selected = recomputed[recomputed["temporal_depth"] == depth]
            row = {
                "source": source,
                "temporal_depth": depth,
                "max_tp": max_tp,
                "n_seeds": len(SEEDS),
            }
            for metric in METRIC_KEYS:
                values = selected[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)
        saved_summary = pd.read_csv(source_dir / "summary.csv")
        _assert_close(
            saved_summary,
            summary,
            ["source", "temporal_depth", "max_tp", "n_seeds"],
            [f"{metric}_{suffix}" for metric in METRIC_KEYS for suffix in ("mean", "std")],
            f"{source} summary",
        )
        all_predictions.append(predictions)
        all_metrics.append(recomputed)
        all_summaries.append(summary)
    return (
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_metrics, ignore_index=True),
        pd.concat(all_summaries, ignore_index=True),
    )


def _oof_and_thresholds(new_root, selection, pool_ids, test_ids, contracts):
    pool_set = set(pool_ids)
    test_set = set(test_ids)
    rows = []
    thresholds = []
    for depth, max_tp in DEPTHS:
        variant = selection["depths"][depth]["variant"]
        slug = depth.lower().replace("-", "_")
        for seed in SEEDS:
            fold_frames = []
            fold_train_aurocs = []
            best_epochs = []
            parameter_counts = []
            for contract in contracts:
                fold = contract["fold"]
                run_dir = (
                    new_root / "train/formal" / slug / f"variant_{variant}"
                    / f"seed_{seed}" / f"fold_{fold}"
                )
                train = pd.read_csv(
                    run_dir / "train_predictions.csv", dtype={"patient_id": str}
                )
                val = pd.read_csv(
                    run_dir / "val_predictions.csv", dtype={"patient_id": str}
                )
                _validate_prediction_contract(
                    train, contract["train_ids"], seed=seed, fold=fold,
                    split="fold_train", depth=depth, max_tp=max_tp,
                    label=f"formal {depth} seed {seed} fold {fold} train",
                )
                _validate_prediction_contract(
                    val, contract["val_ids"], seed=seed, fold=fold,
                    split="oof_val", depth=depth, max_tp=max_tp,
                    label=f"formal {depth} seed {seed} fold {fold} validation",
                )
                summary = json.loads((run_dir / "summary.json").read_text())
                train_metrics = compute_metrics(
                    train["label"], train["probability"], threshold=0.5
                )
                val_metrics = compute_metrics(
                    val["label"], val["probability"], threshold=0.5
                )
                for metric in METRIC_KEYS:
                    if not np.isclose(
                        train_metrics[metric], summary["train_metrics"][metric],
                        rtol=1e-10, atol=1e-10, equal_nan=True,
                    ):
                        raise ValueError(f"raw train {metric} mismatch: {run_dir}")
                    if not np.isclose(
                        val_metrics[metric], summary["validation_metrics"][metric],
                        rtol=1e-10, atol=1e-10, equal_nan=True,
                    ):
                        raise ValueError(f"raw validation {metric} mismatch: {run_dir}")
                if (
                    summary.get("schema") != "pillar_full978_independent_depth_cv_v1"
                    or summary.get("stage") != "formal"
                    or summary.get("variant") != variant
                    or summary.get("temporal_depth") != depth
                    or int(summary.get("seed", -1)) != seed
                    or int(summary.get("fold", -1)) != fold
                ):
                    raise ValueError(f"formal fold summary mismatch: {run_dir}")
                fold_frames.append(val)
                fold_train_aurocs.append(float(train_metrics["auroc"]))
                best_epochs.append(
                    int(summary["selection"]["best_epoch_zero_based"])
                )
                parameter_counts.append(int(summary["parameters"]))

            frame = pd.concat(fold_frames, ignore_index=True)
            if (
                len(frame) != len(pool_ids)
                or set(frame["patient_id"]) != pool_set
                or frame["patient_id"].duplicated().any()
                or set(frame["patient_id"]) & test_set
            ):
                raise ValueError(f"formal OOF coverage mismatch: {depth} seed {seed}")
            saved_oof = pd.read_csv(
                new_root / "oof/formal" / slug / f"variant_{variant}"
                / f"seed_{seed}" / "oof_predictions.csv",
                dtype={"patient_id": str},
            )
            _assert_close(
                saved_oof, frame,
                ["patient_id", "seed", "fold", "split", "temporal_depth", "max_tp"],
                ["label", "probability", "residual_logit", "clinical_prior_probability"],
                f"formal raw-fold OOF {depth} seed {seed}",
            )
            metrics = compute_metrics(frame["label"], frame["probability"], threshold=0.5)
            mean_train = float(np.mean(fold_train_aurocs))
            if len(set(parameter_counts)) != 1:
                raise ValueError(f"parameter count differs across folds: {depth} seed {seed}")
            rows.append(
                {
                    "stage": "formal",
                    "temporal_depth": depth,
                    "max_tp": max_tp,
                    "variant": variant,
                    "seed": seed,
                    "mean_fold_train_auroc": mean_train,
                    "train_oof_gap": mean_train - float(metrics["auroc"]),
                    "mean_best_epoch": float(np.mean(best_epochs)),
                    "parameters": parameter_counts[0],
                    **metrics,
                }
            )
            fpr, tpr, candidate_thresholds = roc_curve(
                frame["label"], frame["probability"], drop_intermediate=False
            )
            objective = tpr - fpr
            eligible = np.flatnonzero(objective == objective.max())
            finite = eligible[np.isfinite(candidate_thresholds[eligible])]
            if len(finite):
                eligible = finite
            index = eligible[
                np.argmin(np.abs(candidate_thresholds[eligible] - 0.5))
            ]
            thresholds.append(
                {
                    "temporal_depth": depth,
                    "max_tp": max_tp,
                    "variant": variant,
                    "seed": seed,
                    "selection_set": "formal_oof_876",
                    "selection_objective": "youden_j_equals_max_balanced_accuracy",
                    "threshold": float(candidate_thresholds[index]),
                    "oof_balanced_accuracy_at_threshold": float(
                        (1.0 + objective[index]) / 2.0
                    ),
                    "test_data_used": False,
                }
            )
    recomputed = pd.DataFrame(rows)
    saved = pd.read_csv(new_root / "formal_oof_metrics_all.csv")
    _assert_close(
        saved,
        recomputed,
        ["stage", "temporal_depth", "max_tp", "variant", "seed"],
        [
            "mean_fold_train_auroc", "train_oof_gap", "mean_best_epoch",
            "parameters", *METRIC_KEYS,
        ],
        "formal OOF metrics",
    )
    summary = (
        recomputed.groupby(["temporal_depth", "max_tp", "variant"], sort=False)
        .agg(
            oof_auroc_mean=("auroc", "mean"),
            oof_auroc_std=("auroc", "std"),
            oof_prauc_mean=("prauc", "mean"),
            mean_fold_train_auroc=("mean_fold_train_auroc", "mean"),
            train_oof_gap_mean=("train_oof_gap", "mean"),
            mean_best_epoch=("mean_best_epoch", "mean"),
            parameters=("parameters", "first"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    summary["positive_train_oof_gap"] = summary["train_oof_gap_mean"].clip(lower=0)
    saved_summary = pd.read_csv(new_root / "formal_variant_summary.csv")
    _assert_close(
        saved_summary, summary,
        ["temporal_depth", "max_tp", "variant"],
        [
            "oof_auroc_mean", "oof_auroc_std", "oof_prauc_mean",
            "mean_fold_train_auroc", "train_oof_gap_mean", "mean_best_epoch",
            "parameters", "n_seeds", "positive_train_oof_gap",
        ],
        "formal variant summary",
    )
    return recomputed, pd.DataFrame(thresholds), summary


def _threshold_test_metrics(predictions, thresholds):
    rows = []
    threshold_index = thresholds.set_index(["temporal_depth", "seed"])
    for source in SOURCES:
        for depth, max_tp in DEPTHS:
            for seed in SEEDS:
                frame = predictions[
                    (predictions["source"] == source)
                    & (predictions["temporal_depth"] == depth)
                    & (predictions["seed"].astype(int) == seed)
                ]
                threshold = float(threshold_index.loc[(depth, seed), "threshold"])
                selected_metrics = compute_metrics(
                    frame["label"], frame["probability"], threshold=threshold
                )
                fixed_metrics = compute_metrics(
                    frame["label"], frame["probability"], threshold=0.5
                )
                comparison = {}
                for metric in ("acc", "sens", "spec", "bacc"):
                    comparison[f"fixed_0_5_{metric}"] = fixed_metrics[metric]
                    comparison[f"{metric}_delta_vs_fixed_0_5"] = (
                        selected_metrics[metric] - fixed_metrics[metric]
                    )
                rows.append(
                    {
                        "source": source,
                        "temporal_depth": depth,
                        "max_tp": max_tp,
                        "seed": seed,
                        "threshold": threshold,
                        "threshold_selected_on": "formal_oof_876",
                        **selected_metrics,
                        **comparison,
                    }
                )
    metrics = pd.DataFrame(rows)
    aliases = {
        "acc": "accuracy", "sens": "sensitivity",
        "spec": "specificity", "bacc": "balanced_accuracy",
    }
    summary_rows = []
    for keys, frame in metrics.groupby(
        ["source", "temporal_depth", "max_tp"], sort=False
    ):
        row = dict(zip(["source", "temporal_depth", "max_tp"], keys))
        row["threshold_mean"] = float(frame["threshold"].mean())
        row["threshold_std"] = float(frame["threshold"].std(ddof=1))
        for metric, alias in aliases.items():
            for column, prefix in (
                (metric, alias),
                (f"fixed_0_5_{metric}", f"fixed_0_5_{alias}"),
                (f"{metric}_delta_vs_fixed_0_5", f"{alias}_delta_vs_fixed_0_5"),
            ):
                row[f"{prefix}_mean"] = float(frame[column].mean())
                row[f"{prefix}_std"] = float(frame[column].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return metrics, summary


def _qc_sensitivity(predictions):
    rows = []
    for source in SOURCES:
        for depth, max_tp in DEPTHS:
            for seed in SEEDS:
                base = predictions[
                    (predictions["source"] == source)
                    & (predictions["temporal_depth"] == depth)
                    & (predictions["seed"].astype(int) == seed)
                ]
                for cohort in COHORTS:
                    excluded = excluded_patient_ids(cohort, depth)
                    frame = base[~base["patient_id"].isin(excluded)]
                    rows.append(
                        {
                            "analysis_status": ANALYSIS_STATUS,
                            "source": source,
                            "temporal_depth": depth,
                            "max_tp": max_tp,
                            "seed": seed,
                            "cohort": cohort,
                            "n_excluded": len(excluded),
                            "n_patients": len(frame),
                            "n_positive": int(frame["label"].sum()),
                            "n_negative": int((frame["label"] == 0).sum()),
                            **compute_metrics(
                                frame["label"], frame["probability"], threshold=0.5
                            ),
                        }
                    )
    metrics = pd.DataFrame(rows)
    summary_rows = []
    for keys, frame in metrics.groupby(
        ["analysis_status", "source", "temporal_depth", "max_tp", "cohort"],
        sort=False,
    ):
        row = dict(
            zip(
                ["analysis_status", "source", "temporal_depth", "max_tp", "cohort"],
                keys,
            )
        )
        for column in ("n_excluded", "n_patients", "n_positive", "n_negative"):
            row[column] = int(frame[column].iloc[0])
        for metric in METRIC_KEYS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    reference = summary[
        summary["cohort"] == "full_locked102"
    ][["source", "temporal_depth", "auroc_mean", "prauc_mean"]].rename(
        columns={"auroc_mean": "full_auroc", "prauc_mean": "full_prauc"}
    )
    summary = summary.merge(reference, on=["source", "temporal_depth"], validate="many_to_one")
    summary["auroc_delta_vs_full"] = summary["auroc_mean"] - summary["full_auroc"]
    summary["prauc_delta_vs_full"] = summary["prauc_mean"] - summary["full_prauc"]
    return metrics, summary


def _old_new(new_metrics, old_root):
    old_frames = []
    for source in SOURCES:
        frame = pd.read_csv(
            old_root / "independent" / "evaluation" / source / "metrics_per_seed.csv"
        )
        old_frames.append(frame[frame["temporal_depth"].isin(dict(DEPTHS))])
    old = pd.concat(old_frames, ignore_index=True)
    keys = ["source", "temporal_depth", "max_tp", "seed"]
    paired = old.merge(new_metrics, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    for metric in METRIC_KEYS:
        paired[f"{metric}_delta"] = paired[f"{metric}_new"] - paired[f"{metric}_old"]
    rows = []
    for (source, depth), frame in paired.groupby(["source", "temporal_depth"], sort=False):
        row = {"source": source, "temporal_depth": depth, "n_seeds": len(frame)}
        for metric in ("auroc", "prauc", "acc", "sens", "spec", "bacc"):
            row[f"{metric}_old_mean"] = float(frame[f"{metric}_old"].mean())
            row[f"{metric}_old_std"] = float(frame[f"{metric}_old"].std(ddof=1))
            row[f"{metric}_new_mean"] = float(frame[f"{metric}_new"].mean())
            row[f"{metric}_new_std"] = float(frame[f"{metric}_new"].std(ddof=1))
            row[f"{metric}_delta_mean"] = float(frame[f"{metric}_delta"].mean())
            row[f"{metric}_delta_std"] = float(frame[f"{metric}_delta"].std(ddof=1))
            row[f"{metric}_wins"] = int((frame[f"{metric}_delta"] > 0).sum())
        rows.append(row)
    return paired, pd.DataFrame(rows)


def _combined_temporal_suite(new_summary, old_root):
    rows = []
    for source in SOURCES:
        old = pd.read_csv(
            old_root / "independent" / "evaluation" / source / "summary.csv"
        )
        for depth in ("T0", "T0-T1"):
            value = old[old["temporal_depth"] == depth].iloc[0].to_dict()
            value["training_policy"] = "kept_fixed_split_independent"
            rows.append(value)
        for depth in ("T0-T2", "T0-T3"):
            value = new_summary[
                (new_summary["source"] == source)
                & (new_summary["temporal_depth"] == depth)
            ].iloc[0].to_dict()
            value["strategy"] = "independent_cv_5fold"
            value["training_policy"] = "new_depth_specific_five_fold_ensemble"
            rows.append(value)
    return pd.DataFrame(rows)


def _format_table(frame, columns, digits=4):
    labels = [label for _, label in columns]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for row in frame.itertuples(index=False):
        values = []
        mapping = row._asdict()
        for column, _ in columns:
            value = mapping[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(output_dir, selection, formal_summary, new_summary, comparison, threshold_summary, qc):
    selection_rows = []
    for depth, values in selection["depths"].items():
        selection_rows.append(
            {
                "temporal_depth": depth,
                "variant": values["variant"],
                "parameters": int(
                    formal_summary[formal_summary["temporal_depth"] == depth][
                        "parameters"
                    ].iloc[0]
                ),
            }
        )
    result = new_summary.copy()
    result["auroc"] = result.apply(
        lambda row: f"{row.auroc_mean:.4f} +/- {row.auroc_std:.4f}", axis=1
    )
    result["prauc"] = result.apply(
        lambda row: f"{row.prauc_mean:.4f} +/- {row.prauc_std:.4f}", axis=1
    )
    qc_visible = qc[qc["cohort"] == "exclude_depth_visible_qc_patients"].copy()
    report = f"""# Independent-depth five-fold CV audit

## Contract

- T0-T2 and T0-T3 use separate parameters; neither shares weights with the other depth.
- Candidate and epoch selection use the 876-patient development pool. Test images, labels, clinical values, and embeddings are loaded only after selection.
- Each seed-level test probability is the mean of five fold models. Mean +/- SD is then calculated across seeds 42-51, not across 1,020 independent patients.
- The locked102 cohort was inspected in earlier project experiments, so these are retrospective method-development results rather than untouched prospective confirmation.
- Formal OOF reuses the tune fold partition and is selection-conditioned, not nested-CV performance.

## Selected models

{_format_table(pd.DataFrame(selection_rows), [("temporal_depth", "Depth"), ("variant", "Variant"), ("parameters", "Parameters")], 0)}

## Intact locked102 results

{_format_table(result, [("source", "Source"), ("temporal_depth", "Depth"), ("auroc", "AUROC mean +/- SD"), ("prauc", "PR-AUC mean +/- SD")])}

## Change from the prior independent models

{_format_table(comparison, [("source", "Source"), ("temporal_depth", "Depth"), ("auroc_old_mean", "Old AUROC"), ("auroc_new_mean", "New AUROC"), ("auroc_delta_mean", "Delta"), ("auroc_wins", "Seeds improved")])}

At the fixed 0.5 threshold, the T0-T2 model becomes more specific and less sensitive. The following thresholds are selected independently for each seed from its 876-patient OOF predictions, then applied unchanged to both test sources. Fixed-0.5 results remain primary.

{_format_table(threshold_summary, [("source", "Source"), ("temporal_depth", "Depth"), ("threshold_mean", "OOF threshold"), ("sensitivity_mean", "Sensitivity"), ("specificity_mean", "Specificity"), ("balanced_accuracy_mean", "Balanced accuracy")])}

## Post-hoc QC exclusion sensitivity

The QC cohorts are model-independent but were defined after examining this test collection. They cannot replace locked102 or guide model selection.

{_format_table(qc_visible, [("source", "Source"), ("temporal_depth", "Depth"), ("n_excluded", "Excluded"), ("auroc_mean", "Excluded AUROC"), ("auroc_delta_vs_full", "Delta vs full"), ("prauc_delta_vs_full", "PR-AUC delta")])}

All depth-visible exclusion arms lower both mean AUROC and mean PR-AUC. This does not support patient deletion as a performance remedy. Because exclusion changes the case mix, it does not establish how much the flagged scans causally contribute to overfitting. Repairing or prospectively masking a failed visit remains preferable to deleting a patient.

## Remaining overfitting

Formal mean fold-train AUROC is {formal_summary.iloc[0].mean_fold_train_auroc:.4f} at T0-T2 and {formal_summary.iloc[1].mean_fold_train_auroc:.4f} at T0-T3; pooled OOF AUROC is {formal_summary.iloc[0].oof_auroc_mean:.4f} and {formal_summary.iloc[1].oof_auroc_mean:.4f}. The corresponding gaps, {formal_summary.iloc[0].train_oof_gap_mean:.4f} and {formal_summary.iloc[1].train_oof_gap_mean:.4f}, show that each fitted fold model retains overfitting. Test predictions are substantially more stable than in the prior single-split experiment, but this package does not include a same-checkpoint non-ensemble control that could isolate how much of that change comes from ensembling alone.
"""
    _atomic_text(output_dir / "independent_cv_audit.md", report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    new_root = args.new_root.resolve()
    old_root = args.old_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else new_root / "analysis"
    )
    config = json.loads((new_root / "EXPERIMENT_COMPLETE.json").read_text())
    if config.get("schema") != "pillar_full978_independent_depth_cv_v1" or not config.get("complete"):
        raise ValueError("independent CV experiment is not complete")
    selection = json.loads((new_root / "selected_variants.json").read_text())
    cohort = ROOT / "data/mewm_ispy2_full978_locked102"
    train_ids = _read_ids(cohort / "splits/train_ids.txt")
    val_ids = _read_ids(cohort / "splits/val_ids.txt")
    test_ids = _read_ids(cohort / "splits/test_ids.txt")
    pool_ids = train_ids + val_ids
    if (
        len(pool_ids) != 876
        or len(test_ids) != 102
        or set(pool_ids) & set(test_ids)
        or len(set(pool_ids)) != 876
    ):
        raise ValueError("development/test split contract mismatch")
    contracts = _fold_contracts(new_root, pool_ids)
    tune_checkpoints = list((new_root / "train/tune").glob("**/best.pt"))
    formal_checkpoints = list((new_root / "train/formal").glob("**/best.pt"))
    prediction_files = list((new_root / "evaluation").glob("**/test_predictions.csv"))
    if (len(tune_checkpoints), len(formal_checkpoints), len(prediction_files)) != (80, 100, 40):
        raise ValueError("formal artifact count mismatch")
    validated_checkpoints = _validate_checkpoint_grid(new_root, selection, contracts)

    predictions, test_metrics, test_summary = _recompute_test(new_root, test_ids)
    validated_test_runs = _validate_individual_test_artifacts(
        new_root, predictions, test_ids
    )
    oof_metrics, thresholds, formal_summary = _oof_and_thresholds(
        new_root, selection, pool_ids, test_ids, contracts
    )
    threshold_metrics, threshold_summary = _threshold_test_metrics(
        predictions, thresholds
    )
    qc_metrics, qc_summary = _qc_sensitivity(predictions)
    paired, comparison = _old_new(test_metrics, old_root)
    combined = _combined_temporal_suite(test_summary, old_root)

    _atomic_csv(output_dir / "test_metrics_recomputed.csv", test_metrics)
    _atomic_csv(output_dir / "test_summary_recomputed.csv", test_summary)
    _atomic_csv(output_dir / "formal_oof_metrics_recomputed.csv", oof_metrics)
    _atomic_csv(output_dir / "formal_variant_summary_recomputed.csv", formal_summary)
    _atomic_csv(output_dir / "oof_selected_thresholds.csv", thresholds)
    _atomic_csv(output_dir / "oof_threshold_test_metrics.csv", threshold_metrics)
    _atomic_csv(output_dir / "oof_threshold_test_summary.csv", threshold_summary)
    _atomic_csv(output_dir / "qc_exclusion_metrics_per_seed.csv", qc_metrics)
    _atomic_csv(output_dir / "qc_exclusion_summary.csv", qc_summary)
    _atomic_csv(output_dir / "old_vs_new_per_seed.csv", paired)
    _atomic_csv(output_dir / "old_vs_new_summary.csv", comparison)
    _atomic_csv(output_dir / "combined_four_depth_summary.csv", combined)
    _write_report(
        output_dir,
        selection,
        formal_summary,
        test_summary,
        comparison,
        threshold_summary,
        qc_summary,
    )
    _atomic_json(
        output_dir / "ANALYSIS_COMPLETE.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "development_patients": len(pool_ids),
            "locked_test_patients": len(test_ids),
            "tune_checkpoints": len(tune_checkpoints),
            "formal_checkpoints": len(formal_checkpoints),
            "checkpoint_metadata_validated": validated_checkpoints,
            "test_prediction_files": len(prediction_files),
            "individual_test_runs_validated": validated_test_runs,
            "metrics_recomputed_from_patient_probabilities": True,
            "fold_train_and_oof_metrics_recomputed_from_raw_predictions": True,
            "test_used_for_variant_or_epoch_selection": False,
            "qc_analysis_status": ANALYSIS_STATUS,
        },
    )
    print(f"analysis complete: {output_dir}")


if __name__ == "__main__":
    main()

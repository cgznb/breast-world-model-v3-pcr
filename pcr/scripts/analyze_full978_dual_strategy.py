"""Audit and compare the full978 independent and shared temporal strategies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import METRIC_KEYS, compute_metrics


ROOT = Path(__file__).resolve().parents[1]
DUAL = ROOT / "results/mewm_ispy2_full978_locked102_dual_strategy"
TABLE2 = ROOT / "results/mewm_ispy2_full978_locked102_table2"
GENERATED_TABLE2 = (
    ROOT
    / "results/mewm_ispy2_full978_locked102_table2_generated_future_dce0_euler20_full296"
)
ANTI = ROOT / "results/mewm_ispy2_full978_locked102_anti_overfit"
ANALYSIS = DUAL / "analysis"
DEPTHS = ("T0", "T0-T1", "T0-T2", "T0-T3")
SEEDS = tuple(range(42, 52))
METHODS = (
    "independent_original",
    "independent_contiguous_regularized",
    "shared_full_plus_random_5fold",
    "shared_unique_contiguous_fixed_split",
)


def _atomic_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _normalize_predictions(frame, method, source):
    result = frame[["patient_id", "label", "probability", "seed", "temporal_depth"]].copy()
    result["patient_id"] = result["patient_id"].astype(str)
    result["label"] = result["label"].astype(int)
    result["seed"] = result["seed"].astype(int)
    result["method"] = method
    result["source"] = source
    return result


def _load_predictions():
    frames = []
    old_real = pd.read_csv(TABLE2 / "all_predictions.csv", dtype={"patient_id": str})
    old_real = old_real[old_real["split"] == "test"]
    frames.append(_normalize_predictions(old_real, METHODS[0], "real"))

    old_generated = pd.read_csv(
        GENERATED_TABLE2 / "all_test_predictions.csv", dtype={"patient_id": str}
    )
    old_generated = old_generated[
        old_generated["input_variant"] == "real_t0_generated_future_dce0"
    ]
    frames.append(_normalize_predictions(old_generated, METHODS[0], "generated"))

    for source in ("real", "generated"):
        old_shared = pd.read_csv(
            ANTI / "evaluation" / source / "all_test_predictions.csv",
            dtype={"patient_id": str},
        )
        frames.append(_normalize_predictions(old_shared, METHODS[2], source))
        for strategy, method in (
            ("independent", METHODS[1]),
            ("shared_contiguous", METHODS[3]),
        ):
            new = pd.read_csv(
                DUAL / strategy / "evaluation" / source / "all_test_predictions.csv",
                dtype={"patient_id": str},
            )
            frames.append(_normalize_predictions(new, method, source))
    predictions = pd.concat(frames, ignore_index=True)
    return predictions


def _audit_predictions(predictions):
    expected_ids = set(
        line.strip()
        for line in (
            ROOT / "data/mewm_ispy2_full978_locked102/splits/test_ids.txt"
        ).read_text().splitlines()
        if line.strip()
    )
    expected_pairs = {(patient_id, seed, depth) for patient_id in expected_ids for seed in SEEDS for depth in DEPTHS}
    for method in METHODS:
        for source in ("real", "generated"):
            selected = predictions[
                (predictions["method"] == method) & (predictions["source"] == source)
            ]
            actual_pairs = set(
                zip(selected["patient_id"], selected["seed"], selected["temporal_depth"])
            )
            if len(selected) != len(expected_pairs) or actual_pairs != expected_pairs:
                raise RuntimeError(f"prediction coverage mismatch for {method} {source}")
            if not selected["label"].isin([0, 1]).all():
                raise RuntimeError("predictions contain non-binary labels")
            if not np.isfinite(selected["probability"].to_numpy(dtype=float)).all():
                raise RuntimeError("predictions contain non-finite probabilities")
            if not selected["probability"].between(0, 1).all():
                raise RuntimeError("predictions contain out-of-range probabilities")
    labels_per_patient = predictions.groupby("patient_id")["label"].nunique()
    if not labels_per_patient.eq(1).all():
        raise RuntimeError("patient labels disagree across methods or sources")

    for method in METHODS:
        real_t0 = predictions[
            (predictions["method"] == method)
            & (predictions["source"] == "real")
            & (predictions["temporal_depth"] == "T0")
        ].sort_values(["seed", "patient_id"])
        generated_t0 = predictions[
            (predictions["method"] == method)
            & (predictions["source"] == "generated")
            & (predictions["temporal_depth"] == "T0")
        ].sort_values(["seed", "patient_id"])
        if not np.array_equal(
            real_t0["probability"].to_numpy(), generated_t0["probability"].to_numpy()
        ):
            raise RuntimeError(f"T0 real/generated predictions differ for {method}")
    return sorted(expected_ids)


def _recompute_metrics(predictions):
    rows = []
    for (method, source, seed, depth), frame in predictions.groupby(
        ["method", "source", "seed", "temporal_depth"], sort=False
    ):
        rows.append(
            {
                "method": method,
                "source": source,
                "seed": int(seed),
                "temporal_depth": depth,
                **compute_metrics(frame["label"], frame["probability"], threshold=0.5),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(METHODS) * 2 * len(SEEDS) * len(DEPTHS):
        raise RuntimeError("recomputed metric grid is incomplete")
    return result


def _summarize_metrics(metrics):
    rows = []
    for (method, source, depth), frame in metrics.groupby(
        ["method", "source", "temporal_depth"], sort=False
    ):
        row = {
            "method": method,
            "source": source,
            "temporal_depth": depth,
            "n_seeds": len(frame),
        }
        for metric in METRIC_KEYS:
            values = frame[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_deltas(metrics):
    comparisons = {
        "new_independent_minus_original": (METHODS[1], METHODS[0]),
        "new_shared_minus_new_independent": (METHODS[3], METHODS[1]),
        "new_shared_minus_old_shared_5fold": (METHODS[3], METHODS[2]),
        "old_shared_5fold_minus_original": (METHODS[2], METHODS[0]),
    }
    rows = []
    keys = ["source", "seed", "temporal_depth"]
    for name, (left_name, right_name) in comparisons.items():
        left = metrics[metrics["method"] == left_name]
        right = metrics[metrics["method"] == right_name]
        paired = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
        for row in paired.to_dict("records"):
            output = {"comparison": name, **{key: row[key] for key in keys}}
            for metric in METRIC_KEYS:
                output[f"{metric}_delta"] = float(row[f"{metric}_left"] - row[f"{metric}_right"])
            rows.append(output)
    detail = pd.DataFrame(rows)
    summary_rows = []
    for (comparison, source, depth), frame in detail.groupby(
        ["comparison", "source", "temporal_depth"], sort=False
    ):
        row = {
            "comparison": comparison,
            "source": source,
            "temporal_depth": depth,
            "n_seeds": len(frame),
        }
        for metric in METRIC_KEYS:
            values = frame[f"{metric}_delta"].to_numpy(dtype=float)
            row[f"{metric}_delta_mean"] = float(np.mean(values))
            row[f"{metric}_delta_std"] = float(np.std(values, ddof=1))
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def _formal_development_metrics():
    selection = json.loads((DUAL / "selected_variants.json").read_text())
    rows = []
    old_metrics = pd.read_csv(TABLE2 / "table2_metrics_per_seed.csv")
    old_t0_val = old_metrics[
        (old_metrics["split"] == "val") & (old_metrics["temporal_depth"] == "T0")
    ]
    for row in old_t0_val.itertuples(index=False):
        rows.append(
            {
                "strategy": "independent",
                "seed": int(row.seed),
                "temporal_depth": "T0",
                "train_auroc": np.nan,
                "validation_auroc": float(row.auroc),
                "best_epoch_zero_based": np.nan,
            }
        )
    for strategy in ("independent", "shared_contiguous"):
        for seed in SEEDS:
            if strategy == "independent":
                specs = [
                    (depth, selection[strategy][depth]["variant"]) for depth in DEPTHS[1:]
                ]
            else:
                specs = [(None, selection[strategy]["variant"])]
            for depth, variant in specs:
                run_dir = DUAL / strategy / "train" / "formal"
                if depth is not None:
                    run_dir = run_dir / depth.lower().replace("-", "_")
                summary = json.loads(
                    (run_dir / f"variant_{variant}" / f"seed_{seed}" / "summary.json").read_text()
                )
                train_by_depth = {row["temporal_depth"]: row for row in summary["train_metrics"]}
                val_by_depth = {
                    row["temporal_depth"]: row for row in summary["validation_metrics"]
                }
                selected_depths = [depth] if depth is not None else list(DEPTHS)
                for selected_depth in selected_depths:
                    rows.append(
                        {
                            "strategy": strategy,
                            "seed": int(seed),
                            "temporal_depth": selected_depth,
                            "train_auroc": float(train_by_depth[selected_depth]["auroc"]),
                            "validation_auroc": float(val_by_depth[selected_depth]["auroc"]),
                            "best_epoch_zero_based": int(
                                summary["selection"]["best_epoch_zero_based"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _generalization_summary(development, metric_summary):
    old_comparison = pd.read_csv(ANTI / "analysis" / "old_vs_new.csv").set_index(
        "temporal_depth"
    )
    rows = []
    for strategy, method in (
        ("independent", METHODS[1]),
        ("shared_contiguous", METHODS[3]),
    ):
        for depth in DEPTHS:
            selected = development[
                (development["strategy"] == strategy)
                & (development["temporal_depth"] == depth)
            ]
            train_values = selected["train_auroc"].dropna().to_numpy(dtype=float)
            if strategy == "independent" and depth == "T0":
                train_mean = float(old_comparison.loc[depth, "old_train_auroc_mean"])
                train_std = np.nan
            else:
                train_mean = float(np.mean(train_values))
                train_std = float(np.std(train_values, ddof=1))
            validation_values = selected["validation_auroc"].to_numpy(dtype=float)
            test_row = metric_summary[
                (metric_summary["method"] == method)
                & (metric_summary["source"] == "real")
                & (metric_summary["temporal_depth"] == depth)
            ].iloc[0]
            train_validation_gap = train_mean - float(np.mean(validation_values))
            train_test_gap = train_mean - float(test_row["auroc_mean"])
            if train_validation_gap > 0.10:
                status = "severe"
            elif train_validation_gap > 0.05:
                status = "moderate"
            else:
                status = "low"
            rows.append(
                {
                    "strategy": strategy,
                    "temporal_depth": depth,
                    "train_auroc_mean": train_mean,
                    "train_auroc_std": train_std,
                    "validation_auroc_mean": float(np.mean(validation_values)),
                    "validation_auroc_std": float(np.std(validation_values, ddof=1)),
                    "real_test_auroc_mean": float(test_row["auroc_mean"]),
                    "real_test_auroc_std": float(test_row["auroc_std"]),
                    "train_validation_gap": train_validation_gap,
                    "train_test_gap": train_test_gap,
                    "overfit_status_by_train_validation_gap": status,
                    "best_epoch_mean": float(selected["best_epoch_zero_based"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _selection_summary():
    selection = json.loads((DUAL / "selected_variants.json").read_text())
    rows = [
        {
            "strategy": "independent",
            "temporal_depth": "T0",
            "variant": "reused_original",
            "parameters": 203842,
            "embedding_dropout": 0.0,
            "dropout": 0.2,
            "weight_decay": 0.0001,
            "proj_dim": 128,
            "sig_dim": 64,
        }
    ]
    for depth in DEPTHS[1:]:
        selected = selection["independent"][depth]
        config = selected["effective_config"]
        rows.append(
            {
                "strategy": "independent",
                "temporal_depth": depth,
                "variant": selected["variant"],
                "parameters": 88610 if selected["variant"] == "compact" else 203842,
                **{
                    key: config[key]
                    for key in (
                        "embedding_dropout",
                        "dropout",
                        "weight_decay",
                        "proj_dim",
                        "sig_dim",
                    )
                },
            }
        )
    selected = selection["shared_contiguous"]
    config = selected["effective_config"]
    rows.append(
        {
            "strategy": "shared_contiguous",
            "temporal_depth": "all_unique_contiguous_prefixes",
            "variant": selected["variant"],
            "parameters": 88610 if selected["variant"] == "compact" else 203842,
            **{
                key: config[key]
                for key in (
                    "embedding_dropout",
                    "dropout",
                    "weight_decay",
                    "proj_dim",
                    "sig_dim",
                )
            },
        }
    )
    return pd.DataFrame(rows)


def _markdown_table(frame, columns, digits=4):
    selected = frame[columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()]
    return "\n".join([header, divider, *body])


def _write_report(summary, generalization, deltas, selection):
    real = summary[summary["source"] == "real"]
    generated = summary[summary["source"] == "generated"]
    independent_delta = deltas[
        (deltas["comparison"] == "new_independent_minus_original")
        & (deltas["source"] == "real")
    ]
    lines = [
        "# Full978 Dual Temporal Strategy Audit",
        "",
        "## Contract",
        "",
        "- Fixed development split: 778 train and 98 validation patients.",
        "- Locked test: exactly the 102-patient BiFlow validation set; no patient was removed.",
        "- T0-T1, T0-T2, and T0-T3 are separate checkpoints in the independent strategy; T0 reuses the verified original checkpoint.",
        "- The shared strategy trains one checkpoint on each unique contiguous T0-starting patient-depth pair once per epoch. T0,T2,T3 is therefore evaluated as T0 only.",
        "- Losses are averaged within depth and then equally across represented depths. The old full-plus-random-prefix duplicate sampling is absent.",
        "- Variant and epoch selection use only train/validation data. Test metrics are retrospective because this locked test had been examined in earlier experiments.",
        "- The real/generated arms use 296 generated future embeddings, but four post-gap tokens are deliberately ignored; 292 future tokens belong to valid contiguous prefixes.",
        "",
        "## Real Test AUROC",
        "",
        _markdown_table(
            real,
            ["method", "temporal_depth", "auroc_mean", "auroc_std", "prauc_mean", "prauc_std"],
        ),
        "",
        "## Generated Test AUROC",
        "",
        _markdown_table(
            generated,
            ["method", "temporal_depth", "auroc_mean", "auroc_std", "prauc_mean", "prauc_std"],
        ),
        "",
        "## New Strategy Generalization",
        "",
        _markdown_table(
            generalization,
            [
                "strategy",
                "temporal_depth",
                "train_auroc_mean",
                "validation_auroc_mean",
                "real_test_auroc_mean",
                "train_validation_gap",
                "train_test_gap",
                "overfit_status_by_train_validation_gap",
            ],
        ),
        "",
        "## Independent Improvement Versus Original",
        "",
        _markdown_table(
            independent_delta,
            ["temporal_depth", "auroc_delta_mean", "auroc_delta_std", "prauc_delta_mean", "prauc_delta_std"],
        ),
        "",
        "## Selected Regularization",
        "",
        _markdown_table(
            selection,
            [
                "strategy",
                "temporal_depth",
                "variant",
                "parameters",
                "embedding_dropout",
                "dropout",
                "weight_decay",
                "proj_dim",
                "sig_dim",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The regularized independent strategy improves real-test AUROC over the original independent models at T0-T1, T0-T2, and T0-T3, while reducing seed variability. It still does not produce a monotonic temporal curve: T0-T2 is slightly below T0-T1 and T0-T3 only returns to approximately T0 performance.",
        "",
        "The new shared-contiguous fixed-split strategy is much more stable than the independent strategy and has only a small T0-T2 dip. Its T0-T3 real AUROC is higher than its T0 AUROC, but the increase is modest. This supports the continuous-prefix semantics, not a claim that every added visit helps every patient.",
        "",
        "The old shared full-plus-random model remains numerically stronger, but it also uses five-fold development training and five-model probability ensembling. Comparing it with the new fixed-split shared model does not isolate prefix sampling: validation coverage and ensembling differ at the same time.",
        "",
        "Generated future imaging remains weaker than real future imaging for both new strategies. T0 is exactly unchanged because it is real in both arms.",
    ]
    _atomic_text(ANALYSIS / "dual_strategy_audit.md", "\n".join(lines) + "\n")


def main():
    completion = json.loads((DUAL / "EXPERIMENT_COMPLETE.json").read_text())
    if completion.get("complete") is not True:
        raise RuntimeError("dual-strategy experiment is incomplete")
    predictions = _load_predictions()
    test_ids = _audit_predictions(predictions)
    metrics = _recompute_metrics(predictions)
    summary = _summarize_metrics(metrics)
    delta_detail, delta_summary = _paired_deltas(metrics)
    development = _formal_development_metrics()
    generalization = _generalization_summary(development, summary)
    selection = _selection_summary()

    _atomic_csv(ANALYSIS / "method_metrics_recomputed.csv", metrics)
    _atomic_csv(ANALYSIS / "method_summary.csv", summary)
    _atomic_csv(ANALYSIS / "paired_deltas_per_seed.csv", delta_detail)
    _atomic_csv(ANALYSIS / "paired_delta_summary.csv", delta_summary)
    _atomic_csv(ANALYSIS / "formal_development_metrics.csv", development)
    _atomic_csv(ANALYSIS / "new_generalization_summary.csv", generalization)
    _atomic_csv(ANALYSIS / "selected_regularization.csv", selection)
    _write_report(summary, generalization, delta_summary, selection)
    _atomic_json(
        ANALYSIS / "ANALYSIS_COMPLETE.json",
        {
            "schema": "pillar_full978_dual_strategy_analysis_v1",
            "complete": True,
            "methods": list(METHODS),
            "sources": ["real", "generated"],
            "seeds": list(SEEDS),
            "temporal_depths": list(DEPTHS),
            "test_patients": len(test_ids),
            "metric_rows_recomputed": len(metrics),
            "test_used_for_training_or_selection": False,
        },
    )
    print(
        summary[
            ["method", "source", "temporal_depth", "auroc_mean", "auroc_std", "prauc_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

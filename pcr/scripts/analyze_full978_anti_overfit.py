"""Audit and summarize the full978 shared-prefix anti-overfit experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import _fold_specs, _prior_from_state
from src.data import TABULAR_FEATURE_NAMES, load_all_splits, load_ids
from src.metrics import METRIC_KEYS, compute_metrics
from src.tdn import TDN
from src.temporal import mask_torch_temporal_prefix


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric_row(frame):
    return compute_metrics(frame["label"], frame["probability"], threshold=0.5)


def _assert_close_metrics(actual, expected, context):
    for key in METRIC_KEYS:
        left = float(actual[key])
        right = float(expected[key])
        if not np.isclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"metric mismatch for {context} {key}: {left} != {right}")


def _audit_formal(config, result_dir, specs, pool_ids, metadata):
    anti = config["anti_overfit"]
    depths = anti["temporal_depths"]
    seeds = [int(value) for value in anti["formal_seeds"]]
    selection = json.loads((result_dir / "selected_variant.json").read_text())
    variant = str(selection["selected_variant"])
    checkpoint_count = 0
    train_metric_rows = []
    checkpoint_rows = []
    oof_metric_rows = []
    expected_pool = set(pool_ids)

    for seed in seeds:
        oof_parts = []
        for spec in specs:
            fold = int(spec["fold"])
            run_dir = (
                result_dir / "train" / "formal" / f"variant_{variant}"
                / f"seed_{seed}" / f"fold_{fold}"
            )
            required = (
                "best.pt", "history.csv", "train_predictions.csv", "val_predictions.csv",
                "train_metrics.csv", "val_metrics.csv", "TRAINING_COMPLETE.json",
            )
            if not all((run_dir / name).is_file() for name in required):
                raise FileNotFoundError(f"incomplete formal fold: {run_dir}")
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            if (
                checkpoint.get("variant") != variant
                or checkpoint.get("stage") != "formal"
                or int(checkpoint.get("seed", -1)) != seed
                or int(checkpoint.get("fold", -1)) != fold
                or checkpoint.get("test_data_loaded_during_training") is not False
                or checkpoint.get("clinical_prior", {}).get("fitted_on") != "fold_train_only"
                or checkpoint.get("clinical_prior", {}).get("feature_names")
                != list(TABULAR_FEATURE_NAMES)
            ):
                raise ValueError(f"formal checkpoint contract mismatch: {run_dir}")
            checkpoint_count += 1

            train_predictions = pd.read_csv(
                run_dir / "train_predictions.csv", dtype={"patient_id": str}
            )
            validation_predictions = pd.read_csv(
                run_dir / "val_predictions.csv", dtype={"patient_id": str}
            )
            for depth in depths:
                name = str(depth["name"])
                train_depth = train_predictions[train_predictions["temporal_depth"] == name]
                val_depth = validation_predictions[
                    validation_predictions["temporal_depth"] == name
                ]
                if train_depth["patient_id"].tolist() != spec["train_ids"]:
                    raise ValueError(f"fold-train IDs changed: seed={seed} fold={fold} {name}")
                if val_depth["patient_id"].tolist() != spec["val_ids"]:
                    raise ValueError(f"fold-validation IDs changed: seed={seed} fold={fold} {name}")
                expected_labels = metadata.loc[val_depth["patient_id"], "pCR"].to_numpy(int)
                if not np.array_equal(val_depth["label"].to_numpy(int), expected_labels):
                    raise ValueError("OOF labels disagree with metadata")
                train_metric_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "temporal_depth": name,
                        **_metric_row(train_depth),
                    }
                )
            oof_parts.append(validation_predictions)

            history = pd.read_csv(run_dir / "history.csv")
            best_epoch = int(checkpoint["selection"]["best_epoch_zero_based"])
            best_row = history[history["epoch"] == best_epoch]
            if len(best_row) != 1:
                raise ValueError("selected epoch is missing from history")
            checkpoint_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "epochs_trained": len(history),
                    "best_validation_score": float(checkpoint["selection"]["best_score"]),
                    "residual_scale_at_best": float(best_row.iloc[0]["residual_scale"]),
                }
            )

        oof = pd.concat(oof_parts, ignore_index=True)
        stored_oof = pd.read_csv(
            result_dir / "oof" / "formal" / f"variant_{variant}"
            / f"seed_{seed}" / "oof_predictions.csv",
            dtype={"patient_id": str},
        )
        pd.testing.assert_frame_equal(
            oof.reset_index(drop=True), stored_oof.reset_index(drop=True), check_dtype=False
        )
        for depth in depths:
            name = str(depth["name"])
            selected = oof[oof["temporal_depth"] == name]
            if (
                len(selected) != len(pool_ids)
                or set(selected["patient_id"]) != expected_pool
                or selected["patient_id"].duplicated().any()
            ):
                raise ValueError(f"OOF coverage mismatch: seed={seed} {name}")
            model_metrics = _metric_row(selected)
            oof_metric_rows.append(
                {
                    "seed": seed,
                    "temporal_depth": name,
                    **model_metrics,
                    "clinical_prior_auroc": float(
                        roc_auc_score(selected["label"], selected["clinical_prior_probability"])
                    ),
                    "clinical_prior_prauc": float(
                        average_precision_score(
                            selected["label"], selected["clinical_prior_probability"]
                        )
                    ),
                }
            )

    recomputed = pd.DataFrame(oof_metric_rows)
    stored = pd.read_csv(result_dir / "formal_oof_metrics_all.csv")
    for row in recomputed.itertuples(index=False):
        match = stored[
            (stored["seed"] == row.seed)
            & (stored["temporal_depth"] == row.temporal_depth)
        ]
        if len(match) != 1:
            raise ValueError("stored formal OOF metric row is missing or duplicated")
        _assert_close_metrics(row._asdict(), match.iloc[0].to_dict(), "formal OOF")
    return (
        variant,
        checkpoint_count,
        pd.DataFrame(train_metric_rows),
        recomputed,
        pd.DataFrame(checkpoint_rows),
    )


def _audit_evaluations(config, result_dir, test_ids, metadata):
    anti = config["anti_overfit"]
    depths = anti["temporal_depths"]
    seeds = [int(value) for value in anti["formal_seeds"]]
    expected_test = set(test_ids)
    metric_rows = []
    all_predictions = {}
    for source in config["evaluation_sources"]:
        source_parts = []
        stored_metrics = pd.read_csv(result_dir / "evaluation" / source / "metrics_per_seed.csv")
        for seed in seeds:
            run_dir = result_dir / "evaluation" / source / f"seed_{seed}"
            ensemble = pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
            fold_predictions = pd.read_csv(
                run_dir / "fold_predictions.csv", dtype={"patient_id": str}
            )
            source_parts.append(ensemble)
            for depth in depths:
                name = str(depth["name"])
                selected = ensemble[ensemble["temporal_depth"] == name]
                raw = fold_predictions[fold_predictions["temporal_depth"] == name]
                if (
                    len(selected) != len(test_ids)
                    or set(selected["patient_id"]) != expected_test
                    or selected["patient_id"].duplicated().any()
                    or len(raw) != len(test_ids) * 5
                    or set(raw.groupby("patient_id").size()) != {5}
                ):
                    raise ValueError(f"test coverage mismatch: {source} seed={seed} {name}")
                expected_labels = metadata.loc[selected["patient_id"], "pCR"].to_numpy(int)
                if not np.array_equal(selected["label"].to_numpy(int), expected_labels):
                    raise ValueError("test labels disagree with metadata")
                averaged = raw.groupby("patient_id")[
                    ["probability", "clinical_prior_probability"]
                ].mean()
                ordered = averaged.loc[selected["patient_id"]]
                if not np.allclose(
                    selected[["probability", "clinical_prior_probability"]],
                    ordered,
                    rtol=0,
                    atol=2 * np.finfo(np.float32).eps,
                ):
                    raise ValueError("test prediction is not the five-fold probability mean")
                metrics = _metric_row(selected)
                match = stored_metrics[
                    (stored_metrics["seed"] == seed)
                    & (stored_metrics["temporal_depth"] == name)
                ]
                if len(match) != 1:
                    raise ValueError("stored test metric row is missing or duplicated")
                _assert_close_metrics(metrics, match.iloc[0].to_dict(), f"{source} test")
                metric_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "temporal_depth": name,
                        **metrics,
                        "clinical_prior_auroc": float(
                            roc_auc_score(
                                selected["label"], selected["clinical_prior_probability"]
                            )
                        ),
                    }
                )
        all_predictions[source] = pd.concat(source_parts, ignore_index=True)

    real_t0 = all_predictions["real"].query("temporal_depth == 'T0'").sort_values(
        ["seed", "patient_id"]
    )
    generated_t0 = all_predictions["generated"].query(
        "temporal_depth == 'T0'"
    ).sort_values(["seed", "patient_id"])
    if not np.array_equal(real_t0["probability"].to_numpy(), generated_t0["probability"].to_numpy()):
        raise ValueError("generated-input evaluation changed T0 probabilities")
    return pd.DataFrame(metric_rows), all_predictions


def _old_training_metrics(old_config, old_result_dir):
    cohort = _repo_path(old_config["mewm_adapter"]["output_dir"])
    split_ids = {
        name: load_ids(cohort / "splits" / f"{name}_ids.txt")
        for name in ("train", "val", "test")
    }
    data = load_all_splits(
        _repo_path(old_config["mewm_adapter"]["embeddings_dir"]),
        cohort / "metadata_enriched.csv",
        split_ids["train"],
        split_ids["val"],
        split_ids["test"],
    )["train"]
    embeddings = torch.from_numpy(data["embs"])
    masks = torch.from_numpy(data["masks"])
    clinical = torch.from_numpy(data["clinical"])
    days = torch.from_numpy(data["days"])
    rows = []
    for depth in old_config["experiment"]["temporal_depths"]:
        name = str(depth["name"])
        slug = name.lower().replace("-", "_")
        prefix_embeddings, prefix_masks, prefix_days = mask_torch_temporal_prefix(
            embeddings, masks, days, int(depth["max_tp"])
        )
        for seed in old_config["experiment"]["seeds"]:
            checkpoint = torch.load(
                old_result_dir / slug / f"seed_{seed}" / "global_tab_temporal" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = TDN({"downstream": checkpoint["effective_config"]})
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            prior = torch.from_numpy(
                _prior_from_state(data["clinical"], checkpoint["clinical_prior"])
            )
            with torch.inference_mode():
                probability = torch.sigmoid(
                    model(
                        prefix_embeddings,
                        prefix_masks,
                        clinical,
                        days=prefix_days,
                        prior_logit=prior,
                    )
                ).numpy()
            rows.append(
                {
                    "seed": int(seed),
                    "temporal_depth": name,
                    "train_auroc": float(roc_auc_score(data["labels"], probability)),
                }
            )
    return pd.DataFrame(rows)


def _summary_tables(
    config, old_config, old_result_dir, train_metrics, oof_metrics, test_metrics,
    checkpoint_stats, predictions, metadata, patient_manifest,
):
    depths = [str(value["name"]) for value in config["anti_overfit"]["temporal_depths"]]
    old_train = _old_training_metrics(old_config, old_result_dir)
    old_stored = pd.read_csv(old_result_dir / "table2_metrics_per_seed.csv")
    rows = []
    for depth in depths:
        old_train_values = old_train[old_train["temporal_depth"] == depth]["train_auroc"]
        old_val_values = old_stored[
            (old_stored["temporal_depth"] == depth) & (old_stored["split"] == "val")
        ]["auroc"]
        old_test_values = old_stored[
            (old_stored["temporal_depth"] == depth) & (old_stored["split"] == "test")
        ]["auroc"]
        new_train_per_seed = (
            train_metrics[train_metrics["temporal_depth"] == depth]
            .groupby("seed")["auroc"].mean()
        )
        new_oof = oof_metrics[oof_metrics["temporal_depth"] == depth]["auroc"]
        real_test = test_metrics[
            (test_metrics["source"] == "real")
            & (test_metrics["temporal_depth"] == depth)
        ]["auroc"]
        generated_test = test_metrics[
            (test_metrics["source"] == "generated")
            & (test_metrics["temporal_depth"] == depth)
        ]["auroc"]
        rows.append(
            {
                "temporal_depth": depth,
                "old_train_auroc_mean": old_train_values.mean(),
                "old_validation_auroc_mean": old_val_values.mean(),
                "old_real_test_auroc_mean": old_test_values.mean(),
                "old_real_test_auroc_std": old_test_values.std(ddof=1),
                "old_train_minus_test_gap": old_train_values.mean() - old_test_values.mean(),
                "new_fold_train_auroc_mean": new_train_per_seed.mean(),
                "new_oof_auroc_mean": new_oof.mean(),
                "new_oof_auroc_std": new_oof.std(ddof=1),
                "new_real_test_auroc_mean": real_test.mean(),
                "new_real_test_auroc_std": real_test.std(ddof=1),
                "new_generated_test_auroc_mean": generated_test.mean(),
                "new_generated_test_auroc_std": generated_test.std(ddof=1),
                "new_train_minus_oof_gap": new_train_per_seed.mean() - new_oof.mean(),
                "new_train_minus_test_gap": new_train_per_seed.mean() - real_test.mean(),
            }
        )
    comparison = pd.DataFrame(rows)

    paired = test_metrics.pivot(
        index=["seed", "temporal_depth"], columns="source", values=list(METRIC_KEYS)
    )
    delta_rows = []
    for (seed, depth), row in paired.iterrows():
        delta_rows.append(
            {
                "seed": int(seed),
                "temporal_depth": depth,
                **{
                    f"{metric}_generated_minus_real": float(
                        row[(metric, "generated")] - row[(metric, "real")]
                    )
                    for metric in METRIC_KEYS
                },
            }
        )
    deltas = pd.DataFrame(delta_rows)
    delta_summary_rows = []
    for depth, group in deltas.groupby("temporal_depth", sort=False):
        row = {"temporal_depth": depth, "n_seeds": len(group)}
        for metric in METRIC_KEYS:
            values = group[f"{metric}_generated_minus_real"]
            row[f"{metric}_delta_mean"] = values.mean()
            row[f"{metric}_delta_std"] = values.std(ddof=1)
        delta_summary_rows.append(row)
    delta_summary = pd.DataFrame(delta_summary_rows)

    lookup_source = patient_manifest.set_index("patient_id")["source"]
    lookup_triple_negative = metadata["TripleNeg"]
    subgroup_rows = []
    for source_name, frame in predictions.items():
        annotated = frame.copy()
        annotated["data_source"] = annotated["patient_id"].map(lookup_source)
        annotated["triple_negative"] = annotated["patient_id"].map(
            lookup_triple_negative
        ).astype(int)
        for grouping, column in (
            ("registration_source", "data_source"),
            ("triple_negative", "triple_negative"),
        ):
            for (seed, depth, group_name), group in annotated.groupby(
                ["seed", "temporal_depth", column], sort=False
            ):
                subgroup_rows.append(
                    {
                        "input_source": source_name,
                        "grouping": grouping,
                        "group": str(group_name),
                        "seed": int(seed),
                        "temporal_depth": depth,
                        "patients": len(group),
                        "pcr_positive": int(group["label"].sum()),
                        "auroc": float(roc_auc_score(group["label"], group["probability"])),
                        "prauc": float(
                            average_precision_score(group["label"], group["probability"])
                        ),
                    }
                )
    subgroup_per_seed = pd.DataFrame(subgroup_rows)
    subgroup_summary = (
        subgroup_per_seed.groupby(
            ["input_source", "grouping", "group", "temporal_depth"],
            sort=False,
            as_index=False,
        )
        .agg(
            patients=("patients", "first"),
            pcr_positive=("pcr_positive", "first"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            prauc_mean=("prauc", "mean"),
            prauc_std=("prauc", "std"),
        )
    )

    checkpoint_summary = pd.DataFrame(
        [
            {
                "formal_checkpoints": len(checkpoint_stats),
                "best_epoch_mean": checkpoint_stats["best_epoch"].mean(),
                "best_epoch_min": checkpoint_stats["best_epoch"].min(),
                "best_epoch_max": checkpoint_stats["best_epoch"].max(),
                "epochs_trained_mean": checkpoint_stats["epochs_trained"].mean(),
                "residual_scale_at_best_mean": checkpoint_stats[
                    "residual_scale_at_best"
                ].mean(),
                "residual_scale_at_best_max": checkpoint_stats[
                    "residual_scale_at_best"
                ].max(),
            }
        ]
    )
    return comparison, deltas, delta_summary, subgroup_per_seed, subgroup_summary, checkpoint_summary


def _markdown_report(selection, comparison, delta_summary, subgroup_summary, checkpoint_summary):
    def table(frame, columns, digits=4):
        selected = frame[columns].copy()
        for column in selected.select_dtypes(include=[np.number]).columns:
            selected[column] = selected[column].map(lambda value: f"{value:.{digits}f}")
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["---"] * len(columns)) + "|"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in selected.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    comparison_columns = [
        "temporal_depth", "old_train_auroc_mean", "old_real_test_auroc_mean",
        "new_fold_train_auroc_mean", "new_oof_auroc_mean", "new_real_test_auroc_mean",
        "new_generated_test_auroc_mean",
    ]
    gap_columns = [
        "temporal_depth", "old_train_minus_test_gap", "new_train_minus_oof_gap",
        "new_train_minus_test_gap", "old_real_test_auroc_std", "new_real_test_auroc_std",
    ]
    delta_columns = [
        "temporal_depth", "auroc_delta_mean", "auroc_delta_std",
        "prauc_delta_mean", "prauc_delta_std",
    ]
    real_subgroups = subgroup_summary[
        subgroup_summary["input_source"] == "real"
    ][
        [
            "grouping", "group", "temporal_depth", "patients", "pcr_positive",
            "auroc_mean", "auroc_std", "prauc_mean",
        ]
    ]
    checkpoint = checkpoint_summary.iloc[0]
    return f"""# Full978 Shared-Prefix Anti-Overfit Audit

## Contract

- Development pool: 876 patients in five mutually exclusive OOF folds.
- Locked test: exactly the 102-patient BiFlow validation set.
- Variant selection used OOF predictions only; selected variant: `{selection['selected_variant']}`.
- Formal evaluation: ten seeds, five fold models per seed, patient probabilities averaged before metrics.
- Generated evaluation changes only the 296 available future T1-T3 embeddings; T0 predictions are exactly unchanged.
- This is a retrospective method-improvement result because the locked test had already been inspected.

## Main Results

{table(comparison, comparison_columns)}

## Generalization Gap And Stability

{table(comparison, gap_columns)}

The 50 formal checkpoints selected epochs {int(checkpoint['best_epoch_min'])} to
{int(checkpoint['best_epoch_max'])} (mean {checkpoint['best_epoch_mean']:.2f}).
The learned residual scale at the selected epoch averaged
{checkpoint['residual_scale_at_best_mean']:.4f} and never exceeded
{checkpoint['residual_scale_at_best_max']:.4f}.

## Generated Minus Real

{table(delta_summary, delta_columns)}

## Real-Test Subgroups

{table(real_subgroups, list(real_subgroups.columns))}

## Interpretation

Shared-prefix training plus five-fold ensembling restores a monotonic aggregate
AUROC trend without removing any locked-test patient. The long-sequence
train-to-test gap and seed variance are materially smaller than for the four
independently trained models. The generated future input remains weaker than
real imaging at T1 and T2, while most of the gap closes by T3. Subgroup estimates
are exploratory and noisy because each subgroup is substantially smaller than
the full 102-patient test cohort.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/mewm_ispy2_full978_locked102_anti_overfit.yaml"
    )
    parser.add_argument(
        "--old-config", default="configs/mewm_ispy2_full978_locked102_table2.yaml"
    )
    parser.add_argument("--result-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    old_config_path = _repo_path(args.old_config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    with old_config_path.open() as handle:
        old_config = yaml.safe_load(handle)
    result_dir = _repo_path(args.result_dir or config["anti_overfit"]["output_dir"])
    output_dir = _repo_path(args.output_dir) if args.output_dir else result_dir / "analysis"
    old_result_dir = _repo_path(old_config["experiment"]["output_dir"])
    cohort = _repo_path(config["data"]["cohort_dir"])
    metadata = pd.read_csv(
        _repo_path(config["data"]["metadata_csv"]), dtype={"pid": str}
    ).set_index("pid")
    patient_manifest = pd.read_csv(
        cohort / "patient_manifest.csv", dtype={"patient_id": str}
    )
    specs, pool_ids, test_ids, fold_audit = _fold_specs(config)
    if set(pool_ids) & set(test_ids) or len(pool_ids) != 876 or len(test_ids) != 102:
        raise ValueError("development/test split contract failed")
    selection = json.loads((result_dir / "selected_variant.json").read_text())
    if selection.get("test_data_used") is not False:
        raise ValueError("variant selection used test data")
    tune_checkpoints = list((result_dir / "train" / "tune").glob("**/best.pt"))
    if len(tune_checkpoints) != 30:
        raise ValueError(f"expected 30 tune checkpoints, found {len(tune_checkpoints)}")

    (
        variant,
        formal_checkpoint_count,
        train_metrics,
        oof_metrics,
        checkpoint_stats,
    ) = _audit_formal(config, result_dir, specs, pool_ids, metadata)
    if formal_checkpoint_count != 50:
        raise ValueError(f"expected 50 formal checkpoints, found {formal_checkpoint_count}")
    test_metrics, predictions = _audit_evaluations(
        config, result_dir, test_ids, metadata
    )
    (
        comparison,
        deltas,
        delta_summary,
        subgroup_per_seed,
        subgroup_summary,
        checkpoint_summary,
    ) = _summary_tables(
        config,
        old_config,
        old_result_dir,
        train_metrics,
        oof_metrics,
        test_metrics,
        checkpoint_stats,
        predictions,
        metadata,
        patient_manifest,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "old_vs_new.csv", comparison)
    _atomic_csv(output_dir / "formal_train_metrics_per_fold.csv", train_metrics)
    _atomic_csv(output_dir / "formal_oof_metrics_recomputed.csv", oof_metrics)
    _atomic_csv(output_dir / "test_metrics_recomputed.csv", test_metrics)
    _atomic_csv(output_dir / "generated_minus_real_per_seed.csv", deltas)
    _atomic_csv(output_dir / "generated_minus_real_summary.csv", delta_summary)
    _atomic_csv(output_dir / "subgroup_metrics_per_seed.csv", subgroup_per_seed)
    _atomic_csv(output_dir / "subgroup_summary.csv", subgroup_summary)
    _atomic_csv(output_dir / "checkpoint_summary.csv", checkpoint_summary)
    _atomic_csv(output_dir / "checkpoint_selection.csv", checkpoint_stats)
    _atomic_csv(output_dir / "fold_assignments.csv", fold_audit)
    _atomic_text(
        output_dir / "anti_overfit_audit.md",
        _markdown_report(
            selection, comparison, delta_summary, subgroup_summary, checkpoint_summary
        ),
    )
    _atomic_json(
        output_dir / "ANALYSIS_COMPLETE.json",
        {
            "schema": "pillar_full978_shared_prefix_anti_overfit_analysis_v1",
            "selected_variant": variant,
            "tune_checkpoints": len(tune_checkpoints),
            "formal_checkpoints": formal_checkpoint_count,
            "formal_seeds": len(config["anti_overfit"]["formal_seeds"]),
            "folds": len(specs),
            "development_patients": len(pool_ids),
            "locked_test_patients": len(test_ids),
            "evaluation_sources": list(config["evaluation_sources"]),
            "test_used_for_selection": False,
            "complete": True,
        },
    )
    print(comparison.to_string(index=False))
    print(f"analysis written to {output_dir}")


if __name__ == "__main__":
    main()

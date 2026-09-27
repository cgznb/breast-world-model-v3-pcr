"""Paired real/generated evaluation with a separate five-fold mean for every seed."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src import first_post_optimization as opt
from src import registered_three_phase_fixed_pcr as fixed
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import identity, now, read_json, write_json
from src.registered_three_phase_generated_pcr import mean_auc_kernel
from src.registered_three_phase_optimization_report import holdout_splits, require_frozen


def mc4_within_model(raw):
    keys = ["model", "depth", "seed", "outer", "patient_id", "label"]
    expected = {"real", *[f"{m}_draw_{d}" for m in ("direct", "rollout", "previous_real") for d in range(4)]}
    if raw.duplicated(keys + ["source"]).any() or set(raw.source) != expected:
        raise ValueError("Missing or duplicate raw source/draw predictions")
    groups = raw.groupby(keys).source.agg(list)
    if any(set(g) != expected or len(g) != len(expected) for g in groups):
        raise ValueError("Every fold/patient must have real input and all12generated draws")
    result = [raw[raw.source == "real"].copy()]
    for mode in ("direct", "rollout", "previous_real"):
        part = raw[raw.source.str.startswith(mode + "_draw_")]
        mean = part.groupby(keys, as_index=False).probability.mean()
        mean["source"] = mode + "_mc4"
        result.append(mean)
    return pd.concat(result, ignore_index=True)


def fold_metrics(predictions, ids):
    rows = []
    for (model, depth, seed, outer, source), frame in predictions.groupby(["model", "depth", "seed", "outer", "source"]):
        if len(frame) != len(ids) or set(frame.patient_id) != set(ids) or not frame.patient_id.is_unique:
            raise ValueError("Paired source/model patient populations differ")
        rows.append({"model": model, "depth": int(depth), "seed": int(seed), "outer": int(outer),
                     "source": source, "patients": len(ids), "positives": int(frame.label.sum()),
                     **opt.metric_values(frame.label.to_numpy(), frame.probability.to_numpy())})
    return pd.DataFrame(rows)


def fivefold_summary(frame, folds, seeds, depths, sources):
    keys = ["model", "depth", "seed", "source", "outer"]
    expected = {(m, d, s, source, f) for m in fixed.ARMS for d in depths for s in seeds for source in sources for f in folds}
    actual = set(frame[keys].itertuples(index=False, name=None))
    if len(frame) != len(expected) or actual != expected:
        raise ValueError("Every separate seed/window/source requires exactly the same five folds")
    metrics = [k for k in ("auroc", "logloss", "brier", "train_auroc", "train_logloss", "train_brier", "auroc_gap") if k in frame]
    rows = []
    for (model, depth, seed, source), part in frame.groupby(keys[:-1]):
        row = {"model": model, "depth": int(depth), "window": opt.DEPTH_NAMES[depth],
               "seed": int(seed), "source": source, "folds": len(folds)}
        for metric in metrics:
            row[metric + "_mean"] = float(part[metric].mean())
            row[metric + "_fold_sd"] = float(part[metric].std(ddof=1))
        for r in part.itertuples():
            row[f"fold_{r.outer + 1}_auroc"] = r.auroc
        rows.append(row)
    return pd.DataFrame(rows)


def development(cfg, refs):
    output = Path(cfg["output_dir"])
    rows, predictions = [], []
    for ref in refs:
        saved = pd.read_csv(Path(ref["path"]) / "predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
        row = {"model": ref["arm"], "depth": ref["depth"], "seed": ref["seed"],
               "outer": ref["outer"], "source": "real"}
        for role in ("train", "validation"):
            part = saved[saved.role == role].copy()
            metrics = opt.metric_values(part.label.to_numpy(), part.probability.to_numpy())
            row.update({("train_" if role == "train" else "") + k: v for k, v in metrics.items()})
            if role == "validation":
                part["model"], part["depth"], part["seed"], part["outer"] = ref["arm"], ref["depth"], ref["seed"], ref["outer"]
                predictions.append(part)
        row["auroc_gap"] = row["train_auroc"] - row["auroc"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    folds = [f["fold"] for f in read_json(output / "nested_folds.json")]
    summary = fivefold_summary(frame, folds, cfg["formal_seeds"], cfg["depths"], ["real"])
    oof = pd.concat(predictions, ignore_index=True)
    ids = read_json(output / "cohort.json")["split"]["train"]
    pooled = []
    for (model, depth, seed), part in oof.groupby(["model", "depth", "seed"]):
        if len(part) != len(ids) or set(part.patient_id) != set(ids) or not part.patient_id.is_unique:
            raise ValueError("Development OOF does not cover each patient exactly once")
        pooled.append({"model": model, "depth": depth, "seed": seed,
                       **opt.metric_values(part.label.to_numpy(), part.probability.to_numpy())})
    root = output / "evaluation/development"
    _atomic_csv(root / "fold_metrics.csv", frame)
    _atomic_csv(root / "seed_fivefold_summary.csv", summary)
    _atomic_csv(root / "oof_predictions.csv", oof)
    _atomic_csv(root / "pooled_oof_metrics.csv", pd.DataFrame(pooled))
    return summary


def paired_intervals(cfg, predictions):
    ids = sorted(predictions.patient_id.unique())
    labels = predictions.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
    if (predictions.groupby("patient_id").label.nunique() != 1).any():
        raise ValueError("Paired outcome labels differ")
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    wp = rng.multinomial(positive, np.full(positive, 1 / positive), size=cfg["bootstrap_samples"])
    wn = rng.multinomial(negative, np.full(negative, 1 / negative), size=cfg["bootstrap_samples"])
    comparisons = [(("primary", source), ("reference", source)) for source in fixed.SOURCES]
    comparisons += [(("primary", source), ("primary", "real")) for source in fixed.SOURCES[1:]]
    rows = []
    for (depth, seed), part in predictions.groupby(["depth", "seed"]):
        points, samples = {}, {}
        for (model, source), frame in part.groupby(["model", "source"]):
            matrix = frame.pivot(index="outer", columns="patient_id", values="probability")[ids].to_numpy()
            if matrix.shape != (5, len(ids)) or not np.isfinite(matrix).all():
                raise ValueError("Five paired models are required for each individual seed")
            kernel = mean_auc_kernel(labels, matrix)
            point = float(np.mean([roc_auc_score(labels, p) for p in matrix]))
            if abs(point - kernel.mean()) > 1e-12:
                raise ValueError("Fold-mean AUROC kernel mismatch")
            points[(model, source)] = point
            samples[(model, source)] = ((wp @ kernel) * wn).sum(axis=1) / (positive * negative)
        for left, right in comparisons:
            low, high = np.quantile(samples[left] - samples[right], [.025, .975])
            rows.append({"depth": int(depth), "seed": int(seed), "left_model": left[0], "left_source": left[1],
                         "right_model": right[0], "right_source": right[1],
                         "mean_fold_auc_difference": points[left] - points[right],
                         "ci95_low": float(low), "ci95_high": float(high)})
    return pd.DataFrame(rows)


def evaluate_holdout(cfg, refs):
    output = Path(cfg["output_dir"])
    splits = {k: v for k, v in holdout_splits(cfg).items() if k != "copy_T0"}
    by_path = {}
    for ref in refs:
        by_path.setdefault(ref["path"], []).append(ref)
    rows = []
    for index, (path, usages) in enumerate(by_path.items(), 1):
        saved = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        task = saved["task"]
        if set(splits["real"]["pids"]) & (set(task["train_ids"]) | set(task["val_ids"])):
            raise ValueError("Historical evaluation patients entered classifier fitting")
        model = shared.load_model(saved)
        for source, split in splits.items():
            probability, _, _ = shared.predict_checkpoint(saved, split, model=model)
            for ref in usages:
                rows.append(pd.DataFrame({"model": ref["arm"], "depth": ref["depth"], "seed": ref["seed"],
                    "outer": ref["outer"], "source": source, "patient_id": split["pids"],
                    "label": split["labels"].astype(int), "probability": probability.astype(np.float64)}))
        if index % 50 == 0:
            shared.progress(cfg, "paired_holdout_inference", completed=index, total=len(by_path))
    raw = pd.concat(rows, ignore_index=True)
    predictions = mc4_within_model(raw)
    metrics = fold_metrics(predictions, splits["real"]["pids"])
    folds = [f["fold"] for f in read_json(output / "nested_folds.json")]
    summary = fivefold_summary(metrics, folds, cfg["formal_seeds"], cfg["depths"], fixed.SOURCES)
    keys = ["model", "depth", "seed", "source", "patient_id", "label"]
    ensemble = predictions.groupby(keys, as_index=False).probability.mean()
    ensemble_metrics = []
    for (model, depth, seed, source), frame in ensemble.groupby(keys[:4]):
        ensemble_metrics.append({"model": model, "depth": depth, "seed": seed, "source": source,
                                 **opt.metric_values(frame.label.to_numpy(), frame.probability.to_numpy())})
    root = output / "evaluation/holdout"
    _atomic_csv(root / "draw_fold_predictions.csv", raw)
    _atomic_csv(root / "mc4_fold_predictions.csv", predictions)
    _atomic_csv(root / "fold_metrics.csv", metrics)
    _atomic_csv(root / "seed_fivefold_summary.csv", summary)
    _atomic_csv(root / "probability_ensemble_predictions.csv", ensemble)
    _atomic_csv(root / "probability_ensemble_metrics.csv", pd.DataFrame(ensemble_metrics))
    _atomic_csv(root / "paired_seed_intervals.csv", paired_intervals(cfg, predictions))
    return summary


def verify_outputs(cfg):
    output = Path(cfg["output_dir"])
    root = output / "evaluation/holdout"
    raw = pd.read_csv(root / "draw_fold_predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
    saved = pd.read_csv(root / "mc4_fold_predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
    replay = mc4_within_model(raw)
    keys = ["model", "depth", "seed", "outer", "source", "patient_id", "label"]
    merged = replay.merge(saved, on=keys, suffixes=("_replay", "_saved"), validate="one_to_one")
    if len(merged) != len(saved):
        raise ValueError("Stored within-model MC4 aggregation is incomplete")
    maximum = float(abs(merged.probability_replay - merged.probability_saved).max())
    if maximum > 1e-12:
        raise ValueError("Stored within-model MC4 probabilities differ")
    ids = read_json(output / "cohort.json")["split"]["val"]
    rescored = fold_metrics(saved, ids)
    original = pd.read_csv(root / "fold_metrics.csv", float_precision="round_trip")
    group = ["model", "depth", "seed", "outer", "source"]
    paired = rescored.merge(original, on=group, suffixes=("_replay", "_saved"), validate="one_to_one")
    metric_error = max(float(abs(paired[k + "_replay"] - paired[k + "_saved"]).max()) for k in ("auroc", "logloss", "brier"))
    summary_error = 0.0
    folds = [f["fold"] for f in read_json(output / "nested_folds.json")]
    for population, sources in (("development", ["real"]), ("holdout", fixed.SOURCES)):
        directory = output / "evaluation" / population
        fold_frame = pd.read_csv(directory / "fold_metrics.csv", float_precision="round_trip")
        recalculated = fivefold_summary(fold_frame, folds, cfg["formal_seeds"], cfg["depths"], sources)
        stored = pd.read_csv(directory / "seed_fivefold_summary.csv", float_precision="round_trip")
        for col in recalculated.select_dtypes(include="number"):
            summary_error = max(summary_error, float(abs(recalculated[col] - stored[col]).max()))
    shared_error = 0.0
    for model in fixed.ARMS:
        t0 = saved[(saved.model == model) & (saved.depth == 1)]
        matrix = t0.pivot(index=["seed", "outer", "patient_id"], columns="source", values="probability")
        shared_error = max(shared_error, float((matrix.max(axis=1) - matrix.min(axis=1)).max()))
        t1 = saved[(saved.model == model) & (saved.depth == 2) & (saved.source != "real")]
        matrix = t1.pivot(index=["seed", "outer", "patient_id"], columns="source", values="probability")
        shared_error = max(shared_error, float((matrix.max(axis=1) - matrix.min(axis=1)).max()))
    if metric_error > 1e-10 or summary_error > 1e-10 or shared_error > 1e-7:
        raise ValueError("Five-fold metric replay or shared input equality failed")
    result = {"passed": True, "checked_utc": now(), "raw_prediction_rows": len(raw),
              "mc4_fold_prediction_rows": len(saved), "holdout_fold_metric_rows": len(original),
              "mc4_replay_max_error": maximum, "metric_replay_max_error": metric_error,
              "fivefold_summary_max_error": summary_error, "shared_T0_T1_max_error": shared_error,
              "fold_mean_computed_after_scoring": True, "seeds_reported_separately": True}
    write_json(output / "evaluation/verification.json", result)
    return result


def markdown_tables(frame, seeds, sources, title):
    lines = [f"# {title}", "", "Every row is one classifier seed. Cells contain mean AUROC +/- sample SD across five folds.",
             "No averaging across seeds. Fold SD is not a patient confidence interval.", ""]
    for depth, window in opt.DEPTH_NAMES.items():
        lines.extend([f"## {window}", "", "| Seed | " + " | ".join(sources) + " |",
                      "|---:|" + "---:|" * len(sources)])
        for seed in seeds:
            part = frame[(frame.depth == depth) & (frame.seed == seed)].set_index("source")
            cells = [f"{part.loc[s, 'auroc_mean']:.5f} +/- {part.loc[s, 'auroc_fold_sd']:.5f}" for s in sources]
            lines.append("| " + str(seed) + " | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_reports(cfg, dev, held):
    output = Path(cfg["output_dir"])
    for arm in fixed.ARMS:
        _atomic_text(output / "evaluation" / f"{arm}_holdout_per_seed.md", markdown_tables(
            held[held.model == arm], cfg["formal_seeds"], fixed.SOURCES,
            f"{arm}: matched102-patient real/generated inputs"))
        _atomic_text(output / "evaluation" / f"{arm}_development_per_seed.md", markdown_tables(
            dev[dev.model == arm], cfg["formal_seeds"], ["real"],
            f"{arm}:764-patient real-input development cross-validation"))
    recipes = []
    for arm in fixed.ARMS:
        for depth in cfg["depths"]:
            effective = fixed.effective_config(cfg, arm, depth)
            recipes.append({"arm": arm, "window": opt.DEPTH_NAMES[depth], **effective})
    _atomic_csv(output / "evaluation/uniform_recipes.csv", pd.DataFrame(recipes))
    primary = held[held.model == "primary"]
    reference = held[held.model == "reference"]
    keys = ["depth", "window", "seed", "source"]
    comparison = primary.merge(reference, on=keys, suffixes=("_primary", "_reference"), validate="one_to_one")
    comparison["auroc_difference"] = comparison.auroc_mean_primary - comparison.auroc_mean_reference
    _atomic_csv(output / "evaluation/fixed_recipe_comparison.csv", comparison)
    lines = ["# Uniform Five-Fold Three-Phase ROI32 pCR Results", "",
        "The network, loss, optimizer, learning-rate schedule and40-epoch duration are identical across all five folds and ten seeds of each window/arm. No per-fold candidate or checkpoint selection is used.",
        "All400 arm/fold/seed references are evaluated;350unique models are trained from initialization because primary/reference T0 are identical.", "",
        "## Fixed Recipes", "",
        "Shared frozen Pillar1152-D features, physical ROI adapter, contiguous visit availability, clinical prior,1-layer/4-headTDN. Original764/102patients and patient folds retained. Batch64, AdamW, LR0.0003,40-epoch cosine to0, exactly40epochs for every fit. No PCA or encoder/generator training.",
        "Primary T0: original32/16head, no clinical token, bounded residual1.0, dropout0.25, no residual L2.",
        "Primary T0-T1/T0-T2:32/16head with clinical token, dropout0.35, embedding dropout0.15, bounded residual1.5(initial0.25), residual L2=0.05, projection/other decay0.01/0.003, ordinary BCE.",
        "Primary T0-T3: retain64/32head and clinical token; dropout0.30, embedding dropout0.15, residual L2=0.05, projection/other decay0.005/0.001, ordinary BCE. Its residual remains unbounded with the original learned scale.",
        "Reference: original network/loss/regularization recipe for each window under the same40-epoch schedule. Its balanced loss rule, where applicable, estimates class weight only from each training fold. Primary uses positive weight1.0 throughout.", "",
        "## What Each Result Means", "",
        "Primary historical comparison: evaluate every fold-trained classifier on the SAME102patients under real, direct-T0, rollout and real-previous inputs. For each classifier seed, score five separate models and average their five AUROCs. These five models are not five disjoint test partitions of the102patients.",
        "For generated inputs, each fold model first averages probabilities from four complete generated sequences(MC4), then AUROC is calculated. Probabilities are never averaged across folds before the main metric. Alternative probability-ensemble scores are explicitly saved separately.",
        "Development: score the five disjoint original outer-validation folds of the764real-image patients, then average five AUROCs within each seed. Pooled OOF AUROC is saved separately. Do not compare764-patient real CV against102-patient generated evaluation as if they share a population.",
        "Direct: realT0 independently generates each future visit. Rollout: realT0 starts a per-draw generated-latent chain. Real-previous: each future visit is generated using its actual observed predecessor. All classifier inputs retain realT0 and replace all future slots with generated features. Real-previous has additional observed follow-up information.", "",
        "## Tables", "",
        "- evaluation/primary_holdout_per_seed.md: main four-source comparison, ten separate seeds for each window.",
        "- evaluation/reference_holdout_per_seed.md: identical evaluation for the fixed reference.",
        "- evaluation/primary_development_per_seed.md: real-input development CV.",
        "- evaluation/holdout/seed_fivefold_summary.csv: mean, fold SD and each of the five AUROCs, plus logloss/Brier.",
        "- evaluation/holdout/paired_seed_intervals.csv: paired patient bootstrap differences for each separate seed.",
        "- evaluation/holdout/probability_ensemble_metrics.csv: alternative five-model probability ensemble, not the main fold-mean result.", "",
        "## Scientific Limits", "",
        "The prior nested study evaluated a model-selection procedure. This study evaluates fixed recipes, as requested; they are different estimands. The reference is refitted at the same budget and is not the old validation-selected checkpoint collection.",
        "Recipes are informed by previous development experiments. The data and102historical patients have been reused, including generator validation. This is an exploratory fixed-recipe internal comparison, not a fresh independent validation or proof that overfitting was solved.",
        "No current-run outcome selects a recipe or seed. Both primary and reference results are retained, including regressions. No mean across ten seeds is used in the main tables.",
        "Paired2000 stratified patient bootstraps resample the same patients jointly for all five models. They are conditional on fitted models, omit refitting/selection uncertainty and have no multiple-comparison adjustment. Fold SD is not a confidence interval.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True, constrained_layout=True)
    colors = plt.colormaps["tab10"]
    for depth, axis in enumerate(axes, 1):
        for i, seed in enumerate(cfg["formal_seeds"]):
            values = primary[(primary.depth == depth) & (primary.seed == seed)].set_index("source").loc[list(fixed.SOURCES)]
            axis.plot(range(4), values.auroc_mean, marker="o", linewidth=1, markersize=3,
                      color=colors(i), alpha=.85, label=str(seed))
        axis.set_title(opt.DEPTH_NAMES[depth], fontsize=11)
        axis.set_xticks(range(4), ["Real", "Direct T0", "Rollout", "Real previous"], rotation=30, ha="right")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean of five model AUROCs (same102patients)")
    axes[0].legend(title="Seed", fontsize=7, ncol=2, frameon=False)
    fig.savefig(output / "evaluation/separate_seed_comparison.png", dpi=180)
    fig.savefig(output / "evaluation/separate_seed_comparison.pdf")
    plt.close(fig)


def evaluate_and_report(cfg):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    require_frozen(cfg)
    output = Path(cfg["output_dir"])
    inventory = dict(read_json(Path(cfg["previous_run"]) / "evaluation_input_inventory.json"))
    shared.verify_inventory(inventory)
    inventory[str(Path(__file__))] = identity(__file__)
    shared.frozen_json(output / "evaluation_input_inventory.json", inventory)
    refs = read_json(output / "model_references.json")
    dev = development(cfg, refs)
    held = evaluate_holdout(cfg, refs)
    verification = verify_outputs(cfg)
    write_reports(cfg, dev, held)
    shared.verify_inventory(inventory)
    require_frozen(cfg)
    return verification

"""Evaluate frozen ROI32 classifiers on nested OOF and cached historical inputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split
from scripts.run_full978_independent_cv import _canonical_split
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as study
from src.data import EmbStore
from src.first_post_pcr_data import identity, now, read_json, write_json
from src.registered_three_phase_generated_pcr import mean_auc_kernel, replaced_split

MAIN_SOURCES = ["real", "copy_T0", "direct_mc4", "rollout_mc4", "previous_real_mc4"]
MODEL_NAMES = {"baseline": "Matched nested baseline", "selected": "Inner-selected model",
               "historical_original": "Original fitted model", "clinical_prior": "Clinical prior"}


def require_frozen(cfg):
    output = Path(cfg["output_dir"])
    if not (output / "FROZEN_MODELS.json").is_file() or not (output / "TRAINING_COMPLETE.json").is_file():
        raise ValueError("Weights and training verification must be frozen before holdout evaluation")
    frozen = read_json(output / "FROZEN_MODELS.json")
    if frozen["holdout_used_for_selection"]:
        raise ValueError("Holdout entered model selection")
    study.verify_inventory(frozen["artifacts"])
    return frozen


def references(cfg):
    return read_json(Path(cfg["output_dir"]) / "selection" / f"{study.STUDY}_outer_models.json")


def summarize_predictions(predictions, expected_ids, expected_seeds):
    rows = []
    for (model, source, depth, seed), frame in predictions.groupby(["model", "source", "depth", "seed"]):
        if (len(frame) != len(expected_ids) or set(frame.patient_id) != set(expected_ids)
                or not frame.patient_id.is_unique):
            raise ValueError("Patient coverage changed in a reported metric")
        rows.append({"model": model, "source": source, "depth": int(depth), "seed": int(seed),
                     "patients": len(frame), "positive": int(frame.label.sum()),
                     **opt.metric_values(frame.label.to_numpy(), frame.probability.to_numpy())})
    metrics = pd.DataFrame(rows)
    for _, frame in metrics.groupby(["model", "source", "depth"]):
        if set(frame.seed) != set(expected_seeds) or len(frame) != len(expected_seeds):
            raise ValueError("Missing or repeated classifier seed")
    summary = metrics.groupby(["model", "source", "depth"])[["auroc", "logloss", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return metrics, summary.reset_index()


def development_report(cfg):
    output = Path(cfg["output_dir"])
    predictions, fits = [], []
    for ref in references(cfg):
        path = Path(ref["path"])
        record = read_json(path / "COMPLETE.json")
        saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        val = saved[saved.role == "validation"].copy()
        val["model"], val["source"], val["depth"], val["seed"], val["outer"] = (
            ref["arm"], "real", ref["depth"], ref["seed"], ref["outer"])
        predictions.append(val)
        fits.append({"model": ref["arm"], "depth": ref["depth"], "seed": ref["seed"],
                     "outer": ref["outer"], "candidate": record["task"]["candidate"],
                     "epochs": record["selected_epochs"], "parameters": record["parameters"],
                     "gap": record["auroc_gap"],
                     **{f"train_{k}": v for k, v in record["train"].items()},
                     **{f"validation_{k}": v for k, v in record["validation"].items()}})
        if ref["arm"] == "baseline":
            clinical = val.copy()
            clinical["model"] = "clinical_prior"
            clinical["probability"] = clinical.clinical_prior_probability
            predictions.append(clinical)
    frame, per_fit = pd.concat(predictions, ignore_index=True), pd.DataFrame(fits)
    folds = read_json(output / "nested_folds.json")
    ids = folds[0]["train_ids"] + folds[0]["val_ids"]
    metrics, summary = summarize_predictions(frame, ids, cfg["formal_seeds"])
    destination = output / "evaluation/development"
    _atomic_csv(destination / "predictions.csv", frame)
    _atomic_csv(destination / "fit_metrics.csv", per_fit)
    _atomic_csv(destination / "seed_metrics.csv", metrics)
    _atomic_csv(destination / "summary.csv", summary)
    comparisons = paired_comparisons(cfg, frame, [("selected", "baseline")])
    _atomic_csv(destination / "paired_differences.csv", comparisons)
    return frame, per_fit, summary, comparisons


def holdout_splits(cfg):
    require_frozen(cfg)
    baseline, direct, sequential = (Path(cfg[k]) for k in ("baseline_run", "direct_run", "sequential_run"))
    cohort = read_json(baseline / "cohort.json")
    ids = cohort["split"]["val"]
    for root in (direct, sequential):
        if not read_json(root / "evaluation/independent_audit.json")["passed"]:
            raise ValueError("Generated inputs require a completed independent audit")
    routes = read_json(direct / "routes.json")
    if isinstance(routes, dict):
        routes = routes["routes"]
    real = _canonical_split(_load_split(baseline / "embeddings/real", baseline / "holdout_metadata.csv", ids), 4)
    splits = {"real": real, "copy_T0": replaced_split(real, routes, "copy_T0")}
    for mode in ("direct", "rollout", "previous_real"):
        root = direct / "embeddings" if mode == "direct" else sequential / mode / "embeddings"
        for draw in range(4):
            splits[f"{mode}_draw_{draw}"] = replaced_split(real, routes, mode, EmbStore(str(root / f"draw_{draw}")))
    return splits


def aggregate_holdout(fold_predictions, fold_ids):
    keys = ["model", "source", "depth", "seed", "patient_id", "label"]
    if fold_predictions.duplicated(keys + ["outer"]).any():
        raise ValueError("Duplicate fold predictions")
    counts = fold_predictions.groupby(keys).outer.agg(list)
    if any(set(x) != set(fold_ids) or len(x) != len(fold_ids) for x in counts):
        raise ValueError("Every holdout prediction requires every outer fold")
    result = fold_predictions.groupby(keys, as_index=False).probability.mean()
    ensembles = []
    for mode in ("direct", "rollout", "previous_real"):
        draws = result[result.source.str.startswith(mode + "_draw_")]
        base_keys = [k for k in keys if k != "source"]
        counts = draws.groupby(base_keys).source.agg(list)
        expected = {f"{mode}_draw_{d}" for d in range(4)}
        if not len(counts) or any(set(x) != expected or len(x) != 4 for x in counts):
            raise ValueError("MC4 requires exactly four complete-sequence probability draws")
        frame = draws.groupby(base_keys, as_index=False).probability.mean()
        frame["source"] = mode + "_mc4"
        ensembles.append(frame)
    return pd.concat([result, *ensembles], ignore_index=True)


def evaluate_holdout(cfg):
    output = Path(cfg["output_dir"])
    inputs = holdout_splits(cfg)
    refs = references(cfg)
    by_path = {}
    for ref in refs:
        by_path.setdefault(ref["path"], []).append(ref)
    rows, completed = [], 0
    for path, usages in by_path.items():
        saved = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        task = saved["task"]
        if set(inputs["real"]["pids"]) & (set(task["train_ids"]) | set(task["val_ids"])):
            raise ValueError("Historical holdout entered fitting")
        model = study.load_model(saved)
        predictions = {}
        for source, split in inputs.items():
            probability, _, prior = study.predict_checkpoint(saved, split, model=model)
            predictions[source] = probability
        for ref in usages:
            for source, probability in predictions.items():
                rows.append(pd.DataFrame({"model": ref["arm"], "source": source,
                    "depth": ref["depth"], "seed": ref["seed"], "outer": ref["outer"],
                    "patient_id": split["pids"], "label": split["labels"].astype(int),
                    "probability": probability.astype(np.float64),
                    "clinical_probability": (1 / (1 + np.exp(-prior))).astype(np.float64)}))
        completed += 1
        if completed % 25 == 0:
            study.progress(cfg, "frozen_holdout_inference", completed=completed, total=len(by_path))
    folds = pd.concat(rows, ignore_index=True)
    fold_ids = [f["fold"] for f in read_json(output / "nested_folds.json")]
    predictions = aggregate_holdout(folds, fold_ids)
    clinical = folds[(folds.model == "baseline") & (folds.source == "real")].copy()
    clinical["probability"] = clinical.clinical_probability
    clinical["model"] = "clinical_prior"
    keys = ["model", "source", "depth", "seed", "patient_id", "label"]
    clinical = clinical.groupby(keys, as_index=False).probability.mean()
    old = pd.read_csv(Path(cfg["sequential_run"]) / "evaluation/predictions.csv", dtype={"patient_id": str})
    old = old[old.source.isin(MAIN_SOURCES)].copy()
    old["depth"] = old.temporal_depth.map({v: k for k, v in opt.DEPTH_NAMES.items()})
    old["model"] = "historical_original"
    old = old[keys + ["probability"]]
    predictions = pd.concat([predictions, clinical, old], ignore_index=True)
    selected = predictions[predictions.source.isin(MAIN_SOURCES)].copy()
    ids = inputs["real"]["pids"]
    metrics, summary = summarize_predictions(selected, ids, cfg["formal_seeds"])
    destination = output / "evaluation/holdout"
    _atomic_csv(destination / "fold_predictions.csv", folds)
    _atomic_csv(destination / "all_draw_predictions.csv", predictions)
    _atomic_csv(destination / "predictions.csv", selected)
    _atomic_csv(destination / "seed_metrics.csv", metrics)
    _atomic_csv(destination / "summary.csv", summary)
    comparisons = paired_comparisons(cfg, selected, [("selected", "baseline"), ("selected", "historical_original")])
    _atomic_csv(destination / "paired_differences.csv", comparisons)
    return selected, summary, comparisons


def paired_comparisons(cfg, predictions, comparisons):
    rng, rows = np.random.default_rng(cfg["bootstrap_seed"]), []
    for (source, depth), frame in predictions.groupby(["source", "depth"]):
        ids = sorted(frame.patient_id.unique())
        if (frame.groupby("patient_id").label.nunique() != 1).any():
            raise ValueError("Paired outcome labels changed")
        y = frame.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
        npos, nneg = int(y.sum()), int((1 - y).sum())
        wp = rng.multinomial(npos, np.full(npos, 1 / npos), size=cfg["bootstrap_samples"])
        wn = rng.multinomial(nneg, np.full(nneg, 1 / nneg), size=cfg["bootstrap_samples"])
        points, samples = {}, {}
        for name in {s for pair in comparisons for s in pair}:
            part = frame[frame.model == name]
            if part.empty:
                continue
            matrix = part.pivot(index="seed", columns="patient_id", values="probability").reindex(
                index=cfg["formal_seeds"], columns=ids).to_numpy()
            if not np.isfinite(matrix).all():
                raise ValueError("Missing paired prediction")
            kernel = mean_auc_kernel(y, matrix)
            point = float(np.mean([roc_auc_score(y, p) for p in matrix]))
            if abs(point - kernel.mean()) > 1e-12:
                raise ValueError("Bootstrap AUROC kernel disagrees with sklearn")
            points[name], samples[name] = point, ((wp @ kernel) * wn).sum(axis=1) / (npos * nneg)
        for left, right in comparisons:
            if left not in points or right not in points:
                continue
            low, high = np.quantile(samples[left] - samples[right], [.025, .975])
            rows.append({"source": source, "depth": int(depth), "comparison": left + "_minus_" + right,
                         "auroc_difference": points[left] - points[right],
                         "ci95_low": float(low), "ci95_high": float(high),
                         "patients": len(ids), "bootstrap_samples": cfg["bootstrap_samples"]})
    return pd.DataFrame(rows)


def verify_outputs(cfg):
    output = Path(cfg["output_dir"])
    max_error, rows = 0.0, 0
    for split in ("development", "holdout"):
        destination = output / "evaluation" / split
        predictions = pd.read_csv(destination / "predictions.csv", dtype={"patient_id": str})
        ids = sorted(predictions.patient_id.unique())
        metrics, summary = summarize_predictions(predictions, ids, cfg["formal_seeds"])
        for name, expected in (("seed_metrics.csv", metrics), ("summary.csv", summary)):
            actual = pd.read_csv(destination / name)
            for column in expected.select_dtypes(include="number"):
                error = float(np.max(np.abs(actual[column] - expected[column])))
                max_error = max(max_error, error)
                if error > 1e-10:
                    raise ValueError("Prediction-to-metric replay failed")
            rows += len(expected)
    destination = output / "evaluation/holdout"
    folds = pd.read_csv(destination / "fold_predictions.csv", dtype={"patient_id": str})
    fold_ids = [f["fold"] for f in read_json(output / "nested_folds.json")]
    replay = aggregate_holdout(folds, fold_ids)
    actual = pd.read_csv(destination / "all_draw_predictions.csv", dtype={"patient_id": str})
    keys = ["model", "source", "depth", "seed", "patient_id", "label"]
    merged = replay.merge(actual, on=keys, validate="one_to_one", suffixes=("_replay", "_saved"))
    if len(merged) != len(replay):
        raise ValueError("Saved fold/MC4 aggregation is incomplete")
    aggregation_error = float(abs(merged.probability_replay - merged.probability_saved).max())
    if aggregation_error > 1e-12:
        raise ValueError("Fold/MC4 aggregation changed")
    equal_errors = []
    for model in ("baseline", "selected"):
        part = actual[(actual.model == model) & (actual.depth == 1)]
        matrix = part.pivot(index=["seed", "patient_id"], columns="source", values="probability")
        equal_errors.append(float((matrix.max(axis=1) - matrix.min(axis=1)).max()))
        part = actual[(actual.model == model) & (actual.depth == 2) &
                      actual.source.isin(["direct_mc4", "rollout_mc4", "previous_real_mc4"])]
        matrix = part.pivot(index=["seed", "patient_id"], columns="source", values="probability")
        equal_errors.append(float((matrix.max(axis=1) - matrix.min(axis=1)).max()))
    if max(equal_errors) > 1e-7:
        raise ValueError("Shared T0 or shared generated T1 differs across sources")
    audit = {"passed": True, "checked_utc": now(), "metric_rows_replayed": rows,
             "maximum_metric_error": max_error, "fold_and_mc4_replay_error": aggregation_error,
             "shared_input_prediction_error": max(equal_errors),
             "outer_fit_prediction_replay": read_json(output / "TRAINING_COMPLETE.json")["audit"],
             "bootstrap_note": "Paired stratified patient resampling of mean seed AUROC; conditional on fitted models; no multiplicity adjustment."}
    write_json(output / "evaluation/verification.json", audit)
    return audit


def plot_results(cfg, development, fit_metrics, holdout):
    output = Path(cfg["output_dir"])
    colors = {"baseline": "#c96d37", "selected": "#168273", "clinical_prior": "#767676",
              "historical_original": "#4764ab"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for axis, summary, title in ((axes[0], development, "Nested development OOF"),
                                  (axes[1], holdout[holdout.source == "real"], "Historical real-input holdout")):
        for model, part in summary.groupby("model"):
            part = part.sort_values("depth")
            axis.errorbar(part.depth, part.auroc_mean, yerr=part.auroc_std.fillna(0), marker="o",
                          capsize=3, color=colors[model], label=MODEL_NAMES[model])
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("AUROC (mean +/- seed SD)")
    gap = fit_metrics.groupby(["model", "depth", "seed"]).gap.mean().reset_index()
    for model, part in gap.groupby("model"):
        summary = part.groupby("depth").gap.agg(["mean", "std"])
        axes[2].errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), marker="o",
                         capsize=3, color=colors[model], label=MODEL_NAMES[model])
    axes[2].set_title("Nested train-validation AUROC gap", fontsize=11)
    axes[2].set_ylabel("Train AUROC - validation AUROC")
    for axis in axes:
        axis.set_xticks([1, 2, 3, 4], list(opt.DEPTH_NAMES.values()))
        axis.grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output / "evaluation/comparison.png", dpi=180)
    fig.savefig(output / "evaluation/comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True, constrained_layout=True)
    sources = ["real", "direct_mc4", "rollout_mc4", "previous_real_mc4"]
    for depth, axis in enumerate(axes, 1):
        for model in ("historical_original", "baseline", "selected"):
            part = holdout[(holdout.depth == depth) & (holdout.model == model)].set_index("source").loc[sources]
            axis.errorbar(range(4), part.auroc_mean, yerr=part.auroc_std.fillna(0),
                          color=colors[model], marker="o", capsize=3, label=MODEL_NAMES[model])
        axis.set_xticks(range(4), ["Real", "Direct T0", "Rollout", "Real previous"], rotation=30, ha="right")
        axis.set_title(opt.DEPTH_NAMES[depth], fontsize=11)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Historical holdout AUROC")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(output / "evaluation/generated_comparison.png", dpi=180)
    fig.savefig(output / "evaluation/generated_comparison.pdf")
    plt.close(fig)


def write_report(cfg, development, fits, holdout, dev_differences, held_differences):
    output = Path(cfg["output_dir"])
    lines = ["# Reduced three-phase ROI32 pCR optimization", "",
        "Frozen Pillar features from real three-phase 32x128x128 ROIs; original 764 development / 102 historical holdout patients.",
        "All four windows, original five patient folds and classifier seeds 42-51. No encoder, generator or crop retraining.", "",
        "Candidate selection: three inner folds and seeds 42/43 for each outer fold/window. AUROC within 0.005 of best, then minimum mean log loss. Training duration is the rounded-up median selected inner epoch.",
        "Outer validation never selects architecture, duration, weights, calibration or thresholds. All models are frozen and replayed before historical holdout evaluation.",
        "Candidates: original recipe; compact TDN with bounded residual, L2 and stronger dropout/decay; the same compact TDN with training-only whitened PCA32; clinical logistic reference.",
        "This is a combined repair experiment, not an isolated causal ablation of each regularizer.", "",
        "## Nested Development Results", "",
        "Matched baseline uses the original recipe with inner-selected duration, so its OOF scores differ from the original validation-selected checkpoints.", "",
        "| Window | Model | Mean OOF AUROC | Seed SD | Log loss | Brier | Train-validation gap |",
        "|---|---|---:|---:|---:|---:|---:|"]
    gaps = fits.groupby(["model", "depth"]).gap.mean()
    for depth in cfg["depths"]:
        for model in ("baseline", "selected", "clinical_prior"):
            row = development[(development.depth == depth) & (development.model == model)].iloc[0]
            gap = f"{gaps.loc[(model, depth)]:.5f}" if (model, depth) in gaps else "n/a"
            lines.append(f"| {opt.DEPTH_NAMES[depth]} | {MODEL_NAMES[model]} | {row.auroc_mean:.5f} | {row.auroc_std:.5f} | {row.logloss_mean:.5f} | {row.brier_mean:.5f} | {gap} |")
    lines.extend(["", "## Historical Holdout Results", "",
        "These 102 patients were previously used for generator validation and repeated analysis. Results are exploratory, not a fresh external test. No new holdout-based tuning was performed.", "",
        "| Window | Input | Original model AUROC | Matched baseline AUROC | Selected model AUROC | Selected - original |",
        "|---|---|---:|---:|---:|---:|"])
    for depth in cfg["depths"]:
        for source in MAIN_SOURCES:
            part = holdout[(holdout.depth == depth) & (holdout.source == source)].set_index("model")
            old, base, new = [part.loc[m, "auroc_mean"] for m in ("historical_original", "baseline", "selected")]
            lines.append(f"| {opt.DEPTH_NAMES[depth]} | {source} | {old:.5f} | {base:.5f} | {new:.5f} | {new-old:+.5f} |")
    lines.extend(["", "## Paired Uncertainty", "",
        "Intervals resample the same patients across all seeds/models, retaining the mean-of-seed-AUROCs estimand. They condition on fitted models, omit refitting/selection uncertainty, and are not multiplicity-adjusted.", "",
        "| Population | Window | Comparison | AUROC difference | Paired 95% interval |",
        "|---|---|---|---:|---|"])
    for population, frame in (("Nested OOF", dev_differences), ("Historical real holdout", held_differences[held_differences.source == "real"])):
        for row in frame.itertuples():
            lines.append(f"| {population} | {opt.DEPTH_NAMES[row.depth]} | {row.comparison} | {row.auroc_difference:+.5f} | [{row.ci95_low:+.5f}, {row.ci95_high:+.5f}] |")
    selections = read_json(output / "selection" / f"{study.STUDY}.json")["selections"]
    lines.extend(["", "## Inner Choices", "", "| Window | Outer fold | Candidate | Fixed epochs |", "|---|---:|---|---:|"])
    for row in selections:
        lines.append(f"| {opt.DEPTH_NAMES[row['depth']]} | {row['outer']} | {row['selected']['candidate']} | {row['selected']['fixed_epochs']} |")
    lines.extend(["", "## Interpretation Boundaries", "",
        "A smaller training gap is not itself a performance gain. Judge nested OOF discrimination together with log loss/Brier and paired uncertainty.",
        "Seed standard deviation describes optimization variability on the same patients, not independent cohort replication. No seed was selected or discarded.",
        "MC4 averages four complete-sequence probabilities after fivefold ensembling; images/features are not averaged before classification.",
        "Rollout starts at real T0 and carries each generated latent forward. Real-previous generation uses the observed predecessor and has additional follow-up information. Classifiers still receive real T0 plus generated future slots.",
        "All workflows retain the original observed-prefix availability and recorded intervals/treatment metadata. This is not a newly audited prospective baseline-only deployment.",
        "PCA and clinical priors are fitted separately on the appropriate training patients. Only permitted visits enter the feature transform.",
        "Per-seed results: evaluation/development/seed_metrics.csv and evaluation/holdout/seed_metrics.csv. Full original, matched and selected comparisons remain available.", ""])
    _atomic_text(output / "README.md", "\n".join(lines))


def evaluate_and_report(cfg):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    require_frozen(cfg)
    output = Path(cfg["output_dir"])
    # Bind only completed feature artifacts; never regenerate images or re-fit a classifier here.
    input_files = []
    for key in ("direct_run", "sequential_run"):
        root = Path(cfg[key])
        input_files.extend(root.glob("**/embeddings/**/*.pt"))
        input_files.extend(root / p for p in ("COMPLETE.json", "routes.json", "evaluation/independent_audit.json"))
    input_files.append(Path(cfg["sequential_run"]) / "evaluation/predictions.csv")
    input_files.extend([Path(__file__), Path(cfg["baseline_run"]) / "holdout_metadata.csv"])
    study.frozen_json(output / "evaluation_input_inventory.json", {str(p): identity(p) for p in sorted(set(input_files))})
    _, fits, dev_summary, dev_diff = development_report(cfg)
    _, held_summary, held_diff = evaluate_holdout(cfg)
    verification = verify_outputs(cfg)
    plot_results(cfg, dev_summary, fits, held_summary)
    write_report(cfg, dev_summary, fits, held_summary, dev_diff, held_diff)
    study.verify_inventory(read_json(output / "evaluation_input_inventory.json"))
    study.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
    return verification

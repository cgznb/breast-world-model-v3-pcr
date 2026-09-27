"""Seed-specific historical comparisons for the frozen 300/50 classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from scripts.report_registered_three_phase_versions import rank_auc
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src import first_post_optimization as opt
from src import registered_three_phase_budget as study
from src import registered_three_phase_fixed_report as previous
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import identity, now, read_json, repo_path, write_json
from src.registered_three_phase_generated_pcr import mean_auc_kernel
from src.registered_three_phase_optimization_report import holdout_splits, require_frozen

SOURCES = ("real", "direct_mc4", "rollout_mc4", "previous_real_mc4")
KEYS = ["model", "depth", "seed", "source"]
FOLDS = [f"F{i}_auroc" for i in range(1, 6)]
VERSIONS = {"v1_original": "V1_Old", "v1_300_50": "V1_New",
            "v4_latest": "V4_Old", "v4_300_50": "V4_New"}


def csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def summarize_folds(frame, cfg, sources):
    expected = {(arm, d, seed, source, fold) for arm, depths in cfg["windows"].items()
                for d in depths for seed in cfg["formal_seeds"] for source in sources for fold in range(5)}
    actual = set(frame[KEYS + ["outer"]].itertuples(index=False, name=None))
    if actual != expected or len(frame) != len(expected):
        raise ValueError("Missing or duplicate declared model/window/seed/source/fold")
    metrics = [k for k in ("auroc", "logloss", "brier", "train_auroc", "train_logloss",
                           "train_brier", "auroc_gap") if k in frame]
    rows = []
    for key, part in frame.groupby(KEYS):
        row = dict(zip(KEYS, key))
        row.update(window=opt.DEPTH_NAMES[key[1]], folds=5)
        for metric in metrics:
            row[metric + "_mean"] = float(part[metric].mean())
            row[metric + "_fold_sd"] = float(part[metric].std(ddof=1))
        row.update({f"F{r.outer + 1}_auroc": r.auroc for r in part.itertuples()})
        rows.append(row)
    return pd.DataFrame(rows)


def development(cfg, refs):
    rows, predictions = [], []
    for ref in refs:
        frame = csv(Path(ref["path"]) / "predictions.csv")
        row = {"model": ref["arm"], "depth": ref["depth"], "seed": ref["seed"],
               "source": "real", "outer": ref["outer"]}
        for role in ("train", "validation"):
            part = frame[frame.role == role].copy()
            metrics = opt.metric_values(part.label, part.probability)
            row.update({("train_" if role == "train" else "") + k: v for k, v in metrics.items()})
            if role == "validation":
                predictions.append(part.assign(model=ref["arm"], depth=ref["depth"],
                                               seed=ref["seed"], outer=ref["outer"]))
        row["auroc_gap"] = row["train_auroc"] - row["auroc"]
        rows.append(row)
    metrics = pd.DataFrame(rows)
    summary = summarize_folds(metrics, cfg, ["real"])
    oof = pd.concat(predictions, ignore_index=True)
    expected = set(read_json(Path(cfg["output_dir"]) / "cohort.json")["split"]["train"])
    pooled = []
    for (arm, depth, seed), part in oof.groupby(["model", "depth", "seed"]):
        if len(part) != len(expected) or set(part.patient_id) != expected or not part.patient_id.is_unique:
            raise ValueError("Each development patient needs exactly one selection-validation prediction")
        pooled.append({"model": arm, "depth": depth, "seed": seed,
                       **opt.metric_values(part.label, part.probability)})
    root = Path(cfg["output_dir"]) / "evaluation/development"
    for name, frame in (("fold_metrics", metrics), ("seed_summary", summary),
                        ("selection_validation_predictions", oof), ("pooled_selection_validation", pd.DataFrame(pooled))):
        _atomic_csv(root / f"{name}.csv", frame)
    return summary


def ensemble_metrics(predictions):
    if predictions.duplicated(KEYS + ["outer", "patient_id"]).any():
        raise ValueError("Repeated fold probability")
    groups = predictions.groupby(KEYS + ["patient_id", "label"]).outer.agg(list)
    if any(len(x) != 5 or set(x) != set(range(5)) for x in groups):
        raise ValueError("Every patient requires five fold probabilities")
    ensemble = predictions.groupby(KEYS + ["patient_id", "label"], as_index=False).probability.mean()
    rows = []
    for key, part in ensemble.groupby(KEYS):
        rows.append({**dict(zip(KEYS, key)), **opt.metric_values(part.label, part.probability)})
    return ensemble, pd.DataFrame(rows)


def matched_intervals(cfg, predictions):
    part = predictions[predictions.depth == 4]
    ids = sorted(part.patient_id.unique())
    labels = part.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    wp = rng.multinomial(positives, np.full(positives, 1 / positives), cfg["bootstrap_samples"])
    wn = rng.multinomial(negatives, np.full(negatives, 1 / negatives), cfg["bootstrap_samples"])
    rows = []
    for (seed, source), group in part.groupby(["seed", "source"]):
        matrices = {arm: frame.pivot(index="outer", columns="patient_id", values="probability").loc[range(5), ids].to_numpy()
                    for arm, frame in group.groupby("model")}
        for statistic in ("fold_mean", "probability_ensemble"):
            kernels = {arm: mean_auc_kernel(labels, matrix if statistic == "fold_mean" else matrix.mean(axis=0, keepdims=True))
                       for arm, matrix in matrices.items()}
            difference = kernels["v4"] - kernels["v1"]
            samples = ((wp @ difference) * wn).sum(axis=1) / (positives * negatives)
            low, high = np.quantile(samples, [.025, .975])
            rows.append({"seed": seed, "source": source, "statistic": statistic,
                         "v4_minus_v1": float(difference.mean()),
                         "ci95_low": float(low), "ci95_high": float(high)})
    return pd.DataFrame(rows)


def evaluate_holdout(cfg, refs):
    root = Path(cfg["output_dir"]) / "evaluation/holdout"
    receipt = root / "PREDICTIONS_COMPLETE.json"
    if receipt.exists():
        shared.verify_inventory(read_json(receipt)["artifacts"])
        raw = csv(root / "draw_fold_predictions.csv")
    else:
        splits = {k: v for k, v in holdout_splits(cfg).items() if k != "copy_T0"}
        rows = []
        for index, ref in enumerate(refs, 1):
            state = torch.load(Path(ref["path"]) / "model.pt", map_location="cpu", weights_only=False)
            task = state["task"]
            if set(splits["real"]["pids"]) & (set(task["train_ids"]) | set(task["val_ids"])):
                raise ValueError("Historical evaluation patient entered classifier fitting")
            model = shared.load_model(state)
            for source, split in splits.items():
                probability, _, _ = shared.predict_checkpoint(state, split, model=model)
                rows.append(pd.DataFrame({"model": ref["arm"], "depth": ref["depth"], "seed": ref["seed"],
                    "outer": ref["outer"], "source": source, "patient_id": split["pids"],
                    "label": split["labels"].astype(int), "probability": probability.astype(float)}))
            if index % 50 == 0:
                shared.progress(cfg, "historical_inference", completed=index, total=len(refs))
        raw = pd.concat(rows, ignore_index=True)
        _atomic_csv(root / "draw_fold_predictions.csv", raw)
        write_json(receipt, {"created_utc": now(), "artifacts": {
            str(root / "draw_fold_predictions.csv"): identity(root / "draw_fold_predictions.csv")}})
    predictions = previous.mc4_within_model(raw)
    ids = read_json(Path(cfg["output_dir"]) / "cohort.json")["split"]["val"]
    metrics = previous.fold_metrics(predictions, ids)
    summary = summarize_folds(metrics, cfg, SOURCES)
    ensemble, ensemble_scores = ensemble_metrics(predictions)
    summary = summary.merge(ensemble_scores[KEYS + ["auroc"]].rename(columns={"auroc": "probability_ensemble_auroc"}),
                            on=KEYS, validate="one_to_one")
    summary["patients"], summary["positive"] = len(ids), int(raw[raw.source == "real"].drop_duplicates("patient_id").label.sum())
    intervals = matched_intervals(cfg, predictions)
    for name, frame in (("mc4_fold_predictions", predictions), ("fold_metrics", metrics),
                        ("seed_summary", summary), ("probability_ensemble_predictions", ensemble),
                        ("probability_ensemble_metrics", ensemble_scores), ("matched_t3_intervals", intervals)):
        _atomic_csv(root / f"{name}.csv", frame)
    return summary, intervals


def verify_results(cfg, summary):
    root = Path(cfg["output_dir"]) / "evaluation/holdout"
    raw, mc4 = csv(root / "draw_fold_predictions.csv"), csv(root / "mc4_fold_predictions.csv")
    metrics, ensemble = csv(root / "fold_metrics.csv"), csv(root / "probability_ensemble_predictions.csv")
    keys = KEYS[:3] + ["outer", "patient_id", "label"]
    errors = []
    if (raw.groupby("patient_id").label.nunique() != 1).any():
        raise ValueError("Outcome labels differ across inputs or models")
    patients = raw.drop_duplicates("patient_id")
    if len(patients) != 102 or patients.label.sum() != 32:
        raise ValueError("Historical patient or positive count changed")
    for mode in ("direct", "rollout", "previous_real"):
        draws = raw[raw.source.str.startswith(mode + "_draw_")].pivot(index=keys, columns="source", values="probability").sort_index()
        saved = mc4[mc4.source == mode + "_mc4"].set_index(keys).sort_index().probability
        pd.testing.assert_index_equal(draws.index, saved.index)
        if draws.shape[1] != 4 or draws.isna().any().any():
            raise ValueError("Incomplete MC4 input")
        errors.append(float(np.max(np.abs(draws.to_numpy(dtype=float).sum(axis=1) / 4 - saved.to_numpy()))))
    observed = raw[raw.source == "real"].set_index(keys).sort_index().probability
    saved = mc4[mc4.source == "real"].set_index(keys).sort_index().probability
    pd.testing.assert_series_equal(observed, saved)
    score_lookup = metrics.set_index(KEYS + ["outer"]).sort_index()
    summary_lookup = summary.set_index(KEYS).sort_index()
    for key, part in mc4.groupby(KEYS + ["outer"]):
        errors.append(abs(rank_auc(part.label, part.probability) - score_lookup.loc[key, "auroc"]))
    for key, part in mc4.groupby(KEYS):
        scores = score_lookup.loc[key].auroc.to_numpy(dtype=float)
        row = summary_lookup.loc[key]
        errors.extend([abs(float(scores.mean()) - row.auroc_mean),
                       abs(float(scores.std(ddof=1)) - row.auroc_fold_sd)])
        for fold in range(5):
            errors.append(abs(score_lookup.loc[(*key, fold), "auroc"] - row[f"F{fold + 1}_auroc"]))
        matrix = part.pivot(index="patient_id", columns="outer", values="probability").sort_index()
        saved = ensemble[(ensemble.model == key[0]) & (ensemble.depth == key[1])
                         & (ensemble.seed == key[2]) & (ensemble.source == key[3])].set_index("patient_id").sort_index()
        pd.testing.assert_index_equal(matrix.index, saved.index)
        expected = matrix.to_numpy(dtype=float).sum(axis=1) / 5
        errors.append(float(np.max(np.abs(expected - saved.probability.to_numpy()))))
        errors.append(abs(rank_auc(saved.label, expected) - row.probability_ensemble_auroc))
    for depth, sources in ((1, SOURCES), (2, SOURCES[1:])):
        part = mc4[(mc4.depth == depth) & mc4.source.isin(sources)]
        matrix = part.pivot(index=keys, columns="source", values="probability")
        errors.append(float(np.max(np.ptp(matrix.to_numpy(), axis=1))))
    if max(errors) > 1e-12:
        raise ValueError(f"Independent metrics or probability aggregation differs: {max(errors)}")
    devroot = Path(cfg["output_dir"]) / "evaluation/development"
    dev, folds = csv(devroot / "selection_validation_predictions.csv"), csv(devroot / "fold_metrics.csv")
    lookup = folds.set_index(["model", "depth", "seed", "outer"])
    for key, part in dev.groupby(["model", "depth", "seed", "outer"]):
        errors.append(abs(rank_auc(part.label, part.probability) - lookup.loc[key, "auroc"]))
    if max(errors) > 1e-12:
        raise ValueError("Development metric replay failed")
    return {"passed": True, "raw_prediction_rows": len(raw), "mc4_prediction_rows": len(mc4),
            "fold_auc_replays": len(metrics), "ensemble_auc_replays": len(summary),
            "development_auc_replays": len(folds), "max_metric_or_probability_error": max(errors),
            "identical_t0_inputs_and_t1_generated_inputs_verified": True}


def compare_versions(cfg, new, development_summary):
    old = csv(Path(cfg["comparison_run"]) / "all_versions.csv")
    old = old[old.version.isin(["v1_original", "v4_latest"])].copy()
    current = new.assign(version=new.model.map({"v1": "v1_300_50", "v4": "v4_300_50"}))
    columns = ["version", "window", "seed", "source", "patients", "positive", "folds",
               *FOLDS, "auroc_mean", "auroc_fold_sd", "probability_ensemble_auroc"]
    combined = pd.concat([old[columns], current[columns]], ignore_index=True)
    differences = []
    keys = ["window", "seed", "source"]
    for left, right in (("v1_300_50", "v1_original"), ("v4_300_50", "v4_latest"), ("v4_300_50", "v1_300_50")):
        paired = combined[combined.version == left].merge(combined[combined.version == right], on=keys,
                                                           suffixes=("_new", "_reference"), validate="one_to_one")
        paired["comparison"] = left + "_minus_" + right
        for metric in ("auroc_mean", "probability_ensemble_auroc"):
            paired["delta_" + metric] = paired[metric + "_new"] - paired[metric + "_reference"]
        differences.append(paired[["comparison", *keys, "auroc_mean_reference", "auroc_mean_new", "delta_auroc_mean",
                                    "probability_ensemble_auroc_reference", "probability_ensemble_auroc_new",
                                    "delta_probability_ensemble_auroc"]])
    differences = pd.concat(differences, ignore_index=True)
    devold = csv(Path(cfg["comparison_run"]) / "development_real_only.csv")
    devold = devold[devold.version.isin(["v1_original", "v4_latest"])]
    devnew = development_summary.assign(version=development_summary.model.map({"v1": "v1_300_50", "v4": "v4_300_50"}))
    dev = pd.concat([devold, devnew], ignore_index=True)
    return combined, differences, dev


def table(frame, metric, with_sd=False):
    lines = ["| Seed | Real | Direct T0 | Rollout | Previous real |", "|---|---:|---:|---:|---:|"]
    for seed, part in frame.groupby("seed"):
        part = part.set_index("source")
        cells = [f"{part.loc[s, metric]:.5f}" + (f" +/- {part.loc[s, 'auroc_fold_sd']:.5f}" if with_sd else "") for s in SOURCES]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    return lines


def write_reports(cfg, combined, differences, dev, intervals):
    output = Path(cfg["output_dir"])
    for name, frame in (("all_versions", combined), ("version_differences", differences), ("development_comparison", dev)):
        _atomic_csv(output / f"{name}.csv", frame)
    sheets = {}
    for version, prefix in VERSIONS.items():
        frame = combined[combined.version == version]
        lines = [f"# {prefix}: {version}", "", "Same102historical patients/32positives; each seed is separate.",
                 "Generated MC4 probabilities are averaged within each model first.", ""]
        for metric, label in (("auroc_mean", "Mean of five model AUCs +/- fold SD"),
                              ("probability_ensemble_auroc", "AUC of five-model mean probability")):
            lines += [f"## {label}", ""]
            for window, part in frame.groupby("window", sort=True):
                lines += [f"### {window}", "", *table(part, metric, metric == "auroc_mean"), ""]
            scores = frame.pivot(index=["window", "seed"], columns="source", values=metric).reindex(columns=SOURCES).reset_index()
            sheets[prefix + ("_FoldMean" if metric == "auroc_mean" else "_ProbMean")] = scores
        sheets[prefix + "_FiveFolds"] = frame[["window", "seed", "source", *FOLDS, "auroc_mean", "auroc_fold_sd"]]
        _atomic_text(output / f"{version}.md", "\n".join(lines))
    epochs = csv(output / "training_epochs.csv")
    epoch_summary = epochs.groupby(["model", "window"]).agg(
        fits=("seed", "size"), trained_min=("epochs_trained", "min"), trained_median=("epochs_trained", "median"),
        trained_max=("epochs_trained", "max"), best_min=("selected_epochs", "min"),
        best_median=("selected_epochs", "median"), best_max=("selected_epochs", "max"),
        stopped_by_patience=("stop_reason", lambda x: int((x == "patience").sum()))).reset_index()
    _atomic_csv(output / "epoch_summary.csv", epoch_summary)
    sheets.update(Changes=differences, Development=dev, Training_Epochs=epochs, Epoch_Summary=epoch_summary, T3_Matched_CI=intervals)
    with pd.ExcelWriter(output / "pcr_300_50_comparison.xlsx") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "C2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="305B49")
            for index, column in enumerate(frame.columns, 1):
                ws.column_dimensions[get_column_letter(index)].width = min(40, max(15, len(column) + 2))
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.00000"
    for name, expected in sheets.items():
        actual = pd.read_excel(output / "pcr_300_50_comparison.xlsx", sheet_name=name)
        pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True), check_dtype=False,
                                      check_names=False, rtol=1e-12, atol=1e-12)
    wins = []
    for (comparison, window, source), frame in differences.groupby(["comparison", "window", "source"]):
        for statistic in ("auroc_mean", "probability_ensemble_auroc"):
            values = frame["delta_" + statistic].to_numpy()
            wins.append({"comparison": comparison, "window": window, "source": source, "statistic": statistic,
                         "higher": int((values > 1e-12).sum()), "equal": int((np.abs(values) <= 1e-12).sum()),
                         "lower": int((values < -1e-12).sum()), "min_difference": float(values.min()),
                         "max_difference": float(values.max())})
    _atomic_csv(output / "seed_change_counts.csv", pd.DataFrame(wins))
    lines = ["# V1/V4: 300 Epoch Cap, Patience50", "",
             "250new fits completed: V1 four windows, V4 T0-T3; seeds42-51 and original five folds.",
             "300is a maximum, not a fixed duration. Best validation-AUC weights are restored after stopping.", "",
             "[Workbook: versions and statistics separated](pcr_300_50_comparison.xlsx)", "",
             "| Version | Results |", "|---|---|",
             *[f"| {prefix} | [Both AUC definitions]({version}.md) |" for version, prefix in VERSIONS.items()], "",
             "[All numeric results](all_versions.csv) | [Per-seed changes](version_differences.csv)",
             "[Development comparison](development_comparison.csv) | [Training epochs](training_epochs.csv)",
             "[Matched T3 patient-bootstrap intervals](evaluation/holdout/matched_t3_intervals.csv)", "",
             "## Training duration", "", epoch_summary.to_markdown(index=False), "",
             "## Matched new V4 versus new V1", "",
             "Both T3 arms use identical folds, initialization, optimizer,300/50stopping, and validation-AUC selection.",
             "The only recipe difference is V4 residual L2=0.1. Every seed is retained.", "",
             pd.DataFrame(wins).query("comparison == 'v4_300_50_minus_v1_300_50'").drop(columns=["comparison", "window"]).to_markdown(index=False), "",
             "## Interpretation limits", "",
             "Fold SD measures variation across fitted models on the same102patients; it is not a patient confidence interval.",
             "Bootstrap intervals condition on the fitted models, omit training uncertainty, and are not multiplicity-adjusted.",
             "Development folds choose checkpoints and therefore report selection-validation performance; pooled rows are not a five-model probability ensemble.",
             "Generated inputs are assessed only on the historical102patients, not generated development-CV folds.",
             "Historical102patients also served as generator validation and were repeatedly evaluated; this is exploratory evidence.",
             "Old/new contrasts change training budget, checkpoint selection (oldV4fixed40), cosine schedule and V1T2min_delta.",
             "Original V1 used seed*1000+fold*10+depth; both new arms use seed*10000+fold*100+depth, matching the V4 engine.",
             "Thus old/new V1 changes cannot be attributed solely to extra epochs. The matched new V1/V4 T3 comparison controls these factors.",
             "Longer training and higher selection-validation AUC alone do not establish reduced overfitting or better generalization.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return len(sheets)


def evaluate_and_report(cfg, refs):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(cfg["output_dir"])
    require_frozen(cfg)
    runtime_paths = [Path(__file__), repo_path("src/registered_three_phase_fixed_report.py"),
                     repo_path("src/registered_three_phase_optimization_report.py"),
                     repo_path("scripts/report_registered_three_phase_versions.py")]
    runtime = {str(p): identity(p) for p in runtime_paths}
    shared.frozen_json(output / "evaluation_runtime.json", runtime)
    shared.verify_inventory(runtime)
    dev = development(cfg, refs)
    new, intervals = evaluate_holdout(cfg, refs)
    shared.progress(cfg, "metric_verification")
    audit = verify_results(cfg, new)
    combined, differences, development_comparison = compare_versions(cfg, new, dev)
    audit["workbook_sheets_roundtripped"] = write_reports(cfg, combined, differences, development_comparison, intervals)
    audit.update(checked_utc=now(), all_versions_rows=len(combined), across_seed_averaging=False)
    write_json(output / "verification.json", audit)
    paths = [p for p in output.rglob("*") if p.is_file()
             and not p.is_relative_to(output / "fits") and not p.is_relative_to(output / "smoke")
             and p.name not in ("workflow.lock", "progress.json", "COMPLETE.json")]
    return {"evaluation_verification": audit, "result_inventory": {str(p): identity(p) for p in paths}}

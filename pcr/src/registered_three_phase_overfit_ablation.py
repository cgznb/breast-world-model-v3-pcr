"""Single-factor T3 regularization experiments on development patients only."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from scripts.run_full978_anti_overfit import _atomic_csv, _prior_from_state, _read_metadata
from src import first_post_optimization as opt
from src import registered_three_phase_fixed_pcr as fixed
from src import registered_three_phase_optimization as shared
from src.first_post_pcr_data import identity, now, public, read_json, repo_path, write_json

SCHEMA = "registered_three_phase_roi32_overfit_ablation_v4"
RUNTIME = tuple(dict.fromkeys((*fixed.RUNTIME,
    "src/registered_three_phase_overfit_ablation.py",
    "scripts/run_registered_three_phase_overfit_ablation.py")))


def load_config(path):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema"] != SCHEMA or cfg["depth"] != 4 or cfg["epochs"] != 40:
        raise ValueError("This experiment requires the declared T3/40-epoch protocol")
    if cfg["formal_seeds"] != list(range(42, 52)):
        raise ValueError("All ten original classifier seeds are required")
    cfg["config_path"] = str(path)
    for key in ("output_dir", "baseline_run", "reference_run", "reference_config", "historical_summary"):
        cfg[key] = str(repo_path(cfg[key]).resolve())
    for key in ("baseline_run", "reference_run"):
        if Path(cfg["output_dir"]).is_relative_to(cfg[key]):
            raise ValueError("A separate experiment directory is required")
    return cfg


def build_tasks(cfg, folds):
    reference_cfg = fixed.load_config(cfg["reference_config"])
    if reference_cfg["output_dir"] != cfg["reference_run"] or reference_cfg["baseline_run"] != cfg["baseline_run"]:
        raise ValueError("Reference experiment points to another cohort or output")
    prior_tasks, _ = fixed.build_tasks(reference_cfg, folds)
    references = [t for t in prior_tasks if t["depth"] == cfg["depth"] and t["candidate"] == "reference"]
    tasks, refs = [], []
    for reference in references:
        if reference["fixed_epochs"] != cfg["epochs"]:
            raise ValueError("Reference and intervention fitting budgets must match")
        refs.append({"arm": "reference", "seed": reference["seed"], "outer": reference["outer"],
                     "path": str(opt.task_path(cfg["reference_run"], reference)), "task": reference})
        for arm, overrides in cfg["arms"].items():
            if arm == "reference" or not overrides:
                raise ValueError("Each intervention must have a distinct name and explicit changes")
            task = copy.deepcopy(reference)
            task["candidate"] = arm
            task["effective"].update(overrides)
            if any(task["effective"][k] != reference["effective"][k]
                   for k in ("epochs", "scheduler_t_max", "epoch_selection", "positive_class_weight",
                             "use_clinical_token", "use_prior", "lr", "batch_size")):
                raise ValueError("Interventions cannot alter the matching conditions")
            tasks.append(task)
            refs.append({"arm": arm, "seed": task["seed"], "outer": task["outer"],
                         "path": str(opt.task_path(cfg["output_dir"], task)), "task": task})
    if len(references) != 5 * len(cfg["formal_seeds"]):
        raise ValueError("Missing reference fold or seed")
    return tasks, refs


def prepare(cfg):
    output, baseline = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    cohort = read_json(baseline / "cohort.json")
    folds = read_json(baseline / "folds.json")
    if isinstance(folds, dict):
        folds = list(folds.values())
    dev, held = cohort["split"]["train"], cohort["split"]["val"]
    opt.validate_partitions(dev, held, folds)
    if (len(dev), len(held)) != (cfg["expected_development"], cfg["expected_holdout"]):
        raise ValueError("Original cohort sizes changed")
    tasks, refs = build_tasks(cfg, folds)
    old_freeze = read_json(Path(cfg["reference_run"]) / "FROZEN_MODELS.json")["artifacts"]
    shared.verify_inventory(old_freeze)
    for ref in refs:
        if ref["arm"] == "reference" and not opt.task_complete(Path(ref["path"]), ref["task"]):
            raise ValueError("Reference weights, predictions, or configuration changed")
    metadata = _read_metadata(baseline / "metadata_enriched.csv", dev)
    contract = public({
        "schema": SCHEMA, "config": cfg, "created_from": "original ROI32 cohort and fixed-v3 reference only",
        "runtime": {str(repo_path(p)): identity(repo_path(p)) for p in RUNTIME},
        "config_identity": identity(cfg["config_path"]),
        "new_fits": len(tasks), "reused_fits": len(refs) - len(tasks),
        "development_patients": len(dev), "historical_holdout_patients_excluded": len(held),
        "outer_validation_used_for_checkpoint_selection": False,
        "holdout_labels_or_features_loaded": False,
        "selection": "none; report all prespecified arms and all seeds, without automatic promotion",
        "evaluation": "per-seed mean of five disjoint development-fold AUROCs and sample fold SD",
        "limitations": "Previously studied development cohort; exploratory ablation, not fresh validation.",
    })
    shared.frozen_json(output / "contract.json", contract)
    shared.frozen_json(output / "nested_folds.json", folds)
    shared.frozen_json(output / "model_references.json", public(refs))
    metadata_path = output / "development_metadata.csv"
    if metadata_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(metadata_path, dtype={"pid": str}),
                                      metadata.reset_index(drop=True), check_dtype=False)
    else:
        _atomic_csv(metadata_path, metadata)
    inventory_path = output / "input_inventory.json"
    if inventory_path.exists():
        shared.verify_inventory(read_json(inventory_path))
    else:
        inventory = dict(old_freeze)
        for path in baseline.rglob("*"):
            if path.is_file():
                inventory[str(path)] = identity(path)
        inventory.update(contract["runtime"])
        for path in (Path(cfg["config_path"]), output / "development_metadata.csv",
                     output / "nested_folds.json", output / "model_references.json", output / "contract.json"):
            inventory[str(path)] = identity(path)
        shared.frozen_json(inventory_path, inventory)
    return folds, tasks, refs


def summarize_folds(frame, arms, seeds, folds):
    expected = {(a, s, f) for a in arms for s in seeds for f in folds}
    actual = set(frame[["arm", "seed", "outer"]].itertuples(index=False, name=None))
    if len(frame) != len(expected) or actual != expected:
        raise ValueError("Each arm and seed requires all five folds exactly once")
    rows = []
    for (arm, seed), part in frame.groupby(["arm", "seed"]):
        row = {"arm": arm, "seed": int(seed), "window": "T0-T3", "folds": len(folds)}
        for metric in ("train_auroc", "validation_auroc", "auroc_gap", "validation_logloss", "validation_brier"):
            row[metric + "_mean"] = float(part[metric].mean())
            row[metric + "_fold_sd"] = float(part[metric].std(ddof=1))
        for fold in part.itertuples():
            row[f"F{fold.outer + 1}_auroc"] = fold.validation_auroc
        rows.append(row)
    return pd.DataFrame(rows)


def paired_intervals(cfg, oof):
    """Patient bootstrap within each disjoint fold and class, keeping models fixed."""
    if oof.duplicated(["arm", "seed", "patient_id"]).any():
        raise ValueError("Every patient must have one OOF prediction per arm/seed")
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    weights, samples, points = {}, {}, {}
    for (seed, outer), group in oof.groupby(["seed", "outer"]):
        reference = group[group.arm == "reference"].sort_values("patient_id")
        ids, labels = reference.patient_id.tolist(), reference.label.to_numpy()
        if not ids or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Bootstrap requires a reference and both outcome classes")
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        if outer not in weights:
            wp = rng.multinomial(n_pos, np.full(n_pos, 1 / n_pos), size=cfg["bootstrap_samples"])
            wn = rng.multinomial(n_neg, np.full(n_neg, 1 / n_neg), size=cfg["bootstrap_samples"])
            weights[outer] = (ids, labels.copy(), wp, wn)
        expected_ids, expected_y, wp, wn = weights[outer]
        if ids != expected_ids or not np.array_equal(labels, expected_y):
            raise ValueError("Fold membership or labels differ between seeds")
        for arm, part in group.groupby("arm"):
            part = part.sort_values("patient_id")
            if part.patient_id.tolist() != ids or not np.array_equal(part.label.to_numpy(), labels):
                raise ValueError("Paired patients or labels differ between arms")
            probability = part.probability.to_numpy()
            positive, negative = probability[labels == 1, None], probability[None, labels == 0]
            kernel = (positive > negative).astype(float) + .5 * (positive == negative)
            score = roc_auc_score(labels, probability)
            if abs(score - kernel.mean()) > 1e-12:
                raise ValueError("Independent rank-kernel AUROC replay failed")
            key = (arm, int(seed))
            points.setdefault(key, []).append(score)
            samples.setdefault(key, []).append(((wp @ kernel) * wn).sum(axis=1) / (n_pos * n_neg))
    rows = []
    for (arm, seed), values in samples.items():
        if len(values) != 5:
            raise ValueError("Bootstrap requires the same five complete folds")
        if arm == "reference":
            continue
        ref = ("reference", seed)
        delta = np.mean(values, axis=0) - np.mean(samples[ref], axis=0)
        low, high = np.quantile(delta, [.025, .975])
        rows.append({"arm": arm, "seed": seed,
                     "delta_validation_auroc": float(np.mean(points[(arm, seed)]) - np.mean(points[ref])),
                     "conditional_ci95_low": float(low), "conditional_ci95_high": float(high)})
    return pd.DataFrame(rows)


def audit_predictions(cfg, refs):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    raw = opt.load_development(cfg["output_dir"], str(opt.source_directory(cfg, "physical_roi")))
    rows, predictions, identities = [], [], {}
    max_error, max_metric_error = 0.0, 0.0
    for ref in refs:
        path, task = Path(ref["path"]), ref["task"]
        if not opt.task_complete(path, task):
            raise ValueError("An incomplete or modified model entered evaluation")
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        if checkpoint["epochs_trained"] != task["fixed_epochs"]:
            raise ValueError("Fitting budget changed")
        model = shared.load_model(checkpoint)
        saved = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
        summary = read_json(path / "COMPLETE.json")
        row = {"arm": ref["arm"], "seed": ref["seed"], "outer": ref["outer"],
               "parameters": checkpoint["parameters"], "epochs": checkpoint["epochs_trained"]}
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            split = opt.select_patients(raw, ids, task["depth"])
            prior = _prior_from_state(split["clinical"], checkpoint["clinical_prior"])
            probability, residual = opt.predict(model, opt.tensor_split(split, prior, "cpu"))
            part = saved[saved.role == role].set_index("patient_id").loc[split["pids"]].reset_index()
            if not np.array_equal(part.label.to_numpy(), split["labels"]):
                raise ValueError("Saved outcomes differ from development metadata")
            error = float(np.max(np.abs(part.probability.to_numpy() - probability)))
            max_error = max(max_error, error)
            if error > 1e-6 or not np.allclose(part.residual_logit, residual, atol=1e-6, rtol=0):
                raise ValueError("Saved-weight probability/residual replay failed")
            metrics = opt.metric_values(split["labels"], probability)
            for key, value in metrics.items():
                max_metric_error = max(max_metric_error, abs(value - summary[role][key]))
                row[f"{role}_{key}"] = value
            row[f"{role}_residual_rms"] = float(np.sqrt(np.mean(residual ** 2)))
            if role == "validation":
                part["arm"], part["seed"], part["outer"] = ref["arm"], ref["seed"], ref["outer"]
                predictions.append(part)
        row["auroc_gap"] = row["train_auroc"] - row["validation_auroc"]
        rows.append(row)
        for name in ("model.pt", "history.csv", "predictions.csv", "COMPLETE.json"):
            identities[str(path / name)] = identity(path / name)
    if max_metric_error > 1e-7:
        raise ValueError("Replayed model metrics differ from saved summaries")
    audit = {"models": len(refs), "max_prediction_error": max_error, "max_metric_error": max_metric_error}
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), identities, audit


def original_windows(cfg):
    path = Path(cfg["baseline_run"]) / "diagnostics/overfitting/fit_details.csv"
    fits = pd.read_csv(path)
    original = fits.groupby(["depth", "seed"], as_index=False).agg(
        training_auroc=("train_auc", "mean"), selected_validation_auroc=("validation_auc", "mean"),
        training_validation_gap=("auc_gap", "mean"))
    historic = pd.read_csv(cfg["historical_summary"])
    historic = historic[historic.source == "real"][["depth", "seed", "auroc_mean", "auroc_fold_sd"]]
    historic = historic.rename(columns={"auroc_mean": "historical_holdout_fold_mean_auroc",
                                        "auroc_fold_sd": "historical_holdout_fold_sd"})
    result = original.merge(historic, on=["depth", "seed"], validate="one_to_one")
    if len(result) != 40:
        raise ValueError("Original four-window/ten-seed context is incomplete")
    return result


def report(cfg, folds, refs):
    output = Path(cfg["output_dir"])
    frame, oof, identities, audit = audit_predictions(cfg, refs)
    arms = ["reference", *cfg["arms"]]
    summary = summarize_folds(frame, arms, cfg["formal_seeds"], [f["fold"] for f in folds])
    for (arm, seed), part in oof.groupby(["arm", "seed"]):
        if len(part) != cfg["expected_development"] or not part.patient_id.is_unique:
            raise ValueError("OOF coverage must equal the complete development population")
    intervals = paired_intervals(cfg, oof)
    reference = summary[summary.arm == "reference"].set_index("seed")
    comparisons = summary[summary.arm != "reference"].copy()
    for key in ("validation_auroc", "validation_logloss", "validation_brier", "auroc_gap"):
        comparisons[f"delta_{key}"] = comparisons[f"{key}_mean"] - comparisons.seed.map(reference[f"{key}_mean"])
    comparisons = comparisons.merge(intervals.drop(columns="delta_validation_auroc"), on=["arm", "seed"], validate="one_to_one")
    comparisons["gap_reduced"] = comparisons.delta_auroc_gap < -1e-12
    comparisons["validation_auc_within_tolerance"] = comparisons.delta_validation_auroc >= -cfg["descriptive_auc_tolerance"]
    comparisons["validation_logloss_no_worse"] = comparisons.delta_validation_logloss <= 1e-12
    comparisons["descriptive_three_checks_pass"] = comparisons[["gap_reduced", "validation_auc_within_tolerance",
                                                                 "validation_logloss_no_worse"]].all(axis=1)
    for name, table in (("fold_metrics", frame), ("per_seed_fivefold", summary), ("oof_predictions", oof),
                        ("paired_comparisons", comparisons), ("original_window_context", original_windows(cfg))):
        _atomic_csv(output / f"{name}.csv", table)
    with pd.ExcelWriter(output / "development_ablation.xlsx") as workbook:
        summary.to_excel(workbook, sheet_name="Per Seed Five Folds", index=False)
        frame.to_excel(workbook, sheet_name="Individual Folds", index=False)
        comparisons.to_excel(workbook, sheet_name="Paired Comparisons", index=False)
        original_windows(cfg).to_excel(workbook, sheet_name="Original Windows", index=False)
    lines = ["# T0-T3 Development-Only Overfitting Ablation", "",
             "Four prespecified single-factor interventions, original TDN framework, 764 development patients,",
             "five original patient folds and all ten seeds separately. Every model uses 40 fixed epochs.",
             "The 50 reference fits are reused from fixed-v3. No new holdout or generated-input evaluation was run.", "",
             "## Validation AUROC: Mean of Five Folds +/- Sample Fold SD", "",
             "| Seed | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for seed in cfg["formal_seeds"]:
        part = summary[summary.seed == seed].set_index("arm")
        values = [f"{part.loc[a].validation_auroc_mean:.4f} +/- {part.loc[a].validation_auroc_fold_sd:.4f}" for a in arms]
        lines.append(f"| {seed} | " + " | ".join(values) + " |")
    lines += ["", "## Matched Development Evidence", "",
              "Counts below describe ten seeds using the same patients; they are not ten independent trials.",
              "The last column is a descriptive screen, not proof of noninferiority or a model-selection rule.", "",
              "| Arm | AUC improved | Gap reduced | Logloss improved | Gap reduced, AUC drop <=0.005, logloss no worse |",
              "|---|---:|---:|---:|---:|"]
    for arm, part in comparisons.groupby("arm", sort=False):
        lines.append(f"| {arm} | {int((part.delta_validation_auroc > 1e-12).sum())}/10 | "
                     f"{int(part.gap_reduced.sum())}/10 | {int((part.delta_validation_logloss < -1e-12).sum())}/10 | "
                     f"{int(part.descriptive_three_checks_pass.sum())}/10 |")
    lines += ["", "## Interpretation Limits", "",
              "A smaller training-validation gap alone can mean underfitting. Read validation AUC, logloss and Brier together.",
              "All arms preserve class weighting, clinical input, learning rate and fitting duration. Capacity is changed only",
              "in compact_head; prefix_dropout randomly masks future prefixes only during training.",
              "Current validation folds never select epochs. Original v1 validation did select checkpoints and is optimistic;",
              "use the matched fixed-epoch reference for intervention comparisons, not the original v1 peak scores.",
              "Bootstrap intervals resample patients within each fold and class with fixed fitted models. They omit retraining",
              "uncertainty and multiplicity adjustment. The development cohort was studied previously, so this is exploratory.",
              "No arm, seed or checkpoint is automatically promoted. More visits need not produce strictly increasing AUC.",
              "The original 102-patient holdout was previously generator validation. Its old aggregate scores appear only",
              "as context; no holdout labels, features or predictions enter new fitting or candidate selection.", "",
              "[Excel](development_ablation.xlsx) | [Per-seed CSV](per_seed_fivefold.csv) | [Every fold](fold_metrics.csv)",
              "[Paired differences](paired_comparisons.csv) | [Original windows](original_window_context.csv)",
              "[Verification](verification.json)", ""]
    (output / "README.md").write_text("\n".join(lines))
    shared.verify_inventory(read_json(output / "input_inventory.json"))
    shared.frozen_json(output / "FROZEN_MODELS.json", {"artifacts": identities})
    verification = {"schema": SCHEMA, "passed": True, "completed_utc": now(), **audit,
                    "fold_metrics": len(frame), "per_seed_summaries": len(summary), "paired_comparisons": len(comparisons),
                    "all_fits_fixed_epochs": cfg["epochs"], "outer_validation_selects_weights": False,
                    "new_holdout_inference": False, "generated_input_inference": False,
                    "all_input_identities_unchanged": True, "bootstrap": "paired stratified patients within folds, fixed models"}
    write_json(output / "verification.json", verification)
    return verification

"""Development diagnostics and post-freeze evaluation for temporal pCR v3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text, _load_split, _prior_from_state
from scripts.run_full978_independent_cv import _canonical_split
from src import first_post_optimization as opt
from src.first_post_pcr_data import identity, now, read_json, write_json
from src.single_phase_repeat_retraining import unchanged
from src.single_phase_temporal_models import build_model, window_split
from src.single_phase_temporal_training import SCHEMA


def diagnose(cfg):
    previous = Path(cfg["previous_run"])
    source = Path(cfg["baseline_run"])
    output = Path(cfg["output_dir"]) / "diagnosis_v2"
    cohort = read_json(source / "cohort.json")
    split = _load_split(source / "embeddings/real", source / "metadata_enriched.csv", cohort["split"]["train"])
    contiguous = _canonical_split(split, 4)
    available = window_split(split, split["pids"], 4, {"visit_policy": "available"})
    masks, used = split["masks"] > 0, contiguous["masks"] > 0
    recovered = (available["masks"] > 0) & ~used
    predictions = pd.read_csv(previous / "reports/outer_oof_predictions.csv", dtype={"patient_id": str})
    predictions = predictions[predictions.arm == "selected"]
    rows = []
    for (depth, seed), frame in predictions.groupby(["depth", "seed"]):
        rows.append({"depth": int(depth), "seed": int(seed),
            **{f"model_{k}": v for k, v in opt.metric_values(frame.label, frame.probability).items()},
            **{f"prior_{k}": v for k, v in opt.metric_values(frame.label, frame.clinical_prior_probability).items()},
            "mean_absolute_probability_correction": float(np.abs(frame.probability - frame.clinical_prior_probability).mean())})
    frame = pd.DataFrame(rows)
    summary = frame.groupby("depth").mean().drop(columns="seed").reset_index()
    _atomic_csv(output / "prior_comparison_by_seed.csv", frame)
    _atomic_csv(output / "prior_comparison.csv", summary)
    payload = {"schema": SCHEMA, "created_utc": now(), "branch": cfg["branch"],
        "available_visits": masks.sum(axis=0).tolist(), "v2_used_visits": used.sum(axis=0).tolist(),
        "available_policy_visits": (available["masks"] > 0).sum(axis=0).tolist(),
        "discarded_visits_v2": int((masks & ~used).sum()),
        "recovered_visits_with_T0": int(recovered.sum()), "recovered_patients_with_T0": int(recovered.any(axis=1).sum()),
        "no_T0_patients": int((~masks[:, 0]).sum()), "development_comparison": summary.to_dict("records"),
        "feature_sd_quantiles": np.quantile(split["embs"][masks].std(axis=0), [0, .1, .5, .9, 1]).tolist(),
        "holdout_used_for_candidate_selection": False,
        "interpretation": "Clinical-prior dominance is observed for registered v2. First-post later inputs improve development OOF, but the historical holdout did not show a consistent gain. More seeds quantify initialization variation, not new patient evidence."}
    write_json(output / "summary.json", payload)
    return payload


def _aggregate(metrics, keys):
    columns = [c for c in ("auroc", "logloss", "brier", "prior_auroc", "train_auroc",
                           "fold_validation_auroc", "gap", "epochs") if c in metrics]
    result = metrics.groupby(keys)[columns].agg(["mean", "std"])
    result.columns = ["_".join(c) for c in result.columns]
    return result.reset_index()


def seed_group_summary(metrics, keys):
    groups = []
    for name, keep in (("all_30", metrics.seed >= 42), ("original_10", metrics.seed <= 51),
                       ("additional_20", metrics.seed >= 52)):
        selected = metrics[keep]
        if not selected.empty:
            groups.append(_aggregate(selected, keys).assign(seed_group=name, seed_count=selected.seed.nunique()))
    return pd.concat(groups, ignore_index=True)


def report_development(cfg):
    output = Path(cfg["output_dir"])
    references = read_json(output / "selection/temporal_outer_models.json")
    folds = read_json(output / "nested_folds.json")
    expected = set(folds[0]["train_ids"] + folds[0]["val_ids"])
    rows, frames = [], []
    for reference in references:
        path = Path(reference["path"])
        result = read_json(path / "COMPLETE.json")
        if not opt.task_complete(path, result["task"]):
            raise ValueError("Incomplete outer fit in report")
        task = result["task"]
        fields = {k: reference[k] for k in ("arm", "depth", "seed", "outer")}
        prediction = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        frames.append(prediction.loc[prediction.role == "validation"].assign(**fields))
        rows.append({**fields, "candidate": task["candidate"], "epochs": result["selected_epochs"],
            "input_depth": min(task["depth"], task["effective"].get("input_depth", task["depth"])),
            "gap": result["auroc_gap"],
            **{f"train_{k}": v for k, v in result["train"].items()},
            **{f"validation_{k}": v for k, v in result["validation"].items()}})
    predictions, fits = pd.concat(frames, ignore_index=True), pd.DataFrame(rows)
    metrics = []
    for (arm, depth, seed), frame in predictions.groupby(["arm", "depth", "seed"]):
        if len(frame) != len(expected) or set(frame.patient_id) != expected or not frame.patient_id.is_unique:
            raise ValueError("Outer OOF coverage changed")
        fit = fits[(fits.arm == arm) & (fits.depth == depth) & (fits.seed == seed)]
        metrics.append({"arm": arm, "depth": int(depth), "seed": int(seed),
            **opt.metric_values(frame.label, frame.probability),
            "prior_auroc": opt.metric_values(frame.label, frame.clinical_prior_probability)["auroc"],
            "train_auroc": fit.train_auroc.mean(), "fold_validation_auroc": fit.validation_auroc.mean(),
            "gap": fit.gap.mean(), "epochs": fit.epochs.mean()})
    metrics = pd.DataFrame(metrics)
    summary = _aggregate(metrics, ["depth", "arm"])
    groups = seed_group_summary(metrics, ["depth", "arm"])
    for name, frame in (("outer_oof_predictions", predictions), ("fit_metrics", fits),
                        ("seed_metrics", metrics), ("summary", summary), ("seed_group_summary", groups)):
        _atomic_csv(output / "reports" / f"{name}.csv", frame)
    payload = {"schema": SCHEMA, "branch": cfg["branch"], "development_patients": len(expected),
        "summary": summary.to_dict("records"), "seed_group_summary": groups.to_dict("records"),
        "holdout_evaluated": False, "selection_nested_inside_outer_training": True,
        "baseline_definition": "raw compact32 TDN, contiguous visits, v2 unbounded_l2 recipe",
        "t0_only_selected_fold_windows": fits.loc[(fits.arm == "selected") & (fits.depth > 1) & (fits.input_depth == 1),
                                                  ["depth", "outer"]].drop_duplicates().to_dict("records")}
    write_json(output / "reports/summary.json", payload)
    return payload


def freeze_models(cfg):
    output = Path(cfg["output_dir"])
    rows = read_json(output / "selection/temporal_outer_models.json")
    references = []
    for row in rows:
        if row["arm"] != "selected":
            continue
        path = Path(row["path"]) / "model.pt"
        task = read_json(path.parent / "COMPLETE.json")["task"]
        references.append({"depth": opt.DEPTH_NAMES[row["depth"]], "max_tp": row["depth"],
            "input_max_tp": min(row["depth"], task["effective"].get("input_depth", row["depth"])),
            "seed": row["seed"], "fold": row["outer"], "path": str(path), "identity": identity(path)})
    expected = {(d, s, f) for d in cfg["depths"] for s in cfg["formal_seeds"] for f in range(5)}
    if len(references) != len(expected) or {(r["max_tp"], r["seed"], r["fold"]) for r in references} != expected:
        raise ValueError("Incomplete selected model set")
    unchanged(output / "frozen_models.json", {"schema": SCHEMA, "models": references,
        "holdout_used_for_selection": False, "factory": "src.single_phase_temporal_models.build_model"})
    return references


def evaluate_real(cfg, references):
    output, source = Path(cfg["output_dir"]), Path(cfg["baseline_run"])
    heldout = read_json(source / "cohort.json")["split"]["val"]
    raw = _load_split(source / "embeddings/real", source / "holdout_metadata.csv", heldout)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    rows = []
    for reference in references:
        if identity(reference["path"]) != reference["identity"]:
            raise ValueError("Selected classifier changed")
        checkpoint = torch.load(reference["path"], map_location="cpu", weights_only=False)
        task = checkpoint["task"]
        if (set(task["train_ids"]) | set(task["val_ids"])) & set(heldout):
            raise ValueError("Holdout entered classifier fitting")
        split = window_split(raw, heldout, task["depth"], task["effective"])
        prior = _prior_from_state(split["clinical"], checkpoint["clinical_prior"])
        model = build_model(task["effective"], task["depth"]).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model_state"])
        probability, _ = opt.predict(model, opt.tensor_split(split, prior, "cuda:0"))
        rows.append(pd.DataFrame({"patient_id": split["pids"], "label": split["labels"].astype(int),
            "probability": probability.astype(np.float64), "depth": reference["depth"],
            "seed": reference["seed"], "fold": reference["fold"]}))
    folds = pd.concat(rows, ignore_index=True)
    if not (folds.groupby(["depth", "seed", "patient_id"]).fold.nunique() == 5).all():
        raise ValueError("Every holdout ensemble must contain five folds")
    predictions = folds.groupby(["depth", "seed", "patient_id", "label"], as_index=False).probability.mean()
    rows = []
    for (depth, seed), frame in predictions.groupby(["depth", "seed"]):
        if len(frame) != len(heldout) or set(frame.patient_id) != set(heldout):
            raise ValueError("Holdout coverage changed")
        rows.append({"depth": depth, "seed": int(seed), "patients": len(heldout),
                     **opt.metric_values(frame.label, frame.probability)})
    metrics = pd.DataFrame(rows)
    summary = _aggregate(metrics, ["depth"])
    groups = seed_group_summary(metrics, ["depth"])
    old = pd.read_csv(Path(cfg["previous_run"]) / "evaluation/real/seed_metrics.csv")
    matched = metrics.merge(old, on=["depth", "seed", "patients"], suffixes=("_v3", "_v2"), validate="one_to_one")
    for metric in ("auroc", "logloss", "brier"):
        matched[f"delta_{metric}"] = matched[f"{metric}_v3"] - matched[f"{metric}_v2"]
    if len(matched) != 40:
        raise ValueError("Matched v2 comparison requires the original ten seeds")
    for name, frame in (("fold_predictions", folds), ("predictions", predictions), ("seed_metrics", metrics),
                        ("summary", summary), ("seed_group_summary", groups), ("matched_v2_comparison", matched)):
        _atomic_csv(output / "evaluation/real" / f"{name}.csv", frame)
    write_json(output / "evaluation/real/COMPLETE.json", {"schema": SCHEMA, "complete": True,
        "models": len(references), "patients": len(heldout), "completed_utc": now(),
        "optimizer_updates": 0, "generated_evaluated": False,
        "aggregation": "five_fold_probability_mean_then_mean_seed_metric"})
    return summary


def write_report(cfg):
    output = Path(cfg["output_dir"])
    dev = pd.read_csv(output / "reports/summary.csv")
    dev = dev[dev.arm == "selected"].set_index("depth")
    heldout = pd.read_csv(output / "evaluation/real/summary.csv").set_index("depth")
    groups = pd.read_csv(output / "evaluation/real/seed_group_summary.csv")
    matched = pd.read_csv(output / "evaluation/real/matched_v2_comparison.csv")
    lines = [f"# {cfg['branch']}: temporal pCR v3", "",
        "Frozen repeated-single-phase Pillar features; training-fold transforms; original nested five-fold cohort.",
        "Thirty formal seeds (42-71), including twenty new seeds (52-71). All seeds are retained.",
        "Model/epoch/transform fitting uses only inner training/validation. Outer duration is fixed before scoring.",
        "The compact raw TDN is a comparison arm; selected candidates may use standardized/PCA features or explicit changes.",
        "T0-only candidates are marked in reports/summary.json; a long requested window need not select a temporal model.", "",
        "| Window | OOF AUC | Clinical OOF AUC | Train/fold AUC gap | Holdout AUC | Seed SD |",
        "|---|---:|---:|---:|---:|---:|"]
    for depth, name in opt.DEPTH_NAMES.items():
        a, b = dev.loc[depth], heldout.loc[name]
        lines.append(f"| {name} | {a.auroc_mean:.5f} | {a.prior_auroc_mean:.5f} | {a.gap_mean:.5f} | {b.auroc_mean:.5f} | {b.auroc_std:.5f} |")
    lines += ["", "| Window | Original 10 seeds | Additional 20 seeds | Matched v3-v2 AUC change |",
              "|---|---:|---:|---:|"]
    for name in opt.DEPTH_NAMES.values():
        selected = groups[groups.depth == name].set_index("seed_group")
        delta = matched.loc[matched.depth == name, "delta_auroc"].mean()
        lines.append(f"| {name} | {selected.loc['original_10', 'auroc_mean']:.5f} | {selected.loc['additional_20', 'auroc_mean']:.5f} | {delta:+.5f} |")
    lines += ["", "These historical internal holdouts were previously used for upstream validation and repeated evaluation.",
        "They are not independent end-to-end tests. Seed SD is not a patient confidence interval; do not select seeds on holdout scores.",
        "Input images/crops are unchanged. This study does not activate the pending T0-mask crop fallback or evaluate generated MRI.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))

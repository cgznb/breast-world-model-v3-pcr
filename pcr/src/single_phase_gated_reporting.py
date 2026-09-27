"""Freeze all pilot comparisons and report development OOF results only."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src import first_post_optimization as opt
from src.first_post_pcr_data import identity, now, read_json, write_json
from src.single_phase_gated_training import CANDIDATES, SCHEMA, canonical_spec, task_complete
from src.single_phase_repeat_retraining import unchanged


def aggregate(metrics):
    columns = [k for k in ("auroc", "logloss", "brier", "prior_auroc", "reference_auroc", "train_auroc",
                           "fold_validation_auroc", "gap", "temporal_gate", "epochs", "trainable_parameters") if k in metrics]
    table = metrics.groupby(["arm", "depth"])[columns].agg(["mean", "std"])
    table.columns = ["_".join(c) for c in table.columns]
    return table.reset_index()


def report_development(cfg, selection):
    output = Path(cfg["output_dir"])
    folds = read_json(output / "nested_folds.json")
    expected = set(folds[0]["train_ids"] + folds[0]["val_ids"])
    references, frames, fits, cached = [], [], [], {}
    choices = {(r["depth"], r["outer"]): r["selected"]["candidate"] for r in selection["selections"]}
    for depth in cfg["depths"]:
        for outer in folds:
            for arm in (*CANDIDATES, "selected"):
                candidate = choices[depth, outer["fold"]] if arm == "selected" else arm
                for seed in cfg["formal_seeds"]:
                    spec = canonical_spec(candidate, depth, outer["fold"], -1, seed, "outer")
                    path = opt.task_path(output, spec)
                    if path not in cached:
                        result = read_json(path / "COMPLETE.json")
                        if not task_complete(path, result["task"]):
                            raise ValueError("Incomplete model in development report")
                        cached[path] = (result, pd.read_csv(path / "predictions.csv", dtype={"patient_id": str}))
                    result, prediction = cached[path]
                    task = result["task"]
                    fields = {"arm": arm, "depth": depth, "seed": seed, "outer": outer["fold"]}
                    reference = {**fields, "candidate": candidate, "model_candidate": task["candidate"],
                        "training_max_tp": task["depth"], "max_tp": depth,
                        "path": str(path / "model.pt"), "identity": identity(path / "model.pt")}
                    references.append(reference)
                    part = prediction[(prediction.depth == depth) & (prediction.role == "validation")]
                    frames.append(part.assign(**fields))
                    values = result["windows"][str(depth)]
                    fits.append({**fields, "candidate": candidate, "model_candidate": task["candidate"],
                        "epochs": result["selected_epochs"], "parameters": result["parameters"],
                        "trainable_parameters": result["trainable_parameters"], "gap": values["gap"],
                        **{f"train_{k}": v for k, v in values["train"].items()},
                        **{f"validation_{k}": v for k, v in values["validation"].items()}})
    predictions, fit_metrics = pd.concat(frames, ignore_index=True), pd.DataFrame(fits)
    rows = []
    for (arm, depth, seed), part in predictions.groupby(["arm", "depth", "seed"]):
        if len(part) != len(expected) or set(part.patient_id) != expected or not part.patient_id.is_unique:
            raise ValueError("Development OOF coverage changed")
        subset = fit_metrics[(fit_metrics.arm == arm) & (fit_metrics.depth == depth) & (fit_metrics.seed == seed)]
        rows.append({"arm": arm, "depth": int(depth), "seed": int(seed),
            **opt.metric_values(part.label, part.probability),
            "prior_auroc": opt.metric_values(part.label, part.clinical_prior_probability)["auroc"],
            "reference_auroc": opt.metric_values(part.label, part.reference_probability)["auroc"],
            "train_auroc": subset.train_auroc.mean(), "fold_validation_auroc": subset.validation_auroc.mean(),
            "gap": subset.gap.mean(), "epochs": subset.epochs.mean(),
            "trainable_parameters": subset.trainable_parameters.mean(), "temporal_gate": part.temporal_gate.mean()})
    metrics = pd.DataFrame(rows)
    summary = aggregate(metrics)
    for name, frame in (("outer_oof_predictions", predictions), ("fit_metrics", fit_metrics),
                        ("seed_metrics", metrics), ("summary", summary)):
        _atomic_csv(output / "reports" / f"{name}.csv", frame)
    old_path = Path(cfg["previous_run"]) / "reports/seed_metrics.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[(old.arm == "selected") & old.seed.isin(cfg["formal_seeds"])].drop(columns="arm")
        matched = metrics.merge(old[["depth", "seed", "auroc", "logloss", "brier", "gap"]],
                                on=["depth", "seed"], suffixes=("_v4", "_v3"), validate="many_to_one")
        if len(matched) != len(metrics):
            raise ValueError("Previous-run seed comparison is incomplete")
        for name in ("auroc", "logloss", "brier", "gap"):
            matched[f"delta_{name}"] = matched[f"{name}_v4"] - matched[f"{name}_v3"]
        _atomic_csv(output / "reports/matched_v3_comparison.csv", matched)
    selected = [r for r in references if r["arm"] == "selected"]
    expected_keys = {(d, s, f["fold"]) for d in cfg["depths"] for s in cfg["formal_seeds"] for f in folds}
    for arm in (*CANDIDATES, "selected"):
        subset = [r for r in references if r["arm"] == arm]
        if len(subset) != len(expected_keys) or {(r["depth"], r["seed"], r["outer"]) for r in subset} != expected_keys:
            raise ValueError("Model reference coverage changed")
    unchanged(output / "frozen_models.json", {"schema": SCHEMA, "models": selected,
        "comparison_models": [r for r in references if r["arm"] != "selected"],
        "factory": "src.single_phase_gated_models.build_model", "holdout_used_for_selection": False,
        "note": "Load with checkpoint training depth, then predict at reference max_tp; shared paths intentionally repeat."})
    payload = {"schema": SCHEMA, "branch": cfg["branch"], "created_utc": now(),
        "development_patients": len(expected), "seeds": cfg["formal_seeds"],
        "distinct_outer_models": len(cached), "selected_model_references": len(selected),
        "selected_unique_checkpoints": len({r["path"] for r in selected}),
        "selection_counts": dict(Counter(choices.values())), "summary": summary.to_dict("records"),
        "holdout_evaluated": False, "generated_evaluated": False, "automatic_thirty_seed_extension": False}
    write_json(output / "reports/summary.json", payload)
    lines = [f"# {cfg['branch']}: gated pCR v4 development pilot", "",
        "Five fixed seeds (42-46), five outer folds, three inner folds and tuning seeds42/43.",
        "Frozen original single-phase Pillar features and patient splits. No historical holdout inference.",
        "All candidates retain available visits anchored at T0; registered compact baseline therefore differs",
        "from the older contiguous baseline where intermediate visits are absent.",
        "Plain compact versus wide64 isolates capacity. Gated candidates also add a frozen T0 reference,",
        "patient-specific bounded correction, gate penalty and follow-up dropout. Shared64 additionally",
        "fits four patient-normalized prefix losses in one model. These are recorded ablations, not proven gains.", "",
        "| Arm | Window | OOF AUC | Log loss | Train AUC | Fold AUC | Gap |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.arm} | {opt.DEPTH_NAMES[row.depth]} | {row.auroc_mean:.5f} | {row.logloss_mean:.5f} | {row.train_auroc_mean:.5f} | {row.fold_validation_auroc_mean:.5f} | {row.gap_mean:.5f} |")
    lines += ["", "T0 references are fitted within each current training partition and remain frozen in gated models.",
        "Shared models use one fixed training duration chosen by inner mean prefix log loss.",
        "Selection uses inner AUROC within0.005 of the best, followed by minimum log loss; outer duration is fixed.",
        "Seed SD measures initialization variation, not patient uncertainty. No automatic extension to thirty seeds.",
        "Prior crops and v1-v3 consumers remain unchanged; no encoder fine-tuning or generated-image evaluation.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return payload

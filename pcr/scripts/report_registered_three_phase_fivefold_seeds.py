"""Report each seed's five outer-validation folds, keeping all seeds separate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as study
from src.first_post_pcr_data import identity, now, read_json, write_json


def report(cfg):
    output = Path(cfg["output_dir"])
    if not read_json(output / "evaluation/independent_audit.json")["passed"]:
        raise ValueError("A completed, audited study is required")
    frozen = read_json(output / "FROZEN_MODELS.json")["artifacts"]
    study.verify_inventory(frozen)
    contract = read_json(output / "contract.json")
    study.verify_inventory(contract["runtime"])
    if identity(cfg["config_path"]) != contract["config_identity"]:
        raise ValueError("The frozen study configuration changed")
    folds = {f["fold"]: f for f in read_json(output / "nested_folds.json")}
    selections = {(r["depth"], r["outer"]): r["selected"] for r in
                  read_json(output / "selection" / f"{study.STUDY}.json")["selections"]}
    refs = read_json(output / "selection" / f"{study.STUDY}_outer_models.json")
    refs = [r for r in refs if r["arm"] == "selected"]
    expected = {(d, seed, fold) for d in cfg["depths"] for seed in cfg["formal_seeds"] for fold in folds}
    actual = {(r["depth"], r["seed"], r["outer"]) for r in refs}
    if len(refs) != len(expected) or actual != expected:
        raise ValueError("Missing or repeated selected model reference")
    metrics, architectures, signatures = [], {}, {}
    max_error = 0.0
    for ref in refs:
        path = Path(ref["path"])
        summary = read_json(path / "COMPLETE.json")
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        task = checkpoint["task"]
        if task != summary["task"]:
            raise ValueError("Checkpoint/task mismatch")
        effective = task["effective"]
        choice = selections[(ref["depth"], ref["outer"])]
        if (task["candidate"] != choice["candidate"]
                or task["fixed_epochs"] != choice["fixed_epochs"]
                or task["depth"] != ref["depth"] or task["seed"] != ref["seed"]
                or task["outer"] != ref["outer"]):
            raise ValueError("Actual model differs from frozen selection")
        if (task["train_ids"] != folds[ref["outer"]]["train_ids"]
                or task["val_ids"] != folds[ref["outer"]]["val_ids"]):
            raise ValueError("Actual model differs from the original outer fold")
        if effective["kind"] != "tdn" or effective.get("feature_transform", "identity") != "identity":
            raise ValueError("Unexpected selected architecture")
        signature = {"effective": effective, "candidate": task["candidate"],
                     "fixed_epochs": task["fixed_epochs"]}
        key = (ref["depth"], ref["outer"])
        if key in signatures and signatures[key] != signature:
            raise ValueError("Model recipe changes between seeds of the same fold")
        signatures[key] = signature
        state = checkpoint["model_state"]
        parameter_count = sum(t.numel() for t in state.values())
        if (tuple(state["proj.net.0.weight"].shape) != (effective["proj_dim"], 1152)
                or tuple(state["head.weight"].shape) != (1, effective["sig_dim"])
                or parameter_count != summary["parameters"]):
            raise ValueError("Saved weights disagree with the reported architecture")
        architectures[key] = {
            "depth": ref["depth"], "window": opt.DEPTH_NAMES[ref["depth"]],
            "fold": ref["outer"] + 1, "outer_id": ref["outer"],
            "candidate": task["candidate"], "fixed_epochs": task["fixed_epochs"],
            "input_dim": effective["input_dim"], "projection_dim": effective["proj_dim"],
            "signature_dim": effective["sig_dim"], "layers": effective["n_layers"],
            "attention_heads": effective["n_heads"], "clinical_prior": effective["use_prior"],
            "clinical_token": effective.get("use_clinical_token", True),
            "dropout": effective["dropout"], "embedding_dropout": effective["embedding_dropout"],
            "residual_logit_limit": effective.get("residual_logit_limit"),
            "residual_l2_weight": effective["residual_l2_weight"],
            "positive_class_weight": effective.get("positive_class_weight", "balanced_fold_train"),
            "inner_epoch_selection": effective["epoch_selection"],
            "tdn_parameters": parameter_count,
        }
        predictions = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
        row = {"window": opt.DEPTH_NAMES[ref["depth"]], "depth": ref["depth"],
               "seed": ref["seed"], "fold": ref["outer"] + 1, "outer_id": ref["outer"],
               "candidate": task["candidate"], "epochs": task["fixed_epochs"]}
        for role, ids in (("train", task["train_ids"]), ("validation", task["val_ids"])):
            part = predictions[predictions.role == role]
            if len(part) != len(ids) or set(part.patient_id) != set(ids) or not part.patient_id.is_unique:
                raise ValueError("Saved patient predictions do not match their fold")
            y, p = part.label.to_numpy(), part.probability.to_numpy()
            recomputed = {"auroc": roc_auc_score(y, p), "logloss": log_loss(y, p, labels=[0, 1]),
                          "brier": brier_score_loss(y, p)}
            for metric, value in recomputed.items():
                max_error = max(max_error, abs(value - summary[role][metric]))
                row[f"{role}_{metric}"] = float(value)
            row[f"{role}_patients"] = len(part)
        row["auroc_gap"] = row["train_auroc"] - row["validation_auroc"]
        metrics.append(row)
    if max_error > 1e-10:
        raise ValueError("Saved metrics differ from independent prediction scoring")
    details = pd.DataFrame(metrics).sort_values(["depth", "seed", "fold"]).reset_index(drop=True)
    aggregate = []
    for (depth, seed), part in details.groupby(["depth", "seed"]):
        if len(part) != 5 or set(part.fold) != set(range(1, 6)):
            raise ValueError("Each seed/window must contain exactly five folds")
        row = {"window": opt.DEPTH_NAMES[depth], "depth": int(depth), "seed": int(seed), "folds": 5}
        for metric in ("validation_auroc", "validation_logloss", "validation_brier",
                       "train_auroc", "train_logloss", "train_brier", "auroc_gap"):
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_fold_sd"] = float(part[metric].std(ddof=1))
        for r in part.itertuples():
            row[f"fold_{r.fold}_validation_auroc"] = r.validation_auroc
        aggregate.append(row)
    summary = pd.DataFrame(aggregate).sort_values(["depth", "seed"]).reset_index(drop=True)
    architecture = pd.DataFrame(architectures.values()).sort_values(["depth", "fold"])
    destination = output / "evaluation/per_seed_fivefold"
    _atomic_csv(destination / "fold_metrics.csv", details)
    _atomic_csv(destination / "seed_fivefold_summary.csv", summary)
    _atomic_csv(destination / "selected_architectures.csv", architecture)
    lines = ["# Selected Models: Five-Fold Results for Each Seed", "",
        "Population:764 development patients with real three-phase ROI32 inputs.",
        "Each entry averages five outer-validation AUROCs with equal fold weights.",
        "SD is the sample standard deviation across those five folds (ddof=1), not across seeds.",
        "No averaging across classifier seeds. Pooled OOF AUROC and102-patient five-model ensemble AUROC are different statistics.",
        "Candidates and fixed training durations were selected using inner data only. The same fold recipe is used for every classifier seed.", "",
        "## Network", "",
        "Three-phase ROI32 -> original physical adapter -> frozen Pillar ->1152-D visit embeddings -> learned projection/signature -> elapsed-time encoding ->1-layer/4-head TDN -> residual logit + fixed fold-trained clinical prior -> sigmoid.",
        "All selected models retain raw Pillar embeddings; PCA and clinical-only candidates were not selected. Clinical prior is always present, even when the extra clinical token is disabled.", "",
        "| Window | Fold | Candidate | Projection/signature | Clinical token | Parameters | Fixed epochs |",
        "|---|---:|---|---|---|---:|---:|"]
    for r in architecture.itertuples():
        lines.append(f"| {r.window} | {r.fold} | {r.candidate} | {r.projection_dim}/{r.signature_dim} | {r.clinical_token} | {r.tdn_parameters} | {r.fixed_epochs} |")
    lines.extend(["", "## Mean AUROC and Fold SD", "",
                  "| Seed | T0 | T0-T1 | T0-T2 | T0-T3 |", "|---:|---:|---:|---:|---:|"])
    for seed in cfg["formal_seeds"]:
        part = summary[summary.seed == seed].set_index("depth")
        cells = [f"{part.loc[d, 'validation_auroc_mean']:.5f} +/- {part.loc[d, 'validation_auroc_fold_sd']:.5f}" for d in cfg["depths"]]
        lines.append("| " + str(seed) + " | " + " | ".join(cells) + " |")
    for depth in cfg["depths"]:
        lines.extend(["", f"## {opt.DEPTH_NAMES[depth]} Fold Details", "",
                      "| Seed | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean | Fold SD |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for _, row in summary[summary.depth == depth].iterrows():
            values = [row[f"fold_{f}_validation_auroc"] for f in range(1, 6)]
            values += [row.validation_auroc_mean, row.validation_auroc_fold_sd]
            lines.append("| " + str(int(row.seed)) + " | " + " | ".join(f"{v:.5f}" for v in values) + " |")
    _atomic_text(destination / "README.md", "\n".join(lines) + "\n")
    reloaded = pd.read_csv(destination / "seed_fivefold_summary.csv")
    fold_columns = [f"fold_{f}_validation_auroc" for f in range(1, 6)]
    means = reloaded[fold_columns].to_numpy().mean(axis=1)
    deviations = reloaded[fold_columns].to_numpy().std(axis=1, ddof=1)
    error = max(float(abs(means - reloaded.validation_auroc_mean).max()),
                float(abs(deviations - reloaded.validation_auroc_fold_sd).max()))
    if error > 1e-12:
        raise ValueError("Exported five-fold means or deviations do not replay")
    study.verify_inventory(frozen)
    write_json(destination / "verification.json", {
        "passed": True, "checked_utc": now(), "selected_fit_records": len(details),
        "separate_seed_window_summaries": len(summary), "fold_recipes": len(architecture),
        "fold_count_per_summary": 5, "across_seed_averaging": False,
        "prediction_metric_max_error": max_error, "exported_aggregation_max_error": error,
        "same_fold_recipe_across_all_seeds": True, "frozen_models_and_original_results_unchanged": True,
        "training_or_inference_performed": False,
    })
    print("\n".join(lines[lines.index("## Mean AUROC and Fold SD"):lines.index("## T0 Fold Details")]))
    print(f"Verified {len(details)} fits, {len(summary)} separate seed/window averages and {len(architecture)} recipes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_pcr_optimization_v2.yaml")
    report(study.load_config(parser.parse_args().config))

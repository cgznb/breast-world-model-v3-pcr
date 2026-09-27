"""Isolated cached-feature study with nested selection and four-source reporting."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src.first_post_pcr_data import disk_gate, identity, now, read_json, repo_path, save_tensor, write_json
from src.single_phase_anchor_training import (
    STATISTICAL, fit_neural, fit_probe, load_data, model_from_bundle, predict_bundle,
    preprocessing, task_complete, task_path,
)
from src.single_phase_fmbcmri_data import BRANCHES, unchanged_json
from src.single_phase_fmbcmri_training import METRICS, metrics
from src.single_phase_response_data import load_config as source_config, load_data as source_data, subset
from src.single_phase_response_reporting import WINDOWS, legacy_comparison, metric_rows, summarize, validate_prediction_frame
from src.single_phase_response_training import prediction_frame

SCHEMA = "single_phase_anchor_pcr_v3"


def load_config(path, branch):
    path = repo_path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema"] != SCHEMA or branch not in BRANCHES or cfg["gpu"] != 0 or cfg["device"] != "cpu":
        raise ValueError("Unexpected protocol, branch or compute placement")
    cfg.update(branch=branch, config_path=str(path), study_dir=str(repo_path(cfg["output_dir"])))
    cfg["output_dir"] = str(Path(cfg["study_dir"]) / branch)
    cfg["source_dir"] = str(repo_path(cfg["source_run"]) / branch)
    original = source_config(cfg["source_config"], branch)
    if Path(original["output_dir"]).resolve() != Path(cfg["source_dir"]).resolve():
        raise ValueError("Feature source/config disagree")
    cfg["source_cfg"] = original
    cfg["baseline_run"] = original["baseline_run"]
    return cfg


def progress(cfg, stage, **details):
    value = dict(schema=SCHEMA, branch=cfg["branch"], pid=os.getpid(), updated_utc=now(),
                 stage=stage, device="cpu", gpu_allowed=0, **details)
    write_json(Path(cfg["output_dir"]) / "progress.json", value)
    import json
    print(json.dumps(value), flush=True)


def prepare(cfg):
    output, source = Path(cfg["output_dir"]), Path(cfg["source_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    if not (source / "COMPLETE.json").exists():
        raise ValueError("The completed source feature study is required")
    study = read_json(source / "cohort.json")
    original_files = ("contract.json", "cohort.json", "metadata.csv", "folds.json", "nested_folds.json",
                      "TRAIN_FEATURES_COMPLETE.json", "COMPLETE.json")
    contract = dict(config={k: v for k, v in cfg.items() if k != "source_cfg"},
                    source_files={name: identity(source / name) for name in original_files})
    unchanged_json(output / "contract.json", contract)
    for name in ("folds.json", "nested_folds.json"):
        unchanged_json(output / name, read_json(source / name))
    unchanged_json(output / "split.json", study["split"])
    if (output / "PREPARED.json").exists():
        saved = read_json(output / "PREPARED.json")
        for path, stamp in saved["data_files"].items():
            if identity(path) != stamp:
                raise ValueError("A prepared feature pack changed")
        return study
    references, packed = {}, {}
    for roi in ("context", "tumor"):
        data = source_data(cfg["source_cfg"], study, roi, "train")
        for paths in data.pop("token_paths"):
            for path in paths:
                if path is not None:
                    references[path] = identity(path)
        if not np.isfinite(data["embs"]).all() or data["embs"].shape[-1] != 768:
            raise ValueError("Invalid frozen encoder features")
        norms = np.linalg.norm(data["embs"], axis=-1)
        if np.any(np.abs(norms[data["masks"] > 0] - 1) > 1e-5):
            raise ValueError("Observed visit features must be normalized and nonzero")
        path = output / "data" / f"train_{roi}.pt"
        save_tensor(path, data)
        packed[str(path)] = identity(path)
        if roi == "context":
            first = data
        else:
            if first["pids"] != data["pids"]:
                raise ValueError("ROI patient order differs")
            for key in ("labels", "masks", "days", "clinical"):
                np.testing.assert_allclose(first[key], data[key], rtol=0, atol=0, equal_nan=True)
    folds, nested = read_json(output / "folds.json"), read_json(output / "nested_folds.json")
    population, holdout = set(study["split"]["train"]), set(study["split"]["val"])
    covered = []
    if population & holdout or len(folds) != 5:
        raise ValueError("Invalid original partition")
    for outer, inner in zip(folds, nested):
        a, b = set(outer["train_ids"]), set(outer["val_ids"])
        if a & b or a | b != population or outer["fold"] != inner["fold"]:
            raise ValueError("Invalid outer fold")
        covered.extend(outer["val_ids"])
        inner_covered = []
        for item in inner["inner_folds"]:
            x, y = set(item["train_ids"]), set(item["val_ids"])
            if x & y or x | y != a:
                raise ValueError("An inner fold crosses the outer training partition")
            inner_covered.extend(item["val_ids"])
        if len(inner_covered) != len(a) or set(inner_covered) != a:
            raise ValueError("Inner validation does not cover outer training exactly once")
    if len(covered) != len(population) or set(covered) != population:
        raise ValueError("Outer OOF coverage is incomplete")
    write_json(output / "source_features.json", references)
    write_json(output / "PREPARED.json", dict(complete=True, data_files=packed, development=len(population),
               holdout=len(holdout), input_dim=768, source_feature_files=len(references), holdout_pixels_read=False))
    progress(cfg, "prepared", development=len(population), holdout=len(holdout), feature_files=len(references))
    return study


def inner_tasks(cfg, names=None):
    tasks = []
    for outer in read_json(Path(cfg["output_dir"]) / "nested_folds.json"):
        for name, arm in cfg["arms"].items():
            if names is not None and name not in names:
                continue
            for inner in outer["inner_folds"]:
                for seed in cfg["tuning_seeds"] if arm["kind"] not in STATISTICAL else cfg["tuning_seeds"][:1]:
                    tasks.append(dict(stage="inner", arm=name, fold=outer["fold"], inner_fold=inner["inner"], seed=seed,
                                      train_ids=inner["train_ids"], val_ids=inner["val_ids"], fixed_epochs=None))
    return tasks


def formal_tasks(cfg, names=None, write_selection=True):
    tasks, selection = [], []
    inner = inner_tasks(cfg, names)
    for outer in read_json(Path(cfg["output_dir"]) / "folds.json"):
        for name, arm in cfg["arms"].items():
            if names is not None and name not in names:
                continue
            selected = []
            if arm["kind"] not in STATISTICAL:
                for task in inner:
                    if task["fold"] == outer["fold"] and task["arm"] == name:
                        if not task_complete(cfg, task):
                            raise ValueError("All six inner fits are required before fixing refit duration")
                        selected.append(read_json(task_path(cfg, task) / "summary.json")["selected_epochs"])
                if len(selected) != 6:
                    raise ValueError("Expected six inner duration selections")
            epochs = math.ceil(float(np.median(selected))) if selected else 0
            selection.append(dict(arm=name, fold=outer["fold"], inner_epochs=selected, selected_epochs=epochs))
            # Zero-epoch neural refits remain present for every declared formal seed.
            for seed in cfg["seeds"] if arm["kind"] not in STATISTICAL else cfg["seeds"][:1]:
                tasks.append(dict(stage="formal", arm=name, fold=outer["fold"], seed=seed,
                                  train_ids=outer["train_ids"], val_ids=outer["val_ids"], fixed_epochs=epochs))
    if write_selection:
        path = Path(cfg["output_dir"]) / "epoch_selection.json"
        previous = read_json(path) if path.exists() else []
        merged = {(r["arm"], r["fold"]): r for r in previous + selection}
        write_json(path, list(merged.values()))
    return tasks


def choose_candidate(losses, candidates, min_improvement):
    if not all(np.isfinite(losses[name]) for name in candidates):
        raise ValueError("Candidate selection needs finite inner losses")
    best = min(losses.values())
    return next(name for name in candidates if losses[name] <= best + min_improvement)


def select_inner(cfg):
    candidates = cfg["selection"]["candidates"]
    frames = []
    for task in inner_tasks(cfg, candidates):
        if not task_complete(cfg, task):
            raise ValueError("Inner candidate selection requires completed inner predictions")
        part = pd.read_csv(task_path(cfg, task) / "predictions.csv", dtype={"patient_id": str})
        for seed in cfg["tuning_seeds"] if cfg["arms"][task["arm"]]["kind"] in STATISTICAL else [task["seed"]]:
            frames.append(part.assign(arm=task["arm"], fold=task["fold"], seed=seed))
    data = pd.concat(frames, ignore_index=True)
    choices, rows = [], []
    outer_folds = read_json(Path(cfg["output_dir"]) / "folds.json")
    for outer in outer_folds:
        fold = outer["fold"]
        expected = set(outer["train_ids"])
        for (_, _, _), group in data[data.fold == fold].groupby(["arm", "prefix", "seed"]):
            if set(group.patient_id) != expected or group.patient_id.duplicated().any():
                raise ValueError("Inner selection predictions have invalid patient coverage")
        for prefix in range(1, 5):
            losses = {}
            for name in candidates:
                group = data[(data.fold == fold) & (data.prefix == prefix) & (data.arm == name) & data.complete]
                values = [metrics(part.label, part.probability)["logloss"] for _, part in group.groupby("seed")]
                losses[name] = float(np.mean(values))
                rows.append(dict(fold=fold, prefix=prefix, arm=name, validation_logloss=losses[name],
                                 patients=group.patient_id.nunique(), seeds=len(values)))
            chosen = choose_candidate(losses, candidates, cfg["selection"]["min_improvement"])
            choices.append(dict(fold=fold, prefix=prefix, selected=chosen, inner_logloss=losses[chosen],
                                clinical_logloss=losses["clinical"], source="outer_training_inner_oof_only"))
    write_json(Path(cfg["output_dir"]) / "selection.json", choices)
    _atomic_csv(Path(cfg["output_dir"]) / "inner_candidate_scores.csv", pd.DataFrame(rows))
    return choices


def add_selected(frame, choices, candidates):
    lookup = {(int(r["fold"]), int(r["prefix"])): r["selected"] for r in choices}
    base = frame[frame.arm.isin(candidates)].copy()
    wanted = ["clinical" if int(available) == 0 else lookup[(int(fold), min(int(prefix), int(available)))]
              for fold, prefix, available in zip(base.fold, base.prefix, base.available_prefix)]
    result = base.loc[base.arm.to_numpy() == np.asarray(wanted)].copy()
    result["selected_arm"] = result.arm
    result["arm"] = "inner_selected"
    return pd.concat([frame, result], ignore_index=True)


def collect_oof(cfg, choices):
    frames = []
    for task in formal_tasks(cfg, write_selection=False):
        if not task_complete(cfg, task):
            raise ValueError("OOF reporting requires all fixed refits")
        part = pd.read_csv(task_path(cfg, task) / "predictions.csv", dtype={"patient_id": str})
        for seed in cfg["seeds"] if cfg["arms"][task["arm"]]["kind"] in STATISTICAL else [task["seed"]]:
            frames.append(part.assign(arm=task["arm"], fold=task["fold"], seed=seed, source="real", population="oof"))
    return add_selected(pd.concat(frames, ignore_index=True), choices, cfg["selection"]["candidates"])


def freeze(cfg, choices):
    references = []
    for task in formal_tasks(cfg, write_selection=False):
        if not task_complete(cfg, task):
            raise ValueError("Cannot freeze an incomplete experiment")
        path = task_path(cfg, task) / "model.pt"
        references.append(dict(task=task, path=str(path), identity=identity(path)))
    result = dict(models=references, choices=choices, holdout_used_for_selection=False, frozen_utc=now())
    path = Path(cfg["output_dir"]) / "FROZEN_MODELS.json"
    if path.exists():
        previous = read_json(path)
        if previous["models"] != references or previous["choices"] != choices:
            raise ValueError("Frozen refits or inner selections changed")
        return previous
    write_json(path, result)
    return result


def evaluate_holdout(cfg, study):
    frozen = read_json(Path(cfg["output_dir"]) / "FROZEN_MODELS.json")
    data_cache, frames, errors = {}, [], []
    for number, reference in enumerate(frozen["models"]):
        task, path = reference["task"], reference["path"]
        if identity(path) != reference["identity"]:
            raise ValueError("Frozen classifier changed")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        roi = cfg["arms"][task["arm"]]["roi"]
        t0 = []
        for source in ("real", "symm", "bifm", "copy"):
            if (roi, source) not in data_cache:
                # This reader opens saved feature tensors and metadata, never MRI pixels.
                data_cache[(roi, source)] = source_data(cfg["source_cfg"], study, roi, "val", source)
            data = data_cache[(roi, source)]
            probability = predict_bundle(bundle, data)
            t0.append(probability[:, 0])
            for seed in cfg["seeds"] if bundle["kind"] in STATISTICAL else [task["seed"]]:
                frames.append(prediction_frame(data, probability, "holdout").assign(
                    arm=task["arm"], fold=task["fold"], seed=seed, source=source, population="holdout"))
        error = max(float(np.max(np.abs(t0[0] - p))) for p in t0[1:])
        if error > 1e-6:
            raise ValueError("Four-source T0 predictions differ")
        errors.append(error)
        if number % 25 == 0:
            progress(cfg, "evaluating_fixed_holdout", completed=number + 1, total=len(frozen["models"]))
    frame = add_selected(pd.concat(frames, ignore_index=True), frozen["choices"], cfg["selection"]["candidates"])
    _atomic_csv(Path(cfg["output_dir"]) / "evaluation/holdout_fold_predictions.csv", frame)
    keys = ["arm", "source", "population", "seed", "prefix", "patient_id", "label", "complete", "available_prefix"]
    if not (frame.groupby(keys).fold.nunique() == 5).all():
        raise ValueError("A holdout seed lacks five distinct outer folds")
    result = frame.groupby(keys, as_index=False).probability.mean()
    write_json(Path(cfg["output_dir"]) / "evaluation/t0_invariance.json",
               dict(passed=True, models=len(errors), maximum_error=max(errors)))
    return result


def report(cfg, study, oof, holdout=None):
    root = Path(cfg["output_dir"]) / ("evaluation" if holdout is not None else "development_review")
    frame = pd.concat([oof, holdout], ignore_index=True) if holdout is not None else oof
    validate_prediction_frame(frame, {"oof": study["split"]["train"], "holdout": study["split"]["val"]}, cfg["seeds"])
    _atomic_csv(root / "predictions.csv", frame)
    saved = pd.read_csv(root / "predictions.csv", dtype={"patient_id": str}, low_memory=False)
    rows, replay = metric_rows(frame), metric_rows(saved)
    error = float(np.nanmax(np.abs(rows[list(METRICS)].to_numpy() - replay[list(METRICS)].to_numpy())))
    if error > 1e-12:
        raise ValueError("Saved predictions do not reproduce reported metrics")
    summary = summarize(replay)
    _atomic_csv(root / "metrics_per_seed.csv", replay)
    _atomic_csv(root / "summary.csv", summary)
    legacy_comparison(cfg, saved, root)
    previous = pd.read_csv(Path(cfg["source_dir"]) / "evaluation/summary.csv")
    previous = previous[(previous.source == "real") & (previous.subset == "all_prefix") &
                        previous.arm.isin(["clinical", "fusion_gru_context", "fusion_gru_context_lora"])]
    _atomic_csv(root / "v2_reference.csv", previous)
    write_json(root / "verification.json", dict(passed=True, metric_replay_max_error=error,
               holdout_included=holdout is not None, seed_count=len(cfg["seeds"]), bootstrap_samples=0))
    lines = [f"# Anchored single-phase pCR: {cfg['branch']}", "",
             "Frozen FM-BCMRI CLS768 and the exact v2 physical-ROI feature caches. No MRI/generator/encoder refitting.", "",
             "The clinical predictor is fitted only on the current training partition and remains frozen. T0 image residuals start at zero. Follow-up models embed the matching frozen T0 model, start at its predictions and use bounded visit-specific corrections. Epoch zero is eligible for inner selection; zero-epoch refits intentionally retain the anchor.", "",
             "PCA32 is fitted on training T0 visits for the baseline, and training observed visits for follow-up heads/probes. Image-only probes are fixed-C regularized logistic models. Temporal heads use patient-mean ordinary BCE over observed follow-ups plus a fixed residual penalty. Validation and reporting use unpenalized log loss.", "",
             "Five original outer folds; three inner folds and seeds42/43 select the ceiling median of six best epochs. Ten formal neural seeds42-51. Deterministic clinical/probe predictions repeat across seed rows. Each holdout seed first averages its five fold probabilities. SD is seed variability, not patient-sampling uncertainty. No bootstrap.", "",
             "inner_selected chooses clinical, T0, linear follow-up or gated follow-up only from the outer-training inner predictions, preferring the earlier-listed simpler model within0.0001 log loss. It uses the last observed contiguous prefix for missing visits. Choices never use outer OOF or fixed-holdout labels; fallback does not guarantee out-of-sample improvement.", "",
             "Historical fixed holdouts were previously used upstream and inspected when designing these experiments. Native ROIs/archives retain retrospective visit-localization limitations. This is an internal comparison, not an independent external test. No monotonic individual risk or AUROC is imposed.", "",
             "## Real MRI AUROC: mean +/- seed SD", "",
             "| Population | Arm | T0 | T0-T1 | T0-T2 | T0-T3 |", "|---|---|---:|---:|---:|---:|"]
    selected = summary[(summary.source == "real") & (summary.subset == "all_prefix")]
    for (population, arm), part in selected.groupby(["population", "arm"]):
        part = part.set_index("prefix")
        lines.append(f"| {population} | {arm} | " + " | ".join(
            f"{part.loc[t, 'auroc_mean']:.4f} +/- {part.loc[t, 'auroc_std']:.4f}" for t in range(1, 5)) + " |")
    lines += ["", "All four-source AUROC/AUPRC/log loss/Brier metrics, per-seed values and complete-window/same-complete-four cohorts are in metrics_per_seed.csv and summary.csv. Matched v1 changes are in comparison_to_v1_per_seed.csv. The prespecified v2 controls are in v2_reference.csv."]
    _atomic_text(root / "report.md", "\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    populations = sorted(selected.population.unique())
    figure, axes = plt.subplots(1, len(populations), figsize=(7 * len(populations), 4.6), squeeze=False)
    display = ["clinical", "anchor_t0_context", "anchor_linear_context", "anchor_gated_context", "inner_selected"]
    for axis, population in zip(axes[0], populations):
        for arm in display:
            part = selected[(selected.population == population) & (selected.arm == arm)].sort_values("prefix")
            axis.errorbar(part.prefix, part.auroc_mean, yerr=part.auroc_std, marker="o", capsize=2, label=arm)
        axis.set_xticks(range(1, 5), WINDOWS.values())
        axis.set_title(f"{cfg['branch']}: {population}")
        axis.set_ylabel("AUROC; error bars: seed SD")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "real_prefix_comparison.png", dpi=170)
    figure.savefig(root / "real_prefix_comparison.pdf")
    plt.close(figure)
    return summary


def smoke(cfg):
    root = Path(cfg["output_dir"]) / "smoke"
    if (root / "COMPLETE.json").exists():
        return read_json(root / "COMPLETE.json")
    checks = []
    settings = dict(cfg["training"], epochs=2, patience=2, batch_size=4)
    for roi in ("context", "tumor"):
        data = load_data(cfg, roi)
        ids = []
        for label in (0, 1):
            ids.extend([p for i, p in enumerate(data["pids"]) if data["labels"][i] == label and data["masks"][i].sum() == 4][:4])
        train, val = subset(data, ids[:3] + ids[4:7]), subset(data, [ids[3], ids[7]])
        task = dict(stage="smoke", arm=f"anchor_t0_{roi}", fold=0)
        prep_t0 = preprocessing(cfg, task, train)
        base = fit_neural(train, None, settings, "t0", prep_t0, 142, root / roi / "base", fixed_epochs=2)
        spec = dict(statistics=base["statistics"], state=model_from_bundle(base).base.state_dict())
        for kind in ("t0", "linear", "gated"):
            prep = prep_t0 if kind == "t0" else preprocessing(cfg, dict(task, arm=f"anchor_gated_{roi}"), train)
            options = dict(base_spec=None if kind == "t0" else spec, fixed_epochs=2)
            first = fit_neural(train, None, settings, kind, prep, 142, root / roi / kind / "continuous", **options)
            try:
                fit_neural(train, None, settings, kind, prep, 142, root / roi / kind / "resume", pause_after=1, **options)
            except InterruptedError:
                pass
            recovered = fit_neural(train, None, settings, kind, prep, 142, root / roi / kind / "resume", **options)
            if first["history"] != recovered["history"] or any(not torch.equal(v, recovered["state"][k]) for k, v in first["state"].items()):
                raise ValueError("Real-feature recovery is not exact")
            path = root / roi / kind / "portable.pt"
            save_tensor(path, first)
            replay = predict_bundle(torch.load(path, map_location="cpu", weights_only=False), val)
            error = float(np.max(np.abs(predict_bundle(first, val) - replay)))
            if error > 1e-6:
                raise ValueError("Real-feature portable replay failed")
            checks.append(dict(roi=roi, kind=kind, exact_recovery=True, reload_error=error,
                               gradient_audit=first["gradient_audit"]))
    result = dict(passed=True, checks=checks, gpu_used=False, holdout_used=False)
    write_json(root / "COMPLETE.json", result)
    progress(cfg, "smoke_passed", checks=len(checks))
    return result

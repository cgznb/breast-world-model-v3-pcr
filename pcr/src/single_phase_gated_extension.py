"""Extend completed v4 seeds without modifying their fits or inner choices."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src import first_post_optimization as opt
from src import single_phase_gated_training as train
from src.first_post_pcr_data import identity, read_json, repo_path, write_json
from src.single_phase_gated_reporting import aggregate, report_development
from src.single_phase_repeat_retraining import unchanged

SCHEMA = "single_phase_gated_pcr_v4_seed_extension"


def load_config(path, branch):
    path = repo_path(path).resolve()
    master = yaml.safe_load(path.read_text())
    if (set(master) != {"schema", "parent_config", "output_dir", "formal_seeds"}
            or master["schema"] != SCHEMA or master["formal_seeds"] != list(range(42, 72))):
        raise ValueError("The extension changes only the output directory and seeds42-71")
    parent = train.load_config(master["parent_config"], branch)
    output = (repo_path(master["output_dir"]) / branch).resolve()
    source = Path(parent["output_dir"]).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Extension output must be separate from the completed pilot")
    return {**parent, "output_dir": str(output), "config_path": str(path),
        "parent_config_path": parent["config_path"], "parent_run": str(source),
        "initial_seeds": parent["formal_seeds"], "formal_seeds": master["formal_seeds"],
        "additional_seeds": [s for s in master["formal_seeds"] if s not in parent["formal_seeds"]]}


def check_identities(root, recorded):
    for relative, saved in recorded.items():
        if identity(Path(root) / relative) != saved:
            raise ValueError(f"Frozen extension input changed: {relative}")


def reuse_parent_models(cfg):
    source, output = Path(cfg["parent_run"]), Path(cfg["output_dir"])
    frozen = read_json(source / "frozen_models.json")
    folds = read_json(source / "nested_folds.json")
    expected = {(d, s, f["fold"]) for d in cfg["depths"] for s in cfg["initial_seeds"] for f in folds}
    references = frozen["models"] + frozen["comparison_models"]
    for arm in (*train.CANDIDATES, "selected"):
        rows = [r for r in references if r["arm"] == arm]
        if len(rows) != len(expected) or {(r["depth"], r["seed"], r["outer"]) for r in rows} != expected:
            raise ValueError("Parent model reference coverage changed")
    artifacts, destinations = {}, {}
    for reference in references:
        path = Path(reference["path"])
        if identity(path) != reference["identity"]:
            raise ValueError("Parent checkpoint identity changed")
        directory = path.parent
        if directory in destinations:
            continue
        result = read_json(directory / "COMPLETE.json")
        task = result["task"]
        if (task["stage"] != "outer" or task["seed"] not in cfg["initial_seeds"]
                or directory.resolve() != opt.task_path(source, task).resolve()
                or task["effective"] != train.effective_config(cfg, task["candidate"], task["depth"])
                or not train.task_complete(directory, task)):
            raise ValueError("Parent fit is incomplete or incompatible")
        destination = opt.task_path(output, task)
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != directory.resolve():
                raise ValueError("Inherited model destination is not the original fit link")
        destinations[directory] = destination
        for name in ("COMPLETE.json", *result["artifacts"]):
            file = directory / name
            artifacts[str(file.relative_to(source))] = identity(file)
    # Only individual completed seed directories are linked; new fits have local paths.
    for directory, destination in destinations.items():
        if not destination.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(directory.resolve(), target_is_directory=True)
    return artifacts


def prepare_extension(cfg):
    source, output = Path(cfg["parent_run"]), Path(cfg["output_dir"])
    parent_cfg = train.load_config(cfg["parent_config_path"], cfg["branch"])
    parent = read_json(source / "gated_contract.json")
    complete = read_json(source / "COMPLETE.json")
    if (parent["config"] != parent_cfg or complete["formal_seeds"] != cfg["initial_seeds"]
            or not complete["complete"] or complete["holdout_evaluated"] or complete["generated_evaluated"]):
        raise ValueError("A completed compatible development-only pilot is required")
    baseline = read_json(source / "contract.json")
    check_identities(repo_path("."), {**baseline["sources"], **parent["runtime"]})
    check_identities(cfg["baseline_run"], baseline["inputs"])
    check_identities(opt.source_directory(cfg, "physical_roi"), read_json(source / "physical_roi_feature_inventory.json"))
    selection = read_json(source / "selection/models.json")
    if not selection["selection_only_uses_inner_validation"] or selection["holdout_loaded"]:
        raise ValueError("Extension must inherit inner-only model selection")
    files = ["COMPLETE.json", "gated_contract.json", "contract.json", "nested_folds.json",
             "physical_roi_feature_inventory.json", "development_metadata.csv", "frozen_models.json",
             "selection/models.json", "selection/ranking.csv", "reports/seed_metrics.csv",
             "reports/outer_oof_predictions.csv", "reports/fit_metrics.csv", "reports/summary.csv"]
    artifacts = {name: identity(source / name) for name in files}
    contract_path = output / "extension_contract.json"
    if contract_path.exists():
        check_identities(source, read_json(contract_path)["parent_artifacts"])
    folds = train.prepare(cfg)
    if (folds != read_json(source / "nested_folds.json")
            or read_json(output / "physical_roi_feature_inventory.json") != read_json(source / "physical_roi_feature_inventory.json")):
        raise ValueError("Extension folds or frozen feature inventory differ from the pilot")
    artifacts.update(reuse_parent_models(cfg))
    runtime = ["src/single_phase_gated_extension.py", "scripts/extend_single_phase_gated_pcr.py",
               cfg["config_path"], cfg["parent_config_path"]]
    unchanged(contract_path, {"schema": SCHEMA, "config": cfg, "parent_artifacts": artifacts,
        "runtime": {name: identity(repo_path(name)) for name in runtime},
        "selection_reused_without_retuning": True, "user_requested_thirty_seeds": True,
        "holdout_evaluated": False, "generated_evaluated": False})
    unchanged(output / "selection/models.json", selection)
    _atomic_csv(output / "selection/ranking.csv", pd.DataFrame(selection["ranking"]))
    return folds, selection


def additional_tasks(cfg, folds, selection, gated=False):
    additional = copy.deepcopy(cfg)
    additional["formal_seeds"] = cfg["additional_seeds"]
    planned = train.tasks(additional, folds, "outer", gated=gated, ranking=selection["ranking"])
    output = Path(cfg["output_dir"]).resolve()
    for task in planned:
        path = opt.task_path(output, task)
        if (task["seed"] in cfg["initial_seeds"] or path.is_symlink()
                or output not in path.resolve().parents):
            raise ValueError("New training cannot write to an inherited pilot fit")
    return planned


def report_extension(cfg, selection):
    output, source = Path(cfg["output_dir"]), Path(cfg["parent_run"])
    result = report_development(cfg, selection)
    metrics = pd.read_csv(output / "reports/seed_metrics.csv")
    original = pd.read_csv(source / "reports/seed_metrics.csv")
    keys = ["arm", "depth", "seed"]
    inherited = metrics[metrics.seed.isin(cfg["initial_seeds"])]
    pd.testing.assert_frame_equal(original.sort_values(keys).reset_index(drop=True),
                                  inherited.sort_values(keys).reset_index(drop=True), check_exact=True)
    groups = []
    for name, seeds in (("initial", cfg["initial_seeds"]), ("additional", cfg["additional_seeds"]),
                        ("all", cfg["formal_seeds"])):
        table = aggregate(metrics[metrics.seed.isin(seeds)])
        groups.append(table.assign(seed_group=name, seed_count=len(seeds)))
    _atomic_csv(output / "reports/seed_group_summary.csv", pd.concat(groups, ignore_index=True))
    result.update(parent_run=str(source), initial_seeds=cfg["initial_seeds"], additional_seeds=cfg["additional_seeds"],
                  selection_reused_without_retuning=True, user_requested_thirty_seeds=True)
    write_json(output / "reports/summary.json", result)
    lines = [f"# {cfg['branch']}: gated pCR v4 seed extension", "",
        f"{len(cfg['formal_seeds'])} fixed seeds: {cfg['formal_seeds']}.",
        f"Retained seeds: {cfg['initial_seeds']}; additional seeds: {cfg['additional_seeds']}.",
        "Original nested folds, inner selections, fixed outer epochs and frozen Pillar features are retained.",
        "All four arms are evaluated. Original fits are linked read-only by convention; new fits use local paths.",
        "No holdout or generated-image inference. Additional seeds measure initialization variation, not more patients.", "",
        "| Arm | Window | OOF AUC | Log loss | Train AUC | Fold AUC | Gap |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregate(metrics).itertuples():
        lines.append(f"| {row.arm} | {opt.DEPTH_NAMES[row.depth]} | {row.auroc_mean:.5f} | {row.logloss_mean:.5f} | {row.train_auroc_mean:.5f} | {row.fold_validation_auroc_mean:.5f} | {row.gap_mean:.5f} |")
    lines += ["", "reports/seed_group_summary.csv separates initial, additional and all seeds.",
              "reports/seed_metrics.csv retains every seed/window; selection never uses holdout results.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))
    return result


def verify_parent_preserved(cfg):
    contract = read_json(Path(cfg["output_dir"]) / "extension_contract.json")
    check_identities(cfg["parent_run"], contract["parent_artifacts"])
    check_identities(repo_path("."), contract["runtime"])

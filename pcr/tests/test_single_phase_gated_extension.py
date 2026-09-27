import copy
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src import first_post_optimization as opt
from src import single_phase_gated_extension as extension
from src import single_phase_gated_training as train
from src.first_post_pcr_data import identity
from src.single_phase_gated_reporting import report_development
from test_first_post_optimization import synthetic_split
from test_single_phase_gated_pcr import base_config, gate_config


def test_extension_configuration_preserves_the_original_protocol(tmp_path):
    master = yaml.safe_load(Path("configs/single_phase_gated_pcr_v4_30seeds.yaml").read_text())
    master["output_dir"] = str(tmp_path / "extension")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(master))
    cfg = extension.load_config(path, "registered_dce0")
    original = train.load_config(master["parent_config"], "registered_dce0")
    assert cfg["formal_seeds"] == list(range(42, 72))
    assert cfg["initial_seeds"] == list(range(42, 47))
    assert cfg["additional_seeds"] == list(range(47, 72))
    for key in ("default_candidate", "candidates", "tuning_seeds", "depths", "baseline_run", "previous_run"):
        assert cfg[key] == original[key]
    assert not cfg["evaluate_holdout"]
    master["formal_seeds"] = list(range(42, 71))
    path.write_text(yaml.safe_dump(master))
    with pytest.raises(ValueError, match="seeds42-71"):
        extension.load_config(path, "registered_dce0")
    master["formal_seeds"] = list(range(42, 72))
    master["output_dir"] = str(Path(original["output_dir"]).parent)
    path.write_text(yaml.safe_dump(master))
    with pytest.raises(ValueError, match="separate"):
        extension.load_config(path, "registered_dce0")


def test_seed_extension_reuses_fits_and_preserves_predictions(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.backends.mha, "get_fastpath_enabled", lambda: False)
    raw = synthetic_split(48)
    raw["masks"][0, 0] = 0
    ids = raw["pids"]
    folds = [{"fold": 0, "train_ids": ids[:24], "val_ids": ids[24:]},
             {"fold": 1, "train_ids": ids[24:], "val_ids": ids[:24]}]
    parent, output = tmp_path / "parent", tmp_path / "extension"
    for directory in (parent, output):
        opt.write_json(directory / "nested_folds.json", folds)
    cfg = {"output_dir": str(parent), "baseline_run": str(parent), "previous_run": str(parent / "absent"),
        "branch": "unit", "depths": [1, 2, 3, 4], "tuning_seeds": [42], "formal_seeds": [42],
        "auroc_tolerance": .005, "default_candidate": base_config(),
        "candidates": {"baseline": {"kind": "tdn"}, "wide64": {"kind": "tdn", "n_layers": 2},
            "gated64": {"kind": "tdn", **{k: v for k, v in gate_config().items() if k != "t0_effective"}},
            "shared64": {"kind": "tdn", **{k: v for k, v in gate_config(True).items() if k != "t0_effective"}}}}
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)
    original_effective = opt.effective_config

    def small_effective(*args):
        result = original_effective(*args)
        result["input_dim"] = 12
        return result

    monkeypatch.setattr(opt, "effective_config", small_effective)
    ranking = [{"candidate": arm, "depth": depth, "outer": fold["fold"], "fixed_epochs": 2}
               for arm in train.CANDIDATES for depth in cfg["depths"] for fold in folds]
    selection = {"ranking": ranking, "selections": [
        {"depth": depth, "outer": fold["fold"], "selected": {"candidate": "gated64"}}
        for depth in cfg["depths"] for fold in folds]}
    for gated in (False, True):
        for task in train.tasks(cfg, folds, "outer", gated=gated, ranking=ranking):
            train.run_task(task, parent, parent, "cpu")
    report_development(cfg, selection)
    before = {str(path.relative_to(parent)): identity(path) for path in parent.rglob("*") if path.is_file()}
    cfg = {**cfg, "output_dir": str(output), "parent_run": str(parent), "formal_seeds": [42, 43],
           "initial_seeds": [42], "additional_seeds": [43]}
    retained = extension.reuse_parent_models(cfg)
    assert len(retained) == 24 * 4
    assert extension.reuse_parent_models(cfg) == retained
    all_tasks = []
    for gated in (False, True):
        planned = extension.additional_tasks(cfg, folds, selection, gated=gated)
        assert {task["seed"] for task in planned} == {43}
        for task in planned:
            train.run_task(task, output, parent, "cpu")
            assert train.run_task(task, output, parent, "cpu")["skipped"]
        all_tasks.extend(planned)
    assert len(all_tasks) == 24
    result = extension.report_extension(cfg, selection)
    assert result["distinct_outer_models"] == 48
    assert result["selected_model_references"] == 16
    assert result["selection_reused_without_retuning"] and not result["holdout_evaluated"]
    assert {str(path.relative_to(parent)): identity(path) for path in parent.rglob("*") if path.is_file()} == before
    original = pd.read_csv(parent / "reports/outer_oof_predictions.csv")
    extended = pd.read_csv(output / "reports/outer_oof_predictions.csv")
    pd.testing.assert_frame_equal(original, extended[extended.seed == 42].reset_index(drop=True), check_exact=True)
    groups = pd.read_csv(output / "reports/seed_group_summary.csv")
    assert groups.groupby("seed_group").seed_count.first().to_dict() == {"additional": 1, "all": 2, "initial": 1}
    refs = opt.read_json(output / "frozen_models.json")["comparison_models"]
    shared = [r for r in refs if r["arm"] == "shared64"]
    assert len(shared) == 16 and len({r["path"] for r in shared}) == 4
    for ref in refs:
        path = Path(ref["path"])
        assert (parent if ref["seed"] == 42 else output) in path.resolve().parents
    t0 = extended[(extended.depth == 1) & extended.arm.isin(["baseline", "gated64", "shared64"])]
    pivot = t0.pivot(index=["patient_id", "seed"], columns="arm", values="probability")
    pd.testing.assert_series_equal(pivot.baseline, pivot.gated64, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(pivot.baseline, pivot.shared64, check_names=False, check_exact=True)
    wrong = copy.deepcopy(cfg)
    wrong["additional_seeds"] = [42]
    with pytest.raises(ValueError, match="inherited"):
        extension.additional_tasks(wrong, folds, selection)
    inherited_model = next(output.glob("fits/outer/physical_roi/T0/baseline/outer_*/inner_*/seed_42"))
    target = inherited_model.resolve()
    inherited_model.unlink()
    inherited_model.mkdir()
    with pytest.raises(ValueError, match="destination"):
        extension.reuse_parent_models(cfg)
    inherited_model.rmdir()
    inherited_model.symlink_to(target, target_is_directory=True)
    opt.write_json(output / "extension_contract.json", {"parent_artifacts": before, "runtime": {}})
    extension.verify_parent_preserved(cfg)
    (parent / "reports/summary.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="input changed"):
        extension.verify_parent_preserved(cfg)

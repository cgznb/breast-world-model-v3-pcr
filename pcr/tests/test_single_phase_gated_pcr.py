import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import first_post_optimization as opt
from src import single_phase_gated_training as train
from src.single_phase_gated_models import GatedTemporal, build_model, drop_followups, predict
from src.single_phase_gated_reporting import report_development
from src.single_phase_temporal_models import window_split
from test_first_post_optimization import small_config, synthetic_split


@pytest.fixture(autouse=True)
def cpu_settings():
    torch.set_num_threads(1)
    original = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(original)


def base_config():
    return {**small_config(), "architecture": "tdn", "feature_transform": "identity", "visit_policy": "available",
            "residual_logit_limit": None, "epochs": 2, "patience": 2, "min_delta": .0001,
            "scheduler_t_max": 2, "scheduler_eta_min": 0.0}


def gate_config(shared=False):
    return {**base_config(), "architecture": "gated", "t0_effective": base_config(),
            "temporal_logit_limit": 1.0, "gate_initial_probability": .1, "gate_l1_weight": .01,
            "followup_dropout_probability": .15, "shared_prefix": shared}


def active_gate():
    model = build_model(gate_config(True), 4).eval()
    torch.nn.init.normal_(model.head.weight, std=.2)
    return model


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_shared_prefix_excludes_future_missing_and_unanchored_images(depth):
    raw = synthetic_split(12)
    raw["masks"][0, 1] = 0
    raw["masks"][1, 0] = 0
    data = opt.tensor_split(raw, np.zeros(12), "cpu")
    model = active_gate()
    expected = predict(model, data, depth)
    changed = {k: v.clone() for k, v in data.items()}
    changed["embs"][:, depth:] = float("nan")
    changed["days"][:, depth:] = float("nan")
    changed["embs"][0, 1] = 1e6
    changed["days"][0, 1] = 1e6
    changed["embs"][1, 1:] = 1e6
    changed["days"][1, 1:] = 1e6
    actual = predict(model, changed, depth)
    np.testing.assert_array_equal(expected["probability"], actual["probability"])
    if depth >= 2:
        observed = {k: v.clone() for k, v in data.items()}
        observed["embs"][2:, 1] *= -3
        different = predict(model, observed, depth)["probability"]
        assert np.max(np.abs(different[2:] - expected["probability"][2:])) > 1e-6


def test_gate_initialization_bounds_and_exact_t0_reference():
    raw = synthetic_split(12)
    raw["masks"][0, 1:] = 0
    model = build_model(gate_config(True), 4)
    data = opt.tensor_split(raw, np.zeros(12), "cpu")
    for depth in range(1, 5):
        values = predict(model, data, depth)
        np.testing.assert_array_equal(values["probability"], values["reference_probability"])
    with torch.no_grad():
        model.head.bias.fill_(100)
        model.gate.bias.fill_(100)
    values = predict(model, data, 4)
    assert np.all(np.abs(values["residual_logit"]) <= 1.0)
    assert values["residual_logit"][0] == values["temporal_gate"][0] == 0
    t0 = predict(model, data, 1)
    np.testing.assert_array_equal(t0["probability"], t0["reference_probability"])
    assert not t0["temporal_gate"].any()
    model.train()
    assert model.training and not model.t0.training
    assert not any(p.requires_grad for p in model.t0.parameters())
    with pytest.raises(ValueError, match="exceeds"):
        predict(model, data, 5)


def test_followup_dropout_retains_t0_and_does_not_mutate_input():
    raw = synthetic_split(8)
    raw["masks"][0, 0] = 0
    data = opt.tensor_split(raw, np.zeros(8), "cpu")
    original = data["masks"].clone()
    dropped = drop_followups(data, 1, torch.Generator().manual_seed(42))
    torch.testing.assert_close(data["masks"], original, rtol=0, atol=0)
    torch.testing.assert_close(dropped["masks"][:, 0], original[:, 0], rtol=0, atol=0)
    assert not dropped["masks"][:, 1:].any()


def test_capacity_controls_use_the_existing_model_factory():
    config = {**base_config(), "input_dim": 1152, "proj_dim": 32, "sig_dim": 16, "n_layers": 1}
    compact = build_model(config, 4)
    wide = build_model({**config, "proj_dim": 64, "sig_dim": 64, "n_layers": 2}, 4)
    assert sum(p.numel() for p in compact.parameters()) == 40418
    assert sum(p.numel() for p in wide.parameters()) == 153986


def test_nested_pilot_artifacts_shared_weights_and_selection(tmp_path, monkeypatch):
    raw = synthetic_split(48)
    raw["masks"][0, 0] = 0
    ids = raw["pids"]
    outer = [{"fold": 0, "train_ids": ids[:24], "val_ids": ids[24:]},
             {"fold": 1, "train_ids": ids[24:], "val_ids": ids[:24]}]
    folds = opt.nested_folds(ids, ["HELDOUT"], outer, dict(zip(ids, raw["labels"])), 2, 2026)
    opt.write_json(tmp_path / "nested_folds.json", folds)
    cfg = {"output_dir": str(tmp_path), "baseline_run": str(tmp_path), "previous_run": str(tmp_path / "absent"),
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
    all_tasks = []
    for gated in (False, True):
        planned = train.tasks(cfg, folds, "inner", gated=gated)
        for task in planned:
            train.run_task(task, tmp_path, tmp_path, "cpu")
        all_tasks.extend(planned)
    selection = train.select_models(cfg, folds)
    assert len(selection["selections"]) == 8 and len(selection["ranking"]) == 32
    for gated in (False, True):
        planned = train.tasks(cfg, folds, "outer", gated=gated, ranking=selection["ranking"])
        for task in planned:
            train.run_task(task, tmp_path, tmp_path, "cpu")
        all_tasks.extend(planned)
    assert len(all_tasks) == 72
    result = report_development(cfg, selection)
    assert result["development_patients"] == 48 and result["distinct_outer_models"] == 24
    assert result["selected_model_references"] == 8 and not result["holdout_evaluated"]
    refs = opt.read_json(tmp_path / "frozen_models.json")
    shared = [r for r in refs["comparison_models"] if r["arm"] == "shared64"]
    assert len(shared) == 8 and len({r["path"] for r in shared}) == 2
    oof = pd.read_csv(tmp_path / "reports/outer_oof_predictions.csv")
    assert len(oof) == 48 * 4 * 5 and "HELDOUT" not in set(oof.patient_id)
    t0 = oof[(oof.depth == 1) & oof.arm.isin(["baseline", "gated64", "shared64"])]
    table = t0.pivot(index="patient_id", columns="arm", values="probability")
    np.testing.assert_array_equal(table.baseline, table.gated64)
    np.testing.assert_array_equal(table.baseline, table.shared64)
    for task in all_tasks:
        path = opt.task_path(tmp_path, task)
        assert train.task_complete(path, task)
        assert train.run_task(task, tmp_path, tmp_path, "cpu")["skipped"]
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["clinical_prior_train_count"] == len(task["train_ids"])
        assert checkpoint["neural_train_count"] == len(set(task["train_ids"]) - {ids[0]})
        if task["stage"] == "outer":
            history = pd.read_csv(path / "history.csv")
            assert len(history) == task["fixed_epochs"] and not any(c.startswith("validation_") for c in history)
        if task["effective"].get("architecture") == "gated":
            base = torch.load(task["t0_reference"]["path"], map_location="cpu", weights_only=False)
            for key, value in base["model_state"].items():
                torch.testing.assert_close(checkpoint["model_state"]["t0." + key], value, rtol=0, atol=0)
            assert checkpoint["model_state"]["head.weight"].abs().sum() > 0
            model = build_model(task["effective"], task["depth"])
            model.load_state_dict(checkpoint["model_state"])
            split = window_split(raw, task["val_ids"], task["depth"], task["effective"])
            data = opt.tensor_split(split, opt._prior_from_state(split["clinical"], checkpoint["clinical_prior"]), "cpu")
            recorded = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str})
            for depth in train.windows(task):
                expected = recorded[(recorded.role == "validation") & (recorded.depth == depth)].set_index("patient_id").loc[split["pids"]]
                np.testing.assert_allclose(predict(model, data, depth)["probability"], expected.probability, rtol=0, atol=1e-7)
    task = next(t for t in all_tasks if t["candidate"] == "gated64" and t["stage"] == "outer")
    split = window_split(raw, task["train_ids"], task["depth"], task["effective"])
    with pytest.raises(ValueError, match="must not receive"):
        train.fit_gated(split, split, task, "cpu")
    altered = copy.deepcopy(task)
    altered["train_ids"] = altered["train_ids"][1:]
    with pytest.raises(ValueError, match="partition"):
        train.attach_t0(altered, tmp_path)
    altered = copy.deepcopy(task)
    altered["t0_reference"]["identity"]["size_bytes"] += 1
    assert not train.task_complete(opt.task_path(tmp_path, task), altered)

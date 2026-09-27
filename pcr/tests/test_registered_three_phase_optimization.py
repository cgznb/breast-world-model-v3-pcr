import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import first_post_optimization as opt
from src import registered_three_phase_optimization as study
from src.single_phase_temporal_models import fit_feature_statistics, window_split
from test_first_post_optimization import small_config, synthetic_split


@pytest.mark.parametrize("candidate", ["baseline", "regularized_tdn", "pca32_tdn", "clinical_only"])
def test_prespecified_four_window_recipe(candidate):
    cfg = study.load_config("configs/registered_three_phase_pcr_optimization_v2.yaml")
    for depth in cfg["depths"]:
        effective = opt.effective_config(cfg, candidate, depth)
        if candidate == "clinical_only":
            assert effective["kind"] == "clinical"
        elif candidate == "baseline":
            assert effective["epoch_selection"] == "auroc"
            assert effective["proj_dim"] == (64 if depth == 4 else 32)
        else:
            assert effective["epoch_selection"] == "logloss"
            assert effective["positive_class_weight"] == 1
            assert effective["residual_l2_weight"] > 0
            assert effective["residual_logit_limit"] == 1.5
            assert effective["visit_policy"] == "contiguous"
            assert effective["input_dim"] == 1152


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_pca_fits_only_training_patients_and_allowed_prefix(depth):
    raw = synthetic_split(48)
    raw["masks"][0, 1] = 0
    cfg = {**small_config(), "feature_transform": "pca", "pca_dim": 8,
           "visit_policy": "contiguous"}
    ids = raw["pids"][:32]
    initial = fit_feature_statistics(window_split(raw, ids, depth, cfg), cfg)
    raw["embs"][32:] = 1e6
    raw["labels"][32:] = 1 - raw["labels"][32:]
    raw["embs"][:, depth:] = -1e6
    raw["embs"][0, 1:] = 1e6
    updated = fit_feature_statistics(window_split(raw, ids, depth, cfg), cfg)
    for key in ("mean", "scale", "components"):
        torch.testing.assert_close(initial[key], updated[key], rtol=0, atol=0)
    assert initial["fitted_visits"] == 31 * depth + 1


@pytest.mark.parametrize("kind,transform", [("tdn", "identity"), ("tdn", "pca"), ("clinical", "identity")])
def test_dispatch_fixed_fit_freeze_reload_and_resume(tmp_path, monkeypatch, kind, transform):
    torch.set_num_threads(1)
    raw = synthetic_split(48)
    monkeypatch.setattr(opt, "load_development", lambda *_: raw)
    effective = {**small_config(), "feature_transform": transform, "pca_dim": 8,
                 "visit_policy": "contiguous", "architecture": "tdn", "kind": kind}
    task = {"stage": "outer", "source": "physical_roi", "depth": 3, "candidate": "test",
            "outer": 0, "inner": -1, "seed": 42, "effective": effective,
            "fixed_epochs": 0 if kind == "clinical" else 2,
            "train_ids": raw["pids"][:32], "val_ids": raw["pids"][32:]}
    assert not study.run_task(task, tmp_path, tmp_path)["skipped"]
    assert study.run_task(task, tmp_path, tmp_path)["skipped"]
    path = opt.task_path(tmp_path, task)
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    original = pd.read_csv(path / "predictions.csv", dtype={"patient_id": str}).set_index("patient_id")
    probability, residual, _ = study.predict_checkpoint(checkpoint, raw)
    np.testing.assert_allclose(probability, original.loc[raw["pids"], "probability"], atol=1e-6)
    assert abs(residual).max() <= effective["residual_logit_limit"]
    if kind == "tdn":
        history = pd.read_csv(path / "history.csv")
        assert not any(c.startswith("validation_") for c in history)
    modified = copy.deepcopy(task)
    modified.pop("fixed_epochs")
    with pytest.raises(ValueError, match="fixed duration"):
        study.run_task(modified, tmp_path, tmp_path)


def test_freeze_inventory_rejects_changed_artifact(tmp_path):
    from src.first_post_pcr_data import identity
    path = tmp_path / "input.txt"
    path.write_text("original")
    inventory = {str(path): identity(path)}
    study.verify_inventory(inventory)
    path.write_text("changed content")
    with pytest.raises(ValueError, match="Frozen input changed"):
        study.verify_inventory(inventory)


def test_holdout_refuses_unfrozen_study(tmp_path):
    from src.registered_three_phase_optimization_report import require_frozen
    with pytest.raises(ValueError, match="before holdout"):
        require_frozen({"output_dir": str(tmp_path)})


def test_holdout_aggregation_keeps_probability_draws_and_checks_coverage():
    from src.registered_three_phase_optimization_report import aggregate_holdout
    rows = []
    for mode in ("direct", "rollout", "previous_real"):
        for draw in range(4):
            for fold in range(5):
                rows.append({"model": "selected", "source": f"{mode}_draw_{draw}",
                             "depth": 3, "seed": 42, "patient_id": "P0", "label": 0,
                             "outer": fold, "probability": .1 + .1 * draw + .01 * fold})
    frame = pd.DataFrame(rows)
    result = aggregate_holdout(frame, list(range(5)))
    np.testing.assert_allclose(result[result.source.str.endswith("_mc4")].probability, .27)
    with pytest.raises(ValueError, match="every outer fold"):
        aggregate_holdout(frame.iloc[1:], list(range(5)))
    with pytest.raises(ValueError, match="exactly four"):
        aggregate_holdout(frame[frame.source != "direct_draw_0"], list(range(5)))


def test_paired_bootstrap_uses_mean_seed_auc_and_shared_patients():
    from src.registered_three_phase_optimization_report import paired_comparisons
    from sklearn.metrics import roc_auc_score
    labels = np.array([0, 0, 0, 1, 1, 1])
    baseline = np.array([[.2, .3, .5, .4, .6, .8], [.1, .6, .4, .3, .7, .9]])
    selected = np.array([[.1, .2, .3, .4, .5, .6], [.1, .2, .3, .4, .5, .6]])
    rows = []
    for model, matrix in (("selected", selected), ("baseline", baseline)):
        for i, p in enumerate(matrix):
            for j in range(6):
                rows.append({"model": model, "source": "real", "depth": 1, "seed": 42 + i,
                             "patient_id": f"P{j}", "label": labels[j], "probability": p[j]})
    cfg = {"bootstrap_seed": 26, "bootstrap_samples": 100, "formal_seeds": [42, 43]}
    result = paired_comparisons(cfg, pd.DataFrame(rows), [("selected", "baseline")]).iloc[0]
    expected = np.mean([roc_auc_score(labels, p) for p in selected]) - np.mean(
        [roc_auc_score(labels, p) for p in baseline])
    assert result.auroc_difference == pytest.approx(expected)
    same = paired_comparisons(cfg, pd.DataFrame(rows), [("selected", "selected")]).iloc[0]
    assert same.auroc_difference == same.ci95_low == same.ci95_high == 0


def test_independent_weighted_rank_audit_handles_ties_and_zero_counts():
    from scripts.verify_registered_three_phase_optimization import weighted_auc_samples
    from sklearn.metrics import roc_auc_score
    y = np.array([0, 0, 0, 1, 1, 1])
    matrix = np.array([[.1, .3, .3, .3, .4, .8], [.6, .2, .6, .6, .7, .1]])
    weights = np.array([[1, 1, 1, 1, 1, 1], [0, 3, 0, 2, 0, 1], [2, 0, 1, 0, 1, 2]])
    actual = weighted_auc_samples(y, matrix, weights)
    expected = [np.mean([roc_auc_score(y, p, sample_weight=w) for p in matrix]) for w in weights]
    np.testing.assert_allclose(actual, expected, atol=1e-14)

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.single_phase_anchor_models import AnchoredResponse, ClinicalAnchor, objective
from src.single_phase_anchor_training import fit_neural, model_from_bundle, predict_bundle
from src.single_phase_anchor_workflow import add_selected, choose_candidate
from src.single_phase_response_data import subset
from src.single_phase_response_training import fit_clinical
from src.single_phase_temporal_models import fit_feature_statistics


def fixture_data():
    rng = np.random.default_rng(7)
    data = dict(pids=[f"synthetic_{i}" for i in range(20)],
                embs=rng.normal(size=(20, 4, 768)).astype(np.float32), masks=np.ones((20, 4), np.float32),
                days=np.tile([0, 30, 90, 150], (20, 1)).astype(np.float32),
                clinical=rng.normal(size=(20, 17)).astype(np.float32), labels=np.tile([0, 1], 10).astype(np.float32))
    settings = dict(input_dim=768, feature_transform="pca", pca_dim=8, hidden_dim=8,
                    residual_limit=.75, gate_initial_probability=.1, dropout=.1, epochs=3,
                    patience=2, min_delta=1e-4, batch_size=4, lr=3e-4, weight_decay=.01,
                    residual_l2=.1, grad_clip=1.)
    train, val = subset(data, data["pids"][:16]), subset(data, data["pids"][16:])
    prep = dict(statistics=fit_feature_statistics(train, settings), clinical=fit_clinical(train, 42))
    return train, val, settings, prep


def inputs(data):
    return tuple(torch.from_numpy(data[k]) for k in ("embs", "masks", "clinical", "days"))


def test_zero_start_is_exact_clinical_anchor_and_not_trainable():
    train, _, settings, prep = fixture_data()
    model = AnchoredResponse(settings, "t0", prep["statistics"], prep["clinical"]).eval()
    logits, residual = model(*inputs(train))
    expected = ClinicalAnchor(prep["clinical"])(torch.from_numpy(train["clinical"]))
    torch.testing.assert_close(logits, expected[:, None].expand(-1, 4), rtol=0, atol=0)
    assert not residual.any()
    assert not list(model.base.clinical.parameters())


@pytest.mark.parametrize("kind", ["linear", "gated"])
def test_causal_prefix_and_missing_visit_fallback_with_nonzero_correction(kind):
    train, _, settings, prep = fixture_data()
    base = AnchoredResponse(settings, "t0", prep["statistics"], prep["clinical"])
    spec = dict(statistics=prep["statistics"], state=base.base.state_dict())
    model = AnchoredResponse(settings, kind, prep["statistics"], prep["clinical"], spec).eval()
    with torch.no_grad():
        for output in model.outputs:
            output.weight.fill_(.15)
            output.bias.fill_(.1)
    data = copy.deepcopy(train)
    data["masks"][0] = [1, 0, 1, 1]
    data["masks"][1] = [0, 1, 1, 1]
    before, residual = model(*inputs(data))
    data["embs"][:, 2:] *= 100
    data["days"][:, 2:] = 900
    after = model(*inputs(data))[0]
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    assert torch.equal(before[0, 0].expand(4), before[0])
    assert torch.equal(before[1, 0].expand(4), before[1])
    assert float(residual.detach().abs().max()) <= settings["residual_limit"]
    assert torch.equal(before[:, 0], base(*inputs(train))[0][:, 0])
    model.train()
    assert not model.base.training
    objective(*model(*inputs(train)), torch.from_numpy(train["labels"]),
              torch.from_numpy(train["masks"]), kind).backward()
    assert all(p.grad is None for p in model.base.parameters())


def test_followup_loss_does_not_train_t0_or_no_followup_patients():
    logits = torch.zeros(3, 4, requires_grad=True)
    mask = torch.tensor([[1., 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1]])
    loss = objective(logits, logits * 0, torch.tensor([1., 0, 1]), mask, "gated")
    loss.backward()
    assert not logits.grad[:, 0].any()
    assert not logits.grad[1:].any()
    assert (logits.grad[0, 1:] < 0).all()


def test_epoch_zero_can_retain_anchor_and_portable_prediction_needs_no_paths(tmp_path):
    train, val, settings, prep = fixture_data()
    bundle = fit_neural(train, None, settings, "t0", prep, 42, tmp_path, fixed_epochs=0)
    assert bundle["selected_epochs"] == 0 and bundle["zero_epoch_fallback"]
    assert bundle["binding"]["validation_ids"] == []
    probability = predict_bundle(bundle, val)
    without_labels = {k: v for k, v in val.items() if k not in ("labels", "pids")}
    np.testing.assert_array_equal(probability, predict_bundle(bundle, without_labels))
    expected = torch.sigmoid(ClinicalAnchor(prep["clinical"])(torch.from_numpy(val["clinical"]))).numpy()
    np.testing.assert_array_equal(probability, np.repeat(expected[:, None], 4, axis=1))
    with pytest.raises(ValueError, match="must not receive"):
        fit_neural(train, val, settings, "t0", prep, 42, tmp_path / "bad", fixed_epochs=0)


@pytest.mark.parametrize("kind", ["t0", "gated"])
def test_exact_epoch_recovery_and_frozen_reference(tmp_path, kind):
    train, val, settings, prep = fixture_data()
    spec = None
    if kind != "t0":
        base = fit_neural(train, None, settings, "t0", prep, 42, tmp_path / "base", fixed_epochs=2)
        spec = dict(statistics=base["statistics"], state=model_from_bundle(base).base.state_dict())
    options = dict(fixed_epochs=3, base_spec=spec)
    first = fit_neural(train, None, settings, kind, prep, 42, tmp_path / "a", **options)
    with pytest.raises(InterruptedError):
        fit_neural(train, None, settings, kind, prep, 42, tmp_path / "b", pause_after=1, **options)
    second = fit_neural(train, None, settings, kind, prep, 42, tmp_path / "b", **options)
    assert first["history"] == second["history"]
    assert all(torch.equal(v, second["state"][k]) for k, v in first["state"].items())
    np.testing.assert_array_equal(predict_bundle(first, val), predict_bundle(second, val))
    assert second["gradient_audit"]["nonzero"]
    with pytest.raises(ValueError, match="partition differs"):
        fit_neural(train, None, settings, kind, prep, 43, tmp_path / "b", **options)


def test_selected_policy_uses_last_observed_prefix_not_requested_future_window():
    rows = []
    for arm, p in (("clinical", .2), ("anchor", .8)):
        for available in (0, 1, 2, 4):
            for prefix in range(1, 5):
                rows.append(dict(arm=arm, probability=p, available_prefix=available, prefix=prefix,
                                 fold=0, patient_id=f"synthetic_{available}"))
    choices = [dict(fold=0, prefix=t, selected="anchor" if t == 2 else "clinical") for t in range(1, 5)]
    selected = add_selected(pd.DataFrame(rows), choices, ["clinical", "anchor"])
    selected = selected[selected.arm == "inner_selected"]
    assert len(selected) == 16
    assert (selected[selected.available_prefix <= 1].probability == .2).all()
    assert (selected[(selected.available_prefix == 2) & (selected.prefix >= 2)].probability == .8).all()


def test_inner_policy_keeps_baseline_when_images_do_not_improve_logloss():
    names = ["clinical", "t0", "gated"]
    assert choose_candidate(dict(clinical=.54, t0=.55, gated=.56), names, 1e-4) == "clinical"
    assert choose_candidate(dict(clinical=.54, t0=.53995, gated=.54), names, 1e-4) == "clinical"
    assert choose_candidate(dict(clinical=.54, t0=.53, gated=.51), names, 1e-4) == "gated"
    with pytest.raises(ValueError, match="finite"):
        choose_candidate(dict(clinical=.54, t0=.53, gated=float("nan")), names, 1e-4)


def test_outer_refit_preprocessing_uses_only_its_training_patients(tmp_path):
    from src.first_post_pcr_data import identity, save_tensor, write_json
    from src.single_phase_anchor_training import run_task, task_complete, task_path
    train, val, settings, _ = fixture_data()
    val["embs"] *= 1000
    all_data = {k: train[k] + val[k] if isinstance(train[k], list) else np.concatenate([train[k], val[k]]) for k in train}
    cfg = dict(study_dir=str(tmp_path), output_dir=str(tmp_path), training=settings, probe={},
               arms={"anchor_t0_context": dict(kind="t0", roi="context")})
    save_tensor(tmp_path / "data/train_context.pt", all_data)
    write_json(tmp_path / "runtime_sources.json", dict(synthetic=True))
    task = dict(stage="formal", arm="anchor_t0_context", fold=0, seed=42, train_ids=train["pids"],
                val_ids=val["pids"], fixed_epochs=2)
    run_task(cfg, task)
    root = task_path(cfg, task)
    bundle = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    assert bundle["binding"]["validation_ids"] == []
    assert bundle["clinical"]["fitted_ids"] == train["pids"]
    assert bundle["statistics"]["fitted_visits"] == len(train["pids"])
    np.testing.assert_allclose(bundle["statistics"]["mean"], train["embs"][:, 0].mean(0), atol=2e-7)
    assert set(pd.read_csv(root / "predictions.csv").patient_id) == set(val["pids"])
    before = identity(root / "model.pt")
    run_task(cfg, task)
    assert task_complete(cfg, task) and identity(root / "model.pt") == before

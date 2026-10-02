import math

import pytest
import torch
import torch.nn.functional as F

from conftest import open_gates
from test_training_v2 import cfg_v2
from responsewm.contracts_v2 import EdgeSupervision
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.losses_v2 import (edge_objective, free_rollout_objective,
                                  observed_update_objective, prefix_objective,
                                  representation_objective)
from responsewm.synthetic_v2 import make_synthetic_v2
from responsewm.training_v2 import build_model


@pytest.fixture
def batch_store(tmp_path):
    store = PatientTrajectoryStore(make_synthetic_v2(tmp_path / "cohort", patients=16),
                                   allow_synthetic=True)
    store.fit_statistics()
    return store


def gradients(model):
    return {name: param.grad.detach().clone() for name, param in model.named_parameters()
            if param.grad is not None}


def assert_batch_matches_singles(model, tasks, objective):
    model.zero_grad(set_to_none=True)
    batched = objective(tasks)
    batched.backward()
    actual = gradients(model)
    model.zero_grad(set_to_none=True)
    expected_loss = 0.
    for task in tasks:
        loss = objective([task]) / len(tasks)
        expected_loss += float(loss.detach())
        loss.backward()
    expected = gradients(model)
    assert float(batched.detach()) == pytest.approx(expected_loss, abs=2e-5, rel=2e-5)
    assert actual.keys() == expected.keys()
    assert any(value.abs().sum() > 0 for value in actual.values())
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=3e-5, rtol=3e-4,
                                   msg=lambda message, name=name: f"{name}: {message}")


def deterministic_corruption(z, cfg):
    mask = torch.zeros(len(z), math.prod(cfg.token_grid), device=z.device, dtype=torch.bool)
    mask[:, ::2] = True
    high = F.interpolate(mask.reshape(len(z), 1, *cfg.token_grid).float(),
                         size=z.shape[2:], mode="nearest").bool()
    return z.masked_fill(high, 0), mask


def test_representation_true_batch8_matches_patient_losses_and_gradients(cfg_v2, batch_store, monkeypatch):
    model = build_model(cfg_v2, batch_store)
    model.configure_stage("representation")
    model.eval()
    monkeypatch.setattr("responsewm.representation.corrupt_latent", deterministic_corruption)
    monkeypatch.setattr("responsewm.losses_v2.torch.multinomial",
                        lambda row, count: row.argmax().reshape(1))
    sizes = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: sizes.append(len(args[0])))
    tasks = [(index, 0) for index in (0, 2, 3, 5, 6, 8, 9, 11)]

    def objective(selected):
        inp, sup = batch_store.batch(selected)
        for row, (index, _) in enumerate(selected):
            if index % 2 == 0:
                sup.auxiliary[row][0]["pillar"] = torch.arange(cfg_v2.network.pillar_dim).float()
        return representation_objective(model, inp, sup)[0]

    assert_batch_matches_singles(model, tasks, objective)
    hook.remove()
    assert 8 in sizes


@pytest.mark.parametrize("mismatch", ["stage", "history", "future", "label"])
def test_batched_objectives_reject_incompatible_patient_structures(cfg_v2, batch_store, mismatch):
    model = build_model(cfg_v2, batch_store)
    if mismatch == "stage":
        inp, sup = batch_store.batch([(0, 0), (2, 1)])
        message = "same as_of"
    elif mismatch == "history":
        inp, sup = batch_store.batch([(0, 2), (1, 2)])
        message = "same observed stage history"
    elif mismatch == "future":
        inp, sup = batch_store.batch([(0, 0), (1, 0)])
        message = "same future availability"
    else:
        inp, sup = batch_store.batch([(0, 0), (2, 0)])
        sup.label_mask[1] = False
        message = "same label availability"
    with pytest.raises(ValueError, match=message):
        prefix_objective(model, inp, sup)


def test_missing_followup_rollout_batch_matches_patient_losses_and_gradients(cfg_v2, batch_store):
    model = build_model(cfg_v2, batch_store)
    model.freeze_representation()
    model.configure_stage("joint")
    model.eval()
    open_gates(model)
    tasks = [(1, 0), (4, 0)]

    def objective(selected):
        inp, sup = batch_store.batch(selected)
        assert not sup.future_mask[:, 0].any()
        loss, terms = free_rollout_objective(model, inp, sup, seed=19, max_hops=3,
                                            marginal_weight=.25)
        assert terms["support/target_T1"] == 0
        assert terms["support/target_T2"] == len(selected)
        assert terms["rollout_depth"] == 3
        return loss

    assert_batch_matches_singles(model, tasks, objective)
    assert model.velocity.image.out.weight.grad.abs().sum() > 0
    assert model.velocity.semantic_out[-1].weight.grad.abs().sum() > 0
    assert all(param.grad is None for param in model.target_encoder.parameters())


def test_observed_update_batch_matches_patient_losses_and_gradients(cfg_v2, batch_store):
    model = build_model(cfg_v2, batch_store)
    model.freeze_representation()
    model.configure_stage("joint")
    model.eval()
    open_gates(model)

    def objective(selected):
        inp, sup = batch_store.batch(selected)
        loss, terms = observed_update_objective(model, inp, sup, seed=31)
        assert terms["posterior_stage"] == 3
        assert terms["support/assimilation_updates"] == len(selected)
        return loss

    assert_batch_matches_singles(model, [(0, 2), (2, 2)], objective)
    assert any(param.grad is not None and param.grad.abs().sum() > 0
               for param in model.assimilator.parameters())


def test_paired_edge_batch_matches_patient_losses_and_gradients(cfg_v2, batch_store, monkeypatch):
    model = build_model(cfg_v2, batch_store)
    model.freeze_representation()
    model.configure_stage("flow")
    model.eval()
    open_gates(model)

    def fixed_path(earlier_z, later_z, earlier_s, later_s, generator=None):
        def path(earlier, later):
            return (torch.cat((later * .4, earlier * .6), 1),
                    torch.cat((later, -earlier), 1))
        z, vz = path(earlier_z, later_z)
        s, vs = path(earlier_s, later_s)
        return z, s, vz, vs, earlier_z.new_full((len(earlier_z),), .4)

    monkeypatch.setattr("responsewm.losses_v2.make_joint_path", fixed_path)

    def objective(selected):
        inp, sup = batch_store.batch(selected)
        truth = EdgeSupervision(sup.future[:, 0], 0, 1, "adjacent",
                                sup.anatomy_comparable[:, 0], [{} for _ in selected])
        return edge_objective(model, inp, truth)[0]

    assert_batch_matches_singles(model, [(0, 0), (2, 0)], objective)

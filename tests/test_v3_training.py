"""Regression tests for deployment checkpoint handoff, masking and exact restart."""
import copy
import os
import signal
from collections import Counter

import pytest
import torch

from test_training_v2 import cfg_v2, store_v2
from responsewm.io import load_checkpoint
from responsewm.losses_v3 import latent_distribution_scores, free_rollout_objective
from responsewm.model_v3 import ResponseWorldModelV3
from responsewm.samplers import PatientLandmarkSampler
from responsewm import training_v3 as training


@pytest.fixture
def cfg_v3(cfg_v2):
    cfg = copy.deepcopy(cfg_v2)
    cfg.schema = "responsewm_v3"
    cfg.training.representation_steps = 2
    cfg.training.log_every = 1
    cfg.training.checkpoint_every = 1
    cfg.training.validation_every = 1
    cfg.protocol.image_eval_cases = 0
    return cfg.validate()


@pytest.fixture
def fast_validation(monkeypatch):
    def validate(model, store, cfg, stage, **kwargs):
        return {"selection_score": .5, "generation_score": 1.0,
                "generation_grounding": 0.0, "rows": []}
    monkeypatch.setattr(training, "validate", validate)
    return validate


def test_prior_initialization_and_bounded_readout(cfg_v3, store_v2):
    model = training.build_model(cfg_v3, store_v2).eval()
    store_v2.fit_prior(model)
    inp, _ = store_v2.batch([(0, 0)])
    with torch.no_grad():
        belief = model.initialize(inp)
        prior = model.pcr.prior(inp.clinical, inp.clinical_mask)
        assert torch.equal(model.query_pcr(belief).logit_per_sample[:, 0], prior)
        for sign in (-1, 1):
            model.pcr.output[-1].bias.fill_(sign*100)
            residual = model.query_pcr(belief).logit_per_sample[:, 0]-prior
            assert torch.allclose(residual, torch.full_like(residual, sign*cfg_v3.protocol.residual_logit_limit))


@pytest.mark.parametrize("scope", ["output", "last_block", "all"])
def test_scope_and_conservative_optimizer(cfg_v3, store_v2, scope):
    cfg_v3.protocol.readout_scope = scope
    cfg_v3.protocol.joint_readout_scope = scope
    model = training.build_model(cfg_v3, store_v2)
    for stage in ("readout", "joint"):
        model.configure_stage(stage)
        last = len(model.pcr.blocks)-1
        for name, parameter in model.pcr.named_parameters():
            expected = scope == "all" or name.startswith("output.") or (scope == "last_block" and name.startswith(f"blocks.{last}."))
            assert parameter.requires_grad == expected, (stage, name)
        if stage == "readout":
            assert not any(p.requires_grad for p in model.velocity.parameters())
        groups, rate = training.optimizer_groups(model, cfg_v3, stage)
        head = {id(p) for p in model.pcr.parameters()}
        for group in groups:
            if any(id(p) in head for p in group["params"]):
                assert group["weight_decay"] == cfg_v3.protocol.readout_weight_decay
        if stage == "readout":
            assert rate == cfg_v3.protocol.readout_lr
    model.configure_stage("representation")
    _, rate = training.optimizer_groups(model, cfg_v3, "representation")
    assert rate == cfg_v3.protocol.representation_lr


def test_latent_scores_ignore_missing_truth_and_preserve_gradient():
    samples = torch.randn(2, 2, 3, 2, 2, 2, 2, requires_grad=True)
    target = torch.randn(2, 3, 2, 2, 2, 2)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    scores = latent_distribution_scores(samples, target, mask)
    poisoned = target.clone()
    poisoned[~mask] = 1e6
    other = latent_distribution_scores(samples, poisoned, mask)
    assert all(torch.equal(a, b) for a, b in zip(scores, other))
    sum(scores).backward()
    assert samples.grad[0, :, [0, 2]].abs().sum() > 0
    assert samples.grad[0, :, 1].count_nonzero() == 0
    assert samples.grad[1].count_nonzero() == 0


def test_free_rollout_scores_have_image_gradient_with_missing_visit(cfg_v3, store_v2):
    model = training.build_model(cfg_v3, store_v2)
    model.freeze_representation()
    model.configure_stage("joint")
    inp, sup = store_v2.batch([(1, 0)])
    loss, terms = free_rollout_objective(model, inp, sup, seed=123, marginal_weight=.1)
    assert terms["support/target_T1"] == 0
    assert terms["loss/latent_ensemble_mean_mse"] > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.velocity.image.parameters())


def test_inferior_c_retains_entry_and_d_receives_it(cfg_v3, store_v2, tmp_path, monkeypatch, fast_validation):
    root = tmp_path/"handoff"
    for stage in ("representation", "flow"):
        assert training.train_stage(store_v2, cfg_v3, root, stage)["completed"]
    visits = {"readout": 0, "joint": 0}
    captured = {}
    def validation(model, store, cfg, stage, **kwargs):
        count = visits[stage]
        visits[stage] += 1
        if stage == "joint" and count == 0:
            captured.update({k:v.detach().clone() for k,v in model.state_dict().items()})
        return {"selection_score": .5+count*.1, "generation_score": 1., "rows": []}
    monkeypatch.setattr(training, "validate", validation)
    training.train_stage(store_v2, cfg_v3, root, "readout")
    best = load_checkpoint(root/"readout"/"best.pt")
    last = load_checkpoint(root/"readout"/"last.pt")
    assert best["step"] == 0 and last["completed"]
    assert any(not torch.equal(best["model"][k],last["model"][k]) for k in best["model"] if k.startswith("pcr."))
    training.train_stage(store_v2, cfg_v3, root, "joint")
    assert all(torch.equal(value, captured[key]) for key,value in best["model"].items())
    assert load_checkpoint(root/"joint"/"best.pt")["step"] == 0


def test_joint_guard_rejects_generation_degradation():
    assert training.checkpoint_eligible("readout", 0, 100, 0, None, None)
    assert training.checkpoint_eligible("joint", 0, 100, 0, 1., 1.1)
    assert not training.checkpoint_eligible("joint", 1, 100, 1, 1.2, 1.1)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fixed_origin_sampling_is_patient_uniform_and_keeps_missing_visit_groups(store_v2, batch_size):
    # Give one patient only T0: conditioning the old landmark weights would
    # overcount this patient by a factor of three relative to complete cases.
    missing = store_v2.by_split["train"][1]
    store_v2.patients[missing]["visits"] = store_v2.patients[missing]["visits"][:1]
    sampler = PatientLandmarkSampler(store_v2, 178, include_terminal=False)
    counts = Counter()
    for _ in range(5000):
        tasks = training._landmark_batch(sampler, batch_size, origin_stage=0)
        assert all(origin == 0 for _, origin in tasks)
        available = [{v["stage"] for v in store_v2.patients[index]["visits"]} for index, _ in tasks]
        assert all(stages == available[0] for stages in available)
        counts.update(index for index, _ in tasks)
    expected = 1 / len(store_v2.by_split["train"])
    assert set(counts) == set(store_v2.by_split["train"])
    for count in counts.values():
        assert count / (5000 * batch_size) == pytest.approx(expected, abs=.025)
    state = sampler.state_dict()
    expected_tasks = [training._landmark_batch(sampler, batch_size, origin_stage=0) for _ in range(20)]
    sampler.load_state_dict(state)
    assert expected_tasks == [training._landmark_batch(sampler, batch_size, origin_stage=0) for _ in range(20)]


def test_signal_handler_restores_previous_handler():
    request = training.StopRequest()
    previous = signal.getsignal(signal.SIGTERM)
    with training.checkpoint_signals(request):
        os.kill(os.getpid(), signal.SIGTERM)
        assert request.requested and request.signum == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("validation_every", [1, 3])
def test_stop_at_optimizer_boundary_resumes_exactly(cfg_v3, store_v2, tmp_path, monkeypatch, fast_validation, validation_every):
    cfg_v3.training.strict_determinism = True
    # Both validation-boundary and ordinary optimizer-boundary stops must
    # preserve the selected checkpoint, not only the final trained weights.
    cfg_v3.training.validation_every = validation_every
    cfg_v3.protocol.stage_batch_sizes = {"representation": 2}
    full, interrupted = tmp_path/"full", tmp_path/"interrupted"
    training.train_stage(store_v2, cfg_v3, full, "representation")
    request = training.StopRequest()
    update = ResponseWorldModelV3.update_target
    def request_after_update(model):
        update(model)
        request(signal.SIGTERM)
    monkeypatch.setattr(ResponseWorldModelV3, "update_target", request_after_update)
    result = training._train_stage(store_v2, cfg_v3, interrupted, "representation", stop_request=request)
    assert result["steps"] == 1 and not result["completed"]
    stopped = load_checkpoint(interrupted/"representation"/"last.pt")
    assert stopped["stop_signal"] == signal.SIGTERM and not stopped["completed"]
    monkeypatch.setattr(ResponseWorldModelV3, "update_target", update)
    training.train_stage(store_v2, cfg_v3, interrupted, "representation", resume=True)
    expected = load_checkpoint(full/"representation"/"last.pt")
    actual = load_checkpoint(interrupted/"representation"/"last.pt")
    assert actual["completed"]
    expected_best = load_checkpoint(full/"representation"/"best.pt")
    actual_best = load_checkpoint(interrupted/"representation"/"best.pt")
    assert expected_best["step"] == actual_best["step"]
    assert expected["best_score"] == actual["best_score"]
    assert expected["stale_validations"] == actual["stale_validations"]
    for key, value in expected_best["model"].items():
        assert torch.equal(value, actual_best["model"][key]), key
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    for name in expected["samplers"]:
        assert torch.equal(expected["samplers"][name]["generator"], actual["samplers"][name]["generator"])

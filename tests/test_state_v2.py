from dataclasses import replace
import copy
import io
import inspect
import pytest
import torch
from responsewm.model_v2 import ResponseWorldModelV2
from responsewm.contracts import ForecastInput
from responsewm.contracts_v2 import PatientPrefix
from responsewm.conditioning import IntervalSpec
from responsewm.rollout import NoiseLedger
from responsewm.state import Observation, PatientBelief
from conftest import open_gates


@pytest.fixture
def model_v2(cfg):
    cfg.schema = "responsewm_v2"
    cfg.network.time_basis = "stage_index"
    cfg.network.semantic_depth = 6
    cfg.multistage.global_tokens = 2
    cfg.multistage.memory_tokens = cfg.encoder.anatomy_tokens+cfg.encoder.disease_tokens+2
    cfg.multistage.assimilation_depth = 2
    cfg.multistage.deep_jepa_weights = (.3, .7)
    model = ResponseWorldModelV2(cfg, 3, 0)
    model.freeze_representation()
    model.configure_stage("joint")
    model.eval()
    return model


@pytest.fixture
def prefix():
    return ForecastInput(torch.randn(2, 1, 24, 4, 8, 8), torch.ones(2, 1, dtype=torch.bool),
                         torch.zeros(2, 1), torch.randn(2, 3), torch.ones(2, 3, dtype=torch.bool),
                         torch.tensor([[1., 2., 3.]]).expand(2, -1), torch.ones(2, 3, dtype=torch.bool),
                         torch.empty(2, 3, 0), torch.empty(2, 3, 0, dtype=torch.bool))


def test_observation_seed_has_image_gradient(model_v2, prefix):
    z = prefix.observed.clone().requires_grad_()
    belief = model_v2.initialize(replace(prefix, observed=z))
    assert not torch.equal(belief.memory[0], belief.memory[1])
    model_v2.query_pcr(belief).logit_per_sample.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_no_observation_update_is_exact_identity(model_v2):
    memory = torch.randn(2, model_v2.cfg.multistage.memory_tokens, 24)
    observation = torch.randn(2, 10, 24)
    result = model_v2.assimilator(memory, observation, torch.zeros(2, dtype=torch.bool), torch.ones(2))
    assert torch.equal(result, memory)


def test_canonical_grid_query_subset_and_readonly(model_v2, prefix):
    with torch.no_grad():
        belief = model_v2.initialize(prefix)
        original = belief.anchor_latent.clone()
        a = model_v2.forecast(belief, output_stages=(3,), samples=2, steps=1, noise_ledger=NoiseLedger(12))
        b = model_v2.forecast(belief, output_stages=(1, 2, 3), samples=2, steps=1, noise_ledger=NoiseLedger(12))
    assert a.stage_ids == (1, 2, 3)
    assert a.latent.shape[:3] == (2, 2, 1)
    assert torch.equal(a.latent[:, :, 0], b.latent[:, :, 2])
    assert torch.equal(a.logits, b.logits)
    assert len(belief.evidence_log) == 1 and belief.stage == 0
    assert torch.equal(belief.anchor_latent, original)
    assert torch.equal(a.probability, a.logits.sigmoid().mean(1))


def test_observation_anchor_idempotence_conflict_and_invalidation(model_v2, prefix):
    with torch.no_grad():
        belief = model_v2.initialize(prefix)
        trace = model_v2.forecast(belief, samples=2, steps=1, noise_ledger=NoiseLedger(4))
        prior = trace.all_states[0]
        obs = Observation(1, torch.randn_like(prefix.observed[:, 0]), ("real1", "real1"))
        posterior = model_v2.observe(prior, obs)
        duplicate = model_v2.observe(posterior, obs)
    assert duplicate is posterior
    assert torch.equal(posterior.anchor_latent[:, 0], obs.latent)
    assert torch.equal(posterior.anchor_latent[:, 1], obs.latent)
    assert sum(e.stage == 1 for e in posterior.model_state_log) == 1
    assert len(posterior.evidence_log) == 2
    with pytest.raises(ValueError, match="conflicts"):
        model_v2.observe(posterior, replace(obs, latent=obs.latent+1))
    with pytest.raises(ValueError, match="invalidated"):
        model_v2.query_pcr(belief, mode="marginal_current", future_trace=trace)
    with pytest.raises(ValueError, match="later stage"):
        model_v2.observe(trace.final_state, obs)


def test_noise_batch_order_and_sample_identity(model_v2, prefix):
    reverse = torch.tensor([1, 0])
    shuffled = replace(prefix, **{name: getattr(prefix, name)[reverse] for name in prefix.__dataclass_fields__})
    with torch.no_grad():
        a = model_v2.forecast(model_v2.initialize(prefix), samples=2, steps=1, noise_ledger=NoiseLedger(7))
        b = model_v2.forecast(model_v2.initialize(shuffled), samples=2, steps=1, noise_ledger=NoiseLedger(7))
    assert torch.allclose(a.latent[reverse], b.latent, atol=1e-6)
    assert torch.allclose(a.logits[reverse], b.logits, atol=1e-6)
    assert not torch.equal(a.latent[:, 0], a.latent[:, 1])
    with pytest.raises(ValueError, match="sample identities"):
        model_v2.advance(a.all_states[0], IntervalSpec(1, 2), samples=3)


def test_history_block_causal_outputs_are_stable(model_v2, prefix):
    with torch.no_grad():
        belief = model_v2.initialize(prefix)
        prior = model_v2.advance(belief, IntervalSpec(0, 1), steps=1).final_state
        early = model_v2.history(belief.model_state_log, return_all=True)
        later = model_v2.history(prior.model_state_log, return_all=True)
    assert torch.allclose(early[:, :, 0], later[:, :, 0], atol=1e-6)
    assert not any(isinstance(m, torch.nn.Dropout) and m.p for m in model_v2.history.modules())


def test_multihop_gradient_reaches_first_state(model_v2, prefix):
    open_gates(model_v2)
    trace = model_v2.forecast(model_v2.initialize(prefix), samples=1, steps=1, noise_ledger=NoiseLedger(9))
    z, s = trace.all_states[0].anchor_latent, trace.all_states[0].anchor_disease
    z.retain_grad()
    s.retain_grad()
    trace.logits.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert s.grad is not None and s.grad.abs().sum() > 0


def test_current_query_is_repeatable_and_no_future_signature(model_v2, prefix):
    state = model_v2.initialize(prefix)
    a = model_v2.query_pcr(state)
    b = model_v2.query_pcr(state)
    assert torch.equal(a.logit_per_sample, b.logit_per_sample)
    assert not {"target", "label", "future_mask", "supervision", "patient_id"} & set(inspect.signature(model_v2.forecast).parameters)
    with pytest.raises(ValueError):
        model_v2.advance(state, IntervalSpec(0, 3))
    with pytest.raises(ValueError):
        model_v2.query_pcr(state, mode="forecasted_state")


def test_state_roundtrip_weights_only(model_v2, prefix):
    state = model_v2.initialize(prefix)
    stream = io.BytesIO()
    torch.save(state.state_dict(), stream)
    stream.seek(0)
    restored = PatientBelief.from_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(model_v2.query_pcr(state).probability, model_v2.query_pcr(restored).probability)


def test_online_observation_only_replay_parity(model_v2, prefix):
    z = torch.randn_like(prefix.observed[:, 0])
    with torch.no_grad():
        start = model_v2.initialize(prefix)
        online = model_v2.observe(start, Observation(1, z, ("mri:T1", "mri:T1")))
        replay_input = replace(prefix, observed=torch.cat((prefix.observed, z[:, None]), 1),
                               observed_mask=torch.ones(2, 2, dtype=torch.bool),
                               observed_days=torch.tensor([[0., 1.], [0., 1.]]),
                               future_days=prefix.future_days[:, 1:], future_mask=prefix.future_mask[:, 1:],
                               actions=prefix.actions[:, 1:], action_mask=prefix.action_mask[:, 1:])
        replay = model_v2.initialize(replay_input)
    assert torch.equal(online.memory, replay.memory)
    assert torch.equal(model_v2.query_pcr(online).probability, model_v2.query_pcr(replay).probability)


def test_observe_preserves_history_and_teacher_coordinates(model_v2, prefix):
    z = torch.randn_like(prefix.observed[:, 0])
    with torch.no_grad():
        a = model_v2.initialize(prefix)
        b = model_v2.initialize(replace(prefix, observed=prefix.observed+2))
        observation = Observation(1, z, ("shared", "shared"))
        a = model_v2.observe(a, observation)
        b = model_v2.observe(b, observation)
    assert torch.equal(a.anchor_disease, b.anchor_disease)
    assert not torch.equal(a.memory, b.memory)
    assert not torch.equal(model_v2.query_pcr(a).probability, model_v2.query_pcr(b).probability)


def test_invalid_observation_leaves_whole_state_unchanged(model_v2, prefix):
    belief = model_v2.initialize(prefix)
    observation = Observation(1, torch.randn_like(prefix.observed[:, 0]), ("none", "none"),
                              valid=torch.zeros(2, dtype=torch.bool))
    assert model_v2.observe(belief, observation) is belief


def test_patient_singleton_noise_agrees_with_batch(model_v2, prefix):
    one = replace(prefix, **{name: getattr(prefix, name)[:1] for name in prefix.__dataclass_fields__})
    with torch.no_grad():
        batch = model_v2.forecast(model_v2.initialize(prefix), samples=2, steps=1, noise_ledger=NoiseLedger(7))
        single = model_v2.forecast(model_v2.initialize(one), samples=2, steps=1, noise_ledger=NoiseLedger(7))
    assert torch.allclose(batch.latent[:1], single.latent, atol=1e-6)
    assert torch.allclose(batch.logits[:1], single.logits, atol=1e-6)


def test_future_interval_plan_does_not_change_earlier_physiology(model_v2, prefix):
    model = ResponseWorldModelV2(copy.deepcopy(model_v2.cfg), 3, 1).eval()
    prefix = replace(prefix, actions=torch.zeros(2, 3, 1), action_mask=torch.ones(2, 3, 1, dtype=torch.bool))
    first = IntervalSpec(2, 3, torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool), known_at=0)
    second = replace(first, actions=torch.ones(2, 1))
    with torch.no_grad():
        belief = model.initialize(prefix)
        a = model.forecast(belief, plan=(first,), samples=1, steps=1, noise_ledger=NoiseLedger(5))
        b = model.forecast(belief, plan=(second,), samples=1, steps=1, noise_ledger=NoiseLedger(5))
    assert torch.equal(a.latent[:, :, :2], b.latent[:, :, :2])
    assert not torch.equal(a.latent[:, :, 2], b.latent[:, :, 2])
    with pytest.raises(ValueError, match="learned after"):
        model.forecast(belief, plan=(replace(first, known_at=2),), samples=1, steps=1)


def test_clinical_payload_change_is_not_idempotent(model_v2, prefix):
    belief = model_v2.initialize(prefix)
    with pytest.raises(ValueError, match="conflicts"):
        model_v2.observe(belief, Observation(0, prefix.observed[:, 0], ("mri:T0", "mri:T0"),
                                            clinical=prefix.clinical+1, clinical_mask=prefix.clinical_mask))


def test_checkpointed_causal_rollout_matches_gradients(model_v2, prefix):
    open_gates(model_v2)
    other = copy.deepcopy(model_v2)
    other.cfg.network.checkpoint_blocks = True
    a = model_v2.forecast(model_v2.initialize(prefix), samples=1, steps=1, noise_ledger=NoiseLedger(16))
    b = other.forecast(other.initialize(prefix), samples=1, steps=1, noise_ledger=NoiseLedger(16))
    assert torch.allclose(a.logits, b.logits, atol=1e-6)
    a.logits.sum().backward()
    b.logits.sum().backward()
    assert torch.allclose(model_v2.velocity.semantic_out[-1].weight.grad,
                          other.velocity.semantic_out[-1].weight.grad, atol=1e-6)


def test_training_horizon_is_explicit_and_not_marginal(model_v2, prefix):
    with torch.no_grad():
        belief = model_v2.initialize(prefix)
        trace = model_v2.forecast(belief, output_stages=(2,), through_stage=2, samples=1, steps=1)
    assert trace.stage_ids == (1, 2) and trace.pcr_marginal is None
    with pytest.raises(ValueError, match="every remaining"):
        model_v2.query_pcr(belief, mode="marginal_current", future_trace=trace)
    with pytest.raises(ValueError):
        model_v2.forecast(belief, output_stages=(1.5,), samples=1, steps=1)


def test_no_future_readout_uses_only_real_source_memory(model_v2, prefix):
    model_v2.cfg.network.use_future = False
    open_gates(model_v2)
    belief = model_v2.initialize(prefix)
    first = model_v2.forecast(belief, samples=2, steps=1, noise_ledger=NoiseLedger(1))
    second = model_v2.forecast(belief, samples=2, steps=1, noise_ledger=NoiseLedger(200))
    observed = model_v2.query_pcr(belief)
    assert not torch.equal(first.latent, second.latent)
    assert torch.equal(first.probability, second.probability)
    assert torch.allclose(first.probability, observed.probability, atol=1e-6)
    first.logits.sum().backward()
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model_v2.velocity.parameters())


def test_delayed_observation_is_not_backdated_during_replay(model_v2, prefix):
    observed = torch.cat((prefix.observed, torch.randn_like(prefix.observed)), 1)
    late = PatientPrefix(observed, torch.ones(2, 2, dtype=torch.bool),
                         torch.tensor([[0., 1.], [0., 1.]]), prefix.clinical, prefix.clinical_mask,
                         prefix.future_days[:, 1:], prefix.future_mask[:, 1:],
                         prefix.actions[:, 1:], prefix.action_mask[:, 1:],
                         as_of=torch.ones(2), stage_valid=torch.ones(2, 4, dtype=torch.bool),
                         output_query_mask=torch.tensor([[False, False, True, True]]).expand(2, -1),
                         observed_stage_ids=((0, 1), (0, 1)),
                         observed_available_at=torch.ones(2, 2))
    with pytest.raises(ValueError, match="stage-aligned MRI availability"):
        model_v2.initialize(late)

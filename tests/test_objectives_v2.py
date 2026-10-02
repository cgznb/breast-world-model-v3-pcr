import copy
from dataclasses import replace
import torch
from test_training_v2 import cfg_v2, store_v2
from responsewm.training_v2 import build_model
from responsewm.losses_v2 import (free_rollout_objective,observed_update_objective,distribution_objectives)
from responsewm.rollout import NoiseLedger
from responsewm.state import Observation
from responsewm.conditioning import IntervalSpec
from conftest import open_gates


def test_independent_time_sample_permutation_changes_joint_energy():
    from responsewm.losses import energy_score
    samples=torch.tensor([[[[[0.]],[[0.]]],[[[2.]],[[2.]]]]])
    target=torch.tensor([[[[0.]],[[2.]]]])
    mask=torch.ones(1,2,dtype=torch.bool)
    shuffled=samples.clone()
    shuffled[:,:,1]=samples[:,[1,0],1]
    assert not torch.allclose(energy_score(samples,target,mask),energy_score(shuffled,target,mask))
    for j in range(2):
        assert torch.equal(energy_score(samples[:,:,j:j+1],target[:,j:j+1],mask[:,j:j+1]),
                           energy_score(shuffled[:,:,j:j+1],target[:,j:j+1],mask[:,j:j+1]))


def test_supervision_poison_does_not_change_v2_forecast(cfg_v2,store_v2):
    model=build_model(cfg_v2,store_v2).eval()
    inp,sup=store_v2.batch([(0,0)])
    with torch.no_grad():
        a=model.forecast(model.initialize(inp),samples=2,steps=1,noise_ledger=NoiseLedger(31))
        sup.future.fill_(12345);sup.label.fill_(1-sup.label.item())
        b=model.forecast(model.initialize(inp),samples=2,steps=1,noise_ledger=NoiseLedger(31))
    assert torch.equal(a.latent,b.latent) and torch.equal(a.logits,b.logits)


def test_stage_and_joint_distribution_have_distinct_terms(cfg_v2,store_v2):
    model=build_model(cfg_v2,store_v2).eval()
    inp,sup=store_v2.batch([(0,0)])
    trace=model.forecast(model.initialize(inp),samples=2,steps=1,noise_ledger=NoiseLedger(12))
    _,terms=distribution_objectives(model,trace,inp,sup)
    assert "loss/energy_stage" in terms and "loss/energy_joint" in terms
    assert terms["support/distribution"]==3
    assert abs(terms["loss/energy_stage"]-terms["loss/energy_joint"])>1e-8


def test_observed_update_trains_continuation_from_real_anchor(cfg_v2,store_v2):
    model=build_model(cfg_v2,store_v2)
    model.freeze_representation();model.configure_stage("joint")
    inp,sup=store_v2.batch([(0,0)])
    # Seed selecting T1 or T2 yields an actually observed next paired target.
    for seed in range(10):
        torch.manual_seed(seed)
        loss,terms=observed_update_objective(model,inp,sup,seed=3)
        if any(k.startswith("posterior_continuation/") for k in terms):
            break
    assert any(k.startswith("posterior_continuation/support/edge") for k in terms)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.assimilator.parameters())


def test_shared_head_and_observation_can_change_risk(cfg_v2,store_v2):
    model=build_model(cfg_v2,store_v2).eval()
    inp,_=store_v2.batch([(0,0)])
    head=model.pcr
    called=[]
    handle=head.register_forward_hook(lambda module,args,output:called.append(id(module)))
    state=model.initialize(inp)
    trace=model.forecast(state,samples=1,steps=1)
    prior=trace.all_states[0]
    real=prior.anchor_latent[:,0]+10
    posterior=model.observe(prior,Observation(1,real,("new-T1",)))
    before=model.query_pcr(prior,mode="forecasted_state").probability
    after=model.query_pcr(posterior).probability
    handle.remove()
    assert set(called)=={id(head)} and len(called)>=5
    assert not torch.equal(before,after)


def test_marginal_bce_reaches_both_stream_parameters(cfg_v2,store_v2):
    model=build_model(cfg_v2,store_v2)
    model.freeze_representation();model.configure_stage("joint")
    open_gates(model)
    inp,sup=store_v2.batch([(0,0)])
    trace=model.forecast(model.initialize(inp),samples=2,steps=1)
    from responsewm.losses import marginal_bernoulli_nll
    marginal_bernoulli_nll(trace.logits,sup.label,sup.label_mask).backward()
    assert model.velocity.image.out.weight.grad.abs().sum()>0
    assert model.velocity.semantic_out[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.target_encoder.parameters())

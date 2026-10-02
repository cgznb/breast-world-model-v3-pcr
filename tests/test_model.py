import copy
from dataclasses import replace
import inspect
import pytest
import torch
from responsewm.model import ResponseWorldModel
from responsewm.backbones import MonaiImageBackbone
from responsewm.losses import stage_loss,marginal_bernoulli_nll
from responsewm.contracts import gather_visit
from conftest import open_gates


def test_forward_shapes_probability_and_stochasticity(model,example):
    inp,_=example; model.eval(); open_gates(model)
    with torch.no_grad():
        out=model.forecast(inp,samples=3,steps=2,generator=torch.Generator().manual_seed(12))
    assert out.latent.shape==(2,3,2,24,4,8,8)
    assert out.state.shape==(2,3,2,4,24)
    assert torch.allclose(out.probability,out.logits.sigmoid().mean(1))
    assert not torch.equal(out.latent[:,0],out.latent[:,1])


def test_forecast_signature_contains_no_supervision(model,example):
    names=set(inspect.signature(model.forecast).parameters)
    assert not names & {'target','label','patient_id','supervision','teacher_forcing'}
    with pytest.raises(TypeError):
        model.forecast(example[0],target=example[1].future)


def test_poisoning_supervision_does_not_change_forecast(model,example):
    inp,sup=example; model.eval()
    with torch.no_grad():
        a=model.forecast(inp,samples=2,steps=1,generator=torch.Generator().manual_seed(999))
        sup.future.fill_(10000); sup.label.fill_(99)
        b=model.forecast(inp,samples=2,steps=1,generator=torch.Generator().manual_seed(999))
    assert torch.equal(a.latent,b.latent) and torch.equal(a.logits,b.logits)


def test_masked_history_cannot_contaminate_prediction(model,example):
    inp,_=example; model.eval(); open_gates(model)
    obs=inp.observed.clone(); obs[1,1]=999
    day=inp.observed_days.clone(); day[1,1]=9999
    altered=replace(inp,observed=obs,observed_days=day)
    with torch.no_grad():
        a=model.forecast(inp,samples=1,steps=1,generator=torch.Generator().manual_seed(1))
        b=model.forecast(altered,samples=1,steps=1,generator=torch.Generator().manual_seed(1))
    assert torch.equal(a.latent,b.latent) and torch.equal(a.logits,b.logits)


def test_last_valid_uses_index_not_count():
    x=torch.tensor([[10.,100.,30.]])
    mask=torch.tensor([[True,False,True]])
    assert gather_visit(x,mask).item()==30


def test_fixed_teacher_still_backpropagates_input(model,example):
    z=example[0].observed[:,0].clone().requires_grad_()
    value=model.target_encoder(z).disease
    value.square().sum().backward()
    assert z.grad is not None and z.grad.abs().sum()>0
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_pcr_gradient_reaches_both_streams(model,example):
    inp,sup=example; open_gates(model); model.eval()
    out=model.forecast(inp,samples=2,steps=2,generator=torch.Generator().manual_seed(9))
    loss=marginal_bernoulli_nll(out.logits,sup.label,sup.label_mask)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.velocity.image.parameters() if p.grad is not None)>0
    assert model.velocity.semantic_out[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.encoder.parameters())


def test_multihop_gradient_reaches_first_generated_interval(model,example,monkeypatch):
    inp,sup=example; open_gates(model); model.eval()
    original=model.sample_interval; recorded=[]
    def sample(*args,**kwargs):
        z,s=original(*args,**kwargs); z.retain_grad(); s.retain_grad(); recorded.append((z,s)); return z,s
    monkeypatch.setattr(model,'sample_interval',sample)
    out=model.forecast(inp,samples=1,steps=2)
    out.logits.sum().backward()
    assert recorded[0][0].grad is not None and recorded[0][0].grad.abs().sum()>0
    assert recorded[0][1].grad is not None and recorded[0][1].grad.abs().sum()>0


def test_vector_field_has_no_dropout(model,example):
    model.train(); z=torch.randn(2,48,4,8,8); s=torch.randn(2,8,24); c=torch.randn(2,5,24); tau=torch.rand(2)
    a=model.velocity(z,s,tau,c)
    torch.manual_seed(322)
    b=model.velocity(z,s,tau,c)
    assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_checkpointed_forward_and_backward_match(cfg,example):
    cfg.network.checkpoint_blocks=False
    m=ResponseWorldModel(cfg,3,2); m.freeze_representation(); m.configure_stage('joint'); open_gates(m); m.eval()
    n=copy.deepcopy(m); n.cfg=copy.deepcopy(cfg); n.cfg.network.checkpoint_blocks=True; n.velocity.cfg=n.cfg
    inp,sup=example
    a=m.forecast(inp,samples=1,steps=1,generator=torch.Generator().manual_seed(4))
    b=n.forecast(inp,samples=1,steps=1,generator=torch.Generator().manual_seed(4))
    assert torch.allclose(a.logits,b.logits,atol=1e-6)
    a.logits.sum().backward(); b.logits.sum().backward()
    assert torch.allclose(m.velocity.semantic_out[-1].weight.grad,n.velocity.semantic_out[-1].weight.grad,atol=1e-6)


def test_all_four_loss_paths_are_finite(cfg,example):
    inp,sup=example; m=ResponseWorldModel(cfg,3,2)
    for stage in ('representation','flow','readout','joint'):
        if stage=='flow':
            m.freeze_representation()
        params=m.configure_stage(stage); optimizer=torch.optim.AdamW(params,lr=1e-3)
        optimizer.zero_grad(); loss,_=stage_loss(m,inp,sup,stage)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        optimizer.step(); m.update_target()


def test_reencode_ablation_removes_semantic_shortcut(cfg,example):
    cfg.network.coupling=False; cfg.network.readout_source='reencode'
    m=ResponseWorldModel(cfg,3,2); m.freeze_representation(); m.configure_stage('joint'); m.eval(); open_gates(m)
    result=m.forecast(example[0],samples=1,steps=2)
    result.logits.sum().backward()
    assert m.velocity.image.out.weight.grad is not None and m.velocity.image.out.weight.grad.abs().sum()>0
    assert m.velocity.semantic_out[-1].weight.grad is None or m.velocity.semantic_out[-1].weight.grad.abs().sum()==0


def test_monai_split_parity(cfg):
    monai=pytest.importorskip('monai',reason='MONAI is an optional production dependency; not installed in this CPU environment')
    if monai.__version__!='1.5.1':
        pytest.skip('Pinned MONAI 1.5.1 required')
    cfg.network.channels=(16,32,32); cfg.network.backend='monai'
    image=MonaiImageBackbone(cfg.network,24).eval()
    # Nonzero output checks actual internal forward equivalence, not just zero init.
    with torch.no_grad():
        for p in image.network.out[-1].parameters():
            p.normal_(0,.01)
        z=torch.randn(2,48,4,8,8); tau=torch.rand(2); context=torch.randn(2,4,24)
        original=image.network(z,tau,context=context)
        split=image(z,tau,context)
    assert torch.allclose(original,split,atol=1e-6,rtol=1e-5)


def test_terminal_landmark_without_future(model,example):
    inp,_=example; model.eval()
    terminal=replace(inp,future_days=inp.future_days[:,:0],future_mask=inp.future_mask[:,:0],
                     actions=inp.actions[:,:0],action_mask=inp.action_mask[:,:0])
    with torch.no_grad(): out=model.forecast(terminal,samples=2,steps=1)
    assert out.latent.shape[2]==0 and out.state.shape[2]==0
    assert torch.equal(out.residuals,torch.zeros_like(out.residuals))
    assert torch.allclose(out.probability,out.observed_logit.sigmoid())


def test_no_future_readout_ablation_is_noise_invariant(cfg,example):
    cfg.network.use_future=False
    m=ResponseWorldModel(cfg,3,2).eval();open_gates(m)
    with torch.no_grad():
        a=m.forecast(example[0],samples=1,steps=1,generator=torch.Generator().manual_seed(1))
        b=m.forecast(example[0],samples=1,steps=1,generator=torch.Generator().manual_seed(999))
    assert not torch.equal(a.latent,b.latent)
    assert torch.equal(a.probability,b.probability)

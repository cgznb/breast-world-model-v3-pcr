import copy
import numpy as np
import pytest
import torch
from symm_observation.config import from_dict
from symm_observation.observation_encoder import ThreePhaseObservationEncoder, positions_3d, sine_position
from symm_observation.observation_heads import ObservationPredictor, SpatialReadout, GradientReverse
from symm_observation.observation_views import observation_batch, patch_affine, map_query_positions, block_mask
from symm_observation.observation_ssl import ObservationPretrainer


def test_production_depth_not_toy():
    c=from_dict({})
    assert c.encoder.depth == 12 and c.encoder.predictor_depth == 6 and c.encoder.phase_depth == 2
    assert tuple(c.velocity.channels)==(64,128,256,256)

@pytest.mark.parametrize('section,key,value',[
    ('observation','future_weight',1),('encoder','future_predictor_depth',4),
    ('training','precision','fp16'),('observation','dann_weight',.1),
    ('encoder','teacher_last_k',15),('observation','mask_ratio',1),
    ('velocity','attention_levels',[True]),('sampling','steps',0)])
def test_bad_config_rejected(cfg,section,key,value):
    obj=cfg.to_dict();obj[section][key]=value
    if key=='dann_weight': obj['observation']['domain_definition']='treatment'
    with pytest.raises(ValueError): from_dict(obj)


def test_masked_values_do_not_reach_context(cfg):
    e=ThreePhaseObservationEncoder(cfg.encoder).eval()
    z=torch.randn(1,24,4,4,4)
    visible=torch.ones(1,16,dtype=torch.bool);visible[:,0]=False
    altered=z.clone();altered[:,:,0:1,0:2,0:2]=10000
    f=e(z,visible=visible);g=e(altered,visible=visible)
    assert f.tokens.shape[1]==15
    torch.testing.assert_close(f.tokens,g.tokens,rtol=0,atol=0)


def test_all_three_phase_tokens_dropped_before_mixing(cfg):
    e=ThreePhaseObservationEncoder(cfg.encoder).eval()
    z=torch.randn(1,24,4,4,4)
    visible=torch.ones(1,16,dtype=torch.bool);visible[:,3]=False
    for phase in range(3):
        changed=z.clone();changed[:,phase*8:(phase+1)*8,0:1,2:4,2:4]+=100
        torch.testing.assert_close(e(z,visible=visible).tokens,e(changed,visible=visible).tokens,rtol=0,atol=0)


def test_ragged_batch_padding_does_not_change_valid_tokens(cfg):
    e=ThreePhaseObservationEncoder(cfg.encoder).eval()
    z=torch.randn(2,24,4,4,4);mask=torch.ones(2,16,dtype=torch.bool);mask[0,4:]=False
    together=e(z,visible=mask)
    alone=e(z[:1],visible=mask[:1])
    torch.testing.assert_close(together.tokens[0,:4],alone.tokens[0],rtol=1e-5,atol=1e-6)
    assert together.padding[0,4:].all()


def test_relative_affines_translation_and_rotation():
    base=torch.eye(4,dtype=torch.float64);base[0,0]=2;base[1,1]=3
    aa=patch_affine(base,(0,0,0),(1,2,2))[None]
    bb=patch_affine(base,(2,4,0),(1,2,2))[None]
    q=torch.tensor([[[0.,0.,0.],[1.,1.,1.]]])
    mapped=map_query_positions(aa,bb,q)
    torch.testing.assert_close(mapped,q+torch.tensor([2.,2.,0.]))
    rot=torch.tensor([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],dtype=torch.float64)
    ra=rot[None]@aa;rb=rot[None]@bb
    torch.testing.assert_close(map_query_positions(ra,rb,q),mapped)


def test_real_valued_sine_position():
    x=torch.tensor([[.2,-1.,4.1]])
    y=sine_position(x,48)
    assert y.shape==(1,48) and torch.isfinite(y).all()
    assert not torch.equal(y,sine_position(x+.1,48))


def test_mask_keeps_context_and_valid_targets():
    valid=np.ones(64,dtype=bool);valid[:12]=False
    mask=block_mask(valid,(4,4,4),.7,4,np.random.default_rng(1)).numpy()
    assert 0<mask.sum()<valid.sum()
    assert not mask[:12].any()


def test_single_visit_four_way_ssl_backward(cfg,visits):
    m=ObservationPretrainer(cfg,visits.fit_statistics())
    batch=observation_batch(visits,visits.records('train')[:2],cfg,0,'cpu')
    loss,parts=m(batch,0);loss.backward()
    assert {'local','global','reconstruction'}<=set(parts)
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None)>0
    assert sum(float(p.grad.abs().sum()) for p in m.predictor.parameters() if p.grad is not None)>0
    assert not any('future' in name.lower() or 'clinical' in name.lower() for name,_ in m.named_modules())


def test_teacher_ema_and_eval_mode(cfg,visits):
    m=ObservationPretrainer(cfg,visits.fit_statistics());m.train()
    assert not m.target_encoder.training
    old=next(m.target_encoder.parameters()).clone()
    with torch.no_grad(): next(m.encoder.parameters()).add_(1.)
    decay=m.update_target(1)
    torch.testing.assert_close(next(m.target_encoder.parameters()),old+(1-decay),rtol=1e-5,atol=1e-6)


def test_A_rejects_future_in_batch(cfg,visits):
    m=ObservationPretrainer(cfg,visits.fit_statistics())
    batch=observation_batch(visits,visits.records('train')[:1],cfg,0,'cpu')
    batch['future']=torch.randn(1,24,4,4,4)
    with pytest.raises(ValueError):m(batch,0)


def test_A_does_not_require_any_pair_file(cfg,dataset,visits):
    from pathlib import Path
    Path(dataset['pairs']).unlink()
    rows=[r for r in visits.records('train') if r['patient_id']=='synthetic_03']
    assert len(rows)==1
    m=ObservationPretrainer(cfg,visits.fit_statistics())
    batch=observation_batch(visits,rows,cfg,0,'cpu')
    loss,_=m(batch,0)
    assert torch.isfinite(loss)


def test_global_location_and_domain_parameters_are_used(cfg,visits):
    m=ObservationPretrainer(cfg,visits.fit_statistics()).eval()
    b=observation_batch(visits,visits.records('train')[:1],cfg,0,'cpu')['a']
    f=m.encoder(b['context'],b['valid'],b['visible'])
    q=positions_3d(f.grid,'cpu')[None]
    good=b['good'];a=torch.zeros(1,4)
    p=m.predictor(f,q,a,~good)
    shifted=m.predictor(f,q+1,a,~good)
    transformed=m.predictor(f,q,a+.1,~good)
    assert not torch.equal(p,shifted) and not torch.equal(p,transformed)


def test_reconstruction_no_raw_skip_and_input_grad(cfg):
    e=ThreePhaseObservationEncoder(cfg.encoder)
    from symm_observation.observation_heads import SpatialReadout
    r=SpatialReadout(cfg.encoder,channels=24)
    z=torch.randn(1,24,4,4,4,requires_grad=True)
    f=e(z);out=r(f)
    assert out.shape==z.shape
    out.square().mean().backward()
    assert z.grad is not None and z.grad.abs().sum()>0


def test_gradient_reversal():
    x=torch.tensor([2.,3.],requires_grad=True)
    GradientReverse.apply(x,.25).sum().backward()
    torch.testing.assert_close(x.grad,torch.full((2,),-.25))


def test_activation_checkpoint_gradient(cfg):
    cfg.encoder.checkpoint_blocks=True
    m=ThreePhaseObservationEncoder(cfg.encoder).train()
    z=torch.randn(1,24,4,4,4,requires_grad=True)
    m(z).tokens[:,:,0].mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0

import copy
from pathlib import Path
import pytest
import torch
from symm_observation.flow import make_path, estimate_endpoints, integrate
from symm_observation.conditioning import ConditionSchema
from symm_observation.observation_encoder import ThreePhaseObservationEncoder
from symm_observation.world_model import ThreePhaseWorldModel
from symm_observation.training import train, load_world, load_observation
from symm_observation.utils import load_checkpoint, file_digest, write_json
from symm_observation.cli import sample_file, extract_file


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('time',[0.,.2,.8,1.])
def test_original_symmflow_endpoint_identities(time):
    early=torch.randn(2,24,2,2,2);late=torch.randn_like(early)
    path=make_path(early,late,tau=torch.full((2,),time))
    e,l=estimate_endpoints(path.joint,path.velocity,path.tau)
    torch.testing.assert_close(e,early,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(l,late,atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('method',['euler','heun'])
@pytest.mark.parametrize('direction',[-1,1])
def test_ode_constant_field(method,direction):
    z=torch.randn(1,48,2,2,2)
    out=integrate(lambda x,t,c: torch.ones_like(x),z,None,steps=3,method=method,direction=direction)
    torch.testing.assert_close(out,z+direction)


def test_B_current_context_only(cfg,visits,pairs):
    encoder=ThreePhaseObservationEncoder(cfg.encoder,visits.fit_statistics())
    model=ThreePhaseWorldModel(cfg,encoder,ConditionSchema.fit(pairs.records('train'))).eval()
    batch=pairs.batch(pairs.records('train')[:1],'cpu')
    source=model.observation_encoder.normalize(batch['source_raw'])
    c1=model.build_context(source,batch['conditions'],batch['source_valid'])
    batch['target_raw'].fill_(123456.)
    c2=model.build_context(source,batch['conditions'],batch['source_valid'])
    torch.testing.assert_close(c1,c2,rtol=0,atol=0)
    assert not hasattr(model,'predictor') and not any('future' in n for n,_ in model.named_modules())
    model.train();assert not model.observation_encoder.training


def test_frozen_encoder_allows_endpoint_input_derivative(cfg,visits,pairs):
    encoder=ThreePhaseObservationEncoder(cfg.encoder,visits.fit_statistics())
    model=ThreePhaseWorldModel(cfg,encoder,ConditionSchema.fit(pairs.records('train')))
    z=torch.randn(1,24,8,8,8,requires_grad=True)
    out=model.current_features(z,torch.ones(1,1,8,8,8))
    out.tokens[:,:,0].mean().backward()
    assert z.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in model.observation_encoder.parameters())


def test_stage_A_exact_resume(cfg,dataset,tmp_path):
    continuous=tmp_path/'continuous';resumed=tmp_path/'resumed'
    train('A',cfg,dataset['visits'],continuous)
    train('A',cfg,dataset['visits'],resumed,stop_after=1)
    with pytest.raises(ValueError):load_observation(resumed/'A/last.pt')
    train('A',cfg,dataset['visits'],resumed,resume=True)
    full=load_checkpoint(continuous/'A/last.pt');other=load_checkpoint(resumed/'A/last.pt')
    for key in ('model','optimizer','scheduler','rng'):assert_tree_equal(full[key],other[key])
    assert full['best_score']==other['best_score']
    before=file_digest(resumed/'A/best.pt')
    train('A',cfg,dataset['visits'],resumed,resume=True)
    assert file_digest(resumed/'A/best.pt')==before


def test_stage_B_exact_resume_and_source_only_inference(cfg,dataset,tmp_path,pairs):
    a=tmp_path/'pretrain';full=tmp_path/'bfull';continued=tmp_path/'bresume'
    train('A',cfg,dataset['visits'],a)
    kwargs={'observation_checkpoint':a/'A/best.pt'}
    train('B',cfg,dataset['pairs'],full,**kwargs)
    train('B',cfg,dataset['pairs'],continued,stop_after=1,**kwargs)
    train('B',cfg,dataset['pairs'],continued,resume=True,**kwargs)
    x=load_checkpoint(full/'B/last.pt');y=load_checkpoint(continued/'B/last.pt')
    for key in ('model','teacher','optimizer','scheduler','rng'):assert_tree_equal(x[key],y[key])
    assert x['best_score']==y['best_score']
    pair=pairs.records('test')[0];source=pairs.base/pair['source']['latent_path']
    target=pairs.base/pair['target']['latent_path'];target.unlink()
    cp=tmp_path/'conditions.json';write_json(cp,pair['conditions'])
    report=sample_file(full/'B/best.pt',source,cp,tmp_path/'prediction.npz',
                       codec_id=pairs.codec_id,samples=2,device='cpu')
    assert report['shape']==[2,24,8,8,8] and not target.exists()
    features=extract_file(a/'A/best.pt',source,tmp_path/'features.npz',codec_id=pairs.codec_id)
    assert features['visit_only'] and not features['uses_ssl_predictor']


def test_resume_rejects_config_change(cfg,dataset,tmp_path):
    train('A',cfg,dataset['visits'],tmp_path/'run',stop_after=1)
    cfg.training.lr_a*=2
    with pytest.raises(ValueError):train('A',cfg,dataset['visits'],tmp_path/'run',resume=True)


def test_B_requires_completed_A(cfg,dataset,tmp_path):
    train('A',cfg,dataset['visits'],tmp_path/'a',stop_after=1)
    with pytest.raises(ValueError):train('B',cfg,dataset['pairs'],tmp_path/'b',observation_checkpoint=tmp_path/'a/A/last.pt')


def test_monai_real_backend_if_installed(cfg):
    pytest.importorskip('monai')
    from symm_observation.velocity import build_velocity
    cfg.velocity.backend='monai'
    net=build_velocity(cfg.velocity,cfg.velocity.context_dim)
    out=net(torch.randn(1,48,8,8,8),torch.tensor([.4]),torch.randn(1,11,48))
    assert out.shape==(1,48,8,8,8)
    out.square().mean().backward()

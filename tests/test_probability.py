import pytest
import torch
from responsewm.losses import marginal_bernoulli_nll,bernoulli_nll,energy_score,spatial_alignment
from responsewm.flow import integrate,make_joint_path
from responsewm.metrics import classification_metrics,patient_bootstrap,paired_bootstrap_delta


def test_marginal_is_probability_mean_not_logit_mean():
    logits=torch.tensor([[-4.,1.],[1.,3.]],requires_grad=True)
    y=torch.tensor([1.,0.]); mask=torch.ones(2,dtype=torch.bool)
    actual=marginal_bernoulli_nll(logits,y,mask)
    p=logits.sigmoid().mean(1)
    expected=-(y*p.log()+(1-y)*(1-p).log()).mean()
    assert torch.allclose(actual,expected)
    assert not torch.allclose(p,logits.mean(1).sigmoid())
    actual.backward()
    assert (logits.grad.abs()>0).all()


def test_jensen_difference():
    l=torch.tensor([[-5.,2.,.1],[3.,-2.,.4]])
    y=torch.tensor([1.,0.]); m=torch.ones(2,dtype=torch.bool)
    assert marginal_bernoulli_nll(l,y,m)<marginal_bernoulli_nll(l,y,m,per_sample=True)


def test_extreme_logits_are_stable():
    l=torch.tensor([[10000.,10001.],[-10000.,-10001.]],requires_grad=True)
    loss=marginal_bernoulli_nll(l,torch.tensor([0.,1.]),torch.ones(2,dtype=torch.bool))
    assert torch.isfinite(loss) and loss>9000
    loss.backward(); assert torch.isfinite(l.grad).all()


def test_missing_labels_zero_gradient():
    l=torch.tensor([[1.,-2.],[3.,4.]],requires_grad=True)
    loss=marginal_bernoulli_nll(l,torch.tensor([1.,999.]),torch.tensor([True,False]))
    loss.backward(); assert torch.equal(l.grad[1],torch.zeros(2))
    with pytest.raises(ValueError):
        bernoulli_nll(l[:,0],torch.tensor([2.,0.]),torch.ones(2,dtype=torch.bool))


def test_energy_permutation_and_missing_future():
    x=torch.randn(2,4,3,2,5,requires_grad=True); y=torch.randn(2,3,2,5)
    mask=torch.tensor([[True,False,True],[False,False,False]])
    a=energy_score(x,y,mask)
    b=energy_score(x[:,[2,0,3,1]],y,mask)
    assert torch.allclose(a,b,atol=1e-6)
    a.backward(); assert x.grad[1].abs().sum()==0
    assert x.grad[0,:,1].abs().sum()==0


def test_energy_never_mixes_patients():
    x=torch.randn(2,3,2,2,4); y=torch.randn(2,2,2,4); mask=torch.ones(2,2,dtype=torch.bool)
    total=energy_score(x,y,mask)
    expected=(energy_score(x[:1],y[:1],mask[:1])+energy_score(x[1:],y[1:],mask[1:]))/2
    assert torch.allclose(total,expected)
    with pytest.raises(ValueError):
        energy_score(x[:,:1],y,mask)


def test_spatial_target_cannot_be_global():
    with pytest.raises(ValueError):
        spatial_alignment(torch.randn(2,1,8),torch.randn(2,1,8))


def test_symmflow_endpoints_and_directions():
    a,b=torch.randn(2,24,2,4,4),torch.randn(2,24,2,4,4)
    sa,sb=torch.randn(2,4,8),torch.randn(2,4,8)
    z0,s0,_,_,_=make_joint_path(a,b,sa,sb,tau=torch.zeros(2))
    z1,s1,_,_,_=make_joint_path(a,b,sa,sb,tau=torch.ones(2))
    assert torch.equal(z0[:,24:],a) and torch.equal(s0[:,4:],sa)
    assert torch.equal(z1[:,:24],b) and torch.equal(s1[:,:4],sb)
    def velocity(z,s,t,c):
        return torch.ones_like(z)*2,torch.ones_like(s)*3,z.new_zeros(len(z),2,8)
    for method in ('euler','heun'):
        z,s=integrate(velocity,z0,s0,torch.zeros(2,1,8),4,method)
        assert torch.allclose(z,z0+2,atol=1e-6)
        zr,sr=integrate(velocity,z,s,torch.zeros(2,1,8),4,method,-1)
        assert torch.allclose(zr,z0,atol=1e-6) and torch.allclose(sr,s0,atol=1e-6)


def test_probability_metrics_and_cluster_ci():
    y=[0,1,0,1]; p=[.1,.9,.3,.8]; ids=['a','b','a','b']
    result=classification_metrics(y,p)
    assert result['auroc']==1.0
    ci=patient_bootstrap(y,p,ids,20)
    assert ci['brier']['valid_replicates']==20
    delta=paired_bootstrap_delta(y,p,p,ids,20)
    assert delta['brier']['lower']==0 and delta['brier']['upper']==0

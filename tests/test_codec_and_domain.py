from dataclasses import asdict
import copy
import numpy as np
import pytest
import torch
from symm_observation.codec import VQConfig, MRILevelVQGAN, FrozenThreePhaseCodec, load_codec
from symm_observation.preparation import image_transform, prepare_sidecars
from symm_observation.data import VisitStore, PHASES, VISIT_SCHEMA
from symm_observation.observation_views import observation_batch
from symm_observation.observation_ssl import ObservationPretrainer
from symm_observation.utils import write_json, codec_file_id


def make_codec(tmp_path):
    c=VQConfig(hidden_channels=4,num_groups=4,n_codes=16,nearest_chunk_size=128)
    m=MRILevelVQGAN(c)
    p=tmp_path/'codec.pt'
    torch.save({'schema':'first_post_unregistered_tumor_roi_v1','model_config':asdict(c),'codec_state':m.state_dict()},p)
    return p, FrozenThreePhaseCodec(m)


def test_frozen_vq_shape_input_gradient_and_immutable_codebook(tmp_path):
    path,codec=make_codec(tmp_path)
    codec.train()
    assert not codec.training and not codec.codec.training
    images=torch.randn(1,3,16,16,16)
    latent=codec.encode(images)
    assert latent.shape==(1,24,4,4,4)
    z=latent.detach().requires_grad_(True)
    before={k:v.clone() for k,v in codec.state_dict().items()}
    reconstruction=codec.decode(z)
    assert reconstruction.shape==images.shape
    reconstruction.square().mean().backward()
    assert z.grad is not None and z.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    for k,v in codec.state_dict().items():assert torch.equal(v,before[k])
    reloaded=load_codec(path)
    torch.testing.assert_close(reloaded.encode(images),latent,rtol=0,atol=0)


def test_same_gain_and_offset_preserve_phase_difference_relation():
    x=torch.randn(1,3,8,8,8)
    y=image_transform(x,[.1,.2,0,0])
    torch.testing.assert_close(y[:,1]-y[:,0],1.1*(x[:,1]-x[:,0]),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(image_transform(x,[0,0,0,0]),x)


def test_blur_and_noise_transform_reproducible():
    x=torch.randn(1,3,8,8,8)
    a=image_transform(x,[0,0,.01,.5],torch.Generator().manual_seed(7))
    b=image_transform(x,[0,0,.01,.5],torch.Generator().manual_seed(7))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(a,x)


def test_actual_image_backed_bank_and_A_loss(tmp_path,cfg):
    path,codec=make_codec(tmp_path)
    images=torch.randn(1,3,32,32,32)
    latent=codec.encode(images)[0].numpy()
    np.save(tmp_path/'image.npy',images[0].numpy());np.save(tmp_path/'latent.npy',latent)
    np.save(tmp_path/'valid.npy',np.ones((3,8,8,8),dtype='float32'))
    row={'id':'v','patient_id':'p','visit_id':'p_T0','stage':'T0','split':'train',
         'latent_path':'latent.npy','image_path':'image.npy','valid_path':'valid.npy',
         'phase_alignment_verified':True,'geometry':{'latent_to_lps':np.eye(4).tolist()},
         'image_normalization':{'stored':'normalized','shared_across_phases':True,'mean':0.,'std':1.}}
    manifest={'schema':VISIT_SCHEMA,'phase_order':PHASES,'latent_channels':24,
              'codec_id':codec_file_id(path),'visits':[row]}
    write_json(tmp_path/'visits.json',manifest)
    result=prepare_sidecars(tmp_path/'visits.json',path,tmp_path/'prepared',samples=2)
    store=VisitStore(result['visits_manifest'])
    cfg.observation.domain_enabled=True
    batch=observation_batch(store,store.records('train'),cfg,0,'cpu')
    assert batch['a']['action'].abs().sum()>0
    assert not torch.equal(batch['a']['context'],batch['a']['clean'])
    loss,details=ObservationPretrainer(cfg,store.fit_statistics())(batch,0)
    assert details['kinetics_labels']==1 and torch.isfinite(loss)
    loss.backward()
    altered=copy.deepcopy(store.manifest)
    altered['visits'][0]['augmentations'][0]['transform_space']='latent'
    write_json(tmp_path/'invalid.json',altered)
    with pytest.raises(ValueError):VisitStore(tmp_path/'invalid.json')


def test_domain_requires_image_bank(visits,cfg):
    cfg.observation.domain_enabled=True
    with pytest.raises(ValueError):observation_batch(visits,visits.records('train')[:1],cfg,0,'cpu')


def test_kinetic_proxy_cannot_be_called_measured(visits,tmp_path):
    m=copy.deepcopy(visits.manifest);m['visits'][0]['kinetic_provenance']='codec_reconstruction_proxy'
    write_json(tmp_path/'wrong.json',m)
    with pytest.raises(ValueError):VisitStore(tmp_path/'wrong.json')

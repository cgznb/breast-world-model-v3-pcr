import torch
import numpy as np
import pytest
from responsewm.legacy.codec import VQConfig,MRILevelVQGAN,FrozenThreePhaseCodec,load_codec
from dataclasses import asdict


def test_real_codec_classes_roundtrip_and_input_gradient(tmp_path):
    torch.manual_seed(3)
    cfg=VQConfig(hidden_channels=4,num_groups=4,n_codes=16)
    original=MRILevelVQGAN(cfg)
    cp=tmp_path/'codec.pt'
    torch.save({'schema':'symm_world_codec_v2','model_config':asdict(cfg),'codec_state':original.state_dict()},cp)
    codec=load_codec(cp)
    images=torch.randn(1,3,8,16,16)
    latent=codec.encode(images)
    assert latent.shape==(1,24,2,4,4)
    assert not latent.requires_grad
    latent=latent.detach().requires_grad_()
    old=codec.codec.quantizer.embeddings.clone()
    decoded=codec.decode(latent)
    assert decoded.shape==images.shape
    decoded.square().mean().backward()
    assert latent.grad is not None and latent.grad.abs().sum()>0
    assert all(not p.requires_grad and p.grad is None for p in codec.parameters())
    assert torch.equal(old,codec.codec.quantizer.embeddings)
    codec.train()
    assert not codec.training


def test_codec_rejects_unrecognized_weights(tmp_path):
    cp=tmp_path/'unknown.pt'; torch.save({'random':torch.ones(1)},cp)
    with pytest.raises(ValueError): load_codec(cp)

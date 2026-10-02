import copy
from pathlib import Path
import pytest
import torch
from responsewm.config import load_config
from responsewm.model import ResponseWorldModel
from responsewm.contracts import ForecastInput,Supervision

@pytest.fixture
def cfg():
    value=load_config(Path(__file__).resolve().parents[1]/"configs/smoke.yaml")
    value.training.threads=2
    torch.set_num_threads(2)
    torch.manual_seed(123)
    return value

@pytest.fixture
def example(cfg):
    b,t,f=2,2,2
    x=ForecastInput(torch.randn(b,t,24,4,8,8),torch.tensor([[True,True],[True,False]]),
                    torch.tensor([[0.,15.],[0.,0.]]),torch.randn(b,3),torch.ones(b,3,dtype=torch.bool),
                    torch.tensor([[30.,60.],[30.,60.]]),torch.ones(b,f,dtype=torch.bool),
                    torch.randn(b,f,2),torch.ones(b,f,2,dtype=torch.bool))
    y=Supervision(torch.randn(b,f,24,4,8,8),torch.tensor([[True,True],[False,True]]),
                  torch.tensor([1.,0.]),torch.ones(b,dtype=torch.bool),[[{} for _ in range(t+f)] for _ in range(b)],
                  torch.zeros(b,f,dtype=torch.bool))
    return x,y

@pytest.fixture
def model(cfg):
    value=ResponseWorldModel(cfg,3,2)
    value.freeze_representation()
    value.configure_stage("joint")
    return value


def open_gates(model):
    """Analytic gradient tests emulate warmed-up residual gates, not pretrained weights."""
    with torch.no_grad():
        for bridge in model.velocity.bridges:
            bridge.image_gate.fill_(.1); bridge.state_gate.fill_(.1)
        for block in model.velocity.blocks:
            block.modulation[-1].bias.fill_(.05)
        torch.nn.init.normal_(model.velocity.semantic_out[-1].weight,std=.02)
        if model.cfg.network.backend == "native":
            torch.nn.init.normal_(model.velocity.image.out.weight,std=.01)

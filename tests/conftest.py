from pathlib import Path
import pytest
import torch
from symm_observation.config import load_config
from symm_observation.synthetic import make_synthetic
from symm_observation.data import VisitStore, PairStore

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)

@pytest.fixture
def cfg():
    return load_config(ROOT/'configs/smoke.yaml')

@pytest.fixture
def dataset(tmp_path):
    return make_synthetic(tmp_path/'data')

@pytest.fixture
def visits(dataset):
    return VisitStore(dataset['visits'])

@pytest.fixture
def pairs(dataset):
    return PairStore(dataset['pairs'])

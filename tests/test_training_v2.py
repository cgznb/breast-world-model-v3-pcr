import json
from pathlib import Path
import shutil

import pytest
import torch

from responsewm.config import load_config
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.io import load_checkpoint
from responsewm.losses_v2 import free_rollout_objective, prefix_objective
from responsewm.samplers import BalancedEdgeSampler
from responsewm.synthetic_v2 import make_synthetic_v2
from responsewm.training_v2 import STAGES, build_model, train_stage


@pytest.fixture
def cfg_v2():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    cfg.schema = "responsewm_v2"
    cfg.network.time_basis = "stage_index"
    cfg.network.semantic_depth = 6
    cfg.network.dropout = 0
    cfg.encoder.widths = (12, 24, 48)
    cfg.encoder.depths = (1, 1, 1)
    cfg.encoder.heads = (3, 3, 3)
    cfg.training.batch_size = 1
    cfg.training.validation_cases = 1
    cfg.training.reverse_probability = 0
    cfg.training.warmup = 0
    cfg.multistage.memory_tokens = 8
    cfg.multistage.global_tokens = 2
    cfg.multistage.assimilation_depth = 1
    cfg.multistage.condition_depth = 1
    cfg.multistage.deep_jepa_weights = (.2, .3, 1.)
    cfg.multistage.rollout_every = 1
    cfg.multistage.early_stopping_patience = 0
    cfg.multistage.bridge_auxiliary_weight = 0
    cfg.sampling.train_steps = 1
    cfg.sampling.inference_steps = 1
    cfg.training.flow_steps = 2
    cfg.training.readout_steps = 1
    cfg.training.joint_steps = 1
    torch.set_num_threads(2)
    return cfg.validate()


@pytest.fixture
def store_v2(tmp_path):
    path = make_synthetic_v2(tmp_path / "synthetic")
    store = PatientTrajectoryStore(path, allow_synthetic=True)
    store.fit_statistics()
    return store


def test_zero_label_support_has_zero_pcr_gradient(cfg_v2, store_v2):
    model = build_model(cfg_v2, store_v2)
    model.configure_stage("representation")
    inp, sup = store_v2.batch([(0, 0)])
    sup.label_mask.zero_()
    loss, _, _ = prefix_objective(model, inp, sup)
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.pcr.parameters())


def test_free_rollout_three_hops_and_missing_truth(cfg_v2, store_v2):
    model = build_model(cfg_v2, store_v2)
    model.freeze_representation()
    model.configure_stage("joint")
    inp, sup = store_v2.batch([(1, 0)])
    assert not sup.future_mask[0, 0]
    loss, terms = free_rollout_objective(model, inp, sup, seed=123, max_hops=3, marginal_weight=.1)
    assert terms["rollout_depth"] == 3
    assert terms["support/target_T1"] == 0 and terms["support/target_T2"] == 1
    assert terms["support/target_T3"] == 1
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.velocity.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_four_stage_curriculum_finite_and_checkpoint_schema(cfg_v2, store_v2, tmp_path):
    root = tmp_path / "run"
    for stage in STAGES:
        result = train_stage(store_v2, cfg_v2, root, stage)
        assert result["completed"]
        checkpoint = load_checkpoint(root / stage / "last.pt")
        assert checkpoint["schema"] == "responsewm_checkpoint_v2"
        assert checkpoint["completed"]
        records = [json.loads(line) for line in (root / stage / "training.jsonl").read_text().splitlines()]
        assert records and all(torch.isfinite(torch.tensor(r["loss"])) for r in records)
        assert all(torch.isfinite(torch.tensor(r["gradient_norm"])) for r in records)
    flow = load_checkpoint(root / "flow" / "last.pt")
    assert flow["rollout_updates"] > 0
    flow_records = [json.loads(line) for line in (root / "flow" / "training.jsonl").read_text().splitlines()]
    assert any(record.get("rollout_depth") == 3 for record in flow_records)
    assert (root / "flow" / "best.pt").exists()
    sampler = BalancedEdgeSampler(store_v2, 1)
    assert sum(e.effective_weight for e in sampler.all_edges()) == pytest.approx(1.)


def test_step_boundary_resume_exact(cfg_v2, store_v2, tmp_path):
    cfg_v2.training.strict_determinism = True
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train_stage(store_v2, cfg_v2, full, "representation")
    train_stage(store_v2, cfg_v2, resumed, "representation", stop_after=1)
    train_stage(store_v2, cfg_v2, resumed, "representation", resume=True)
    expected = load_checkpoint(full / "representation" / "last.pt")
    actual = load_checkpoint(resumed / "representation" / "last.pt")
    assert expected["completed"] and actual["completed"]
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    assert torch.equal(expected["samplers"]["landmarks"]["generator"], actual["samplers"]["landmarks"]["generator"])


def test_joint_step_boundary_resume_exact(cfg_v2, store_v2, tmp_path):
    cfg_v2.training.strict_determinism = True
    cfg_v2.training.joint_steps = 2
    cfg_v2.training.validation_every = 3
    full, resumed = tmp_path / "full_joint", tmp_path / "resumed_joint"
    for stage in STAGES[:-1]:
        train_stage(store_v2, cfg_v2, full, stage)
    shutil.copytree(full, resumed)
    train_stage(store_v2, cfg_v2, full, "joint")
    result = train_stage(store_v2, cfg_v2, resumed, "joint", stop_after=1)
    assert not result["completed"]
    train_stage(store_v2, cfg_v2, resumed, "joint", resume=True)
    expected = load_checkpoint(full / "joint" / "last.pt")
    actual = load_checkpoint(resumed / "joint" / "last.pt")
    assert expected["completed"] and actual["completed"]
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    for sampler in ("landmarks", "origins", "edges"):
        assert torch.equal(expected["samplers"][sampler]["generator"], actual["samplers"][sampler]["generator"])


def test_four_stage_physical_batch_eight(cfg_v2, store_v2, tmp_path, monkeypatch):
    cfg_v2.training.batch_size = 8
    cfg_v2.training.accumulation = 1
    cfg_v2.training.log_every = 1
    original_batch = store_v2.batch
    seen = []

    def batch(tasks, *args, **kwargs):
        seen.append(len(tasks))
        return original_batch(tasks, *args, **kwargs)

    monkeypatch.setattr(store_v2, "batch", batch)
    root = tmp_path / "batch_eight"
    for stage in STAGES:
        before = len(seen)
        result = train_stage(store_v2, cfg_v2, root, stage)
        assert result["completed"]
        assert 8 in seen[before:]
        records = [json.loads(line) for line in (root / stage / "training.jsonl").read_text().splitlines()]
        assert all(row["physical_batch_min"] == row["physical_batch_max"] == 8 for row in records)
        assert all(torch.isfinite(torch.tensor(row["loss"])) for row in records)


def test_batch_eight_resume_preserves_model_and_samplers(cfg_v2, store_v2, tmp_path):
    cfg_v2.training.batch_size = 8
    cfg_v2.training.accumulation = 1
    cfg_v2.training.representation_steps = 2
    cfg_v2.training.validation_every = 3
    cfg_v2.training.strict_determinism = True
    full, resumed = tmp_path / "full_batch", tmp_path / "resumed_batch"
    train_stage(store_v2, cfg_v2, full, "representation")
    train_stage(store_v2, cfg_v2, resumed, "representation", stop_after=1)
    train_stage(store_v2, cfg_v2, resumed, "representation", resume=True)
    expected = load_checkpoint(full / "representation" / "last.pt")
    actual = load_checkpoint(resumed / "representation" / "last.pt")
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    for sampler in ("landmarks", "origins", "edges"):
        assert torch.equal(expected["samplers"][sampler]["generator"], actual["samplers"][sampler]["generator"])

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.single_phase_response_data import sample_volume, subset
from src.single_phase_response_generated import deny_real_pixels, observed_crop
from src.single_phase_response_models import (
    LastBlockAdapter, ResponseHead, apply_clinical_transform, fit_clinical_transform, patient_prefix_loss,
)
from src.single_phase_response_training import fit, predict


@pytest.mark.parametrize("kind,clinical", [("linear", False), ("gru", False), ("gru", True)])
def test_prefix_causality_and_missing_visits(kind, clinical):
    torch.manual_seed(42)
    model = ResponseHead(dict(model=kind, clinical=clinical), dict(hidden_dim=16, dropout=0.1)).eval()
    features, c = torch.randn(3, 4, 768), torch.randn(3, 17)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1.]])
    days = torch.tensor([[0, 30, 80, 140.]]).expand(3, -1)
    before = model(features, mask, c, days)
    changed, changed_days = features.clone(), days.clone()
    changed[:, 2:] = torch.randn(3, 2, 768) * 100
    changed_days[:, 2:] = 1000
    after = model(changed, mask, c, changed_days)
    assert torch.equal(before[:, :2], after[:, :2])
    assert torch.equal(before[1, :1].expand(4), before[1])
    assert torch.equal(before[2, :1].expand(4), before[2])
    if not clinical:
        assert torch.equal(before, model(features, mask, c * 10, days))


def test_each_patient_has_equal_total_loss_weight():
    logits = torch.tensor([[0., 2, 3, 4], [1., 1, 1, 1], [5., 5, 5, 5]], requires_grad=True)
    mask = torch.tensor([[1., 0, 0, 0], [1., 1, 1, 1], [0., 1, 1, 1]])
    loss = patient_prefix_loss(logits, torch.zeros(3), mask)
    expected = (torch.nn.functional.softplus(logits[0, 0]) + torch.nn.functional.softplus(logits[1]).mean()) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert not logits.grad[2].any()
    assert not logits.grad[0, 1:].any()


def test_physical_scale_and_real_generated_preprocessing():
    # A physical X ramp sampled on different spacings must give the same local cube.
    a = np.broadcast_to(np.arange(120, dtype=np.float32), (80, 120, 120)).copy()
    b = np.broadcast_to(np.arange(60, dtype=np.float32) * 2, (40, 60, 60)).copy()
    first = sample_volume(a, [40, 60, 60], [1, 1, 1], 48, zscore=False)[0]
    second = sample_volume(b, [20, 30, 30], [2, 2, 2], 48, zscore=False)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=3e-5)
    assert sample_volume(a, [40, 60, 60], [1, 1, 1], 48)[0].shape == (1, 48, 48, 48)
    with pytest.raises(ValueError):
        sample_volume(a, [0, 0, 0], [0, 1, 1], 48)


def test_generated_crop_only_takes_observed_geometry():
    grid = dict(shape_zyx=[96, 256, 256], spacing_xyz_mm=[1, 1, 2], direction_lps=np.eye(3).reshape(-1).tolist(), origin_lps_mm=[0, 0, 0])
    source = dict(timepoint=1, crop_geometry=grid)
    loc = dict(center_lps_mm=[100, 100, 50], sides_mm={"tumor": 60, "context": 80})
    actual = observed_crop("unregistered_first_post", source, loc)
    assert actual["center_zyx"] == [25, 100, 100]
    assert actual["localization_timepoint"] == 1
    assert actual["scale_source_timepoint"] == 0
    assert "target" not in " ".join(actual)


def test_generated_reader_guards_pathlib_and_image_libraries(tmp_path):
    import nibabel as nib
    import SimpleITK as sitk
    with deny_real_pixels():
        for call, path in ((lambda p: Path(p).open("rb"), "target.nii.gz"),
                           (nib.load, "other_phase.nii.gz"), (np.load, "target_mask.npy"),
                           (sitk.ReadImage, "target.nii.gz")):
            with pytest.raises(PermissionError):
                call(tmp_path / path)


def test_clinical_transform_has_training_only_imputation():
    train = np.array([[1, np.nan], [3, 5], [5, 7]], np.float32)
    state = fit_clinical_transform(train)
    before = copy.deepcopy(state)
    apply_clinical_transform(np.array([[100, np.nan]], np.float32), state)
    for k in state:
        np.testing.assert_array_equal(state[k], before[k])
    np.testing.assert_array_equal(state["median"], [3, 6])


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(self.qkv(x)[..., :8])


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()

    def forward(self, x):
        return x + self.attn(x)


def test_lora_initial_parity_and_frozen_base_gradients():
    encoder = nn.Module()
    encoder.blocks, encoder.norm = nn.ModuleList([TinyBlock()]), nn.LayerNorm(8)
    adapter = LastBlockAdapter(encoder, dict(last_blocks=1, rank=2, alpha=2))
    inputs = torch.randn(2, 4, 8)
    expected = torch.nn.functional.normalize(encoder.norm(encoder.blocks[-1](inputs))[:, 0], dim=-1)
    torch.testing.assert_close(adapter(inputs), expected, rtol=0, atol=0)
    adapter.train()
    adapter(inputs)[:, 0].sum().backward()
    assert all(p.grad is None for p in adapter.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters() if p.requires_grad)
    assert not adapter.block.training
    state = adapter.adapter_state()
    assert all(k.endswith((".a", ".b")) for k in state)


def synthetic_data():
    generator = np.random.default_rng(13)
    return dict(pids=[f"test_{i}" for i in range(16)], embs=generator.normal(size=(16, 4, 768)).astype(np.float32),
                masks=np.ones((16, 4), np.float32), clinical=generator.normal(size=(16, 17)).astype(np.float32),
                days=np.tile([0, 30, 70, 140], (16, 1)).astype(np.float32), labels=np.tile([0, 1], 8).astype(np.float32),
                token_paths=[[None] * 4 for _ in range(16)], conflicted=np.array([True] + [False] * 15))


def test_exact_training_resume_and_conflict_sensitivity(tmp_path):
    data = synthetic_data()
    train, val = subset(data, data["pids"][:12]), subset(data, data["pids"][12:])
    cfg = dict(output_dir=str(tmp_path), training=dict(epochs=3, patience=2, min_delta=1e-4, batch_size=4,
               hidden_dim=8, dropout=.2, lr=3e-4, weight_decay=.01, grad_clip=1.), adaptation={})
    arm = dict(model="gru", clinical=True, exclude_conflicts_from_fit=True)
    continuous, a = fit(train, None, cfg, arm, 42, tmp_path / "continuous", fixed_epochs=3)
    with pytest.raises(InterruptedError):
        fit(train, None, cfg, arm, 42, tmp_path / "resume", fixed_epochs=3, pause_after=1)
    resumed, b = fit(train, None, cfg, arm, 42, tmp_path / "resume", fixed_epochs=3)
    for k in a["state"]["head"]:
        assert torch.equal(a["state"]["head"][k], b["state"]["head"][k])
    assert a["history"] == b["history"]
    np.testing.assert_array_equal(predict(continuous, val, a["clinical"], "cpu")[0], predict(resumed, val, b["clinical"], "cpu")[0])
    assert data["pids"][0] not in b["clinical"]["fitted_ids"]
    with pytest.raises(ValueError, match="partition differs"):
        fit(train, None, cfg, arm, 43, tmp_path / "resume", fixed_epochs=3)


def test_fixed_outer_task_has_no_selection_validation_and_resumes_completed(tmp_path, monkeypatch):
    import pandas as pd
    from src.first_post_pcr_data import identity, write_json
    from src.single_phase_response_training import run_task, task_complete, task_path
    data = synthetic_data()
    cfg = dict(output_dir=str(tmp_path), study_dir=str(tmp_path), training=dict(epochs=3, patience=2,
               min_delta=1e-4, batch_size=4, hidden_dim=8, dropout=.2, lr=3e-4, weight_decay=.01, grad_clip=1.),
               adaptation={}, arms={"fusion": dict(model="gru", roi="context", clinical=True)})
    write_json(tmp_path / "cohort.json", {"split": {"train": data["pids"], "val": []}})
    write_json(tmp_path / "contract.json", {"synthetic": True})
    write_json(tmp_path / "runtime_sources.json", {"synthetic": True})
    monkeypatch.setattr("src.single_phase_response_training.load_data", lambda *args: data)
    task = dict(stage="formal", arm="fusion", fold=0, seed=42, train_ids=data["pids"][:12],
                val_ids=data["pids"][12:], fixed_epochs=2)
    run_task(cfg, task)
    path = task_path(cfg, task)
    checkpoint = torch.load(path / "model.pt", weights_only=False)
    assert checkpoint["binding"]["validation_ids"] == []
    assert checkpoint["selected_epochs"] == 2
    assert checkpoint["reload_max_error"] <= 1e-6
    predictions = pd.read_csv(path / "predictions.csv")
    assert set(predictions.patient_id) == set(task["val_ids"])
    assert len(predictions) == 4 * len(task["val_ids"])
    before = identity(path / "model.pt")
    run_task(cfg, task)
    assert identity(path / "model.pt") == before
    assert task_complete(cfg, task)

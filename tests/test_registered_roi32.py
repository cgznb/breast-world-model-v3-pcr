import copy
from pathlib import Path

import pytest
import torch

from symm_observation.config import load_config, stage_budget
from symm_observation.data import VisitStore, PairStore
from symm_observation.training import train, load_observation, validation_records
from symm_observation.utils import load_checkpoint, file_identity, read_json
from test_flow_training import assert_tree_equal


def load_launcher():
    import importlib.util
    path = Path(__file__).resolve().parents[1]/"scripts/launch_registered_roi32.py"
    spec = importlib.util.spec_from_file_location("roi32_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_selection_limits_memory_and_preserves_throughput():
    select = load_launcher().select_profile
    def report(batch, memory, rate):
        return {"passed": True, "batch": batch, "peak_reserved_mib": memory,
                "peak_allocated_mib": memory-300, "device_total_mib": 32607,
                "examples_per_second": rate}
    reports = [report(32, 15000, 80), report(64, 29000, 100),
               report(80, 30000, 97), report(100, 31700, 102)]
    assert select(reports)["batch"] == 80
    reports[2]["examples_per_second"] = 90
    assert select(reports)["batch"] == 64
    with pytest.raises(RuntimeError):
        select([{"passed": False}])


def test_wait_tracks_process_start_not_only_pid():
    import os
    launcher = load_launcher()
    identity = launcher.process_identity(os.getpid())
    assert identity["pid"] == os.getpid() and int(identity["start_ticks"]) > 0
    assert launcher.process_identity(999999999) is None


def test_roi32_crop_and_sample_budgets():
    cfg = load_config(Path(__file__).resolve().parents[1]/"configs/registered_roi32_5090.yaml")
    assert cfg.observation.crop == (6, 24, 24)
    for batch in (16, 32, 64, 80, 100, 128, 160, 200, 250, 256):
        cfg.training.stage_batches = {"A": batch, "B": batch}
        for stage, samples in (("A", 160000), ("B", 800000)):
            budget = stage_budget(cfg, stage)
            assert budget["samples"] == samples
            assert budget["steps"]*budget["effective_batch"] == samples
            assert budget["ema_decay"]**budget["steps"] == pytest.approx(cfg.training.ema_decay**(samples/8))
    cfg.training.stage_batches["A"] = 48
    with pytest.raises(ValueError, match="divisible"):
        stage_budget(cfg, "A")


def test_patient_balanced_selection_covers_distinct_patients():
    rows = [{"id": str(i), "patient_id": p} for p in ("a", "b", "c") for i in range(4)]
    selected = validation_records(rows, 3, True)
    assert {r["patient_id"] for r in selected} == {"a", "b", "c"}
    assert selected == validation_records(rows, 3, True)


def test_preload_preserves_real_batch_and_source_only_access(visits, pairs, monkeypatch):
    vr = visits.records("train")[0]
    original = visits.read(vr, include_aux=True)
    visits.preload()
    cached = visits.read(vr, include_aux=True)
    for key in ("latent", "valid", "kinetics", "segmentation"):
        if key in original:
            torch.testing.assert_close(cached[key], original[key], rtol=0, atol=0)
    selected = pairs.records("train")[:2]
    before = pairs.batch(selected, "cpu")
    pairs.preload()
    after = pairs.batch(selected, "cpu")
    for key in ("source_raw", "target_raw", "source_valid", "target_valid"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    after["source_raw"].fill_(17)
    torch.testing.assert_close(pairs.batch(selected, "cpu")["source_raw"], before["source_raw"], rtol=0, atol=0)
    import symm_observation.data as data
    monkeypatch.setattr(data, "load_visit", lambda *a, **k: pytest.fail("Unexpected disk read"))
    assert pairs.source(selected[0])["latent"].shape == before["source_raw"][0].shape


def test_scaled_large_batch_exact_resume_and_frozen_teacher_handoff(cfg, dataset, tmp_path):
    cfg.training.reference_batch_size = 2
    cfg.training.stage_batches = {"A": 2, "B": 2}
    cfg.training.accumulation = 1
    cfg.training.preload_latents = True
    cfg.training.patient_balanced_validation = True
    cfg.training.stage_a_steps = 3
    full, resumed = tmp_path/"full", tmp_path/"resumed"
    train("A", cfg, dataset["visits"], full)
    train("A", cfg, dataset["visits"], resumed, stop_after=1)
    train("A", cfg, dataset["visits"], resumed, resume=True)
    a, b = load_checkpoint(full/"A/last.pt"), load_checkpoint(resumed/"A/last.pt")
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert_tree_equal(a[key], b[key])
    encoder, meta = load_observation(full/"A/best.pt")
    best = load_checkpoint(full/"A/best.pt")
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, best["model"]["target_encoder."+key])
    assert not any(p.requires_grad for p in encoder.parameters())
    contract = read_json(full/"A/contract.json")
    assert contract["manifest"] == file_identity(dataset["visits"])
    assert not any("sha" in key or "digest" in key for key in contract)
    assert meta["A_contains_longitudinal_predictor"] is False


def test_resume_rejects_changed_latent_file(cfg, dataset, tmp_path):
    train("A", cfg, dataset["visits"], tmp_path/"run", stop_after=1)
    store = VisitStore(dataset["visits"])
    path = store.base/store.visits[0]["latent_path"]
    import os
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000))
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        train("A", cfg, dataset["visits"], tmp_path/"run", resume=True)

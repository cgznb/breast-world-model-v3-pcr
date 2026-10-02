from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from responsewm import imaging_evaluation_v3 as imaging
from responsewm.io import digest


def test_cohort_is_identifier_sorted_complete_and_label_independent():
    def patient(pid, stages, label, split="val"):
        return {"patient_key": pid, "split": split, "visits": [{"stage": i, "available_at": i} for i in stages],
                "target": {"pcr": label}}
    patients = [patient("Z", range(4), 1), patient("A", range(3), 1), patient("B", range(4), 0),
                patient("C", range(4), None), patient("0", range(4), 1, "train")]
    assert [p["patient_key"] for p in imaging.select_complete_patients(patients, count=2)] == ["B", "C"]
    for p in patients:
        p.pop("target")
    assert [p["patient_key"] for p in imaging.select_complete_patients(patients, count=2)] == ["B", "C"]


def test_metrics_preserve_absolute_offset_and_exclude_empty_regions():
    truth = np.arange(24).reshape(3, 2, 2, 2).astype(np.float32)
    got = imaging.image_errors(truth + 7, truth)
    assert got["mae"] == got["bias"] == got["rmse"] == 7
    assert got["mse"] == 49
    assert imaging.image_errors(truth, truth, np.zeros_like(truth, dtype=bool)) is None
    assert imaging._pseudo_valid(np.ones((2, 2, 2), bool), {"measurement": {"retained_fraction": .8}})[0] is False
    with pytest.raises(ValueError, match="boolean"):
        imaging.image_errors(truth, truth, np.ones_like(truth))


def _make_bundle(tmp_path):
    root, output = tmp_path / "source", tmp_path / "bundle"
    for name in ("image_views", "raw_latents", "masks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "codec.pt").write_bytes(b"test codec identity")
    norm = {"mean": 20., "std": 5., "fit_split": "train"}
    phases = ["pre_aqc0", "first_post_aqc1", "metadata_late"]
    images, masks, latents, visits = [], [], [], []
    for stage in range(4):
        stem, key = f"P01_T{stage}", f"P01:T{stage}"
        z = np.full((24, 1, 1, 1), stage, np.float16)
        image = np.full((3, 2, 2, 2), stage, np.float32)
        roi, support = np.ones((1, 2, 2, 2), bool), np.ones_like(image, bool)
        np.savez_compressed(root / "image_views" / f"{stem}.npz", image=image, latent=z, roi=roi, support=support)
        np.save(root / "raw_latents" / f"{stem}.npy", z)
        np.savez_compressed(root / "masks" / f"{stem}.npz", mask=roi[0])
        images.append({"view_id": key, "split": "val", "patient_id": "P01", "path": f"image_views/{stem}.npz"})
        masks.append({"view_id": key, "split": "val", "patient_id": "P01", "path": f"masks/{stem}.npz",
                      "label_kind": "mapped_model_pseudo_mask", "measurement": {"retained_fraction": 1.}})
        latents.append({"id": key, "split": "val", "patient_id": "P01", "source_available_grid": True,
                        "geometry": {"shape_zyx": [2, 2, 2]}, "grid_id": "P01:T0_fixed"})
        visits.append({"stage": stage, "available_at": stage, "latent": str(root / "raw_latents" / f"{stem}.npy")})
    patient = {"patient_key": "P01", "split": "val", "visits": visits, "target": {"pcr": 1}}
    manifest = {"patients": [patient], "phase_order": phases, "latent_shape": [24, 1, 1, 1],
                "vq_identity": "sha256:" + digest(root / "codec.pt")}
    def write(name, value):
        (root / name).write_text(json.dumps(value))
    write("trajectory.json", manifest)
    write("image_manifest.json", {"status": "passed", "phase_order": phases, "image_normalization": norm,
                                  "arrays": {"image": {"shape": [3, 2, 2, 2]}}, "views": images})
    write("mask_manifest.json", {"status": "passed", "views": masks})
    write("manifest.json", {"image_normalization": norm, "views": latents})
    audit = imaging.prepare_bundle(root / "trajectory.json", root, output, count=1)
    return root, output, manifest, audit


def test_portable_bundle_roundtrip_and_latent_mismatch_rejected(tmp_path):
    root, output, manifest, audit = _make_bundle(tmp_path)
    assert audit["patients"] == 1 and audit["visits"] == 4
    bundle = json.loads((output / "manifest.json").read_text())
    assert all(not Path(v["file"]).is_absolute() for v in bundle["visits"])
    assert "not_expert" in bundle["roi_provenance"]["pseudo_roi"]
    np.save(root / "raw_latents/P01_T0.npy", np.ones((24, 1, 1, 1), np.float16))
    with pytest.raises(ValueError, match="differs from the training"):
        imaging.prepare_bundle(root / "trajectory.json", root, tmp_path / "bad", count=1)


@dataclass
class Event:
    stage: int
    origin: str
    disease: torch.Tensor


@dataclass
class FakeState:
    model_state_log: tuple


@dataclass
class FakeTrace:
    latent: torch.Tensor
    image_states: tuple
    internal_full_trace: tuple
    stage_ids: tuple = (1, 2, 3)
    @property
    def final_state(self):
        return self.internal_full_trace[-1]
    @property
    def probability(self):
        return torch.tensor([.4])


class FakeCodec:
    def decode(self, z):
        return z[:, ::8].expand(-1, -1, 2, 2, 2)
    def encode(self, image):
        return (image.mean((2, 3, 4), keepdim=True) + 1).repeat_interleave(8, 1)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("latent_mean", torch.full((1, 24, 1, 1, 1), 2.))
        self.register_buffer("latent_std", torch.full((1, 24, 1, 1, 1), 3.))
    def forward(self, z):
        return SimpleNamespace(disease=z.flatten(1).mean(1).reshape(-1, 1, 1))


class FakeModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.encoder = FakeEncoder()
        self.target_encoder = self.encoder
        self.pcr = SimpleNamespace(prior=lambda clinical, mask: clinical[:, 0])
        self.events = events
    def initialize(self, inp):
        return SimpleNamespace(memory=torch.zeros(1), clinical=torch.zeros(1, 1),
                               clinical_mask=torch.ones(1, 1, dtype=torch.bool))
    def forecast(self, belief, **kwargs):
        self.events.append("forecast_finished")
        # Two raw decoded futures bracketing the real future: ensemble image mean is exact.
        raw = torch.stack([torch.tensor([0., 1., 2.]), torch.tensor([2., 3., 4.])])
        z = ((raw - 2) / 3).reshape(1, 2, 3, 1, 1, 1, 1).expand(1, 2, 3, 24, 1, 1, 1)
        diseases = tuple(z[:, :, j].mean((2, 3, 4, 5))[:, :, None, None] for j in range(3))
        state = FakeState(tuple(Event(i + 1, "predicted", d) for i, d in enumerate(diseases)))
        return FakeTrace(z, diseases, (state,))
    def query_pcr(self, belief, *, mode="observed_landmark", future_trace=None):
        if mode == "observed_landmark":
            return SimpleNamespace(probability=torch.tensor([.6]))
        return SimpleNamespace(probability=future_trace.final_state.model_state_log[-1].disease.mean().sigmoid()[None])


def test_full_imaging_uses_source_only_and_reports_codec_floor_and_roundtrip(tmp_path, monkeypatch):
    _, bundle, manifest, _ = _make_bundle(tmp_path)
    events = []
    class Store:
        patients = manifest["patients"]
        def make_prefix(self, index, stage, device):
            assert stage == 0
            events.append("read_T0")
            return SimpleNamespace(observed=torch.full((1, 1, 24, 1, 1, 1), -2/3))
    store = Store()
    store.manifest = manifest
    model = FakeModel(events)
    config = SimpleNamespace(protocol=SimpleNamespace(imaging_manifest=str(bundle / "manifest.json"), codec_path=None,
                            image_eval_cases=1, validation_seed=17), training=SimpleNamespace(precision="fp32", validate_samples=2),
                            sampling=SimpleNamespace(inference_steps=1))
    def load_test_codec(*args):
        torch.randn(10)  # Real codec construction consumes RNG before loading.
        return FakeCodec()
    monkeypatch.setattr(imaging, "load_codec", load_test_codec)
    monkeypatch.setattr(imaging, "interval_plan", lambda x: ())
    monkeypatch.setattr(imaging, "_figure", lambda *a: None)
    original_load = np.load
    def audited_load(path, **kwargs):
        if "bundle/views" in str(path):
            assert "forecast_finished" in events
            events.append("read_target")
        return original_load(path, **kwargs)
    monkeypatch.setattr(imaging.np, "load", audited_load)
    rng_before = torch.get_rng_state().clone()
    result = imaging.evaluate_imaging(model, store, config, tmp_path / "results")
    assert torch.equal(rng_before, torch.get_rng_state())
    summaries = result["summaries"]
    selected = {r["method"]: r for r in summaries if r["stage"] == 2 and r["reference"] == "actual_MRI"
                and r["region"] == "whole_crop" and r["phase"] == "all"}
    assert selected["codec_reconstruction"]["mae"] == 0
    assert selected["decoded_T0_persistence"]["mae"] == 2
    assert selected["generated_ensemble_image_mean"]["mae"] == pytest.approx(0, abs=1e-6)
    assert selected["generated_sample"]["mae"] == pytest.approx(1)
    assert result["pcr_probes"][0]["decoded_MRI_roundtrip_probability"] != result["pcr_probes"][0]["latent_reencoded_probability"]
    assert result["pcr_probes"][0]["clinical_only_probability"] == .5
    assert result["pcr_probes"][0]["observed_probability"] == pytest.approx(.6)
    assert model.training
    assert (tmp_path / "results/imaging_metrics.csv").exists()

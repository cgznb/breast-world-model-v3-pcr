from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import fcntl
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import ispy2_sampling_sweep as sweep


def route(policy="direct_t0", patient="ISPY2-999999", stage=2):
    source = 0 if policy == "direct_t0" else stage - 1
    return SimpleNamespace(
        patient_id=patient, strategy=policy, source_stage=source, target_stage=stage,
        source_visit_id=f"{patient}:T{source}", target_visit_id=f"{patient}:T{stage}",
        target_key=(patient, stage), source_dce0="generated" if policy == sweep.ROLLOUT else "real",
        delta_days=50,
    )


@pytest.mark.parametrize("key,value", [
    ("solver_steps", 20), ("seed", 9), ("source_stage", 1),
    ("source_dce0", "generated"), ("model", "symmflow"), ("target_dce0_read", True),
])
def test_wrong_step_or_source_cannot_enter_sweep(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(sweep, "OUT", tmp_path)
    item = route()
    payload = sweep.identity("biflow", 2, item, 22026)
    payload[key] = value
    sweep.base._atomic_torch_save(sweep.artifact("biflow", 2, item, "decoded"), payload)
    with pytest.raises(ValueError, match="identity or sampling"):
        sweep.prediction("biflow", 2, item, 22026)


def metric_frame():
    routes = {policy: route(policy) for policy in sweep.POLICIES}
    frame = pd.DataFrame([
        dict(schema=sweep.SCHEMA, model="biflow", steps=2, strategy=policy,
             patient_id=item.patient_id, target_stage=2, generation_source_stage=item.source_stage,
             metric_reference_stage=0, region=region, voxel_count=10, ssim_center_count=2,
             **{column: .5 for column in sweep.METRIC_COLUMNS})
        for policy, item in routes.items() for region in sweep.metrics.REGIONS
    ])
    return routes, frame


@pytest.mark.parametrize("key,value", [
    ("schema", "old"), ("patient_id", "ISPY2-111111"), ("steps", 20),
    ("target_stage", 3), ("metric_reference_stage", 1),
    ("generation_source_stage", 3), ("region", "wrong_mask"), ("voxel_count", 9),
])
def test_metric_shards_require_common_cohort_regions_and_steps(key, value):
    routes, frame = metric_frame()
    sweep.validate_metric_group(frame, "biflow", 2, routes)
    frame.loc[0, key] = value
    with pytest.raises(ValueError):
        sweep.validate_metric_group(frame, "biflow", 2, routes)


@pytest.mark.parametrize("steps", [2, 10, 50])
def test_symmflow_retains_target_branch_and_propagates_steps(monkeypatch, steps):
    base = sweep.base
    monkeypatch.setattr(base.torch, "autocast", lambda **kwargs: nullcontext())
    generator = object.__new__(base.SymmGenerator)
    generator.steps, generator.device = steps, torch.device("cpu")
    generator.codec = SimpleNamespace(encode=lambda value, **kwargs: value)
    generator.conditions = {}
    item = route(sweep.ROLLOUT)
    generator.by_patient = {item.patient_id: {}}
    generator.encoder = lambda *args, **kwargs: torch.zeros(1, 1, 2)

    def sample(z, tokens, **kwargs):
        assert kwargs == dict(num_samples=1, seed=22026, steps=steps,
                              solver="euler", return_joint_state=True)
        joint = torch.cat((torch.full_like(z, 3), torch.full_like(z, 9)), dim=1)
        return SimpleNamespace(samples=torch.full((1, 1, 1, 2, 2, 2), 7.),
                               final_joint_states=joint[None])

    generator.sampler = SimpleNamespace(sample_forward_latent=sample)
    image, latent = generator.predict(item, torch.ones(2, 2, 2, 2), 22026,
                                      generated_source=True, return_latent=True)
    assert image.shape == (1, 2, 2, 2) and torch.all(image == 7)
    assert latent.shape == (1, 1, 2, 2, 2) and torch.all(latent == 3)


def test_explicit_biflow_embeddings_are_replayed_not_historical(tmp_path, monkeypatch):
    base = sweep.base
    monkeypatch.setattr(base, "ROOT", tmp_path)
    config = tmp_path / "configs/mewm_ispy2_full978_locked102_table2.yaml"
    config.parent.mkdir()
    config.write_text("{}")
    ids = [f"ISPY2-{index:06}" for index in range(102)]
    labels = np.array([1] * 32 + [0] * 70)
    probabilities = np.linspace(.1, .9, 102)
    real = dict(pids=ids, labels=labels, masks=np.ones((102, 4)))
    monkeypatch.setattr(base, "_cohort_inputs", lambda _: (
        {"embeddings_dir": "unused"}, tmp_path,
        {"train": ["train"], "val": ["val"], "test": ids}, None, None))
    monkeypatch.setattr(base, "load_all_splits", lambda *args: {"test": real})
    monkeypatch.setattr(base, "_canonical_split", lambda split, depth: split)
    overlays = []

    def overlay(split, directory):
        overlays.append(directory)
        return dict(split, generated=True)

    monkeypatch.setattr(base, "_overlay_generated_test", overlay)
    monkeypatch.setattr(base, "checkpoint_path", lambda *args: tmp_path / "unused.pt")
    monkeypatch.setattr(base.torch, "load", lambda *args, **kwargs: {
        "train_ids": ["train"], "validation_ids": ["val"]})
    monkeypatch.setattr(base, "_validate_checkpoint", lambda *args: None)
    monkeypatch.setattr(base, "_predict_checkpoint", lambda header, split, device:
                        probabilities + (.01 if split.get("generated") else 0))
    canonical = tmp_path / "results/mewm_ispy2_full978_locked102_final_hybrid_policy"
    canonical.mkdir(parents=True)
    pd.DataFrame([dict(temporal_depth=depth, seed=seed, input_scheme="real", patient_id=pid,
                       label=label, probability=p)
                  for depth in base.DEPTHS for seed in range(42, 52)
                  for pid, label, p in zip(ids, labels, probabilities)]).to_csv(
        canonical / "all_test_predictions.csv", index=False)
    pd.DataFrame(dict(temporal_depth=base.DEPTHS, threshold=.5)).to_csv(
        canonical / "development_thresholds.csv", index=False)
    destination = tmp_path / "pcr"
    base.pcr(["biflow"], ["direct_t0"], torch.device("cpu"), destination,
             embedding_sources={("biflow", "direct_t0"): tmp_path / "new_embeddings"})
    actual = pd.read_csv(destination / "predictions.csv")
    generated = actual[actual.model == "biflow"]
    assert overlays == [tmp_path / "new_embeddings"]
    assert len(generated) == 4080
    np.testing.assert_allclose(generated.probability, np.tile(probabilities + .01, 40))
    assert len(pd.read_csv(destination / "fold_predictions.csv")) == 20400


@pytest.mark.parametrize("external", [False, True])
def test_pruning_preserves_selected_patients_and_rejects_external_paths(tmp_path, monkeypatch, external):
    monkeypatch.setattr(sweep, "OUT", tmp_path / "sweep")
    selected, unselected = route(patient="selected"), route(patient="unselected")
    routes = {policy: [] for policy in sweep.POLICIES}
    routes["direct_t0"] = [selected, unselected]
    monkeypatch.setattr(sweep.base, "context", lambda: (
        None, None, routes, {unselected.target_key: 22026}, None, None, None))
    root = sweep.setting_root("biflow", 2)
    sweep.base._atomic_json(root / "audit.json", dict(status="passed", model="biflow", steps=2))
    sweep.base._atomic_json(sweep.OUT / "protocol.json", dict(display_patients=["selected"]))
    paths = [sweep.artifact("biflow", 2, item, "decoded") for item in (selected, unselected)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    latent = dict(sweep.identity("biflow", 2, unselected, 22026), normalization="codebook_minmax",
                  endpoint_latent=torch.ones(1, 1, 8, 24, 64, 64, dtype=torch.float16))
    sweep.base._atomic_torch_save(sweep.artifact("biflow", 2, unselected, "latents"), latent)
    if external:
        external_path = tmp_path / "old_image.pt"
        external_path.write_bytes(b"old image")
        paths[1].unlink()
        paths[1].symlink_to(external_path)
        with pytest.raises(ValueError, match="external decoded"):
            sweep.prune("biflow", 2)
        assert external_path.read_bytes() == b"old image" and paths[1].exists()
        assert not (root / "complete.json").exists()
    else:
        sweep.prune("biflow", 2)
        assert not paths[1].exists() and (root / "complete.json").exists()
    assert paths[0].read_bytes() == b"image"


@pytest.mark.parametrize("fault", ["wrong_normalization", "wrong_shape", "nonfinite"])
def test_retained_latents_are_decodable_under_the_recorded_contract(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(sweep, "OUT", tmp_path)
    item = route()
    value = torch.ones(1, 8, 24, 64, 64, dtype=torch.float32)
    payload = dict(sweep.identity("symmflow", 2, item, 22026),
                   normalization="symmflow_standardized", endpoint_latent=value)
    path = sweep.artifact("symmflow", 2, item, "latents")
    sweep.base._atomic_torch_save(path, payload)
    assert torch.equal(value, sweep.read_latent("symmflow", 2, item, 22026))
    if fault == "wrong_normalization":
        payload["normalization"] = "codebook_minmax"
    elif fault == "wrong_shape":
        payload["endpoint_latent"] = value[0]
    else:
        value[0, 0, 0, 0, 0] = float("nan")
    sweep.base._atomic_torch_save(path, payload)
    with pytest.raises(ValueError, match="Invalid retained"):
        sweep.read_latent("symmflow", 2, item, 22026)


@pytest.mark.parametrize("fail", [False, True])
def test_metric_writers_are_locked_and_release_after_failure(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(sweep, "OUT", tmp_path)
    root = sweep.setting_root("biflow", 2)
    sweep.base._atomic_json(root / "generation.json", dict(complete=True))
    entered, release = Event(), Event()

    def compute(*args):
        entered.set()
        assert release.wait(5)
        if fail:
            raise ValueError("Synthetic metric failure")

    monkeypatch.setattr(sweep, "_image_metrics", compute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sweep.image_metrics, "biflow", 2, 2)
        try:
            assert entered.wait(5)
            with (root / "image_metrics/.metrics.lock").open("a") as competing:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            release.set()
        if fail:
            with pytest.raises(ValueError, match="Synthetic metric failure"):
                future.result(timeout=5)
        else:
            future.result(timeout=5)
    with (root / "image_metrics/.metrics.lock").open("a") as released:
        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from responsewm.data import PHASES
from responsewm.data_v2 import PatientTrajectoryStore, SCHEMA
from responsewm.io import read_json, write_json
from responsewm.samplers import BalancedEdgeSampler, PatientLandmarkSampler


@pytest.fixture
def trajectory(tmp_path):
    rng = np.random.default_rng(123)
    patients = []
    for i, stages in enumerate(((0, 1, 2, 3), (0, 2), (0, 1), (0, 1, 2, 3), (0, 3))):
        visits = []
        for stage in stages:
            path = tmp_path / f"p{i}_T{stage}.npy"
            np.save(path, rng.normal(size=(24, 2, 4, 4)).astype(np.float32))
            visits.append({"stage": stage, "latent": str(path), "available_at": stage})
        patients.append({"patient_key": f"p{i}", "split": "train" if i < 3 else ("val" if i == 3 else "test"),
                         "baseline": {"values": [40 + i, 1, None, 0], "known_at": [0, 2, None, 0]},
                         "visits": visits, "events": [], "interval_plans": [],
                         "target": {"pcr": i % 2, "label_source": "synthetic_fixture"}})
    value = {"schema": SCHEMA, "time_basis": "stage_index", "canonical_stages": [0, 1, 2, 3],
             "clinical_features": ["age", "hr_positive", "her2_positive", "mammaprint_binary"],
             "action_features": [], "phase_order": PHASES, "latent_shape": [24, 2, 4, 4],
             "vq_identity": "sha256:" + "a" * 64, "shared_grid_verified": True, "synthetic": True,
             "patients": patients}
    path = tmp_path / "patients.json"
    write_json(path, value)
    store = PatientTrajectoryStore(path, allow_synthetic=True)
    store.fit_statistics()
    return path, store


def test_patient_split_disjoint(trajectory):
    path, store = trajectory
    value = read_json(path)
    duplicate = copy.deepcopy(value["patients"][0]); duplicate["split"] = "test"
    value["patients"].append(duplicate); write_json(path, value)
    with pytest.raises(ValueError, match="overlaps"):
        PatientTrajectoryStore(path, allow_synthetic=True)


def test_stats_train_only(trajectory):
    path, store = trajectory
    expected = store.statistics
    for i in store.by_split["val"] + store.by_split["test"]:
        for visit in store.patients[i]["visits"]:
            np.save(visit["latent"], np.load(visit["latent"]) + 9999)
    fresh = PatientTrajectoryStore(path, allow_synthetic=True)
    assert fresh.fit_statistics() == expected
    assert expected["train_visit_count"] == 8
    with pytest.raises(ValueError, match="training split"):
        fresh.set_statistics({**expected, "fit_split": "test"})
    with pytest.raises(ValueError, match="codec"):
        fresh.set_statistics({**expected, "vq_identity": "sha256:" + "b" * 64})


def test_known_at_filter_and_labels_absent(trajectory):
    _, store = trajectory
    early = store.make_prefix(0, 0)
    late = store.make_prefix(0, 2)
    assert not early.clinical_mask[0, 1]
    assert early.clinical[0, 1] == 0
    assert late.clinical_mask[0, 1] and late.clinical_known_at[0, 1] == 2
    assert not hasattr(early, "label") and not hasattr(early, "patient_key")
    assert early.actions.shape == (1, 3, 0)
    assert early.to("cpu").validate(4, 0) is not None


def test_four_masks_not_conflated(trajectory):
    _, store = trajectory
    inp, sup = store.batch([(1, 0)], output_stages=[3])
    assert inp.stage_valid.tolist() == [[True] * 4]
    assert inp.observed_stage_ids == ((0,),)
    assert inp.output_query_mask.tolist() == [[False, False, False, True]]
    assert inp.future_days.tolist() == [[1, 2, 3]]
    assert inp.future_mask.tolist() == [[True, True, True]]
    assert sup.target_available_mask.tolist() == [[False, False, True, False]]
    assert sup.future_mask.tolist() == [[False, True, False]]
    assert not hasattr(inp, "target_available_mask")
    assert not hasattr(inp, "label")


def test_nonadjacent_pair_tagging(trajectory):
    _, store = trajectory
    edges = store.edge_index(include_bridges=True)
    assert 1 in edges[(0, 2)] and all(1 not in edges[e] for e in ((0, 1), (1, 2), (2, 3)))
    sampler = BalancedEdgeSampler(store, 1, include_bridges=True)
    bridge = next(e for e in sampler.all_edges() if e.patient_index == 1)
    prefix, truth = store.make_edge_task(bridge)
    assert truth.kind == "bridge_auxiliary" and truth.target_stage == 2
    assert truth.latent.shape == (1, 24, 2, 4, 4)
    assert prefix.observed_stage_ids == ((0,),)


def test_stage_ids_not_observation_count(trajectory):
    _, store = trajectory
    assert store.make_prefix(1, 2).observed_stage_ids == ((0, 2),)
    assert store.make_prefix(2, 1).observed_stage_ids == ((0, 1),)
    terminal, sup = store.batch([(0, 3)])
    assert terminal.future_days.shape == (1, 0) and sup.future.shape[1] == 0
    terminal.validate(4, 0)


def test_source_only_does_not_open_targets(trajectory):
    _, store = trajectory
    before = store.make_prefix(0, 0)
    for visit in store.patients[0]["visits"][1:]:
        Path(visit["latent"]).unlink()
    store.cache.clear()
    store.patients[0]["target"]["pcr"] = 1 - store.patients[0]["target"]["pcr"]
    after = store.make_prefix(0, 0)
    assert torch.equal(before.observed, after.observed)
    assert torch.equal(before.clinical, after.clinical)
    with pytest.raises(FileNotFoundError):
        store.batch([(0, 0)])


def test_future_dates_and_target_input_rejected(trajectory):
    path, store = trajectory
    with pytest.raises(ValueError, match="integer stage"):
        store.make_prefix(0, 0, output_stages=[14.5])
    prefix = store.make_prefix(0, 0)
    days = prefix.future_days.clone(); days[0, 0] = 1.5
    with pytest.raises(ValueError, match="integer stages"):
        replace(prefix, future_days=days).validate(4, 0)
    value = read_json(path); value["patients"][0]["baseline"]["pcr"] = 1
    write_json(path, value)
    with pytest.raises(ValueError, match="Unexpected"):
        PatientTrajectoryStore(path, allow_synthetic=True)


def test_delayed_scan_only_visible_when_received(trajectory):
    _, store = trajectory
    store.patients[0]["visits"][1]["available_at"] = 2
    assert store.make_prefix(0, 1).observed_stage_ids == ((0,),)
    assert store.make_prefix(0, 2).observed_stage_ids == ((0, 1, 2),)
    assert 0 not in store.edge_index()[(1, 2)]


def test_interval_controls_filtered_by_source_known_at(trajectory):
    path, _ = trajectory
    value = read_json(path); value["action_features"] = ["drug_dose"]
    value["patients"][0]["interval_plans"] = [
        {"source_stage": 0, "target_stage": 1, "values": [2.], "known_at": [1], "kind": "planned", "units": ["mg"]},
        {"source_stage": 2, "target_stage": 3, "values": [5.], "known_at": [0], "kind": "planned", "units": ["mg"]}]
    write_json(path, value)
    store = PatientTrajectoryStore(path, allow_synthetic=True); store.fit_statistics()
    prefix = store.make_prefix(0, 0)
    assert prefix.action_mask.tolist() == [[[False], [False], [True]]]
    assert prefix.actions[0, 0, 0] == 0


def test_sampler_expectation_coverage_and_exact_resume(trajectory):
    _, store = trajectory
    sampler = BalancedEdgeSampler(store, 123)
    exhaustive = sampler.all_edges()
    score = lambda e: 1. + e.patient_index + 3 * e.source_stage
    expected = sum(e.effective_weight * score(e) for e in exhaustive)
    draws = sampler.sample(12000)
    estimate = np.mean([e.effective_weight * score(e) for e in draws])
    assert estimate == pytest.approx(expected, abs=.08)
    assert set(sampler.coverage) == {"01", "12", "23"}
    assert all(e.effective_weight == pytest.approx(1) for e in draws)
    state = sampler.state_dict(); following = sampler.sample(20)
    restored = BalancedEdgeSampler(store, 0); restored.load_state_dict(state)
    assert restored.sample(20) == following
    landmarks = PatientLandmarkSampler(store, 1)
    landmarks.sample(5); state = landmarks.state_dict(); expected = landmarks.sample(20)
    restored_landmarks = PatientLandmarkSampler(store, 88); restored_landmarks.load_state_dict(state)
    assert restored_landmarks.sample(20) == expected


def test_migration_preserves_sources_and_all_real_visits(trajectory, tmp_path):
    _, store = trajectory
    common = {key: value for key, value in store.manifest.items() if key not in {"canonical_stages", "patients"}}
    cases = []
    for p in store.patients:
        baseline = copy.deepcopy(p["baseline"])
        baseline["known_at"][1] = 0
        visits = {v["stage"]: v for v in p["visits"]}
        cases.append({"id": p["patient_key"] + "_T0", "patient_id": p["patient_key"], "split": p["split"],
                      "input": {"landmark_day": 0, "observed": [{"day": 0, "available_at": 0, "latent": visits[0]["latent"]}],
                                "clinical": baseline["values"], "clinical_known_at": baseline["known_at"],
                                "queries": [{"day": j, "known_at": 0, "actions": [], "actions_known_at": []} for j in (1, 2, 3)],
                                "source_only_geometry": True},
                      "target": {"pcr": p["target"]["pcr"], "future": [
                          {"day": j, "latent": visits[j]["latent"]} if j in visits else None for j in (1, 2, 3)]}})
    common.update(schema="responsewm_manifest_v1", cases=cases)
    source = tmp_path / "legacy.json"; write_json(source, common)
    content = source.read_bytes()
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_multistage_v2.py"
    spec = importlib.util.spec_from_file_location("prepare_multistage_v2", module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.prepare(source, tmp_path / "converted", allow_synthetic=True)
    migrated = PatientTrajectoryStore(report["manifest"], allow_synthetic=True)
    assert source.read_bytes() == content
    assert len(migrated.patients) == 5
    assert migrated.audit()["split_summary"]["train"]["visits_by_stage"] == {"0": 3, "1": 2, "2": 2, "3": 1}
    assert report["source_preserved"]
    with pytest.raises(FileExistsError):
        module.prepare(source, tmp_path / "converted", allow_synthetic=True)

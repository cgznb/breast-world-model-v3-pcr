from collections import Counter
from dataclasses import replace

import pytest
import torch

from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.samplers import BalancedEdgeSampler, PatientLandmarkSampler
from responsewm.synthetic_v2 import make_synthetic_v2


@pytest.fixture
def batch_store(tmp_path):
    path = make_synthetic_v2(tmp_path / "data", shape=(24, 2, 4, 4), patients=10, missing_labels=True)
    store = PatientTrajectoryStore(path, allow_synthetic=True)
    store.fit_statistics()
    return store


def task_structure(store, task):
    index, stage = task
    patient = store.patients[index]
    observed = tuple(v["stage"] for v in patient["visits"]
                     if v["stage"] <= stage and v["available_at"] <= stage)
    available = {v["stage"] for v in patient["visits"]}
    return stage, observed, tuple(j in available for j in range(stage + 1, 4)), patient["target"]["pcr"] is not None


def test_landmark_batches_have_compatible_masks_and_real_batch_eight(batch_store):
    sampler = PatientLandmarkSampler(batch_store, 42)
    observed_origins = set()
    for _ in range(100):
        tasks = sampler.sample_batch(8)
        assert len(tasks) == 8
        assert len({task_structure(batch_store, task) for task in tasks}) == 1
        prefix, truth = batch_store.batch(tasks)
        assert prefix.observed.shape[0] == truth.future.shape[0] == 8
        assert prefix.clinical.shape == (8, 4)
        assert prefix.observed.shape[2:] == (24, 2, 4, 4)
        assert len(set(prefix.observed_stage_ids)) == 1
        assert torch.equal(truth.future_mask, truth.future_mask[:1].expand_as(truth.future_mask))
        assert torch.equal(truth.label_mask, truth.label_mask[:1].expand_as(truth.label_mask))
        observed_origins.add(int(prefix.as_of[0]))
    assert observed_origins == {0, 1, 2, 3}


def test_landmark_marginals_preserve_patient_and_landmark_weights(batch_store):
    # Delayed metadata creates different task weights inside one compatible bucket.
    batch_store.patients[3]["visits"][1]["available_at"] = 2
    sampler = PatientLandmarkSampler(batch_store, 912)
    counts = Counter(task for _ in range(16000) for task in sampler.sample_batch(8))
    total = counts.total()
    for index, stages in sampler.groups:
        patient_fraction = sum(counts[(index, stage)] for stage in stages) / total
        assert patient_fraction == pytest.approx(1 / len(sampler.groups), abs=.018)
        for stage in stages:
            expected = 1 / (len(sampler.groups) * len(stages))
            assert counts[(index, stage)] / total == pytest.approx(expected, abs=.01)


def test_origin_filter_conditions_original_landmark_distribution(batch_store):
    sampler = PatientLandmarkSampler(batch_store, 181, include_terminal=False)
    counts = Counter(task for _ in range(8000) for task in sampler.sample_batch(8, origin_stage=0))
    normalizer = sum(1 / len(stages) for _, stages in sampler.groups)
    assert all(stage == 0 for _, stage in counts)
    for index, stages in sampler.groups:
        assert counts[(index, 0)] / counts.total() == pytest.approx(1 / len(stages) / normalizer, abs=.02)
    with pytest.raises(ValueError, match="No eligible"):
        sampler.sample_batch(8, origin_stage=3)
    with pytest.raises(ValueError, match="canonical"):
        sampler.sample_batch(8, origin_stage=1.5)


def test_new_sampler_rng_restores_exactly_and_legacy_rng_is_unchanged(batch_store):
    landmarks = PatientLandmarkSampler(batch_store, 46)
    generator = torch.Generator().manual_seed(46)
    expected = []
    for _ in range(20):
        index, stages = landmarks.groups[int(torch.randint(len(landmarks.groups), (), generator=generator))]
        stage = stages[int(torch.randint(len(stages), (), generator=generator))]
        expected.append((index, stage))
    assert landmarks.sample(20) == expected
    landmarks.sample_batch(8)
    state = landmarks.state_dict()
    expected = (landmarks.sample_batch(8), landmarks.sample_batch(8, origin_stage=0), landmarks.sample(9))
    restored = PatientLandmarkSampler(batch_store, 99)
    restored.load_state_dict(state)
    assert (restored.sample_batch(8), restored.sample_batch(8, origin_stage=0), restored.sample(9)) == expected

    edges = BalancedEdgeSampler(batch_store, 46, include_bridges=True)
    generator.manual_seed(46)
    expected = []
    for _ in range(20):
        edge = edges.types[int(torch.randint(len(edges.types), (), generator=generator))]
        indices = edges.groups[edge]
        index = indices[int(torch.randint(len(indices), (), generator=generator))]
        expected.append(edges._entry(edge, index, 1 / (len(edges.types) * len(indices))))
    assert edges.sample(20) == expected
    edges.sample_batch(8)
    state = edges.state_dict()
    expected = (edges.sample_batch(8), edges.sample(9), edges.sample_batch(8))
    restored = BalancedEdgeSampler(batch_store, 99, include_bridges=True)
    restored.load_state_dict(state)
    assert (restored.sample_batch(8), restored.sample(9), restored.sample_batch(8)) == expected
    assert restored.coverage == edges.coverage


def test_edge_batches_preserve_population_probability_and_objective(batch_store):
    sampler = BalancedEdgeSampler(batch_store, 891, include_bridges=True, bridge_weight=.25)
    counts, observed_sum, total = Counter(), 0., 0
    score = lambda edge: 1. + edge.patient_index + 3 * edge.source_stage
    expected = sum(edge.effective_weight * score(edge) for edge in sampler.all_edges())
    for _ in range(12000):
        edges = sampler.sample_batch(8)
        assert len({(e.source_stage, e.target_stage, e.kind) for e in edges}) == 1
        assert len({task_structure(batch_store, (e.patient_index, e.source_stage))[1] for e in edges}) == 1
        for edge in edges:
            key = edge.source_stage, edge.target_stage
            probability = 1 / (len(sampler.types) * len(sampler.groups[key]))
            assert edge.probability == pytest.approx(probability)
            assert edge.effective_weight == pytest.approx(edge.objective_weight / probability)
            counts[(key, edge.patient_index)] += 1
            observed_sum += edge.effective_weight * score(edge)
            total += 1
    assert observed_sum / total == pytest.approx(expected, abs=.13)
    assert sum(sampler.coverage.values()) == total
    for key, patients in sampler.groups.items():
        assert sampler.coverage[f"{key[0]}{key[1]}"] / total == pytest.approx(1 / len(sampler.types), abs=.02)
        for index in patients:
            assert counts[(key, index)] / total == pytest.approx(1 / (len(sampler.types) * len(patients)), abs=.008)


def test_edge_batch_eight_matches_single_item_inputs_and_truth(batch_store):
    sampler = BalancedEdgeSampler(batch_store, 21, include_bridges=True)
    covered = set()
    for _ in range(50):
        edges = sampler.sample_batch(8)
        prefix, truth = batch_store.make_edge_batch(edges)
        assert prefix.observed.shape[0] == 8
        assert truth.latent.shape == (8, 24, 2, 4, 4)
        assert truth.anatomy_comparable.shape == (8,)
        assert len(truth.auxiliary) == 8
        assert len(set(prefix.observed_stage_ids)) == 1
        for row, edge in enumerate(edges):
            single_prefix, single_truth = batch_store.make_edge_task(edge)
            assert torch.equal(prefix.observed[row], single_prefix.observed[0])
            assert torch.equal(prefix.clinical[row], single_prefix.clinical[0])
            assert torch.equal(truth.latent[row], single_truth.latent[0])
            assert (truth.source_stage, truth.target_stage, truth.kind) == (edge.source_stage, edge.target_stage, edge.kind)
        covered.add((truth.source_stage, truth.target_stage))
    assert covered == set(sampler.types)


def test_edge_batches_reject_mixed_edges_and_observed_prefixes(batch_store):
    sampler = BalancedEdgeSampler(batch_store, 4, include_bridges=True)
    edges = sampler.all_edges()
    early = next(e for e in edges if e.source_stage == 0 and e.target_stage == 1)
    late = next(e for e in edges if e.source_stage == 2 and e.patient_index == 0)
    missing_t1 = next(e for e in edges if e.source_stage == 2 and e.patient_index == 1)
    with pytest.raises(ValueError, match="Empty"):
        batch_store.make_edge_batch([])
    with pytest.raises(ValueError, match="same source"):
        batch_store.make_edge_batch([early, late])
    with pytest.raises(ValueError, match="same observed stage prefix"):
        batch_store.make_edge_batch([late, missing_t1])
    with pytest.raises(ValueError, match="kind disagrees"):
        batch_store.make_edge_batch([replace(early, kind="bridge_auxiliary")])
    with pytest.raises(ValueError, match="legal observed source"):
        batch_store.make_edge_batch([replace(early, patient_index=1)])
    batch_store.patients[0]["visits"][2]["available_at"] = 3
    with pytest.raises(ValueError, match="legal observed source"):
        batch_store.make_edge_batch([late])


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_batch_samplers_reject_invalid_count(batch_store, count):
    for sampler in (PatientLandmarkSampler(batch_store, 1), BalancedEdgeSampler(batch_store, 1)):
        with pytest.raises(ValueError, match="positive integer"):
            sampler.sample_batch(count)

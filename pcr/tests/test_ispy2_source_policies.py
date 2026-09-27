from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts import report_ispy2_source_policies as reporting


def predictions():
    return pd.DataFrame([
        dict(model="symmflow", strategy=policy, temporal_depth=depth, seed=42,
             patient_id=pid, label=label, probability=probability)
        for policy in reporting.POLICIES
        for depth in ("T0", "T0-T1", "T0-T2", "T0-T3")
        for pid, label, probability in (("ISPY2-100001", 0, .2), ("ISPY2-100002", 1, .8))
    ])


def test_shared_early_predictions_are_checked_but_later_policies_may_differ():
    frame = predictions()
    frame.loc[(frame.strategy == "adjacent_real") & (frame.temporal_depth == "T0-T2"), "probability"] += .01
    reporting.validate_prediction_relations(frame)
    frame.loc[(frame.strategy == "adjacent_real") & (frame.temporal_depth == "T0-T1"), "probability"] += .01
    with pytest.raises(ValueError, match="must be identical"):
        reporting.validate_prediction_relations(frame)


def test_equal_sized_mismatched_patient_cohort_is_rejected():
    frame = predictions()
    frame.loc[(frame.strategy == "adjacent_real") & (frame.patient_id == "ISPY2-100001"), "patient_id"] = "ISPY2-999999"
    with pytest.raises(ValueError, match="cohorts differ"):
        reporting.validate_prediction_relations(frame)


def test_old_or_duplicated_mask_metric_shards_cannot_be_reused():
    frame = pd.DataFrame([dict(model=arm, region=region, schema=reporting.SCHEMA)
                          for arm in reporting.ARMS for region in reporting.reporting.REGIONS])
    assert reporting.metric_group_complete(frame)
    assert not reporting.metric_group_complete(frame.drop(columns="schema"))
    assert not reporting.metric_group_complete(pd.concat([frame.iloc[:-1], frame.iloc[[0]]]))
    frame["schema"] = "previous_source_masks"
    assert not reporting.metric_group_complete(frame)


@pytest.mark.parametrize("generated_source", [False, True])
def test_rollout_reencodes_generated_dce0_even_when_real_visit_cache_exists(monkeypatch, generated_source):
    comparison = reporting.generation
    monkeypatch.setattr(comparison.torch, "autocast", lambda **kwargs: nullcontext())
    generator = object.__new__(comparison.SymmGenerator)
    generator.device = torch.device("cpu")
    encoded, cached, sampled, conditions = [], [], [], []
    route = SimpleNamespace(patient_id="ISPY2-100001", source_stage=1, target_stage=2,
                            source_visit_id="ISPY2-100001:T1", delta_days=50)
    source = torch.stack([torch.full((2, 2, 2), 5.), torch.full((2, 2, 2), 17.)])

    def encode(value, *, normalize):
        encoded.append(value.clone())
        assert normalize
        return value

    def load(visit_id):
        cached.append(visit_id)
        return torch.full((1, 2, 2, 2), 9.)

    def condition(value, *, batch_size):
        conditions.append(value)
        assert batch_size == 1
        return torch.zeros((1, 1, 2))

    def sample(z, tokens, **kwargs):
        sampled.append(z.clone())
        assert kwargs == dict(num_samples=1, seed=22027, steps=20, solver="euler")
        return SimpleNamespace(samples=z[None])

    generator.codec = SimpleNamespace(encode=encode, normalize_latent=lambda x: x)
    generator.latents = SimpleNamespace(visit_ids={route.source_visit_id}, load=load,
                                       denormalize=lambda x: x)
    generator.encoder = condition
    generator.sampler = SimpleNamespace(sample_forward_latent=sample)
    generator.conditions = {}
    generator.by_patient = {route.patient_id: {"clinical": "fixed"}}
    generator.predict(route, source, 22027, generated_source=generated_source)

    assert conditions[0]["stage_i"] == "T1"
    assert conditions[0]["stage_j"] == "T2"
    assert conditions[0]["delta_days"] == 50
    if generated_source:
        assert not cached
        assert len(encoded) == 1
        assert torch.equal(encoded[0], source[:1][None])
        assert torch.all(sampled[0] == 5.)
    else:
        assert not encoded
        assert cached == [route.source_visit_id]
        assert torch.all(sampled[0] == 9.)

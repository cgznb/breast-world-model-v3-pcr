from pathlib import Path

import pytest
import torch

from responsewm.config import load_config
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.evaluation_v3 import (_image_scores, fixed_cohorts, reencoded_trace_readout,
                                     validate)
from responsewm.model_v3 import ResponseWorldModelV3
from responsewm.rollout import NoiseLedger
from responsewm.synthetic_v2 import make_synthetic_v2


@pytest.fixture
def evaluation_case(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/multistage_smoke_v2.yaml")
    cfg.schema = "responsewm_v3"
    cfg.encoder.depths = (1, 1, 1)
    cfg.network.dropout = 0
    cfg.sampling.inference_steps = 1
    cfg.training.validation_cases = 1  # Must never truncate the fixed cohort.
    cfg.protocol.validation_batch_size = 2
    cfg.protocol.validation_seed = 9987
    torch.set_num_threads(2)
    torch.manual_seed(24)
    store = PatientTrajectoryStore(make_synthetic_v2(tmp_path / "data"), allow_synthetic=True)
    store.fit_statistics()
    model = ResponseWorldModelV3(cfg, store.c, store.a).eval().requires_grad_(False)
    with torch.no_grad():
        model.pcr.output[-1].weight.normal_(std=.1)
    return cfg, store, model


def test_fixed_cohorts_include_missing_followup_and_ignore_label(evaluation_case):
    _, store, _ = evaluation_case
    cohorts = fixed_cohorts(store)
    assert cohorts["primary_t0"]["patients"] == 2
    assert cohorts["common_complete_t0_t3"]["patients"] == 1
    assert len(store.patients[cohorts["primary_t0"]["patient_indices"][0]]["visits"]) == 3
    for index in store.by_split["val"]:
        store.patients[index]["target"]["pcr"] = None
    assert fixed_cohorts(store) == cohorts


def test_validation_fixed_noise_selection_and_source_only_boundary(evaluation_case, monkeypatch):
    cfg, store, model = evaluation_case
    original_batch, original_forecast = store.batch, model.forecast
    predicted = set()
    sizes = []

    def batch(tasks, *args, **kwargs):
        if kwargs.get("supervised", True):
            assert all((index, origin) in predicted for index, origin in tasks), "Future truth loaded before prediction"
        else:
            sizes.append(len(tasks))
        return original_batch(tasks, *args, **kwargs)

    def forecast(belief, **kwargs):
        assert belief.observed_stage_ids == ((0,), (0,))
        result = original_forecast(belief, **kwargs)
        predicted.update((index, 0) for index in store.by_split["val"])
        return result

    monkeypatch.setattr(store, "batch", batch)
    monkeypatch.setattr(model, "forecast", forecast)
    before_rng = torch.get_rng_state().clone()
    first = validate(model, store, cfg, "readout")
    assert torch.equal(before_rng, torch.get_rng_state())
    assert first["patients"] == 2 and 2 in sizes
    assert first["selection_score"] == first["primary_t0"]["modalities"]["generated_future_joint"]["nll"]
    assert first["generation_by_target"]["T1"]["patients"] == 1
    assert first["generation_by_target"]["T2"]["patients"] == 2
    assert first["generation_by_target"]["T3"]["patients"] == 2
    assert all(first["secondary_common_complete"][f"T{s}"]["patients"] == 1 for s in range(4))
    assert set(first["primary_t0"]["modalities"]) == {
        "clinical_only", "real_mri_history", "generated_future_joint", "reencoded_generated_latent"}
    cfg.training.seed += 999
    second = validate(model, store, cfg, "readout")
    assert first == second, "Changing the training seed must not change validation draws"
    assert not model._lineage_versions, "Validation must not accumulate stale lineage metadata"


def test_reencoded_readout_preserves_images_source_memory_and_observations(evaluation_case, monkeypatch):
    cfg, store, model = evaluation_case
    inp = store.batch([(store.by_split["val"][0], 0)], supervised=False)
    with torch.no_grad():
        trace = model.forecast(model.initialize(inp), samples=2, steps=1, noise_ledger=NoiseLedger(17))
    original_query = model.query_pcr
    seen = []

    def query(belief, mode, future_trace):
        assert mode == "marginal_current"
        assert belief is trace.source_state
        assert future_trace.source_state.memory is trace.source_state.memory
        assert torch.equal(future_trace.latent, trace.latent)
        for old, new in zip(trace.final_state.model_state_log, future_trace.final_state.model_state_log):
            if old.origin == "observed":
                assert old is new
            else:
                assert new.disease is trace.image_states[trace.stage_ids.index(old.stage)]
                assert new.raw_tokens is old.raw_tokens
        seen.append(True)
        return original_query(belief, mode, future_trace)

    def forbidden_forecast(*args, **kwargs):
        raise AssertionError("A readout ablation must not regenerate dynamics")

    monkeypatch.setattr(model, "query_pcr", query)
    monkeypatch.setattr(model, "forecast", forbidden_forecast)
    reencoded_trace_readout(model, trace)
    assert seen


def test_ensemble_mean_and_persistence_are_distinct_estimands():
    samples = torch.tensor([[-1., 1.]]).reshape(1, 2, 1, 1, 1, 1)
    truth = torch.zeros(1, 1, 1, 1, 1)
    values = _image_scores(samples, truth, torch.ones_like(truth) * 2)
    assert values["sample_mae"].item() == 1
    assert values["ensemble_mean_mae"].item() == values["ensemble_mean_mse"].item() == 0
    assert values["persistence_mae"].item() == 2
    assert values["persistence_mse"].item() == 4
    assert values["energy_score"].item() == .5


def test_flow_uses_generation_selection_without_outcome_labels(evaluation_case):
    cfg, store, model = evaluation_case
    for index in store.by_split["val"]:
        store.patients[index]["target"]["pcr"] = None
    result = validate(model, store, cfg, "flow")
    expected = sum(v["ensemble_mse_to_persistence_ratio"] for v in result["generation_by_target"].values()) / 3
    assert result["selection_score"] == result["generation_score"] == expected
    assert result["t0_marginal_nll"] is None
    assert result["primary_t0"]["labelled_patients"] == 0


def test_representation_has_no_future_rollout_and_preserves_training_rng(evaluation_case, monkeypatch):
    cfg, store, model = evaluation_case

    def forbidden_forecast(*args, **kwargs):
        raise AssertionError("A selection must not evaluate an untrained generator")

    monkeypatch.setattr(model, "forecast", forbidden_forecast)
    model.train()
    before = torch.get_rng_state().clone()
    result = validate(model, store, cfg, "representation")
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    expected = (result["representation_reconstruction_plus_jepa"] + cfg.multistage.representation_pcr_weight *
                result["primary_t0"]["modalities"]["real_mri_history"]["nll"])
    assert result["selection_score"] == expected
    assert result["generation_score"] is None


def test_full_analysis_uses_same_complete_cohort_and_batch_independent_noise(evaluation_case):
    cfg, store, model = evaluation_case
    batched = validate(model, store, cfg, "readout", full=True)
    cfg.protocol.validation_batch_size = 1
    individual = validate(model, store, cfg, "readout", full=True)
    for origin in range(3):
        assert set(batched["secondary_common_complete"][f"T{origin}"]["modalities"]) == set(
            batched["primary_t0"]["modalities"])
    for a, b in zip(batched["rows"], individual["rows"]):
        assert a["patient_key"] == b["patient_key"]
        assert a["origin_stage"] == b["origin_stage"]
        for field in ("observed_probability", "marginal_probability", "reencoded_marginal_probability"):
            if field in a:
                assert a[field] == pytest.approx(b[field], abs=2e-6)
    assert batched["generation_score"] == pytest.approx(individual["generation_score"], abs=1e-6)

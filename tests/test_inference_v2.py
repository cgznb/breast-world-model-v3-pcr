import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from responsewm.cli import main as cli_main
from responsewm.conditioning import IntervalSpec
from responsewm.config import load_config
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.inference_v2 import forecast, load_state, read_request
from responsewm.io import digest, load_checkpoint, read_json, write_json
from responsewm.rollout import NoiseLedger
from responsewm.synthetic_v2 import make_synthetic_v2
from responsewm.training_v2 import STAGES, train_stage


def input_request(store, index=0):
    contract = {key: copy.deepcopy(store.manifest[key]) for key in (
        "clinical_features", "action_features", "phase_order", "latent_shape", "vq_identity", "time_basis")}
    patient = store.patients[index]
    first = patient["visits"][0]
    contract.update(schema="responsewm_request_v2", input={
        "landmark_day": 0, "observed": [{"day": 0, "available_at": 0, "latent": first["latent"]}],
        "clinical": patient["baseline"]["values"], "clinical_known_at": patient["baseline"]["known_at"],
        "queries": [{"day": 3, "known_at": 0, "actions": [], "actions_known_at": []}],
        "source_only_geometry": True})
    return contract


@pytest.fixture
def request_fixture(tmp_path):
    path = make_synthetic_v2(tmp_path / "cohort")
    store = PatientTrajectoryStore(path, allow_synthetic=True)
    store.fit_statistics()
    request = input_request(store)
    payload = {"metadata": {"data_contract": {key: request[key] for key in (
        "clinical_features", "action_features", "phase_order", "latent_shape", "vq_identity", "time_basis")},
        "statistics": store.statistics}}
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    return store, request_path, payload


def test_independent_request_has_no_future_file_access(request_fixture):
    store, path, payload = request_fixture
    expected, requested = read_request(path, payload)
    for visit in store.patients[0]["visits"][1:]:
        Path(visit["latent"]).unlink()
    actual, actual_requested = read_request(path, payload)
    assert requested == actual_requested == [3]
    assert actual.future_days.tolist() == [[1., 2., 3.]]
    assert actual.output_query_mask.tolist() == [[False, False, False, True]]
    assert torch.equal(expected.observed, actual.observed)


@pytest.mark.parametrize("field,value", [("target", {"pcr": 1}), ("patient_id", "forbidden")])
def test_independent_request_rejects_target_and_identity(request_fixture, field, value):
    _, path, payload = request_fixture
    request = read_json(path); request[field] = value; write_json(path, request)
    with pytest.raises(ValueError, match="Unexpected"):
        read_request(path, payload)


def test_request_rejects_codec_identity_and_actual_future_dates(request_fixture):
    _, path, payload = request_fixture
    request = read_json(path)
    altered = copy.deepcopy(request); altered["vq_identity"] = "sha256:" + "0" * 64
    write_json(path, altered)
    with pytest.raises(ValueError, match="vq_identity"):
        read_request(path, payload)
    altered = copy.deepcopy(request); altered["input"]["queries"][0]["day"] = 14.5
    write_json(path, altered)
    with pytest.raises(ValueError, match="arbitrary days"):
        read_request(path, payload)


def test_v2_configs_register_and_validate():
    root = Path(__file__).resolve().parents[1]
    paths=[root/"configs"/name for name in ("multistage_smoke_v2.yaml", "ispy2_multistage_native_v2.yaml")]
    paths.extend(sorted((root/"configs/ablations_v2").glob("*.yaml")))
    for path in paths:
        cfg = load_config(path)
        assert cfg.schema == "responsewm_v2"
        assert cfg.network.time_basis == "stage_index"
        assert cfg.network.semantic_depth == 6
        assert cfg.training.reverse_probability == 0
        assert cfg.multistage.use_full_future_plan_in_physiology is False
        assert cfg.multistage.truncate_bptt is False
        assert cfg.sampling.train_samples >= 2


def test_five_state_apis_and_cli_without_future_assets(request_fixture, tmp_path, capsys):
    store, request_path, _ = request_fixture
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "multistage_smoke_v2.yaml")
    cfg.encoder.depths = (1, 1, 1)
    cfg.training.validation_cases = 1
    cfg.training.representation_steps = 1
    cfg.training.flow_steps = 1
    cfg.training.readout_steps = 1
    cfg.training.joint_steps = 1
    cfg.sampling.train_steps = 1
    cfg.sampling.inference_steps = 1
    torch.set_num_threads(2)
    run = tmp_path / "run"
    for stage in STAGES:
        assert train_stage(store, cfg, run, stage)["completed"]
    checkpoint = run / "joint" / "best.pt"
    payload = load_checkpoint(checkpoint)
    assert payload["schema"] == "responsewm_checkpoint_v2"
    arriving_latent = tmp_path / "new_observation.npy"
    np.save(arriving_latent, np.load(store.patients[0]["visits"][1]["latent"]))
    for visit in store.patients[0]["visits"][1:]:
        Path(visit["latent"]).unlink()
    state = tmp_path / "state_T0.pt"
    cli_main(["initialize-v2", "--checkpoint", str(checkpoint), "--request", str(request_path), "--output", str(state)])
    state_digest = digest(state)
    observed_risk = tmp_path / "observed_pcr.json"
    cli_main(["query-pcr-v2", "--state", str(state), "--output", str(observed_risk)])
    assert read_json(observed_risk)["interpretation"] == "observed_landmark"
    assert digest(state) == state_digest
    forecast_path = tmp_path / "forecast.npz"
    cli_main(["forecast-v2", "--state", str(state), "--output", str(forecast_path),
              "--output-stages", "3", "--samples", "2", "--steps", "1", "--seed", "19"])
    with np.load(forecast_path) as trace:
        assert trace["latent"].shape == (1, 2, 1, 24, 4, 8, 8)
        assert trace["internal_stages"].tolist() == [1, 2, 3]
        assert np.isfinite(trace["pcr_marginal"]).all()
    assert digest(state) == state_digest
    with pytest.raises(ValueError, match="Codec SHA"):
        forecast(state, tmp_path / "bad_codec.npz", samples=2, steps=1, codec=checkpoint)
    assert not (tmp_path / "bad_codec.npz").exists()
    model, _, belief, _ = load_state(state)
    with torch.no_grad():
        expected=model.forecast(belief,output_stages=(3,),samples=2,steps=1,noise_ledger=NoiseLedger(19))
        raw=expected.latent*model.encoder.latent_std[None,None]+model.encoder.latent_mean[None,None]
        with np.load(forecast_path) as exported:
            np.testing.assert_array_equal(exported["latent"],raw.numpy())
        advanced = model.advance(belief, IntervalSpec(0, 1), samples=2, steps=1, noise_ledger=NoiseLedger(19)).final_state
    assert advanced.stage == 1 and advanced.observed_stage_ids == ((0,),)
    assert belief.stage == 0
    observation_path = tmp_path / "observation.json"
    write_json(observation_path, {"schema": "responsewm_observation_v2", "stage": 1,
                                 "latent": str(arriving_latent), "available_at": 1, "event_id": "arriving_T1",
                                 "vq_identity": store.manifest["vq_identity"]})
    posterior = tmp_path / "state_T1.pt"
    cli_main(["observe-v2", "--state", str(state), "--observation", str(observation_path), "--output", str(posterior)])
    _, _, updated, _ = load_state(posterior)
    assert updated.stage == 1 and updated.observed_stage_ids == ((0, 1),)
    posterior_risk = tmp_path / "posterior_pcr.json"
    cli_main(["query-pcr-v2", "--state", str(posterior), "--output", str(posterior_risk)])
    assert read_json(posterior_risk)["observed_stage_ids"] == [[0, 1]]
    assert digest(state) == state_digest
    capsys.readouterr()

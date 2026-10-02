from pathlib import Path
import torch
from responsewm.config import load_config
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.evaluation_v2 import prequential_evaluation
from responsewm.synthetic_v2 import make_synthetic_v2
from responsewm.training_v2 import build_model


def test_prequential_predicts_before_mri_loading_and_retains_samples(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/multistage_smoke_v2.yaml")
    cfg.encoder.depths = (1, 1, 1)
    cfg.sampling.inference_steps = 1
    torch.set_num_threads(2)
    store = PatientTrajectoryStore(make_synthetic_v2(tmp_path / "data"), allow_synthetic=True)
    store.fit_statistics()
    index = store.by_split["val"][0]
    assert [v["stage"] for v in store.patients[index]["visits"]] == [0, 2, 3]
    store.by_split["val"] = [index]
    model = build_model(cfg, store).eval().requires_grad_(False)
    stages_by_path = {v["latent"]: v["stage"] for v in store.patients[index]["visits"]}
    produced_stage = 0
    transitions, opened = [], []
    original_advance, original_read = model.advance, store.read_latent

    def advance(belief, interval, **kwargs):
        nonlocal produced_stage
        trace = original_advance(belief, interval, **kwargs)
        transitions.append((interval.src_stage, interval.dst_stage, belief.samples, trace.final_state.samples))
        produced_stage = interval.dst_stage
        return trace

    def read_latent(path):
        stage = stages_by_path[str(path)]
        assert stage <= produced_stage, "A real future MRI was loaded before its prior prediction"
        opened.append((stage, produced_stage))
        return original_read(path)

    monkeypatch.setattr(model, "advance", advance)
    monkeypatch.setattr(store, "read_latent", read_latent)
    result = prequential_evaluation(model, store, cfg)
    assert transitions == [(0, 1, 1, 2), (1, 2, 2, 2), (2, 3, 2, 2)]
    rows = result["rows"]
    assert rows[0]["target_stage"] == 1 and rows[0]["new_mri_received"] is False
    assert "observed_probability" not in rows[0]
    assert rows[1]["prior_observed_stage_ids"] == [0]
    assert rows[1]["observed_stage_ids"] == [0, 2]
    assert rows[2]["prior_observed_stage_ids"] == [0, 2]
    assert rows[2]["observed_stage_ids"] == [0, 2, 3]
    assert (2, 2) in opened and (3, 3) in opened
    assert all(torch.isfinite(torch.tensor(row["forecasted_state_probability"])) for row in rows)

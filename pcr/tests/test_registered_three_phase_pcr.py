import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import registered_three_phase_pcr as study


def cohort_inputs():
    patients = ["a", "b", "c", "d", "h", "i", "excluded"]
    metadata = pd.DataFrame({"pid": patients, "pCR": [0, 1, 0, 1, 0, 1, 0],
                             "registered_timepoints": ["T0;T1;T2;T3"] * len(patients)})
    visits = [{"patient_id": pid, "visit_id": f"{pid}:T{tp}", "visit": f"T{tp}",
               "fold": "val" if pid in ("h", "i") else "train", "phase_indices": [0, 1, 4]}
              for pid in patients[:-1] for tp in ((0, 2) if pid == "a" else (0, 1))]
    folds = [{"fold": 0, "train_ids": ["c", "d", "excluded"], "val_ids": ["a", "b"]},
             {"fold": 1, "train_ids": ["a", "b"], "val_ids": ["c", "d", "excluded"]}]
    return {"visits": visits}, metadata, ["a", "b", "c", "d", "excluded"], ["h", "i"], folds


def test_cohort_inherits_patient_folds_and_updates_available_visits():
    cohort, metadata, folds = study.intersect_cohort(*cohort_inputs())
    assert cohort["split"] == {"train": ["a", "b", "c", "d"], "val": ["h", "i"]}
    assert folds[0] == {"fold": 0, "train_ids": ["c", "d"], "val_ids": ["a", "b"]}
    assert metadata.set_index("pid").loc["a", "registered_timepoints"] == "T0;T2"
    assert "excluded" not in set(metadata.pid)


def test_cohort_rejects_a_holdout_patient_in_development():
    args = cohort_inputs()
    args[0]["visits"][8]["fold"] = "train"
    with pytest.raises(ValueError, match="boundaries"):
        study.intersect_cohort(*args)


def test_cohort_rejects_changed_phase_semantics():
    args = cohort_inputs()
    args[0]["visits"][0]["phase_indices"] = [0, 2, 4]
    with pytest.raises(ValueError, match="phase order"):
        study.intersect_cohort(*args)


def test_cohort_rejects_a_missing_baseline():
    args = cohort_inputs()
    args[0]["visits"] = args[0]["visits"][1:]
    with pytest.raises(ValueError, match="baseline"):
        study.intersect_cohort(*args)


def test_denormalization_restores_background_and_preserves_three_phase_order(monkeypatch):
    raw = np.arange(72, dtype=np.float32).reshape(3, 2, 3, 4) + 20
    valid = np.ones(raw.shape, dtype=bool)
    valid[:, 0, 0, 0] = False
    normalized = (raw - 100) / 20
    normalized[~valid] = 0
    seen = []

    def channel(values, spacing):
        seen.append((values.copy(), spacing))
        return torch.from_numpy(values).permute(1, 2, 0)

    monkeypatch.setattr(study, "SHAPE", raw.shape)
    monkeypatch.setattr(study, "PILLAR_SHAPE", (3, 3, 4, 2))
    monkeypatch.setattr(study, "pillar_channel", channel)
    volume, _ = study.build_volume(normalized, valid, {"mean": 100, "std": 20})
    assert tuple(volume.shape) == (3, 3, 4, 2)
    for index, (phase, spacing) in enumerate(seen):
        expected = raw[index].copy()
        expected[~valid[index]] = 0
        np.testing.assert_allclose(phase, expected, atol=1e-5)
        assert spacing == (2.0, 0.7032, 0.7032)


def test_physical_roi_is_padded_without_stretching_to_full_field():
    raw = np.broadcast_to(np.linspace(1, 1000, 128, dtype=np.float32), study.SHAPE).copy()
    volume, audit = study.build_volume(raw, np.ones(raw.shape, bool), {"mean": 0, "std": 1})
    assert tuple(volume.shape) == (3, 384, 384, 192)
    occupied = torch.nonzero(volume[0], as_tuple=False)
    extent = occupied.max(0).values - occupied.min(0).values + 1
    assert 87 <= extent[0] <= 90 and 87 <= extent[1] <= 90 and 62 <= extent[2] <= 64
    assert torch.count_nonzero(volume[:, :100]) == 0
    assert audit["min"] == 0 and audit["max"] <= 1
    assert all(0 < value < 0.025 for value in audit["nonzero_fraction"])


def test_empty_generated_or_real_phase_is_rejected():
    values = np.zeros(study.SHAPE, np.float32)
    valid = np.ones(study.SHAPE, bool)
    valid[1] = False
    with pytest.raises(ValueError, match="empty phase"):
        study.build_volume(values, valid, {"mean": 100, "std": 20})


def test_prediction_aggregation_averages_folds_before_seeds_and_rejects_duplicates():
    rows = [{"temporal_depth": depth, "seed": seed, "patient_id": pid, "label": label,
             "fold": fold, "probability": .1 * fold + .1 * seed}
            for depth in study.DEPTHS for seed in (1, 2) for pid, label in (("h", 0), ("i", 1))
            for fold in range(5)]
    frame = pd.DataFrame(rows)
    result = study.aggregate_predictions(frame, ["h", "i"], [1, 2])
    np.testing.assert_allclose(result[result.seed == 1].probability, .3)
    np.testing.assert_allclose(result[result.seed == 2].probability, .4)
    duplicate = copy.deepcopy(frame)
    duplicate.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="distinct fold"):
        study.aggregate_predictions(duplicate, ["h", "i"], [1, 2])

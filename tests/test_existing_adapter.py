import importlib.util
from pathlib import Path

import numpy as np
import pytest

from responsewm.io import read_json, write_json


spec = importlib.util.spec_from_file_location(
    "existing_adapter", Path(__file__).resolve().parents[1] / "scripts" / "prepare_existing_ispy2.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def dataset(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    views, pairs = [], []
    for pid, split in (("train_patient", "train"), ("val_patient", "val")):
        for visit in ("T0", "T1", "T3"):
            np.save(data / f"{pid}_{visit}.npy", np.ones((24, 8, 32, 32), np.float16))
            views.append({"id": f"{pid}:{visit}", "patient_id": pid, "visit": visit,
                          "split": split, "latent": f"/old/data/{pid}_{visit}.npy",
                          "source_available_grid": True, "grid_id": f"{pid}:fixed_T0",
                          "geometry": {"shape_zyx": [32, 128, 128]}, "labels": {"pcr": None}})
        pairs.append({"patient_id": pid, "conditions": {
            "age": None, "hr_status": "1", "her2_status": "0", "mammaprint": "1"}})
    source = tmp_path / "original.json"
    write_json(source, {"schema": "symm_world_manifest_v2", "phase_order": adapter.PHASES,
                        "views": views, "pairs": pairs})
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"test codec identity")
    return source, data, codec


def test_existing_cache_rebinding_missing_supervision_and_baseline(tmp_path):
    source, data, codec = dataset(tmp_path)
    adapter.prepare(source, data, codec, tmp_path / "out")
    direct = read_json(tmp_path / "out" / "direct_t0_t3.json")
    assert direct["action_features"] == []
    assert direct["time_basis"] == "stage_index"
    assert {c["split"] for c in direct["cases"]} == {"train", "val"}
    case = direct["cases"][0]
    assert case["target"]["pcr"] is None
    assert case["input"]["clinical"] == [None, 1, 0, 1]
    assert case["input"]["clinical_known_at"] == [None, 0, 0, 0]
    assert case["input"]["queries"] == [{"day": 3, "known_at": 0, "actions": [], "actions_known_at": []}]
    assert case["target"]["future"][0]["anatomy_comparable"] is False
    longitudinal = read_json(tmp_path / "out" / "longitudinal.json")
    assert longitudinal["cases"][0]["target"]["future"][1] is None
    assert len(longitudinal["cases"][1]["input"]["observed"]) == 2


def test_adapter_rejects_inconsistent_geometry_and_baseline(tmp_path):
    source, data, codec = dataset(tmp_path)
    original = read_json(source)
    original["views"][1]["grid_id"] = "future_selected_grid"
    write_json(source, original)
    with pytest.raises(ValueError, match="baseline coordinate"):
        adapter.prepare(source, data, codec, tmp_path / "out")
    original["views"][1]["grid_id"] = original["views"][0]["grid_id"]
    original["pairs"].append({"patient_id": "train_patient", "conditions": {
        "age": 42, "hr_status": "1", "her2_status": "0", "mammaprint": "1"}})
    write_json(source, original)
    with pytest.raises(ValueError, match="covariates disagree"):
        adapter.prepare(source, data, codec, tmp_path / "out")

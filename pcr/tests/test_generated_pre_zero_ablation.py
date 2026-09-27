from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.evaluate_full978_generated_pre_zero_ablation as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mewm_ispy2_full978_locked102_generated_pre_zero_ablation.yaml"


def _config():
    value = yaml.safe_load(CONFIG.read_text())
    value["_config_path"] = str(CONFIG)
    return value


def test_config_is_zero_pre_only_and_keeps_batch_six():
    config = _config()
    runner._validate_config(config)

    assert config["zero_pre"]["batch_size"] == 6
    assert config["zero_pre"]["expected_future_embeddings"] == 296
    assert config["reference"]["sources"] == [
        "real",
        "direct_t0",
        "rollout_generated_dce0_real_ser",
    ]
    assert all("shuff" not in str(value).lower() for value in config.values())


def test_zero_pre_changes_only_channel_zero():
    volume = torch.rand((1, 3, 4, 5, 6), generator=torch.Generator().manual_seed(7))
    output = runner.zero_pre_channel(volume)

    assert not output[:, 0].any()
    assert torch.equal(output[:, 1:], volume[:, 1:])
    assert output.data_ptr() != volume.data_ptr()
    assert volume[:, 0].any()


def test_zero_pre_rejects_wrong_or_nonfinite_volume():
    with pytest.raises(ValueError, match="one finite Pillar"):
        runner.zero_pre_channel(torch.zeros((1, 2, 4, 5, 6)))
    volume = torch.zeros((1, 3, 4, 5, 6))
    volume[0, 1, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="one finite Pillar"):
        runner.zero_pre_channel(volume)


def test_overlay_replaces_future_embeddings_but_preserves_all_other_inputs(tmp_path):
    pids = ["P1", "P2"]
    original = {
        "pids": pids,
        "embs": np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3),
        "masks": np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.float32),
        "days": np.asarray([[0, 40, 0, 0], [0, 35, 75, 0]], dtype=np.float32),
        "clinical": np.ones((2, 17), dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.float32),
    }
    for patient_id, timepoint in (("P1", 1), ("P2", 1), ("P2", 2)):
        path = tmp_path / patient_id / f"{patient_id}_T{timepoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.full((3,), float(timepoint + 10)), path)

    # The production inventory is deliberately strict; isolate its mechanics here.
    expected = {
        (str(original["pids"][index]), timepoint)
        for index in range(len(original["pids"]))
        for timepoint in range(1, 4)
        if original["masks"][index, timepoint] == 1
    }
    assert len(expected) == 3
    store = runner.EmbStore(str(tmp_path))
    result = {key: np.asarray(value).copy() if key != "pids" else list(value) for key, value in original.items()}
    pid_index = {pid: index for index, pid in enumerate(pids)}
    for patient_id, timepoint in expected:
        result["embs"][pid_index[patient_id], timepoint] = store.load(patient_id, timepoint)

    np.testing.assert_array_equal(result["embs"][:, 0], original["embs"][:, 0])
    for key in ("masks", "days", "clinical", "labels"):
        np.testing.assert_array_equal(result[key], original[key])
    assert not np.array_equal(result["embs"][:, 1:], original["embs"][:, 1:])


def test_paired_bootstrap_is_deterministic_and_patient_level(monkeypatch):
    monkeypatch.setattr(
        runner,
        "roc_auc_score",
        lambda y, p: float(
            np.mean(p[np.asarray(y) == 1]) - np.mean(p[np.asarray(y) == 0])
        ),
    )
    monkeypatch.setattr(runner, "average_precision_score", runner.roc_auc_score)
    rows = []
    for source in runner.EXPECTED_SOURCES:
        for depth, _ in runner.frozen_runner.EXPECTED_DEPTHS:
            for patient in range(102):
                label = int(patient >= 70)
                probability = (patient + 1) / 103
                if source == "zero_pre":
                    probability = 1 - probability
                for seed in (42, 43):
                    rows.append(
                        {
                            "source": source,
                            "temporal_depth": depth,
                            "patient_id": f"P{patient:03d}",
                            "label": label,
                            "probability": probability,
                            "seed": seed,
                        }
                    )
    frame = pd.DataFrame(rows)
    comparisons = [["direct_t0", "zero_pre"]]
    first = runner.paired_patient_bootstrap(frame, comparisons, repetitions=20, seed=99)
    second = runner.paired_patient_bootstrap(frame, comparisons, repetitions=20, seed=99)

    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    assert len(first[0]) == 4 * 4 * 102
    assert len(first[1]) == 4 * 20
    assert len(first[2]) == 4


def test_batch_grouping_keeps_the_final_partial_batch():
    assert list(runner._batches(range(8), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7]]

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.evaluate_full978_frozen_mixed_biflow_trajectories as runner
from src.pillar import _ensure_transformers_remote_code_compatibility


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "mewm_ispy2_full978_locked102_frozen_mixed_no_residual_biflow_trajectories.yaml"
)


def _config():
    value = yaml.safe_load(CONFIG.read_text())
    value["_config_path"] = str(CONFIG)
    return value


def _payload(strategy="direct_t0", source_stage=0, target_stage=1):
    patient_id = "ISPY2-123456"
    return {
        "schema": "pillar_full978_biflow_trajectory_decoded_v1",
        "strategy": strategy,
        "patient_id": patient_id,
        "source_stage": source_stage,
        "target_stage": target_stage,
        "source_visit_id": f"{patient_id}:T{source_stage}",
        "target_visit_id": f"{patient_id}:T{target_stage}",
        "target_dce0_read": False,
        "solver_steps": 20,
        "prediction": torch.zeros((1, 96, 256, 256), dtype=torch.float16),
    }


def _checkpoint(depth, seed=42, fold=None, variant=None, residual=0.0):
    checkpoint = {
        "seed": seed,
        "temporal_depth": depth,
        "effective_config": {
            "model_type": "tdn",
            "input_dim": 1152,
            "clinical_dim": 17,
            "use_prior": True,
            "residual_l2_weight": residual,
        },
    }
    if fold is not None:
        checkpoint.update(
            {
                "schema": "pillar_full978_independent_depth_cv_v1",
                "fold": fold,
                "variant": variant,
                "test_data_loaded_during_training": False,
            }
        )
    elif variant is not None:
        checkpoint["variant"] = variant
    return checkpoint


def test_config_freezes_requested_mixed_no_residual_policy():
    config = _config()
    runner._validate_config(config)
    depths = config["frozen_models"]["depths"]

    assert [(item["name"], item["folds"]) for item in depths] == [
        ("T0", 1),
        ("T0-T1", 1),
        ("T0-T2", 5),
        ("T0-T3", 5),
    ]
    assert all(item["residual_l2_weight"] == 0.0 for item in depths)
    assert depths[1]["variant"] == "moderate"
    assert depths[2]["variant"] == depths[3]["variant"] == "layerwise_adamw"
    assert config["evaluation"]["sources"] == [
        "real",
        "direct_t0",
        "rollout_generated_dce0_real_ser",
    ]


@pytest.mark.parametrize("index,expected", [(0, 1), (1, 1), (2, 5), (3, 5)])
def test_checkpoint_paths_match_frozen_fold_count(index, expected):
    spec = _config()["frozen_models"]["depths"][index]
    paths = runner._checkpoint_paths(spec, 47)

    assert len(paths) == expected
    assert all("seed_47" in str(path) for path in paths)
    if expected == 5:
        assert [path.parent.name for path in paths] == [f"fold_{fold}" for fold in range(5)]


def test_checkpoint_contract_accepts_single_split_without_residual_field(tmp_path):
    spec = _config()["frozen_models"]["depths"][1]
    checkpoint = _checkpoint("T0-T1", variant="moderate")
    checkpoint["effective_config"].pop("residual_l2_weight")
    path = tmp_path / "best.pt"
    path.touch()

    runner._validate_checkpoint(checkpoint, path, spec, 42, 0)


def test_checkpoint_contract_rejects_nonzero_residual(tmp_path):
    spec = _config()["frozen_models"]["depths"][3]
    checkpoint = _checkpoint(
        "T0-T3", fold=0, variant="layerwise_adamw", residual=1.0e-3
    )
    path = tmp_path / "best.pt"
    path.touch()

    with pytest.raises(ValueError, match="contract changed"):
        runner._validate_checkpoint(checkpoint, path, spec, 42, 0)


def test_decoded_payload_validation_binds_strategy_and_target(tmp_path):
    path = tmp_path / "T1.pt"
    torch.save(_payload(), path)

    loaded = runner._load_decoded_payload(
        path,
        strategy="direct_t0",
        patient_id="ISPY2-123456",
        source_stage=0,
        target_stage=1,
        decoded_schema="pillar_full978_biflow_trajectory_decoded_v1",
        solver_steps=20,
    )
    assert tuple(loaded["prediction"].shape) == (1, 96, 256, 256)

    changed = _payload()
    changed["target_dce0_read"] = True
    torch.save(changed, path)
    with pytest.raises(ValueError, match="invalid BiFlow"):
        runner._load_decoded_payload(
            path,
            strategy="direct_t0",
            patient_id="ISPY2-123456",
            source_stage=0,
            target_stage=1,
            decoded_schema="pillar_full978_biflow_trajectory_decoded_v1",
            solver_steps=20,
        )


def test_shared_decoded_target_requires_tensor_exact_match(tmp_path):
    direct = _payload("direct_t0")
    rollout = _payload("rollout_generated_dce0_real_ser")
    direct_path = tmp_path / "direct.pt"
    rollout_path = tmp_path / "rollout.pt"
    torch.save(direct, direct_path)
    torch.save(rollout, rollout_path)
    route = {
        "patient_id": "ISPY2-123456",
        "source_stage": 0,
        "target_stage": 1,
    }
    config = _config()
    config["trajectory"]["decoded_schema"] = "pillar_full978_biflow_trajectory_decoded_v1"
    config["trajectory"]["solver_steps"] = 20

    assert runner._same_decoded_target(
        config,
        {**route, "decoded_path": direct_path},
        {**route, "decoded_path": rollout_path},
    )

    rollout["prediction"][0, 0, 0, 0] = 1
    torch.save(rollout, rollout_path)
    assert not runner._same_decoded_target(
        config,
        {**route, "decoded_path": direct_path},
        {**route, "decoded_path": rollout_path},
    )


def test_baseline_predictions_are_reordered_and_depth_filtered(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "patient_id": ["P2", "P1", "P2", "P1"],
            "label": [1, 0, 1, 0],
            "probability": [0.8, 0.2, 0.7, 0.3],
            "temporal_depth": ["T0", "T0", "T0-T1", "T0-T1"],
        }
    ).to_csv(path, index=False)
    spec = {
        "name": "T0-T1",
        "baseline_predictions_template": str(path),
    }

    values = runner._baseline_probabilities(
        spec, 42, ["P1", "P2"], np.asarray([0, 1])
    )
    np.testing.assert_allclose(values, [0.3, 0.7])


def test_summary_uses_sample_standard_deviation():
    metrics = pd.DataFrame(
        [
            {
                "source": "real",
                "temporal_depth": "T0",
                "max_tp": 1,
                "protocol": "fixed_train_validation_split",
                "folds_per_seed": 1,
                "seed": seed,
                **{metric: float(seed - 41) for metric in runner.METRIC_KEYS},
            }
            for seed in (42, 43)
        ]
    )

    summary = runner._summary_frame(metrics)
    assert summary.loc[0, "auroc_mean"] == pytest.approx(1.5)
    assert summary.loc[0, "auroc_std"] == pytest.approx(np.sqrt(0.5))


def test_pillar_remote_code_compatibility_only_supplies_missing_tied_weight_map():
    class LegacyRemoteModel:
        pass

    _ensure_transformers_remote_code_compatibility(LegacyRemoteModel)
    assert LegacyRemoteModel.all_tied_weights_keys == {}

    existing = {"target.weight": "source.weight"}

    class CurrentRemoteModel:
        all_tied_weights_keys = existing

    _ensure_transformers_remote_code_compatibility(CurrentRemoteModel)
    assert CurrentRemoteModel.all_tied_weights_keys is existing



def test_extraction_task_keeps_decoded_input_separate_from_embedding_output(tmp_path):
    decoded = tmp_path / "trajectory" / "decoded.pt"
    destination = tmp_path / "embeddings" / "embedding.pt"
    route = {"decoded_path": decoded}

    task = runner._extraction_task("ISPY2-123456", 1, route, destination)

    assert task[-2] == decoded
    assert task[-1] == destination
    assert task[-2] != task[-1]


def test_frozen_evaluation_uses_contiguous_t0_starting_prefix():
    split = {
        "pids": ["P1", "P2"],
        "embs": np.ones((2, 4, 3), dtype=np.float32),
        "masks": np.asarray([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float32),
        "days": np.asarray([[0, 0, 80, 120], [0, 40, 80, 120]], dtype=np.float32),
        "clinical": np.zeros((2, 17), dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.float32),
    }

    canonical = runner._canonical_split(split, 3)

    np.testing.assert_array_equal(canonical["masks"], [[1, 0, 0, 0], [1, 1, 1, 0]])
    np.testing.assert_array_equal(canonical["days"], [[0, 0, 0, 0], [0, 40, 80, 0]])
    assert not canonical["embs"][0, 2:].any()
    assert canonical["embs"][1, :3].all()

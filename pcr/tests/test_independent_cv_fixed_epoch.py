import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.run_full978_independent_cv as independent_cv
from src.data import TABULAR_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "mewm_ispy2_full978_locked102_independent_cv_fixed_epoch.yaml"
)
VARIANT = "fixed_epoch_v2_selected"


def _config():
    return yaml.safe_load(CONFIG.read_text())


def _small_config(tmp_path):
    return {
        "data": {
            "embedding_dim": 4,
            "train_embeddings_dir": str(tmp_path / "embeddings"),
            "metadata_csv": str(tmp_path / "metadata.csv"),
        },
        "downstream": {
            "checkpoint_policy": "final_epoch",
            "proj_dim": 4,
            "sig_dim": 4,
            "n_layers": 1,
            "n_heads": 2,
            "use_prior": True,
            "epochs": 3,
            "patience": 1,
            "min_delta": 10.0,
            "batch_size": 4,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "embedding_dropout": 0.0,
            "grad_clip": 1.0,
            "optimizer": "adam",
            "prefix_dropout_probability": 0.0,
            "residual_l2_weight": 0.0,
            "scheduler_t_max": 10,
            "scheduler_eta_min": 0.0,
        },
        "independent_cv": {
            "temporal_depths": [{"name": "T0-T2", "max_tp": 3}],
        },
        "variants": {VARIANT: {"description": "test"}},
        "evaluation_sources": {"real": {"mode": "direct"}},
    }


def _split(patient_ids, seed):
    rng = np.random.default_rng(seed)
    count = len(patient_ids)
    masks = np.ones((count, 4), dtype=np.float32)
    return {
        "pids": list(patient_ids),
        "embs": rng.normal(size=(count, 4, 4)).astype(np.float32),
        "masks": masks,
        "clinical": rng.normal(
            size=(count, len(TABULAR_FEATURE_NAMES))
        ).astype(np.float32),
        "labels": np.asarray([index % 2 for index in range(count)], dtype=np.float32),
        "days": np.tile([0, 30, 80, 145], (count, 1)).astype(np.float32),
        "dim": 4,
    }


def test_checkpoint_policy_defaults_to_legacy_best_validation():
    assert independent_cv._checkpoint_policy({}) == "best_validation"
    assert independent_cv._recorded_checkpoint_policy({}) == "best_validation"
    assert independent_cv._validation_selection_matches({}, "best_validation")
    assert not independent_cv._validation_selection_matches({}, "final_epoch")
    assert (
        independent_cv._selection_criterion("best_validation")
        == independent_cv.SELECTION_CRITERION
    )
    assert (
        independent_cv._selection_criterion("final_epoch")
        == independent_cv.FINAL_EPOCH_CRITERION
    )
    with pytest.raises(ValueError, match="unsupported checkpoint_policy"):
        independent_cv._checkpoint_policy({"checkpoint_policy": "validation_last"})

    legacy = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "mewm_ispy2_full978_locked102_independent_cv.yaml"
        ).read_text()
    )
    effective = independent_cv._effective_config(
        legacy, "cv_current", "T0-T2"
    )
    assert "checkpoint_policy" not in effective
    assert independent_cv._checkpoint_policy(effective) == "best_validation"

def test_selection_and_training_phase_isolation_are_final_strict(tmp_path):
    legacy = {"depths": {"T0-T2": {"checkpoint_policy": "best_validation"}}}
    final = {"depths": {"T0-T2": {"checkpoint_policy": "final_epoch"}}}
    phase_path = tmp_path / "TRAINING_PHASE_COMPLETE.json"

    assert independent_cv._selection_test_isolation_matches(legacy)
    assert not independent_cv._selection_test_isolation_matches(final)
    assert independent_cv._training_phase_complete(phase_path, legacy, 20, 100)
    assert not independent_cv._training_phase_complete(phase_path, final, 20, 100)

    final["test_embeddings_or_labels_loaded"] = False
    assert independent_cv._selection_test_isolation_matches(final)
    payload = {
        "schema": independent_cv.SCHEMA,
        "selected_variants": final["depths"],
        "tune_checkpoints": 20,
        "formal_checkpoints": 100,
        "test_data_loaded": False,
        "test_embeddings_or_labels_loaded": False,
        "complete": True,
    }
    phase_path.write_text(json.dumps(payload))
    assert independent_cv._training_phase_complete(phase_path, final, 20, 100)

    payload["test_embeddings_or_labels_loaded"] = True
    phase_path.write_text(json.dumps(payload))
    assert not independent_cv._training_phase_complete(phase_path, final, 20, 100)

    legacy["test_embeddings_or_labels_loaded"] = True
    assert not independent_cv._selection_test_isolation_matches(legacy)



def test_fixed_epoch_config_is_isolated_and_keeps_v2_learning_rate_trajectory():
    config = _config()
    section = config["independent_cv"]

    assert section["output_dir"].endswith("independent_cv_fixed_epoch")
    assert section["output_dir"] not in {
        "results/mewm_ispy2_full978_locked102_independent_cv",
        "results/mewm_ispy2_full978_locked102_independent_cv_regularization_v2",
    }
    assert section["folds"] == 5
    assert section["tune_seeds"] == [2026, 2027]
    assert section["formal_seeds"] == list(range(42, 52))
    assert section["temporal_depths"] == [
        {"name": "T0-T2", "max_tp": 3},
        {"name": "T0-T3", "max_tp": 4},
    ]
    assert list(config["variants"]) == [VARIANT]
    assert len(section["tune_seeds"]) * section["folds"] * 2 == 20
    assert len(section["formal_seeds"]) * section["folds"] * 2 == 100

    t2 = independent_cv._effective_config(config, VARIANT, "T0-T2")
    t3 = independent_cv._effective_config(config, VARIANT, "T0-T3")
    assert t2["checkpoint_policy"] == t3["checkpoint_policy"] == "final_epoch"
    assert (t2["proj_dim"], t2["sig_dim"], t2["epochs"]) == (32, 16, 33)
    assert (t3["proj_dim"], t3["sig_dim"], t3["epochs"]) == (64, 32, 17)
    assert t2["scheduler_t_max"] == 80
    assert t3["scheduler_t_max"] == 150
    assert t2["optimizer"] == t3["optimizer"] == "adamw_layerwise"
    assert t2["projection_weight_decay"] == pytest.approx(2.0e-3)
    assert t3["projection_weight_decay"] == pytest.approx(2.0e-3)
    assert t2["residual_l2_weight"] == 0.0
    assert t3["residual_l2_weight"] == pytest.approx(1.0e-3)
    assert independent_cv._parameter_count(config, VARIANT, "T0-T2") == 40_978
    assert independent_cv._parameter_count(config, VARIANT, "T0-T3") == 88_610

    smoke = independent_cv._effective_config(
        config, VARIANT, "T0-T3", smoke=True
    )
    assert smoke["checkpoint_policy"] == "final_epoch"
    assert smoke["epochs"] == smoke["patience"] == smoke["scheduler_t_max"] == 2


def test_final_epoch_runs_full_budget_records_policy_and_never_loads_test(
    monkeypatch, tmp_path
):
    config = _small_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    train_ids = [f"TRAIN{index}" for index in range(8)]
    val_ids = [f"VAL{index}" for index in range(4)]
    train = _split(train_ids, 11)
    val = _split(val_ids, 12)

    def fake_load_split(_embeddings, _metadata, patient_ids):
        return train if list(patient_ids) == train_ids else val

    prior_state = {
        "coef": [[0.0] * len(TABULAR_FEATURE_NAMES)],
        "intercept": [0.0],
        "fitted_on": "fold_train_only",
        "feature_names": list(TABULAR_FEATURE_NAMES),
    }
    monkeypatch.setattr(independent_cv, "_load_split", fake_load_split)
    monkeypatch.setattr(
        independent_cv,
        "_fit_prior",
        lambda train_split, val_split, seed: (
            np.zeros(len(train_split["pids"]), dtype=np.float32),
            np.zeros(len(val_split["pids"]), dtype=np.float32),
            prior_state,
        ),
    )

    def fail_if_test_loaded(*args, **kwargs):
        raise AssertionError("training attempted to load locked-test data")

    monkeypatch.setattr(independent_cv, "_load_test_source", fail_if_test_loaded)
    output_dir = tmp_path / "output"
    depth = {"name": "T0-T2", "max_tp": 3}
    spec = {"fold": 0, "train_ids": train_ids, "val_ids": val_ids}
    message = independent_cv._train_fold_task(
        (
            str(config_path),
            str(output_dir),
            "tune",
            depth,
            VARIANT,
            2026,
            spec,
            "cpu",
            False,
            False,
        )
    )

    assert message.startswith("done tune T0-T2")
    run_dir = independent_cv._training_dir(
        output_dir, "tune", "T0-T2", VARIANT, 2026, 0
    )
    history = pd.read_csv(run_dir / "history.csv")
    checkpoint = torch.load(
        run_dir / "best.pt", map_location="cpu", weights_only=False
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    sentinel = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text())

    assert len(history) == 3
    assert checkpoint["selection"]["best_epoch_zero_based"] == 2
    assert checkpoint["selection"]["epochs_trained"] == 3
    assert checkpoint["selection"]["early_stopping_enabled"] is False
    assert checkpoint["selection"]["criterion"] == independent_cv.FINAL_EPOCH_CRITERION
    assert checkpoint["selection"]["best_score"] == pytest.approx(
        history.iloc[-1]["validation_auroc"]
    )
    for artifact in (checkpoint, summary, sentinel):
        assert artifact["checkpoint_policy"] == "final_epoch"
        assert artifact["validation_used_for_checkpoint_selection"] is False
        assert (
            artifact["test_embeddings_or_labels_loaded_during_training"] is False
        )
    assert (
        checkpoint["selection"]["validation_used_for_checkpoint_selection"]
        is False
    )
    assert (
        checkpoint["selection"]["test_embeddings_or_labels_loaded_during_training"]
        is False
    )
    assert checkpoint["test_data_loaded_during_training"] is False
    assert checkpoint["test_embeddings_or_labels_loaded_during_training"] is False

    expected = {
        "stage": "tune",
        "depth": depth,
        "variant": VARIANT,
        "seed": 2026,
        "fold": 0,
        "effective_config": independent_cv._effective_config(
            config, VARIANT, "T0-T2"
        ),
        "train_ids": train_ids,
        "validation_ids": val_ids,
    }
    assert independent_cv._training_complete(run_dir, expected)

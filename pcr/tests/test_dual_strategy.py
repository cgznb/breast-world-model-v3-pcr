import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.run_full978_dual_strategy as dual
from src.data import TABULAR_FEATURE_NAMES
from src.temporal import (
    canonicalize_temporal_prefix,
    expand_contiguous_torch_temporal_prefixes,
)


def test_discontinuous_t0_t2_t3_is_canonicalized_to_t0_only():
    embeddings = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    masks = np.asarray([[1, 0, 1, 1]], dtype=np.float32)
    days = np.asarray([[0, 0, 75, 145]], dtype=np.float32)

    out_embeddings, out_masks, out_days = canonicalize_temporal_prefix(
        embeddings, masks, days, 4
    )

    np.testing.assert_array_equal(out_masks, [[1, 0, 0, 0]])
    np.testing.assert_array_equal(out_days, [[0, 0, 0, 0]])
    np.testing.assert_array_equal(out_embeddings[:, 1:], 0)


def test_shared_expansion_has_unique_patient_depth_pairs_and_equal_depth_loss():
    embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)
    masks = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    )
    days = torch.tensor(
        [[0.0, 30.0, 80.0, 145.0], [0.0, 0.0, 75.0, 140.0], [0.0, 28.0, 0.0, 0.0]]
    )
    patient_index = torch.arange(3)

    _, _, _, repeated, depths, counts = expand_contiguous_torch_temporal_prefixes(
        embeddings, masks, days, patient_index
    )
    pairs = list(zip(repeated[0].tolist(), depths.tolist()))

    assert counts == (3, 2, 1, 1)
    assert len(pairs) == len(set(pairs)) == 7
    assert (1, 2) not in pairs
    point_losses = torch.tensor([1.0, 3.0, 5.0, 9.0, 11.0, 15.0, 21.0])
    expected = torch.tensor([(1 + 3 + 5) / 3, (9 + 11) / 2, 15, 21]).mean()
    torch.testing.assert_close(dual._equal_depth_loss(point_losses, counts), expected)


def test_equal_depth_loss_rejects_invalid_partition():
    with pytest.raises(ValueError, match="partition mismatch"):
        dual._equal_depth_loss(torch.ones(3), (2, 2))
    with pytest.raises(ValueError, match="no temporal depth"):
        dual._equal_depth_loss(torch.empty(0), (0, 0, 0, 0))


def test_variant_selection_prefers_smaller_gap_within_validation_tolerance():
    rows = [
        {"variant": "baseline", "seed": 1, "validation_auroc": 0.800, "train_auroc": 0.950, "train_validation_gap": 0.150},
        {"variant": "baseline", "seed": 2, "validation_auroc": 0.802, "train_auroc": 0.952, "train_validation_gap": 0.150},
        {"variant": "moderate", "seed": 1, "validation_auroc": 0.795, "train_auroc": 0.850, "train_validation_gap": 0.055},
        {"variant": "moderate", "seed": 2, "validation_auroc": 0.796, "train_auroc": 0.851, "train_validation_gap": 0.055},
        {"variant": "compact", "seed": 1, "validation_auroc": 0.770, "train_auroc": 0.800, "train_validation_gap": 0.030},
        {"variant": "compact", "seed": 2, "validation_auroc": 0.772, "train_auroc": 0.802, "train_validation_gap": 0.030},
    ]

    selected, _, summary = dual._select_with_tolerance(
        rows, ["baseline", "moderate", "compact"], 0.01
    )

    assert selected == "moderate"
    assert summary.set_index("variant").loc["compact", "within_validation_tolerance"] == 0


def _t0_checkpoint(seed=42, max_tp=1):
    return {
        "cell": "global | +tab+temporal",
        "seed": seed,
        "effective_config": {
            "max_tp": max_tp,
            "model_type": "tdn",
            "use_prior": True,
            "input_dim": 1152,
            "clinical_dim": len(TABULAR_FEATURE_NAMES),
        },
        "run_context": {"temporal_depth": "T0", "max_tp": max_tp},
        "clinical_prior": {"feature_names": list(TABULAR_FEATURE_NAMES)},
    }


def test_reused_t0_checkpoint_is_contract_checked(tmp_path):
    root = tmp_path / "old_t0"
    checkpoint_path = root / "seed_42" / "global_tab_temporal" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(_t0_checkpoint(), checkpoint_path)
    config = {
        "data": {"embedding_dim": 1152},
        "strategies": {"independent": {"reuse_t0_from": str(root)}},
    }

    assert dual._validate_reused_t0_checkpoint(config, 42) == checkpoint_path

    torch.save(_t0_checkpoint(max_tp=2), checkpoint_path)
    with pytest.raises(ValueError, match="contract mismatch"):
        dual._validate_reused_t0_checkpoint(config, 42)


def test_generated_296_future_tokens_become_292_contiguous_tokens():
    masks = np.asarray(
        [[1, 0, 1, 1]]
        + [[1, 1, 0, 0]] * 2
        + [[1, 1, 0, 1]] * 2
        + [[1, 1, 1, 0]] * 3
        + [[1, 1, 1, 1]] * 94,
        dtype=np.float32,
    )
    split = {
        "embs": np.ones((102, 4, 2), dtype=np.float32),
        "masks": masks,
        "days": np.ones((102, 4), dtype=np.float32),
    }

    assert int(masks[:, 1:].sum()) == 296
    assert dual._retained_future_tokens(split) == 292


def _prediction_frame(patient_ids, strategy="independent", source=None):
    frame = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": [index % 2 for index in range(len(patient_ids))],
            "probability": np.linspace(0.2, 0.8, len(patient_ids)),
            "clinical_prior_probability": np.linspace(0.3, 0.7, len(patient_ids)),
            "seed": 42,
            "strategy": strategy,
            "temporal_depth": "T0",
            "max_tp": 1,
        }
    )
    if source is not None:
        frame["source"] = source
        frame["split"] = "test"
    return frame


def test_training_completion_binds_patient_predictions_and_sentinel(tmp_path):
    train_ids = ["TRAIN0", "TRAIN1", "TRAIN2", "TRAIN3"]
    val_ids = ["VAL0", "VAL1", "VAL2", "VAL3"]
    effective = {"batch_size": 4}
    depths = [{"name": "T0", "max_tp": 1}]
    checkpoint = {
        "schema": dual.SCHEMA,
        "strategy": "independent",
        "stage": "formal",
        "variant": "moderate",
        "seed": 42,
        "temporal_depth": "T0",
        "effective_config": effective,
        "selection": {"criterion": dual.SELECTION_CRITERION},
        "test_data_loaded_during_training": False,
        "clinical_prior": {
            "fitted_on": "fold_train_only",
            "feature_names": list(TABULAR_FEATURE_NAMES),
        },
        "train_ids": train_ids,
        "validation_ids": val_ids,
    }
    torch.save(checkpoint, tmp_path / "best.pt")
    (tmp_path / "summary.json").write_text("{}\n")
    (tmp_path / "history.csv").write_text("epoch\n0\n")
    _prediction_frame(train_ids).to_csv(tmp_path / "train_predictions.csv", index=False)
    _prediction_frame(val_ids).to_csv(tmp_path / "val_predictions.csv", index=False)
    sentinel = {
        "schema": dual.SCHEMA,
        "strategy": "independent",
        "stage": "formal",
        "variant": "moderate",
        "seed": 42,
        "temporal_depth": "T0",
        "effective_config": effective,
        "complete": True,
    }
    (tmp_path / "TRAINING_COMPLETE.json").write_text(json.dumps(sentinel))
    expected = {
        "strategy": "independent",
        "stage": "formal",
        "variant": "moderate",
        "seed": 42,
        "temporal_depth": "T0",
        "effective_config": effective,
        "depths": depths,
        "train_ids": train_ids,
        "validation_ids": val_ids,
    }

    assert dual._validate_training_artifact(tmp_path, expected)

    _prediction_frame(["VAL0", "VAL1", "VAL2", "OTHER"]).to_csv(
        tmp_path / "val_predictions.csv", index=False
    )
    assert not dual._validate_training_artifact(tmp_path, expected)


def test_evaluation_completion_recomputes_metrics_and_binds_test_ids(tmp_path):
    test_ids = ["TEST0", "TEST1", "TEST2", "TEST3"]
    depths = [{"name": "T0", "max_tp": 1}]
    predictions = _prediction_frame(test_ids, strategy="shared_contiguous", source="real")
    predictions.to_csv(tmp_path / "test_predictions.csv", index=False)
    metrics = pd.DataFrame(
        [
            {
                "strategy": "shared_contiguous",
                "source": "real",
                "seed": 42,
                **dual._metrics_by_depth(predictions)[0],
            }
        ]
    )
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": dual.SCHEMA,
                "strategy": "shared_contiguous",
                "source": "real",
                "seed": 42,
                "test_ids": test_ids,
                "temporal_depths": depths,
                "retained_contiguous_future_tokens": 3,
                "complete": True,
            }
        )
    )
    expected = {
        "strategy": "shared_contiguous",
        "source": "real",
        "seed": 42,
        "test_ids": test_ids,
        "depths": depths,
        "retained_future_tokens": 3,
    }

    assert dual._validate_evaluation_artifact(tmp_path, expected)

    metrics.loc[0, "auroc"] += 0.01
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    assert not dual._validate_evaluation_artifact(tmp_path, expected)


def test_training_path_never_loads_test_embeddings_or_labels(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)

    def split(patient_ids):
        n = len(patient_ids)
        return {
            "pids": patient_ids,
            "embs": rng.normal(size=(n, 4, 4)).astype(np.float32),
            "masks": np.ones((n, 4), dtype=np.float32),
            "clinical": rng.normal(size=(n, len(TABULAR_FEATURE_NAMES))).astype(np.float32),
            "labels": np.asarray([index % 2 for index in range(n)], dtype=np.float32),
            "days": np.tile([0, 30, 80, 145], (n, 1)).astype(np.float32),
        }

    ids = {
        "train": [f"TRAIN{index}" for index in range(6)],
        "val": [f"VAL{index}" for index in range(4)],
        "test": [f"TEST{index}" for index in range(4)],
    }
    data = {"train": split(ids["train"]), "val": split(ids["val"])}
    prior_state = {
        "coef": [[0.0] * len(TABULAR_FEATURE_NAMES)],
        "intercept": [0.0],
        "fitted_on": "fold_train_only",
        "feature_names": list(TABULAR_FEATURE_NAMES),
    }
    monkeypatch.setattr(dual, "_split_ids", lambda config: ids)
    monkeypatch.setattr(dual, "_load_development", lambda config: (data, ids))
    monkeypatch.setattr(
        dual,
        "_fit_prior",
        lambda train, val, seed: (
            np.zeros(len(train["pids"]), dtype=np.float32),
            np.zeros(len(val["pids"]), dtype=np.float32),
            prior_state,
        ),
    )

    def fail_if_test_loaded(*args, **kwargs):
        raise AssertionError("training loaded test embeddings or labels")

    monkeypatch.setattr(dual, "_load_test_source", fail_if_test_loaded)
    config = {
        "data": {"embedding_dim": 4},
        "downstream": {
            "proj_dim": 4,
            "sig_dim": 4,
            "n_layers": 1,
            "n_heads": 2,
            "use_prior": True,
            "epochs": 1,
            "patience": 1,
            "min_delta": 0.0,
            "batch_size": 3,
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "embedding_dropout": 0.0,
            "grad_clip": 1.0,
            "residual_logit_limit": None,
        },
        "experiment": {"temporal_depths": [{"name": "T0", "max_tp": 1}]},
        "variants": {"baseline": {"description": "test"}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "output"

    dual._train_task(
        (
            str(config_path),
            str(output_dir),
            "independent",
            "formal",
            "baseline",
            42,
            "T0",
            "cpu",
            False,
            False,
        )
    )

    checkpoint = torch.load(
        output_dir
        / "independent"
        / "train"
        / "formal"
        / "t0"
        / "variant_baseline"
        / "seed_42"
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["test_embeddings_or_labels_loaded_during_training"] is False
    assert checkpoint["test_membership_validated_during_training"] is True

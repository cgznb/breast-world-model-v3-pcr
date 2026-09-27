import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.run_full978_shared_contiguous_cv as shared
from src.data import TABULAR_FEATURE_NAMES
from src.temporal import expand_contiguous_torch_temporal_prefixes


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mewm_ispy2_full978_locked102_shared_contiguous_cv.yaml"
FIXED_EPOCH_CONFIG = (
    ROOT / "configs/mewm_ispy2_full978_locked102_shared_contiguous_cv_fixed_epoch12.yaml"
)


def _small_config(tmp_path, epochs=1):
    return {
        "data": {
            "embedding_dim": 4,
            "train_embeddings_dir": str(tmp_path / "embeddings"),
            "metadata_csv": str(tmp_path / "metadata.csv"),
            "expected_contiguous_future_embeddings": 3,
        },
        "downstream": {
            "proj_dim": 4,
            "sig_dim": 4,
            "n_layers": 1,
            "n_heads": 2,
            "use_prior": True,
            "epochs": epochs,
            "patience": epochs,
            "min_delta": 0.0,
            "batch_size": 4,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "embedding_dropout": 0.0,
            "grad_clip": 1.0,
            "optimizer": "adam",
            "scheduler_eta_min": 0.0,
            "residual_l2_weight": 0.0,
            "residual_logit_limit": None,
            "residual_scale_init": 0.1,
        },
        "shared_contiguous_cv": {
            "folds": 5,
            "fold_seed": 2026,
            "temporal_depths": [
                {"name": "T0", "max_tp": 1},
                {"name": "T0-T1", "max_tp": 2},
                {"name": "T0-T2", "max_tp": 3},
                {"name": "T0-T3", "max_tp": 4},
            ],
            "prefix_policy": shared.PREFIX_POLICY,
            "loss_weighting": shared.LOSS_WEIGHTING,
            "selection": shared.SELECTION_CRITERION,
            "selection_tolerance": 0.005,
        },
        "variants": {"baseline": {"description": "test"}},
        "evaluation_sources": {"real": {"mode": "direct"}},
    }


def _split(patient_ids, masks):
    rng = np.random.default_rng(17 + len(patient_ids))
    labels = np.asarray([index % 2 for index in range(len(patient_ids))], dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    embeddings = rng.normal(size=(len(patient_ids), 4, 4)).astype(np.float32)
    embeddings *= masks[..., None]
    days = np.tile([0, 30, 80, 145], (len(patient_ids), 1)).astype(np.float32)
    days *= masks
    return {
        "pids": list(patient_ids),
        "embs": embeddings,
        "masks": masks,
        "clinical": rng.normal(
            size=(len(patient_ids), len(TABULAR_FEATURE_NAMES))
        ).astype(np.float32),
        "labels": labels,
        "days": days,
        "dim": 4,
    }


def test_formal_config_declares_isolated_four_variant_five_fold_contract():
    config = yaml.safe_load(CONFIG.read_text())
    section = config["shared_contiguous_cv"]

    assert shared._checkpoint_policy(config) == "best_validation_macro"
    assert section["output_dir"].endswith("locked102_shared_contiguous_cv")
    assert section["folds"] == 5
    assert section["selection_tolerance"] == 0.005
    assert section["prefix_policy"] == shared.PREFIX_POLICY
    assert section["loss_weighting"] == shared.LOSS_WEIGHTING
    assert list(config["variants"]) == [
        "baseline",
        "moderate",
        "compact",
        "ultra_compact_adamw",
    ]
    assert len(section["tune_seeds"]) * section["folds"] * len(config["variants"]) == 40
    assert len(section["formal_seeds"]) * section["folds"] == 50
    assert [shared._parameter_count(config, name) for name in config["variants"]] == [
        203842,
        203842,
        88610,
        40978,
    ]


def test_fixed_epoch_config_is_isolated_moderate_and_preserves_scheduler_curve():
    config = yaml.safe_load(FIXED_EPOCH_CONFIG.read_text())
    section = config["shared_contiguous_cv"]

    assert section["output_dir"].endswith("shared_contiguous_cv_fixed_epoch12")
    assert section["checkpoint_policy"] == shared.FINAL_EPOCH_CHECKPOINT_POLICY
    assert section["folds"] == 5
    assert list(config["variants"]) == ["moderate"]
    effective = shared._effective_config(config, "moderate")
    assert effective["epochs"] == 12
    assert effective["scheduler_t_max"] == 150
    assert effective["proj_dim"] == 128
    assert effective["sig_dim"] == 64
    assert effective["embedding_dropout"] == pytest.approx(0.10)
    assert effective["dropout"] == pytest.approx(0.25)
    assert shared._parameter_count(config, "moderate") == 203842

    smoke = shared._effective_config(config, "moderate", smoke=True)
    assert smoke["epochs"] == 2
    assert smoke["scheduler_t_max"] == 150


def test_checkpoint_policy_defaults_to_best_and_rejects_unknown(tmp_path):
    config = _small_config(tmp_path)
    policies = shared._policies(config)

    assert policies["checkpoint_policy"] == shared.BEST_VALIDATION_CHECKPOINT_POLICY
    assert shared._allows_legacy_checkpoint_policy(config)
    legacy = {
        key: value for key, value in policies.items() if key != "checkpoint_policy"
    }
    assert shared._artifact_policies_match(legacy, policies, allow_legacy=True)
    assert not shared._artifact_policies_match(legacy, policies, allow_legacy=False)

    config["shared_contiguous_cv"]["checkpoint_policy"] = "unknown"
    with pytest.raises(ValueError, match="checkpoint_policy"):
        shared._checkpoint_policy(config)


def test_canonical_split_stops_at_first_missing_visit_without_mutating_input():
    split = _split(["P0", "P1"], [[1, 0, 1, 1], [1, 1, 0, 1]])
    original_embeddings = split["embs"].copy()
    original_masks = split["masks"].copy()

    canonical = shared._canonical_split(split, 4)

    np.testing.assert_array_equal(canonical["masks"], [[1, 0, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(canonical["embs"][:, 2:], 0)
    np.testing.assert_array_equal(split["embs"], original_embeddings)
    np.testing.assert_array_equal(split["masks"], original_masks)


def test_unique_contiguous_expansion_and_equal_depth_loss():
    embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)
    masks = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
    )
    days = torch.tensor(
        [[0.0, 30.0, 80.0, 145.0], [0.0, 0.0, 75.0, 140.0], [0.0, 28.0, 0.0, 150.0]]
    )
    patients = torch.arange(3)

    _, _, _, repeated, depths, counts = expand_contiguous_torch_temporal_prefixes(
        embeddings, masks, days, patients
    )
    pairs = list(zip(repeated[0].tolist(), depths.tolist()))

    assert counts == (3, 2, 1, 1)
    assert len(pairs) == len(set(pairs)) == 7
    assert (1, 2) not in pairs
    losses = torch.tensor([1.0, 3.0, 5.0, 9.0, 11.0, 15.0, 21.0])
    expected = torch.tensor([(1 + 3 + 5) / 3, (9 + 11) / 2, 15, 21]).mean()
    torch.testing.assert_close(shared._equal_depth_loss(losses, counts), expected)


def test_selection_uses_gap_then_parameters_then_auroc_inside_point005_band():
    summary = pd.DataFrame(
        [
            {"variant": "baseline", "macro_oof_auroc_mean": 0.800, "positive_train_oof_gap": 0.080, "parameters": 203842},
            {"variant": "moderate", "macro_oof_auroc_mean": 0.798, "positive_train_oof_gap": 0.030, "parameters": 203842},
            {"variant": "compact", "macro_oof_auroc_mean": 0.797, "positive_train_oof_gap": 0.030, "parameters": 88610},
            {"variant": "ultra", "macro_oof_auroc_mean": 0.794, "positive_train_oof_gap": 0.010, "parameters": 40978},
        ]
    )

    selected, decorated = shared._select_variant(
        summary, ["baseline", "moderate", "compact", "ultra"], 0.005
    )

    assert selected == "compact"
    assert not bool(decorated.set_index("variant").loc["ultra", "within_oof_tolerance"])

    summary.loc[summary["variant"] == "moderate", "positive_train_oof_gap"] = 0.020
    selected, _ = shared._select_variant(
        summary, ["baseline", "moderate", "compact", "ultra"], 0.005
    )
    assert selected == "moderate"


def test_fold_probability_ensemble_requires_and_averages_exactly_five_models():
    depths = [{"name": "T0", "max_tp": 1}, {"name": "T0-T1", "max_tp": 2}]
    patient_ids = ["P0", "P1"]
    rows = []
    for fold in range(5):
        for depth in depths:
            for patient_index, patient_id in enumerate(patient_ids):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "label": patient_index,
                        "probability": 0.1 * patient_index + 0.01 * fold,
                        "residual_logit": float(fold),
                        "clinical_prior_probability": 0.4 + 0.01 * fold,
                        "seed": 42,
                        "fold": fold,
                        "split": "test_fold",
                        "strategy": "shared_contiguous_cv",
                        "temporal_depth": depth["name"],
                        "max_tp": depth["max_tp"],
                        "source": "real",
                    }
                )
    frame = pd.DataFrame(rows)

    ensemble = shared._ensemble_fold_predictions(frame, patient_ids, depths, 5)

    assert len(ensemble) == 4
    assert set(ensemble["fold"]) == {-1}
    assert set(ensemble["split"]) == {"test_fold_ensemble"}
    np.testing.assert_allclose(
        ensemble[ensemble["patient_id"] == "P0"]["probability"], 0.02
    )
    with pytest.raises(RuntimeError, match="do not cover"):
        shared._ensemble_fold_predictions(frame[frame["fold"] != 4], patient_ids, depths, 5)


def test_generated_raw_296_future_tokens_reduce_to_292_contiguous_tokens():
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
        "days": masks.copy(),
    }

    assert int(masks[:, 1:].sum()) == 296
    assert shared._retained_future_tokens(split) == 292


def test_fold_wrapper_passes_only_declared_seed_and_count(monkeypatch, tmp_path):
    config = _small_config(tmp_path)
    captured = {}

    def fake_fold_specs(value):
        captured.update(value["anti_overfit"])
        return [], [], [], pd.DataFrame()

    monkeypatch.setattr(shared, "_shared_fold_specs", fake_fold_specs)

    shared._fold_specs(config)

    assert captured == {"folds": 5, "fold_seed": 2026}


def test_train_fold_uses_unique_contiguous_pairs_and_never_loads_test(
    monkeypatch, tmp_path
):
    config = _small_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    train_ids = [f"TRAIN{index}" for index in range(8)]
    val_ids = [f"VAL{index}" for index in range(4)]
    train_masks = (
        [[1, 1, 1, 1]] * 4
        + [[1, 0, 1, 1]] * 2
        + [[1, 1, 0, 1]] * 2
    )
    train = _split(train_ids, train_masks)
    val = _split(val_ids, [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    def fake_load_split(_embeddings, _metadata, patient_ids):
        return train if list(patient_ids) == train_ids else val

    prior_state = {
        "coef": [[0.0] * len(TABULAR_FEATURE_NAMES)],
        "intercept": [0.0],
        "fitted_on": "fold_train_only",
        "feature_names": list(TABULAR_FEATURE_NAMES),
    }
    monkeypatch.setattr(shared, "_load_split", fake_load_split)
    monkeypatch.setattr(
        shared,
        "_fit_prior",
        lambda train_split, val_split, seed: (
            np.zeros(len(train_split["pids"]), dtype=np.float32),
            np.zeros(len(val_split["pids"]), dtype=np.float32),
            prior_state,
        ),
    )

    def fail_if_test_loaded(*args, **kwargs):
        raise AssertionError("training attempted to load locked-test data")

    monkeypatch.setattr(shared, "_load_test_source", fail_if_test_loaded)
    output_dir = tmp_path / "output"
    message = shared._train_fold_task(
        (
            str(config_path),
            str(output_dir),
            "tune",
            "baseline",
            2026,
            {"fold": 0, "train_ids": train_ids, "val_ids": val_ids},
            "cpu",
            False,
            False,
        )
    )

    assert message.startswith("done tune baseline")
    run_dir = shared._training_dir(output_dir, "tune", "baseline", 2026, 0)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    history = pd.read_csv(run_dir / "history.csv")
    assert checkpoint["expected_prefix_counts"] == [8, 6, 4, 4]
    assert checkpoint["training_view_policy"] == shared.PREFIX_POLICY
    assert checkpoint["loss_weighting"] == shared.LOSS_WEIGHTING
    assert history.loc[0, "train_pairs_T0"] == 8
    assert history.loc[0, "train_pairs_T0-T1"] == 6
    assert history.loc[0, "train_pairs_T0-T2"] == 4
    assert history.loc[0, "train_pairs_T0-T3"] == 4


def test_final_epoch_trains_full_budget_and_binds_artifacts(monkeypatch, tmp_path):
    config = _small_config(tmp_path, epochs=4)
    config["downstream"]["patience"] = 1
    config["downstream"]["min_delta"] = 2.0
    config["shared_contiguous_cv"][
        "checkpoint_policy"
    ] = shared.FINAL_EPOCH_CHECKPOINT_POLICY
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    train_ids = [f"TRAIN{index}" for index in range(8)]
    val_ids = [f"VAL{index}" for index in range(4)]
    train = _split(train_ids, [[1, 1, 1, 1]] * len(train_ids))
    val = _split(val_ids, [[1, 1, 1, 1]] * len(val_ids))

    def fake_load_split(_embeddings, _metadata, patient_ids):
        return train if list(patient_ids) == train_ids else val

    prior_state = {
        "coef": [[0.0] * len(TABULAR_FEATURE_NAMES)],
        "intercept": [0.0],
        "fitted_on": "fold_train_only",
        "feature_names": list(TABULAR_FEATURE_NAMES),
    }
    monkeypatch.setattr(shared, "_load_split", fake_load_split)
    monkeypatch.setattr(
        shared,
        "_fit_prior",
        lambda train_split, val_split, seed: (
            np.zeros(len(train_split["pids"]), dtype=np.float32),
            np.zeros(len(val_split["pids"]), dtype=np.float32),
            prior_state,
        ),
    )

    def fail_if_test_loaded(*args, **kwargs):
        raise AssertionError("training attempted to load locked-test data")

    monkeypatch.setattr(shared, "_load_test_source", fail_if_test_loaded)
    output_dir = tmp_path / "output"
    spec = {"fold": 0, "train_ids": train_ids, "val_ids": val_ids}
    shared._train_fold_task(
        (
            str(config_path),
            str(output_dir),
            "tune",
            "baseline",
            2026,
            spec,
            "cpu",
            False,
            False,
        )
    )

    run_dir = shared._training_dir(output_dir, "tune", "baseline", 2026, 0)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    sentinel = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text())
    history = pd.read_csv(run_dir / "history.csv")
    selection = checkpoint["selection"]

    assert history["epoch"].astype(int).tolist() == [0, 1, 2, 3]
    assert selection["criterion"] == shared.FINAL_EPOCH_CHECKPOINT_CRITERION
    assert selection["checkpoint_epoch_zero_based"] == 3
    assert selection["best_epoch_zero_based"] == 3
    assert selection["epochs_trained"] == 4
    assert selection["early_stopping_enabled"] is False
    assert selection["validation_used_for_checkpoint_selection"] is False
    for artifact in (checkpoint, summary, sentinel):
        assert artifact["checkpoint_policy"] == shared.FINAL_EPOCH_CHECKPOINT_POLICY
        assert artifact["early_stopping_enabled"] is False
        assert artifact["validation_used_for_checkpoint_selection"] is False

    effective = shared._effective_config(config, "baseline")
    expected = {
        "stage": "tune",
        "variant": "baseline",
        "seed": 2026,
        "fold": 0,
        "effective_config": effective,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "depths": shared._depths(config),
        "policies": shared._policies(config),
        "allow_legacy_checkpoint_policy": False,
    }
    assert shared._training_complete(run_dir, expected)

    history.iloc[:-1].to_csv(run_dir / "history.csv", index=False)
    assert not shared._training_complete(run_dir, expected)


def test_evaluate_validates_all_formal_models_before_loading_test(monkeypatch, tmp_path):
    config = _small_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    loaded = {"test": False}

    monkeypatch.setattr(shared, "_training_complete", lambda *args, **kwargs: False)

    def mark_test_load(*args, **kwargs):
        loaded["test"] = True
        raise AssertionError("test should not be loaded")

    monkeypatch.setattr(shared, "_load_test_source", mark_test_load)
    task = (
        str(config_path),
        str(tmp_path / "output"),
        {"selected_variant": "baseline"},
        "real",
        42,
        [{"fold": 0, "train_ids": ["TRAIN"], "val_ids": ["VAL"]}],
        ["TEST"],
        "cpu",
        False,
    )

    with pytest.raises(RuntimeError, match="formal training contract mismatch"):
        shared._evaluate_task(task)
    assert loaded["test"] is False


def test_selection_contract_rejects_any_test_use(tmp_path):
    config = _small_config(tmp_path)
    variants = list(config["variants"])
    selection = {
        "schema": shared.SCHEMA,
        "stage": "tune",
        "criterion": shared.SELECTION_CRITERION,
        "selection_tolerance": 0.005,
        "selected_variant": "baseline",
        "selected_effective_config": shared._effective_config(config, "baseline"),
        "variant_configs": {
            "baseline": shared._effective_config(config, "baseline")
        },
        "tune_seeds": [2026],
        "folds": 5,
        "test_data_used": False,
        "test_embeddings_or_labels_loaded": False,
        **shared._policies(config),
    }

    assert shared._validate_selection(config, selection, variants, [2026], 5, False)
    legacy_selection = dict(selection)
    legacy_selection.pop("checkpoint_policy")
    assert shared._validate_selection(
        config, legacy_selection, variants, [2026], 5, False
    )
    config["shared_contiguous_cv"][
        "checkpoint_policy"
    ] = shared.BEST_VALIDATION_CHECKPOINT_POLICY
    assert not shared._validate_selection(
        config, legacy_selection, variants, [2026], 5, False
    )
    config["shared_contiguous_cv"].pop("checkpoint_policy")
    selection["test_data_used"] = True
    assert not shared._validate_selection(config, selection, variants, [2026], 5, False)


def test_training_phase_completion_declares_no_test_payload_load(
    monkeypatch, tmp_path
):
    config = _small_config(tmp_path)
    section = config["shared_contiguous_cv"]
    section.update(
        {
            "output_dir": str(tmp_path / "canonical"),
            "tune_seeds": [2026],
            "formal_seeds": [42],
            "jobs": 1,
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "output"
    args = SimpleNamespace(
        config=str(config_path),
        phase="train",
        variants=None,
        tune_seeds=None,
        seeds=None,
        sources=None,
        jobs=1,
        device="cpu",
        output_dir=str(output_dir),
        smoke=True,
        force=False,
    )
    specs = [{"fold": fold} for fold in range(5)]
    captured = {}

    monkeypatch.setattr(shared, "parse_args", lambda: args)
    monkeypatch.setattr(
        shared,
        "_fold_specs",
        lambda _config: (
            specs,
            [f"DEV{index}" for index in range(876)],
            [f"TEST{index}" for index in range(102)],
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(shared, "_write_fold_manifests", lambda *args: None)
    monkeypatch.setattr(shared, "_atomic_text", lambda *args: None)
    monkeypatch.setattr(shared, "_run_tasks", lambda *args: None)
    monkeypatch.setattr(
        shared,
        "_build_oof",
        lambda *args: {"selected_variant": "baseline"} if args[6] else None,
    )
    monkeypatch.setattr(
        shared,
        "_atomic_json",
        lambda path, payload: captured.__setitem__(Path(path).name, payload),
    )

    shared.main()

    completion = captured["TRAINING_PHASE_COMPLETE.json"]
    assert completion["test_data_loaded"] is False
    assert completion["test_embeddings_or_labels_loaded"] is False
    assert completion["complete"] is True
    assert "EXPERIMENT_COMPLETE.json" not in captured

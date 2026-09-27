import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_full978_anti_overfit import (
    SCHEMA,
    _evaluation_complete,
    _evaluation_source_contract,
    _fold_specs,
    _load_test_source,
    _read_metadata,
    _training_complete,
)
from src.data import TABULAR_FEATURE_NAMES
from src.tdn import TDN
from src.temporal import (
    canonicalize_temporal_prefix,
    contiguous_prefix_lengths,
    expand_contiguous_torch_temporal_prefixes,
    expand_all_torch_temporal_prefixes,
    mask_torch_temporal_prefix,
)


def _tdn_config(**overrides):
    downstream = {
        "input_dim": 8,
        "proj_dim": 4,
        "clinical_dim": 2,
        "sig_dim": 4,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "use_prior": True,
    }
    downstream.update(overrides)
    return {"downstream": downstream}


def _tdn_inputs(batch=2):
    torch.manual_seed(7)
    return (
        torch.randn(batch, 4, 8),
        torch.ones(batch, 4),
        torch.randn(batch, 2),
        torch.tensor([[0.0, 30.0, 90.0, 150.0]]).expand(batch, -1).clone(),
        torch.linspace(-0.4, 0.6, batch),
    )


def test_mask_prefix_does_not_modify_any_input_in_place():
    embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    masks = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    days = torch.tensor([[0.0, 20.0, 70.0, 140.0], [0.0, 0.0, 82.0, 151.0]])
    originals = tuple(value.clone() for value in (embeddings, masks, days))

    masked_embeddings, masked_masks, masked_days = mask_torch_temporal_prefix(
        embeddings, masks, days, torch.tensor([2, 3])
    )
    masked_embeddings[0, 0, 0] = -999
    masked_masks[0, 0] = -999
    masked_days[0, 0] = -999

    for value, original in zip((embeddings, masks, days), originals):
        torch.testing.assert_close(value, original)


def test_mask_prefix_uses_calendar_positions_and_preserves_missing_visit():
    embeddings = torch.tensor(
        [
            [[10.0], [0.0], [30.0], [40.0]],
            [[11.0], [21.0], [0.0], [41.0]],
        ]
    )
    masks = torch.tensor([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    days = torch.tensor([[0.0, 0.0, 75.0, 150.0], [0.0, 28.0, 0.0, 145.0]])

    out_embeddings, out_masks, out_days = mask_torch_temporal_prefix(
        embeddings, masks, days, torch.tensor([3, 2])
    )

    torch.testing.assert_close(
        out_embeddings.squeeze(-1),
        torch.tensor([[10.0, 0.0, 30.0, 0.0], [11.0, 21.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        out_masks, torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        out_days, torch.tensor([[0.0, 0.0, 75.0, 0.0], [0.0, 28.0, 0.0, 0.0]])
    )


def test_expand_all_prefixes_is_depth_major_and_repeats_patient_tensors():
    embeddings = torch.tensor(
        [
            [[10.0], [20.0], [30.0]],
            [[11.0], [21.0], [31.0]],
        ]
    )
    masks = torch.ones(2, 3)
    days = torch.tensor([[0.0, 20.0, 70.0], [0.0, 30.0, 80.0]])
    labels = torch.tensor([0.0, 1.0])
    clinical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    out_embeddings, out_masks, out_days, repeated, depths = (
        expand_all_torch_temporal_prefixes(
            embeddings, masks, days, labels, clinical
        )
    )
    repeated_labels, repeated_clinical = repeated

    torch.testing.assert_close(depths, torch.tensor([1, 1, 2, 2, 3, 3]))
    torch.testing.assert_close(
        out_embeddings.squeeze(-1),
        torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 20.0, 0.0],
                [11.0, 21.0, 0.0],
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
            ]
        ),
    )
    torch.testing.assert_close(
        out_masks,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    torch.testing.assert_close(
        out_days,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 30.0, 0.0],
                [0.0, 20.0, 70.0],
                [0.0, 30.0, 80.0],
            ]
        ),
    )
    torch.testing.assert_close(repeated_labels, torch.tensor([0.0, 1.0] * 3))
    torch.testing.assert_close(
        repeated_clinical,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]] * 3),
    )


@pytest.mark.parametrize(
    "embeddings,masks,days,depths,match",
    [
        (torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4), 1, "embedding/mask"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 3), torch.zeros(2, 4), 1, "embedding/mask"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 3), 1, "elapsed-day"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 4), [1], "one value"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 4), [1.0, 2.0], "integers"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 4), [True, False], "integers"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 4), [0, 2], "between"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 4), [1, 5], "between"),
    ],
)
def test_mask_prefix_rejects_invalid_inputs(embeddings, masks, days, depths, match):
    with pytest.raises(ValueError, match=match):
        mask_torch_temporal_prefix(embeddings, masks, days, depths)


def test_expand_all_prefixes_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="embeddings must have shape"):
        expand_all_torch_temporal_prefixes(
            torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4)
        )
    with pytest.raises(ValueError, match="mask/day shape"):
        expand_all_torch_temporal_prefixes(
            torch.zeros(2, 4, 3), torch.zeros(2, 3), torch.zeros(2, 4)
        )
    with pytest.raises(ValueError, match="patient tensors"):
        expand_all_torch_temporal_prefixes(
            torch.zeros(2, 4, 3),
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(3),
        )


def test_contiguous_prefix_lengths_stop_at_first_missing_visit():
    masks = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(contiguous_prefix_lengths(masks), [4, 2, 1, 0])


def test_canonical_prefix_removes_every_visit_after_first_gap():
    embeddings = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
    masks = np.asarray([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=np.float32)
    days = np.asarray([[0, 0, 80, 140], [0, 30, 0, 150]], dtype=np.float32)

    out_embeddings, out_masks, out_days = canonicalize_temporal_prefix(
        embeddings, masks, days, 4
    )

    np.testing.assert_array_equal(out_masks, [[1, 0, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(out_days, [[0, 0, 0, 0], [0, 30, 0, 0]])
    np.testing.assert_array_equal(out_embeddings[0, 1:], 0)
    np.testing.assert_array_equal(out_embeddings[1, 2:], 0)


def test_expand_contiguous_prefixes_contains_each_patient_depth_once():
    embeddings = torch.tensor(
        [
            [[10.0], [20.0], [30.0], [40.0]],
            [[11.0], [21.0], [31.0], [41.0]],
            [[12.0], [22.0], [32.0], [42.0]],
        ]
    )
    masks = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    )
    days = torch.tensor(
        [[0.0, 20.0, 70.0, 140.0], [0.0, 0.0, 80.0, 150.0], [0.0, 0.0, 75.0, 145.0]]
    )
    patient_index = torch.arange(3)

    out_embeddings, out_masks, out_days, repeated, depths, counts = (
        expand_contiguous_torch_temporal_prefixes(
            embeddings, masks, days, patient_index
        )
    )

    assert counts == (2, 1, 1, 1)
    torch.testing.assert_close(depths, torch.tensor([1, 1, 2, 3, 4]))
    torch.testing.assert_close(repeated[0], torch.tensor([0, 1, 0, 0, 0]))
    torch.testing.assert_close(
        out_masks,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )
    assert out_embeddings.shape == (5, 4, 1)
    assert out_days.shape == (5, 4)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"residual_logit_limit": 0.0}, "must be positive"),
        ({"residual_logit_limit": -1.0}, "must be positive"),
        (
            {"residual_logit_limit": 0.5, "residual_scale_init": 0.0},
            "between zero and residual_logit_limit",
        ),
        (
            {"residual_logit_limit": 0.5, "residual_scale_init": 0.5},
            "between zero and residual_logit_limit",
        ),
    ],
)
def test_bounded_residual_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        TDN(_tdn_config(**overrides))


def test_bounded_residual_respects_limit_and_returns_effective_residual():
    limit = 0.4
    model = TDN(
        _tdn_config(residual_logit_limit=limit, residual_scale_init=0.1)
    ).eval()
    embeddings, masks, clinical, days, prior = _tdn_inputs()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(100.0)
        logits, residual = model(
            embeddings,
            masks,
            clinical,
            days=days,
            prior_logit=prior,
            return_residual=True,
        )

    assert float(model.residual_scale().detach()) < limit
    assert torch.all(residual.abs() < limit)
    torch.testing.assert_close(logits, prior + residual)


def test_bounded_residual_transform_has_nonzero_finite_gradients():
    model = TDN(
        _tdn_config(residual_logit_limit=0.5, residual_scale_init=0.1)
    ).train()
    embeddings, masks, clinical, days, prior = _tdn_inputs(batch=1)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(0.5)

    logits, residual = model(
        embeddings,
        masks,
        clinical,
        days=days,
        prior_logit=prior,
        return_residual=True,
    )
    (logits.sum() + residual.sum()).backward()

    assert model.alpha_logit.grad is not None
    assert torch.isfinite(model.alpha_logit.grad)
    assert model.alpha_logit.grad.abs() > 0
    assert model.head.bias.grad is not None
    assert torch.isfinite(model.head.bias.grad)
    assert model.head.bias.grad.abs() > 0


def test_legacy_unbounded_default_retains_alpha_fusion_and_tensor_return():
    model = TDN(_tdn_config()).eval()
    embeddings, masks, clinical, days, prior = _tdn_inputs()

    assert model.residual_logit_limit is None
    assert hasattr(model, "alpha")
    assert not hasattr(model, "alpha_logit")
    torch.testing.assert_close(model.residual_scale(), model.alpha)

    with torch.no_grad():
        raw_residual = model(
            embeddings, masks, clinical, days=days, prior_logit=None
        )
        logits = model(
            embeddings, masks, clinical, days=days, prior_logit=prior
        )
        logits_with_residual, residual = model(
            embeddings,
            masks,
            clinical,
            days=days,
            prior_logit=prior,
            return_residual=True,
        )

    assert isinstance(logits, torch.Tensor)
    torch.testing.assert_close(residual, model.alpha * raw_residual)
    torch.testing.assert_close(logits, prior + residual)
    torch.testing.assert_close(logits_with_residual, logits)


def test_return_residual_without_prior_reports_raw_model_logit():
    model = TDN(_tdn_config(use_prior=False)).eval()
    embeddings, masks, clinical, days, _ = _tdn_inputs()

    with torch.no_grad():
        logits = model(embeddings, masks, clinical, days=days)
        paired_logits, residual = model(
            embeddings,
            masks,
            clinical,
            days=days,
            return_residual=True,
        )

    torch.testing.assert_close(paired_logits, logits)
    torch.testing.assert_close(residual, logits)


def test_read_metadata_only_parses_allowed_rows_after_pid_scan(tmp_path, monkeypatch):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"pid": "DEV0", "pCR": 1, "feature": "kept"},
            {"pid": "LOCKED0", "pCR": "not-a-binary-label", "feature": "excluded"},
        ]
    ).to_csv(path, index=False)
    original_read_csv = pd.read_csv
    calls = []

    def tracking_read_csv(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        calls.append((dict(kwargs), frame.copy()))
        return frame

    monkeypatch.setattr(pd, "read_csv", tracking_read_csv)

    selected = _read_metadata(path, ["DEV0"])

    assert selected["pid"].tolist() == ["DEV0"]
    assert selected["pCR"].astype(float).tolist() == [1.0]
    assert len(calls) == 2
    first_options, first_frame = calls[0]
    second_options, second_frame = calls[1]
    assert first_options["usecols"] == ["pid"]
    assert first_frame.columns.tolist() == ["pid"]
    assert first_frame["pid"].tolist() == ["DEV0", "LOCKED0"]
    assert callable(second_options["skiprows"])
    assert second_options["skiprows"](0) is False
    assert second_options["skiprows"](1) is False
    assert second_options["skiprows"](2) is True
    assert second_frame["pid"].tolist() == ["DEV0"]
    assert "LOCKED0" not in set(second_frame["pid"])


def test_read_metadata_rejects_invalid_allowed_or_full_labels(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"pid": "DEV0", "pCR": 0},
            {"pid": "LOCKED0", "pCR": "not-a-binary-label"},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        _read_metadata(path, ["LOCKED0"])
    with pytest.raises(ValueError):
        _read_metadata(path)
    with pytest.raises(ValueError, match="metadata is missing patient MISSING"):
        _read_metadata(path, ["MISSING"])


def test_read_metadata_pid_scan_preserves_global_uniqueness_check(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"pid": "DEV0", "pCR": 0},
            {"pid": "LOCKED0", "pCR": 0},
            {"pid": "LOCKED0", "pCR": 1},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unique pid"):
        _read_metadata(path, ["DEV0"])


def test_fold_specs_preserve_joint_strata_and_balance_visit_counts(tmp_path):
    cohort = tmp_path / "cohort"
    splits = cohort / "splits"
    splits.mkdir(parents=True)
    subtypes = ("TripleNeg", "HER2pos", "HRposHER2neg")
    sources = ("local_ispy2", "qingyuan")
    pool_ids = []
    metadata_rows = []
    manifest_rows = []
    counter = 0
    for label in (0, 1):
        for subtype in subtypes:
            for source in sources:
                for member, visits in enumerate((1, 2, 3, 4, 4)):
                    patient_id = f"P{counter:03d}"
                    counter += 1
                    pool_ids.append(patient_id)
                    metadata_rows.append(
                        {
                            "pid": patient_id,
                            "pCR": label,
                            "HR_HER2_STATUS": subtype,
                            "n_registered_timepoints": visits,
                        }
                    )
                    manifest_rows.append({"patient_id": patient_id, "source": source})
    test_ids = ["TEST0", "TEST1"]
    for index, patient_id in enumerate(test_ids):
        metadata_rows.append(
            {
                "pid": patient_id,
                "pCR": index,
                "HR_HER2_STATUS": subtypes[index],
                "n_registered_timepoints": 4,
            }
        )
        manifest_rows.append({"patient_id": patient_id, "source": sources[index]})
    pd.DataFrame(metadata_rows).to_csv(cohort / "metadata_enriched.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(cohort / "patient_manifest.csv", index=False)
    reference_path = tmp_path / "locked_reference.json"
    reference_path.write_text('{"val": ["TEST0", "TEST1"]}\n')
    (splits / "train_ids.txt").write_text("\n".join(pool_ids[:50]) + "\n")
    (splits / "val_ids.txt").write_text("\n".join(pool_ids[50:]) + "\n")
    (splits / "test_ids.txt").write_text("\n".join(test_ids) + "\n")
    config = {
        "data": {
            "cohort_dir": str(cohort),
            "metadata_csv": str(cohort / "metadata_enriched.csv"),
            "locked_test_reference_json": str(reference_path),
            "locked_test_reference_split": "val",
            "expected_pool_patients": 60,
            "expected_test_patients": 2,
        },
        "anti_overfit": {"folds": 5, "fold_seed": 2026},
    }

    specs, actual_pool, actual_test, audit = _fold_specs(config)

    assert set(actual_pool) == set(pool_ids)
    assert actual_test == test_ids
    assert sorted(patient_id for spec in specs for patient_id in spec["val_ids"]) == sorted(
        pool_ids
    )
    assert all(len(spec["val_ids"]) == 12 for spec in specs)
    assert not any(set(spec["val_ids"]) & set(test_ids) for spec in specs)
    for _, stratum in audit.groupby("base_stratum"):
        assert sorted(stratum["fold"].tolist()) == [0, 1, 2, 3, 4]
    visit_table = pd.crosstab(audit["fold"], audit["n_registered_timepoints"])
    assert (visit_table.max(axis=0) - visit_table.min(axis=0)).max() <= 1

    reference_path.write_text('{"val": ["TEST0", "OTHER"]}\n')
    with pytest.raises(ValueError, match="exactly match"):
        _fold_specs(config)


def _prediction_frame(patient_ids, seed, fold, split):
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": [index % 2 for index in range(len(patient_ids))],
            "probability": np.linspace(0.2, 0.8, len(patient_ids)),
            "seed": seed,
            "fold": fold,
            "split": split,
            "temporal_depth": "T0",
            "max_tp": 1,
        }
    )


def test_training_complete_binds_fold_patient_ids_and_checkpoint_contract(tmp_path):
    train_ids = ["TRAIN0", "TRAIN1"]
    val_ids = ["VAL0", "VAL1"]
    config = {"batch_size": 2}
    depths = [{"name": "T0", "max_tp": 1}]
    checkpoint = {
        "schema": SCHEMA,
        "stage": "formal",
        "variant": "A",
        "seed": 42,
        "fold": 0,
        "effective_config": config,
        "selection": {"criterion": "mean_validation_auroc_across_four_prefixes"},
        "test_data_loaded_during_training": False,
        "clinical_prior": {
            "fitted_on": "fold_train_only",
            "feature_names": list(TABULAR_FEATURE_NAMES),
        },
    }
    torch.save(checkpoint, tmp_path / "best.pt")
    (tmp_path / "history.csv").write_text("epoch\n0\n")
    _prediction_frame(train_ids, 42, 0, "fold_train").to_csv(
        tmp_path / "train_predictions.csv", index=False
    )
    _prediction_frame(val_ids, 42, 0, "oof_val").to_csv(
        tmp_path / "val_predictions.csv", index=False
    )
    (tmp_path / "TRAINING_COMPLETE.json").write_text(
        json.dumps({"schema": SCHEMA, "complete": True})
    )

    assert _training_complete(
        tmp_path, train_ids, val_ids, "formal", "A", 42, 0, config, depths
    )
    assert not _training_complete(
        tmp_path,
        ["TRAIN0", "OTHER"],
        val_ids,
        "formal",
        "A",
        42,
        0,
        config,
        depths,
    )
    checkpoint["test_data_loaded_during_training"] = True
    torch.save(checkpoint, tmp_path / "best.pt")
    assert not _training_complete(
        tmp_path, train_ids, val_ids, "formal", "A", 42, 0, config, depths
    )


def test_evaluation_complete_binds_source_contract_and_fold_count(tmp_path):
    test_ids = ["TEST0", "TEST1"]
    depths = [{"name": "T0", "max_tp": 1}]
    source_contract = {
        "source_config": {"mode": "direct", "embeddings_dir": "embeddings"},
        "metadata_csv": "metadata.csv",
        "availability_csv": "availability.csv",
        "embedding_dim": 1152,
    }
    _prediction_frame(test_ids, 42, -1, "test_fold_ensemble").to_csv(
        tmp_path / "test_predictions.csv", index=False
    )
    (tmp_path / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "complete": True,
                "source": "real",
                "selected_variant": "A",
                "seed": 42,
                "folds_ensembled": 5,
                "source_contract": source_contract,
            }
        )
    )

    assert _evaluation_complete(
        tmp_path, test_ids, "real", 42, depths, "A", 5, source_contract
    )
    assert not _evaluation_complete(
        tmp_path, test_ids, "real", 42, depths, "A", 4, source_contract
    )
    changed_source = {**source_contract, "embedding_dim": 64}
    assert not _evaluation_complete(
        tmp_path, test_ids, "real", 42, depths, "A", 5, changed_source
    )


def test_evaluation_source_contract_tracks_configured_paths():
    config = {
        "data": {
            "metadata_csv": "metadata.csv",
            "availability_csv": "availability.csv",
            "embedding_dim": 1152,
        },
        "evaluation_sources": {
            "generated": {
                "mode": "overlay_future",
                "base_embeddings_dir": "real",
                "overlay_embeddings_dir": "generated",
                "expected_overlay_tokens": 296,
            }
        },
    }

    contract = _evaluation_source_contract(config, "generated")

    assert contract["source_config"]["overlay_embeddings_dir"] == "generated"
    config["evaluation_sources"]["generated"]["overlay_embeddings_dir"] = "changed"
    assert contract["source_config"]["overlay_embeddings_dir"] == "generated"


def _write_overlay_fixture(tmp_path):
    real = tmp_path / "real"
    overlay = tmp_path / "overlay"
    rows = []
    availability = []
    patient_ids = ["P0", "P1"]
    for index, patient_id in enumerate(patient_ids):
        (real / patient_id).mkdir(parents=True)
        (overlay / patient_id).mkdir(parents=True)
        torch.save(torch.tensor([1.0, 0.0, 0.0, 0.0]), real / patient_id / f"{patient_id}_T0.pt")
        torch.save(torch.tensor([0.0, 1.0, 0.0, 0.0]), real / patient_id / f"{patient_id}_T1.pt")
        torch.save(torch.tensor([0.0, 0.0, 1.0, 0.0]), overlay / patient_id / f"{patient_id}_T1.pt")
        rows.append(
            {
                "pid": patient_id,
                "pCR": index,
                "registered_timepoints": "T0;T1",
                "days_T0": 0,
                "days_T1": 30,
            }
        )
        for timepoint in range(4):
            availability.append(
                {
                    "patient_id": patient_id,
                    "visit": f"T{timepoint}",
                    "valid_mask": int(timepoint < 2),
                }
            )
    metadata = tmp_path / "metadata.csv"
    availability_path = tmp_path / "availability.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    pd.DataFrame(availability).to_csv(availability_path, index=False)
    config = {
        "data": {
            "metadata_csv": str(metadata),
            "availability_csv": str(availability_path),
            "embedding_dim": 4,
        },
        "evaluation_sources": {
            "generated": {
                "mode": "overlay_future",
                "base_embeddings_dir": str(real),
                "overlay_embeddings_dir": str(overlay),
                "expected_overlay_tokens": 2,
            }
        },
    }
    return config, patient_ids, availability_path


def test_generated_source_overlays_only_future_embeddings(tmp_path):
    config, patient_ids, _ = _write_overlay_fixture(tmp_path)

    result = _load_test_source(config, "generated", patient_ids)

    np.testing.assert_array_equal(result["embs"][:, 0], [[1, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_array_equal(result["embs"][:, 1], [[0, 0, 1, 0], [0, 0, 1, 0]])
    np.testing.assert_array_equal(result["masks"], [[1, 1, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(result["days"], [[0, 30, 0, 0], [0, 30, 0, 0]])


def test_generated_source_rejects_duplicate_availability_rows(tmp_path):
    config, patient_ids, availability_path = _write_overlay_fixture(tmp_path)
    availability = pd.read_csv(availability_path)
    pd.concat([availability, availability.iloc[[0]]], ignore_index=True).to_csv(
        availability_path, index=False
    )

    with pytest.raises(ValueError, match="duplicate patient visits"):
        _load_test_source(config, "generated", patient_ids)

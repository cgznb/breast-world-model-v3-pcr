import json

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_full978_independent_cv as independent_cv
from src.tdn import TDN


def test_sample_prefix_depths_are_legal_and_can_be_disabled():
    disabled = independent_cv._sample_prefix_depths(32, 4, 0.0, "cpu")
    torch.testing.assert_close(disabled, torch.full((32,), 4, dtype=torch.long))

    generator = torch.Generator(device="cpu").manual_seed(2026)
    shortened = independent_cv._sample_prefix_depths(
        256, 4, 1.0, "cpu", generator=generator
    )
    assert shortened.dtype == torch.long
    assert int(shortened.min()) >= 1
    assert int(shortened.max()) <= 3
    assert set(shortened.tolist()) == {1, 2, 3}

    baseline_only = independent_cv._sample_prefix_depths(8, 1, 1.0, "cpu")
    torch.testing.assert_close(baseline_only, torch.ones(8, dtype=torch.long))
    with pytest.raises(ValueError, match="invalid structured prefix-dropout"):
        independent_cv._sample_prefix_depths(8, 4, 1.1, "cpu")


def test_layerwise_adamw_has_three_disjoint_decay_groups():
    model = TDN(
        {
            "downstream": {
                "input_dim": 16,
                "clinical_dim": 3,
                "proj_dim": 8,
                "sig_dim": 8,
                "n_layers": 1,
                "n_heads": 2,
                "dropout": 0.0,
                "use_prior": True,
            }
        }
    )
    optimizer = independent_cv._make_optimizer(
        model,
        {
            "optimizer": "adamw_layerwise",
            "lr": 3.0e-4,
            "weight_decay": 7.0e-4,
            "projection_weight_decay": 3.0e-3,
        },
    )

    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        3.0e-3,
        7.0e-4,
        0.0,
    ]
    grouped_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == expected_ids

    projection_id = id(dict(model.named_parameters())["proj.net.0.weight"])
    assert projection_id in {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    alpha_id = id(dict(model.named_parameters())["alpha"])
    assert alpha_id in {id(parameter) for parameter in optimizer.param_groups[2]["params"]}


def test_select_variant_prefers_smaller_model_within_point005_tolerance():
    summary = pd.DataFrame(
        [
            {
                "variant": "large_best",
                "oof_auroc_mean": 0.800,
                "parameters": 200_000,
                "positive_train_oof_gap": 0.01,
            },
            {
                "variant": "small_within_tolerance",
                "oof_auroc_mean": 0.796,
                "parameters": 50_000,
                "positive_train_oof_gap": 0.03,
            },
            {
                "variant": "small_outside_tolerance",
                "oof_auroc_mean": 0.794,
                "parameters": 10_000,
                "positive_train_oof_gap": 0.0,
            },
        ]
    )
    order = [
        "large_best",
        "small_within_tolerance",
        "small_outside_tolerance",
    ]

    selected, decorated = independent_cv._select_variant(summary, order, 0.005)

    assert selected == "small_within_tolerance"
    indexed = decorated.set_index("variant")
    assert indexed.loc["small_within_tolerance", "within_oof_tolerance"]
    assert not indexed.loc["small_outside_tolerance", "within_oof_tolerance"]


def test_canonical_split_clears_every_token_at_and_after_first_gap():
    embeddings = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    masks = np.asarray([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float32)
    days = np.asarray([[0, 0, 80, 140], [0, 30, 75, 150]], dtype=np.float32)
    split = {
        "embs": embeddings,
        "masks": masks,
        "days": days,
        "clinical": np.ones((2, 3), dtype=np.float32),
        "labels": np.asarray([0, 1]),
        "pids": ["P0", "P1"],
        "dim": 3,
    }

    result = independent_cv._canonical_split(split, 3)

    np.testing.assert_array_equal(result["masks"], [[1, 0, 0, 0], [1, 1, 1, 0]])
    np.testing.assert_array_equal(result["days"], [[0, 0, 0, 0], [0, 30, 75, 0]])
    np.testing.assert_array_equal(result["embs"][0, 1:], 0)
    np.testing.assert_array_equal(result["embs"][1, 3], 0)
    np.testing.assert_array_equal(split["masks"], masks)


def test_formal_variant_map_uses_each_depths_saved_selection(tmp_path):
    config = {
        "independent_cv": {
            "temporal_depths": [
                {"name": "T0-T2", "max_tp": 3},
                {"name": "T0-T3", "max_tp": 4},
            ]
        },
        "variants": {"variant_a": {}, "variant_b": {}, "unused": {}},
    }
    selection = {
        "schema": independent_cv.SCHEMA,
        "stage": "tune",
        "test_data_used": False,
        "depths": {
            "T0-T2": {"max_tp": 3, "variant": "variant_a"},
            "T0-T3": {"max_tp": 4, "variant": "variant_b"},
        },
    }
    (tmp_path / "selected_variants.json").write_text(json.dumps(selection))

    selected = independent_cv._formal_variant_map(
        config,
        tmp_path,
        "formal",
        ["variant_a", "variant_b", "unused"],
    )

    assert selected == {"T0-T2": "variant_a", "T0-T3": "variant_b"}
    selection["test_embeddings_or_labels_loaded"] = True
    (tmp_path / "selected_variants.json").write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match="formal OOF contract"):
        independent_cv._formal_variant_map(
            config,
            tmp_path,
            "formal",
            ["variant_a", "variant_b", "unused"],
        )
    assert independent_cv._formal_variant_map(
        config, tmp_path, "tune", ["variant_a", "variant_b", "unused"]
    ) is None

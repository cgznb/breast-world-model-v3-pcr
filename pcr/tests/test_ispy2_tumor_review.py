from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import ispy2_tumor_review as review


def shrinking_masks():
    masks = np.zeros((4, 96, 256, 256), dtype=bool)
    for stage, width in enumerate((40, 35, 30, 10)):
        masks[stage, 40:50, 90:90+width, 100:100+width] = True
    return masks


def test_selection_uses_real_change_and_retains_visible_source_and_target():
    masks = shrinking_masks()
    case = review.candidate("case", masks, .001)
    assert case["selected_source"] == 2 and case["preferred_target"] == 3
    assert case["direction"] == "decrease"
    assert case["absolute_change_ml"] == pytest.approx(8)
    assert case["relative_change"] == pytest.approx(-8/9)
    z = case["central_slice"]
    assert masks[2, z].sum() >= 20 and masks[3, z].sum() >= 10
    assert z in case["slices"]
    assert all(0 <= plane < 96 for plane in case["slices"])


@pytest.mark.parametrize("fault", ["empty_target", "no_change", "disjoint_planes"])
def test_selection_rejects_uninformative_real_comparisons(fault):
    masks = shrinking_masks()
    masks[:] = masks[0].copy()
    if fault == "empty_target":
        masks[1:] = False
    elif fault == "disjoint_planes":
        masks[1:] = False
        masks[1:, 70:72, 100:110, 100:110] = True
    assert review.candidate("case", masks, .001) is None


def test_missing_visit_does_not_load_partial_reference_masks():
    loaded = SimpleNamespace(visits={"case:T0": object()})
    cache = SimpleNamespace(load=lambda _: pytest.fail("Incomplete case was loaded"))
    assert review.masks_for_patient(loaded, cache, "case") is None


@pytest.mark.parametrize("origin", [0, 190, 245])
def test_roi_at_image_edge_contains_tumor_and_remains_in_bounds(origin):
    masks = np.zeros((4, 96, 256, 256), dtype=bool)
    masks[:, 45, origin:origin+10, origin:origin+10] = True
    x, y, width, height = review.square_crop(masks, [45])
    assert 64 <= width == height <= 256
    assert 0 <= x <= 256-width and 0 <= y <= 256-height
    assert masks[:, 45, y:y+height, x:x+width].sum() == masks.sum()


def test_actual_rollout_input_uses_previous_output_and_preserves_real_ser():
    shape = (96, 256, 256)
    real = torch.empty((2, *shape), dtype=torch.float16)
    real[0].fill_(2)
    real[1].fill_(7)
    foreground = torch.zeros(shape, dtype=torch.bool)
    foreground[:, 50:180, 60:190] = True
    generated = {1: torch.full((1, *shape), 3., dtype=torch.float16),
                 2: torch.full((1, *shape), 5., dtype=torch.float16)}
    route = SimpleNamespace(source_stage=2, source_dce0="generated")
    actual = review.actual_source(route, real, foreground, generated)
    assert torch.all(actual[0, foreground] == 5)
    assert torch.all(actual[0, ~foreground] == 0)
    assert torch.equal(actual[1], real[1])
    assert torch.all(real[0] == 2)
    route.source_dce0 = "real"
    assert review.actual_source(route, real, foreground, generated) is real


def test_change_metrics_use_real_source_and_handle_undefined_regions():
    source = torch.tensor([0., 2., 100.])
    target = torch.tensor([2., 4., 0.])
    actual = torch.tensor([1., 3., 20.])
    region = np.array([True, True, False])
    values = review.local_change_metrics(target, actual, source, target, region)
    assert values == dict(roi_voxels=2, model_mae=0, actual_input_copy_mae=1,
                         real_source_copy_mae=2, predicted_change_l1=2,
                         true_change_l1=2, change_cosine=pytest.approx(1))
    unchanged = review.local_change_metrics(source, actual, source, target, region)
    assert unchanged["change_cosine"] is None
    empty = review.local_change_metrics(target, actual, source, target, ~np.ones(3, dtype=bool))
    assert empty["roi_voxels"] == 0
    assert all(value is None for key, value in empty.items() if key != "roi_voxels")

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import compare_ispy2_generators as comparison
from scripts import report_ispy2_generators as reporting


@pytest.fixture
def prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "OUT", tmp_path)
    route = SimpleNamespace(patient_id="ISPY2-999999", strategy="direct_t0",
                            source_stage=0, target_stage=1,
                            source_visit_id="ISPY2-999999:T0", target_visit_id="ISPY2-999999:T1")
    values = torch.full((1, 96, 256, 256), -0.25, dtype=torch.float16)
    comparison.save_prediction("bridge025", route, values, seed=22026, step=12000)
    return route, comparison.decoded_path("bridge025", route)


def test_readback_preserves_unclipped_standardized_image(prediction):
    route, _ = prediction
    actual = comparison.read_prediction("bridge025", route)
    assert actual.dtype == torch.float16
    assert torch.all(actual == -0.25)


@pytest.mark.parametrize("key,value", [
    ("patient_id", "ISPY2-999998"),
    ("source_visit_id", "ISPY2-999999:T1"),
    ("target_visit_id", "ISPY2-999999:T2"),
    ("target_dce0_read", True),
    ("solver_steps", 8),
    ("model", "bridge050"),
    ("candidate", 1),
    ("source_dce0", "generated"),
])
def test_mismatched_patient_or_sampling_cannot_enter_comparison(prediction, key, value):
    route, path = prediction
    payload = torch.load(path, weights_only=True)
    payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        comparison.read_prediction("bridge025", route)


def test_nonfinite_image_is_rejected_before_downstream_extraction(prediction):
    route, path = prediction
    payload = torch.load(path, weights_only=True)
    payload["prediction"][0, 0, 0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError):
        comparison.read_prediction("bridge025", route)


def test_common_target_seed_is_checked_on_resume(prediction):
    route, _ = prediction
    comparison.read_prediction("bridge025", route, expected_seed=22026)
    with pytest.raises(ValueError):
        comparison.read_prediction("bridge025", route, expected_seed=22027)


@pytest.mark.parametrize("kind", ["empty", "full", "cuboid", "holes"])
def test_optimized_ssim_centers_preserve_boundary_and_hole_exclusion(kind):
    from mewm_ispy2.ispy2_biflow_cohort_evaluation import valid_ssim_centers
    mask = np.ones((25, 31, 37), dtype=bool)
    if kind == "empty":
        mask[:] = False
    elif kind == "cuboid":
        mask[:3] = False
        mask[:, :2] = False
        mask[:, :, -4:] = False
    elif kind == "holes":
        mask[np.random.default_rng(7).random(mask.shape) < .001] = False
    assert np.array_equal(reporting.comparison_ssim_centers(mask), valid_ssim_centers(mask))


def test_adding_model_reuses_only_complete_existing_metric_groups():
    rows = []
    for model in ("biflow", "bridge025"):
        for region in sorted(reporting.REGIONS):
            rows.append(dict(model=model, region=region, voxel_count=10, ssim_center_count=2,
                             **{key: .5 for key in reporting.METRICS + reporting.CHANGE_METRICS}))
    old = pd.DataFrame(rows)
    reused = reporting.reusable_metrics(old, ["biflow", "bridge025", "symmflow"])
    pd.testing.assert_frame_equal(old, reused)
    incomplete = old.drop(old.index[-1])
    assert set(reporting.reusable_metrics(incomplete, ["biflow", "bridge025"]).model) == {"biflow"}
    duplicate = pd.concat([old, old.iloc[[-1]]], ignore_index=True)
    assert set(reporting.reusable_metrics(duplicate, ["biflow", "bridge025"]).model) == {"biflow"}

import numpy as np

from scripts.evaluate_generated_dce0_full978 import _prior_logits
from src.data import TABULAR_FEATURE_NAMES
from src.generated_dce0 import (
    NativeGeometry,
    crop_full_zyx,
    crop_slices,
    native_geometry_from_meta,
    native_valid_foreground_roi,
    paste_roi_zyx,
)


def test_negative_start_crop_inverse_roundtrip_on_retained_region():
    plan = {
        "input_shape_zyx": [4, 5, 6],
        "output_shape_zyx": [6, 5, 5],
        "crop_start_zyx": [-1, 1, 3],
    }
    full = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    roi = crop_full_zyx(full, plan)
    restored = paste_roi_zyx(roi, plan)
    full_slices, _ = crop_slices(
        plan["input_shape_zyx"], plan["crop_start_zyx"], plan["output_shape_zyx"]
    )
    assert np.array_equal(restored[full_slices], full[full_slices])
    assert np.array_equal(crop_full_zyx(restored, plan), roi)


def test_native_geometry_uses_metadata_not_identity_nifti_affine():
    meta = {
        "n_slices": 4,
        "rows": 5,
        "cols": 6,
        "pixel_spacing": [0.8, 0.7],
        "spacing_between_slices": 2.0,
        "image_orientation_patient": [-1, 0, 0, 0, -1, 0],
        "image_position_patient_first": [10, 20, 30],
    }
    geometry = native_geometry_from_meta(meta, (4, 5, 6))
    assert geometry.shape_zyx == (4, 5, 6)
    assert geometry.spacing_zyx == (2.0, 0.8, 0.7)
    assert np.allclose(np.diag(geometry.affine_ras)[:3], [0.7, 0.8, 2.0])
    assert np.allclose(geometry.affine_ras[:3, 3], [-10, -20, 30])


def test_native_valid_foreground_rebuilds_strict_roi_support():
    volume = np.ones((4, 5, 6), dtype=np.float32)
    volume[:, :, 0] = 0
    geometry = NativeGeometry(
        shape_zyx=volume.shape,
        spacing_zyx=(1.0, 1.0, 1.0),
        affine_ras=np.eye(4, dtype=np.float64),
    )
    plan = {
        "input_shape_zyx": list(volume.shape),
        "output_shape_zyx": list(volume.shape),
        "crop_start_zyx": [0, 0, 0],
    }
    valid = native_valid_foreground_roi(
        volume, plan, geometry, strict_spacing_xyz=(1.0, 1.0, 1.0)
    )
    assert valid.dtype == np.bool_
    assert np.array_equal(valid, volume != 0)


def test_checkpoint_prior_logits_are_clipped_and_float32():
    features = len(TABULAR_FEATURE_NAMES)
    state = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(TABULAR_FEATURE_NAMES),
        "coef": [[100.0] * features],
        "intercept": [-10.0],
    }
    clinical = np.stack(
        [np.zeros(features, dtype=np.float32), np.ones(features, dtype=np.float32)]
    )
    logits = _prior_logits(state, clinical)
    assert logits.dtype == np.float32
    assert logits.tolist() == [-10.0, 30.0]

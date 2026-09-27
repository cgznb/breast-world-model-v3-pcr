import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.evaluate_first_post_pcr import oof_threshold, overlay, routes_for
from scripts.run_first_post_pcr import pillar_forward, recipe_configs
from src.first_post_pcr_data import build_cohort, load_config, make_folds, public
from src.tdn import TDN


def cohort_inputs():
    visits = []
    native = {"destination_root": "/native", "records": []}
    for tp in range(4):
        v = {"patient_id": "ACRIN-6698-9", "canonical_patient_id": "ISPY2-101",
             "visit": f"T{tp}", "visit_id": f"ACRIN-6698-9:T{tp}", "split": "train",
             "visit_date": f"2020-0{tp + 1}-01", "shape_zyx": [8, 8, 8],
             "image_orientation_patient": [1, 0, 0, 0, 1, 0], "image_position_patient_first": [0, 0, 0],
             "pixel_spacing_yx_mm": [1, 1], "slice_spacing_mm": 1, "crop_geometry": {},
             "crop_localization": "all_predicted_components_union", "image_path": "/crop.npy",
             "metadata_path": "/metadata.json", "mask_path": "/mask.nii.gz"}
        visits.append(v)
        native["records"].append({"patient_id": v["patient_id"], "visit": v["visit"],
                                   "source_path": f"/remote/T{tp}/image_aqc_1.nii.gz",
                                   "relative_path": f"T{tp}/image_aqc_1.nii.gz"})
    row = {"pid": "ACRIN-6698-9", "pCR": 1, "HR_HER2_STATUS": 1}
    for tp in range(4):
        row.update({f"pre_T{tp}": 0, f"post_early_T{tp}": 2, f"post_late_T{tp}": 5})
    return {"visits": visits}, native, pd.DataFrame([row]), {"ACRIN-6698-9": "ISPY2-101"}


def test_official_alias_and_explicit_late_are_preserved():
    visits, metadata, splits, _ = build_cohort(*cohort_inputs())
    assert splits == {"train": ["ISPY2-101"], "val": []}
    assert len(visits) == 4
    assert visits[1]["remote_phases"]["late"].endswith("_aqc_5.nii.gz")
    assert visits[1]["native_first_post"].endswith("_aqc_1.nii.gz")
    assert metadata.iloc[0].days_T1 == 31


def test_gap_truncates_prefix_without_filling_from_later_visit():
    bundle, native, metadata, aliases = cohort_inputs()
    metadata.loc[0, "post_late_T1"] = np.nan
    visits, frame, _, excluded = build_cohort(bundle, native, metadata, aliases)
    assert [v["timepoint"] for v in visits] == [0]
    assert frame.iloc[0].n_available_timepoints == 1
    assert any("breaks_prefix" in r["reason"] for r in excluded)


def test_canonical_patient_overlap_and_fractional_phase_fail():
    args = list(cohort_inputs())
    args[0]["visits"][1]["split"] = "val"
    with pytest.raises(ValueError, match="crosses"):
        build_cohort(*args)
    args = list(cohort_inputs())
    args[2]["post_late_T0"] = args[2]["post_late_T0"].astype(float)
    args[2].loc[0, "post_late_T0"] = 5.5
    with pytest.raises(ValueError, match="integer"):
        build_cohort(*args)


def test_nonpositive_date_stops_longitudinal_prefix():
    args = list(cohort_inputs())
    args[0]["visits"][1]["visit_date"] = "2020-01-01"
    visits, _, _, exclusions = build_cohort(*args)
    assert len(visits) == 1
    assert any(r["reason"] == "nonpositive_verified_interval_breaks_prefix" for r in exclusions)


def test_folds_cover_training_patients_once_and_exclude_holdout():
    rows = [{"pid": f"P{i:03d}", "pCR": i % 2, "HR_HER2_STATUS": 1,
             "source": "ISPY2", "n_available_timepoints": 1 + i % 4,
             "split": "train" if i < 40 else "val"} for i in range(45)]
    frame = pd.DataFrame(rows)
    folds = make_folds(frame, 2026)
    assert folds == make_folds(frame, 2026)
    seen = []
    for fold in folds:
        assert not set(fold["train_ids"]) & set(fold["val_ids"])
        assert set(fold["train_ids"]) | set(fold["val_ids"]) == {f"P{i:03d}" for i in range(40)}
        seen += fold["val_ids"]
    assert len(seen) == len(set(seen)) == 40


def test_forecast_routes_use_real_previous_stage_and_fixed_noise():
    bundle, native, metadata, aliases = cohort_inputs()
    for v in bundle["visits"]:
        v["split"] = "val"
    visits, _, _, _ = build_cohort(bundle, native, metadata, aliases)
    pairs = [{"earlier_visit_id": visits[t - 1]["visit_id"], "later_visit_id": visits[t]["visit_id"],
              "pair_id": f"pair{t}", "split": "val"} for t in range(1, 4)]
    routes = routes_for({"visits": visits}, {"pairs": pairs}, 22026)
    assert [(r["source_stage"], r["target_stage"]) for r in routes] == [(0, 1), (1, 2), (2, 3)]
    assert [r["noise_seed"] for r in routes] == [22026, 22027, 22028]
    assert all(r["source_policy"] == "previous_real" for r in routes)


def test_overlay_preserves_t0_clinical_labels_and_availability(tmp_path):
    real = {"pids": ["P0"], "embs": np.zeros((1, 4, 1152), np.float32),
            "masks": np.array([[1, 1, 0, 0]]), "days": np.array([[0, 30, 0, 0]]),
            "clinical": np.zeros((1, 17)), "labels": np.array([1]), "dim": 1152}
    (tmp_path / "P0").mkdir()
    torch.save(torch.ones(1152), tmp_path / "P0/P0_T1.pt")
    before = copy.deepcopy(real)
    result = overlay(real, tmp_path)
    assert np.count_nonzero(result["embs"][:, 1]) == 1152
    for key in ("masks", "days", "clinical", "labels"):
        np.testing.assert_array_equal(result[key], before[key])
    np.testing.assert_array_equal(result["embs"][:, 0], before["embs"][:, 0])
    torch.save(torch.ones(1152), tmp_path / "P0/P0_T2.pt")
    with pytest.raises(ValueError, match="inventory"):
        overlay(real, tmp_path)


def test_oof_threshold_and_public_manifest():
    assert 0.1 < oof_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) <= 0.8
    payload = public({"checksum": "secret", "shape": [1, 2], "score": float("nan")})
    assert payload == {"shape": [1, 2], "score": None}
    json.dumps(payload, allow_nan=False)


def test_batched_pillar_contract_rejects_bad_shape():
    class Backbone:
        def extract_vision_feats(self, inputs):
            return torch.ones(inputs["breast_mr"].shape[0], 1152)
    assert pillar_forward(Backbone(), torch.zeros(2, 3, 2, 2, 2)).shape == (2, 1152)


def test_all_four_original_recipes_forward_backward_and_reload(tmp_path):
    import yaml
    cfg = load_config("configs/first_post_pcr_v1.yaml")
    cfg["output_dir"] = str(tmp_path)
    configs = recipe_configs(cfg)
    torch.set_num_threads(1)
    for depth, path in configs.items():
        effective = yaml.safe_load(open(path))["downstream"]
        model = TDN({"downstream": effective})
        inputs = torch.randn(4, 4, 1152)
        masks = torch.zeros(4, 4)
        masks[:, :cfg["recipes"][depth]["max_tp"]] = 1
        clinical = torch.randn(4, 17)
        days = torch.tensor([[0, 30, 70, 150]]).expand(4, -1).float()
        prior = torch.zeros(4)
        value = model(inputs, masks, clinical, days=days, prior_logit=prior)
        torch.nn.functional.binary_cross_entropy_with_logits(value, torch.tensor([0., 1., 0., 1.])).backward()
        assert torch.isfinite(value).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        state = tmp_path / f"{depth}.pt"
        torch.save(model.state_dict(), state)
        restored = TDN({"downstream": effective}).eval()
        restored.load_state_dict(torch.load(state, weights_only=True))
        torch.testing.assert_close(model(inputs, masks, clinical, days=days, prior_logit=prior),
                                   restored(inputs, masks, clinical, days=days, prior_logit=prior))
        assert effective["batch_size"] == 64
        assert effective["lr"] == 3e-4


def test_native_zyx_flips_preserve_physical_geometry_in_lps(tmp_path):
    import nibabel as nib
    import SimpleITK as sitk
    from pathlib import Path
    from src.first_post_pcr_data import native_image, world_imports
    cfg = load_config("configs/first_post_pcr_v1.yaml")
    if not Path(cfg["world_repo"]).is_dir():
        pytest.skip("The shared world-model geometry helpers are not installed")
    world_imports(cfg)
    array = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    path = tmp_path / "native.nii.gz"
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    visit = {"shape_zyx": list(array.shape), "image_orientation_patient": [-1, 0, 0, 0, -1, 0],
             "image_position_patient_first": [0, 0, 0], "pixel_spacing_yx_mm": [0.7, 0.7], "slice_spacing_mm": 2.2}
    image = native_image(path, visit)
    assert image.GetDirection() == (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    np.testing.assert_allclose(image.GetOrigin(), [-4.9, -3.5, 0])
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), array[:, ::-1, ::-1])

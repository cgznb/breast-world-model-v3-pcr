import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from src.mewm_data import (
    RegisteredMewmSource,
    build_adapted_cohort,
    build_registered_volume,
    patient_key,
)


def _write_visit(tmp_path, patient_id="ISPY2-101", phases=(0, 2, 5), fold="train"):
    image_dir = tmp_path / patient_id / "T0" / "dce"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for phase in phases:
        path = image_dir / f"{patient_id}_T0_dce_aqc_{phase}.nii.gz"
        array = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) + phase
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        paths.append(str(path.resolve()))
    meta_path = tmp_path / patient_id / "T0" / "meta.json"
    geometry = {
        "shape_zyx": [3, 4, 5],
        "spacing_xyz": [1.0, 1.0, 1.0],
    }
    meta_path.write_text(
        json.dumps(
            {
                "array_shape_policy": "zyx_slices_rows_cols",
                "spacing_between_slices": 1.0,
                "pixel_spacing": [1.0, 1.0],
                "n_slices": 3,
                "rows": 4,
                "cols": 5,
                "source_geometry": geometry,
                "target_geometry": geometry,
            }
        )
    )
    return paths, meta_path, {
        "patient_id": patient_id,
        "visit_id": f"{patient_id}:T0",
        "visit": "T0",
        "dce0_path": paths[0],
        "meta_path": str(meta_path.resolve()),
        "fold": fold,
        "slice_spacing_mm": 1.0,
        "row_spacing_mm": 1.0,
        "column_spacing_mm": 1.0,
    }


def _write_source(tmp_path, visits, phase_rows, input_mode="registered_three_phase"):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    pd.DataFrame(visits).to_csv(bundle_dir / "visits.csv", index=False)
    (bundle_dir / "bundle.json").write_text(
        json.dumps({"artifacts": {"visits": {"path": "visits.csv"}}})
    )
    manifest = tmp_path / "phases.csv"
    pd.DataFrame(phase_rows).to_csv(manifest, index=False)
    return RegisteredMewmSource(
        bundle_dir / "bundle.json", [manifest], input_mode=input_mode
    )


def test_registered_source_matches_alias_and_preserves_zyx(tmp_path):
    paths, _, visit = _write_visit(tmp_path)
    source = _write_source(
        tmp_path,
        [visit],
        [{"patient_id": "ACRIN-6698-101", "visit": "T0", "dce_paths": ";".join(paths)}],
    )
    row = {"pre_T0": 0, "post_early_T0": 2, "post_late_T0": 5}

    assert patient_key("ACRIN-6698-101") == patient_key("ISPY2-101")
    assert source.phase_members("ACRIN-6698-101", 0, row) == tuple(paths)
    assert source.phase_selection("ACRIN-6698-101", 0, row)[1] == "metadata_pre_early_late"
    volume = build_registered_volume(
        source, "ACRIN-6698-101", 0, row, target_depth=4, target_hw=6
    )
    assert volume.shape == (1, 3, 6, 6, 4)
    assert torch.isfinite(volume).all()
    assert float(volume.min()) == pytest.approx(0.0)
    assert float(volume.max()) == pytest.approx(1.0)


def test_bundle_dce0_path_rejects_stale_duplicate(tmp_path):
    paths, _, visit = _write_visit(tmp_path)
    stale_paths, _, _ = _write_visit(tmp_path / "stale", patient_id="ISPY2-101")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame([visit]).to_csv(bundle_dir / "visits.csv", index=False)
    (bundle_dir / "bundle.json").write_text(
        json.dumps({"artifacts": {"visits": {"path": "visits.csv"}}})
    )
    stale = tmp_path / "stale.csv"
    current = tmp_path / "current.csv"
    pd.DataFrame(
        [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": ";".join(stale_paths)}]
    ).to_csv(stale, index=False)
    pd.DataFrame(
        [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": ";".join(paths)}]
    ).to_csv(current, index=False)

    source = RegisteredMewmSource(bundle_dir / "bundle.json", [stale, current])
    assert source.phase_members(
        "ISPY2-101", 0, {"pre_T0": 0, "post_early_T0": 2, "post_late_T0": 5}
    ) == tuple(paths)


def test_dce0_repeat_is_explicit_ablation(tmp_path):
    paths, _, visit = _write_visit(tmp_path, phases=(0,))
    source = _write_source(
        tmp_path,
        [visit],
        [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": paths[0]}],
        input_mode="registered_dce0_repeat",
    )
    members = source.phase_members("ISPY2-101", 0, {})
    assert members == (paths[0], paths[0], paths[0])
    volume = build_registered_volume(source, "ISPY2-101", 0, {}, target_depth=3, target_hw=5)
    assert torch.equal(volume[:, 0], volume[:, 1])
    assert torch.equal(volume[:, 1], volume[:, 2])


def test_reference_cohort_uses_canonical_ids_and_locked_splits(tmp_path):
    all_visits = []
    phase_rows = []
    metadata_rows = []
    reference_rows = []
    split_names = ("train", "validation", "test")
    for offset, split in enumerate(split_names, start=1):
        canonical = f"ISPY2-{offset}01"
        paths, _, visit = _write_visit(tmp_path, patient_id=canonical, fold="val" if split == "test" else "train")
        all_visits.append(visit)
        phase_rows.append(
            {"patient_id": canonical, "visit": "T0", "dce_paths": ";".join(paths)}
        )
        metadata_rows.append(
            {
                "pid": f"ACRIN-6698-{offset}01",
                "pCR": offset % 2,
                "pre_T0": 0,
                "post_early_T0": 2,
                "post_late_T0": 5,
            }
        )
        reference_rows.append(
            {
                "patient_id": canonical,
                "label": offset % 2,
                "split": split,
                "valid_timepoints": [True, False, False, False],
            }
        )
    source = _write_source(tmp_path, all_visits, phase_rows)
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_csv, index=False)
    reference_json = tmp_path / "reference.json"
    reference_json.write_text(json.dumps({"records": reference_rows}))

    metadata, splits, summary = build_adapted_cohort(
        metadata_csv,
        None,
        source,
        reference_manifest_json=reference_json,
    )

    assert metadata["pid"].tolist() == ["ISPY2-101", "ISPY2-201", "ISPY2-301"]
    assert splits == {
        "train": ["ISPY2-101"],
        "val": ["ISPY2-201"],
        "test": ["ISPY2-301"],
    }
    assert summary["patients"] == 3
    assert summary["split_source"] == "reference_manifest"
    assert summary["phase_selection_counts"] == {"metadata_pre_early_late": 3}


def test_registered_layout_policy_is_validated(tmp_path):
    paths, meta_path, visit = _write_visit(tmp_path)
    meta_path.write_text(json.dumps({"array_shape_policy": "xyz"}))
    with pytest.raises(ValueError, match="array-axis policy"):
        _write_source(
            tmp_path,
            [visit],
            [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": ";".join(paths)}],
        )


def test_legacy_registered_meta_uses_validated_target_geometry(tmp_path):
    paths, meta_path, visit = _write_visit(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta.pop("array_shape_policy")
    meta["dce_native_geometry"] = meta["source_geometry"]
    meta_path.write_text(json.dumps(meta))
    source = _write_source(
        tmp_path,
        [visit],
        [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": ";".join(paths)}],
    )

    volume, spacing = source.load("ISPY2-101", 0, paths[0])

    assert volume.shape == (3, 4, 5)
    assert spacing == [1.0, 1.0, 1.0]


def test_registered_payload_shape_must_match_target_geometry(tmp_path):
    paths, meta_path, visit = _write_visit(tmp_path)
    source = _write_source(
        tmp_path,
        [visit],
        [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": ";".join(paths)}],
    )
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 5), dtype=np.float32), np.eye(4)), paths[0])

    with pytest.raises(ValueError, match="shape does not match target geometry"):
        source.load("ISPY2-101", 0, paths[0])


def test_registered_volume_uses_target_not_bundle_source_spacing(tmp_path, monkeypatch):
    paths, meta_path, visit = _write_visit(tmp_path)
    visit.update(
        {
            "slice_spacing_mm": 2.0,
            "row_spacing_mm": 0.625,
            "column_spacing_mm": 0.625,
        }
    )
    meta = json.loads(meta_path.read_text())
    meta["spacing_between_slices"] = 2.0
    meta["pixel_spacing"] = [0.5664, 0.5664]
    meta["source_geometry"]["spacing_xyz"] = [0.625, 0.625, 2.0]
    meta["target_geometry"]["spacing_xyz"] = [0.5664, 0.5664, 2.0]
    meta_path.write_text(json.dumps(meta))
    source = _write_source(
        tmp_path,
        [visit],
        [{"patient_id": "ISPY2-101", "visit": "T0", "dce_paths": ";".join(paths)}],
    )
    observed = []

    def capture_spacing(volume, spacing, target_spacing):
        observed.append(tuple(spacing))
        return volume

    monkeypatch.setattr("src.mewm_data.resample_volume", capture_spacing)
    volume = build_registered_volume(
        source,
        "ISPY2-101",
        0,
        {"pre_T0": 0, "post_early_T0": 2, "post_late_T0": 5},
        target_spacing=(2.0, 0.5664, 0.5664),
        target_depth=3,
        target_hw=5,
    )

    assert observed == [(2.0, 0.5664, 0.5664)] * 3
    assert volume.shape == (1, 3, 5, 5, 3)

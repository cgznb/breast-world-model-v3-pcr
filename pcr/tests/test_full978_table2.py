import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.aggregate_full978_table2 import aggregate_table2
from scripts.run_experiments import save_cell_artifacts
from scripts.run_full978_table2 import _add_depth_column
from scripts.prepare_full978_table2 import (
    _manifest_index,
    build_locked_splits,
    select_strict_phase_paths,
)
from src.data import EmbStore, load_split
from src.mewm_data import RegisteredManifestSource, patient_key
from src.tdn import TDN
from src.temporal import truncate_temporal_splits


def _registered_manifest(tmp_path, patient_id="ISPY2-101", phases=(0, 2, 5)):
    visit_dir = tmp_path / patient_id / "T0"
    dce_dir = visit_dir / "dce"
    dce_dir.mkdir(parents=True)
    paths = []
    for phase in phases:
        path = dce_dir / f"{patient_id}_T0_dce_aqc_{phase}.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4)), path)
        paths.append(str(path.resolve()))
    geometry = {"spacing_xyz": [1.0, 1.0, 1.0], "shape_zyx": [3, 4, 5]}
    meta = visit_dir / "meta.json"
    meta.write_text(
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
    manifest = tmp_path / "registered.csv"
    pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "visit": "T0",
                "dce_paths": ";".join(paths),
                "meta_path": str(meta.resolve()),
            }
        ]
    ).to_csv(manifest, index=False)
    return manifest, paths


def test_manifest_source_normalizes_prefix_and_selects_only_declared_phases(tmp_path):
    manifest, paths = _registered_manifest(tmp_path)
    source = RegisteredManifestSource(manifest)
    metadata = {"pre_T0": 0, "post_early_T0": 2, "post_late_T0": 5}

    assert patient_key("ACRIN-6698-101") == patient_key("ISPY2-101")
    assert source.source_patient_id("ACRIN-6698-101") == "ISPY2-101"
    assert source.phase_members("ACRIN-6698-101", 0, metadata) == tuple(paths)
    assert source.phase_selection("ISPY2-101", 0, metadata)[1] == "metadata_pre_early_late"


def test_manifest_source_rejects_duplicate_normalized_visit(tmp_path):
    manifest, _ = _registered_manifest(tmp_path)
    frame = pd.read_csv(manifest)
    duplicate = frame.copy()
    duplicate["patient_id"] = "ACRIN-6698-101"
    pd.concat([frame, duplicate], ignore_index=True).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="duplicate registered manifest visit"):
        RegisteredManifestSource(manifest)
    with pytest.raises(ValueError, match="duplicate local manifest visit"):
        _manifest_index(manifest, "local")


def test_strict_phase_selection_has_no_first_three_fallback(tmp_path):
    manifest, _ = _registered_manifest(tmp_path, phases=(0, 1, 2))
    source = RegisteredManifestSource(manifest)
    metadata = {"pre_T0": 0, "post_early_T0": 2, "post_late_T0": 5}

    assert source.phase_selection("ISPY2-101", 0, metadata) == (None, None)
    with pytest.raises(ValueError, match="metadata-selected DCE phases are absent"):
        select_strict_phase_paths(
            {0: "zero", 1: "one", 2: "two"}, metadata, 0
        )


def test_locked_split_is_exact_and_stratified():
    rows = []
    for index in range(100):
        rows.append({"pid": f"ISPY2-{index:06d}", "pCR": int(index % 4 == 0)})
    metadata = pd.DataFrame(rows)
    locked = metadata["pid"].iloc[:10].tolist()

    splits = build_locked_splits(metadata, locked, seed=2026, val_size=10)

    assert set(splits["test"]) == set(locked)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))


def test_missing_visits_are_zero_embedding_mask_and_elapsed_days(tmp_path):
    embedding_root = tmp_path / "embeddings"
    patient_dir = embedding_root / "ISPY2-101"
    patient_dir.mkdir(parents=True)
    torch.save(torch.ones(8), patient_dir / "ISPY2-101_T0.pt")
    row = pd.Series(
        {
            "pCR": 1,
            "registered_timepoints": "T0",
            "days_T0": 0,
        }
    )

    loaded = load_split(
        EmbStore(embedding_root), ["ISPY2-101"], {"ISPY2-101": row}, 17
    )

    assert loaded["masks"].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert np.count_nonzero(loaded["embs"][:, 1:]) == 0
    assert np.count_nonzero(loaded["days"][:, 1:]) == 0


def test_tdn_accepts_an_all_zero_imaging_mask_when_clinical_is_present():
    config = {
        "downstream": {
            "input_dim": 8,
            "proj_dim": 4,
            "clinical_dim": 2,
            "sig_dim": 4,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0,
            "use_prior": True,
        }
    }
    model = TDN(config).eval()
    with torch.no_grad():
        logits = model(
            torch.zeros(2, 4, 8),
            torch.zeros(2, 4),
            torch.ones(2, 2),
            days=torch.zeros(2, 4),
            prior_logit=torch.zeros(2),
        )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_temporal_depth_masks_and_zeroes_later_tokens(depth):
    source = {
        "train": {
            "embs": np.ones((2, 4, 3), dtype=np.float32),
            "masks": np.ones((2, 4), dtype=np.float32),
            "days": np.tile(np.arange(4, dtype=np.float32), (2, 1)),
            "labels": np.asarray([0, 1], dtype=np.float32),
        }
    }

    result = truncate_temporal_splits(source, depth)["train"]

    assert np.all(result["embs"][:, :depth] == 1)
    assert np.all(result["masks"][:, :depth] == 1)
    assert np.count_nonzero(result["embs"][:, depth:]) == 0
    assert np.count_nonzero(result["masks"][:, depth:]) == 0
    assert np.count_nonzero(result["days"][:, depth:]) == 0
    assert np.all(source["train"]["masks"] == 1)


def test_manifest_source_accepts_explicit_legacy_top_level_geometry(tmp_path):
    manifest, _ = _registered_manifest(tmp_path)
    row = pd.read_csv(manifest).iloc[0]
    meta_path = row["meta_path"]
    meta = json.loads(open(meta_path).read())
    meta.pop("source_geometry")
    meta.pop("target_geometry")
    with open(meta_path, "w") as handle:
        json.dump(meta, handle)

    source = RegisteredManifestSource(manifest)

    assert source.visits[("101", 0)].shape_zyx == (3, 4, 5)


def test_table2_aggregation_requires_explicit_depth_and_exact_ids(tmp_path):
    cohort = tmp_path / "cohort" / "splits"
    cohort.mkdir(parents=True)
    val_ids = ["ISPY2-101", "ISPY2-102"]
    test_ids = ["ISPY2-201", "ISPY2-202"]
    pd.DataFrame(
        [
            {"pid": pid, "pCR": index % 2}
            for index, pid in enumerate(val_ids + test_ids)
        ]
    ).to_csv(tmp_path / "cohort" / "metadata_enriched.csv", index=False)
    (cohort / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    (cohort / "test_ids.txt").write_text("\n".join(test_ids) + "\n")
    output = tmp_path / "results"
    for seed in (42, 43):
        predictions = {
            "val_y": np.asarray([0, 1]),
            "val_prob": np.asarray([0.2, 0.8]),
            "test_y": np.asarray([0, 1]),
            "test_prob": np.asarray([0.1, 0.9]),
            "_model_state": {},
            "_selection": {
                "criterion": "validation_auroc",
                "best_epoch_zero_based": 0,
                "best_validation_auroc": 1.0,
                "epochs_trained": 1,
            },
            "_effective_config": {"max_tp": 1},
            "_clinical_prior": {"fitted_on": "train"},
            "_run_context": {"test_role": "final_evaluation_only"},
        }
        depth_dir = output / "t0"
        save_cell_artifacts(
            depth_dir,
            "global | +tab+temporal",
            predictions,
            {"val": val_ids, "test": test_ids},
            seed,
        )
        cell = depth_dir / f"seed_{seed}" / "global_tab_temporal"
        for split in ("val", "test"):
            _add_depth_column(cell / f"{split}_predictions.csv", "T0")
        summary_path = cell / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["temporal_depth"] = "T0"
        summary["max_tp"] = 1
        summary_path.write_text(json.dumps(summary))

    table = aggregate_table2(
        output,
        tmp_path / "cohort",
        [42, 43],
        [{"name": "T0", "max_tp": 1}],
    )

    assert table.loc[0, "n_seeds"] == 2

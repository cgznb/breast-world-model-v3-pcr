import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.aggregate import main as aggregate_main
from scripts.extract_pillar_mewm import (
    _prefetched_volumes,
    ensure_extraction_contract,
    validate_embedding,
)
from scripts.run_mewm_reproduction import main as reproduction_main
from scripts.run_experiments import (
    _validate_mewm_embedding_store,
    fit_prior,
    main,
    save_cell_artifacts,
    train_eval,
)
from src.data import EmbStore, load_split


def _split(rng, size, dim, clinical_dim):
    labels = np.asarray([index % 2 for index in range(size)], dtype=np.float32)
    embeddings = rng.normal(size=(size, 4, dim)).astype(np.float32)
    embeddings[:, :, 0] += labels[:, None]
    return {
        "pids": [f"P{index}" for index in range(size)],
        "embs": embeddings,
        "dim": dim,
        "masks": np.ones((size, 4), dtype=np.float32),
        "clinical": rng.normal(size=(size, clinical_dim)).astype(np.float32),
        "labels": labels,
        "days": np.tile(np.asarray([0, 30, 90, 150], dtype=np.float32), (size, 1)),
    }


def test_tdn_smoke_and_auditable_artifacts(tmp_path):
    rng = np.random.default_rng(7)
    data = {
        "train": _split(rng, 12, 8, 2),
        "val": _split(rng, 6, 8, 2),
        "test": _split(rng, 6, 8, 2),
    }
    config = {
        "model_type": "tdn",
        "input_dim": 8,
        "clinical_dim": 2,
        "proj_dim": 4,
        "sig_dim": 4,
        "n_layers": 1,
        "n_heads": 2,
        "use_prior": False,
        "epochs": 2,
        "patience": 2,
        "train_loss_floor": 0,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0,
        "dropout": 0,
        "grad_clip": 1,
    }
    predictions = train_eval(config, data, None, seed=11, device=torch.device("cpu"))
    assert predictions["test_prob"].shape == (6,)
    assert np.isfinite(predictions["test_prob"]).all()
    assert predictions["_selection"]["epochs_trained"] == 2

    save_cell_artifacts(
        tmp_path,
        "global | +tab+temporal",
        predictions,
        {"val": data["val"]["pids"], "test": data["test"]["pids"]},
        seed=11,
    )
    cell_dir = tmp_path / "seed_11" / "global_tab_temporal"
    checkpoint = torch.load(cell_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["selection"]["criterion"] == "validation_auroc"
    frame = pd.read_csv(cell_dir / "test_predictions.csv")
    assert frame.columns.tolist() == ["patient_id", "label", "probability", "seed", "split"]
    assert frame["patient_id"].tolist() == data["test"]["pids"]
    summary = json.loads((cell_dir / "summary.json").read_text())
    assert summary["threshold"] == 0.5
    assert set(summary["metrics"]) == {"val", "test"}


def test_registered_metadata_rejects_partial_or_mixed_embeddings(tmp_path):
    root = tmp_path / "embeddings"
    patient_dir = root / "P1"
    patient_dir.mkdir(parents=True)
    torch.save(torch.ones(8), patient_dir / "P1_T0.pt")
    row = pd.Series({"pCR": 1, "registered_timepoints": "T0;T1"})

    with pytest.raises(FileNotFoundError, match="Missing registered embeddings"):
        load_split(EmbStore(root), ["P1"], {"P1": row}, 17)

    torch.save(torch.ones(8), patient_dir / "P1_T1.pt")
    row["registered_timepoints"] = "T0"
    with pytest.raises(ValueError, match="Unexpected registered embeddings"):
        load_split(EmbStore(root), ["P1"], {"P1": row}, 17)


def test_registered_metadata_rejects_nonfinite_embedding(tmp_path):
    root = tmp_path / "embeddings"
    patient_dir = root / "P1"
    patient_dir.mkdir(parents=True)
    torch.save(torch.tensor([1.0, float("nan")]), patient_dir / "P1_T0.pt")
    row = pd.Series({"pCR": 0, "registered_timepoints": "T0"})

    with pytest.raises(ValueError, match="non-finite"):
        load_split(EmbStore(root), ["P1"], {"P1": row}, 17)


def test_extraction_resume_validates_embedding_and_contract(tmp_path):
    output = tmp_path / "embeddings"
    contract = {"schema_version": 1, "input_mode": "registered_three_phase"}
    path = ensure_extraction_contract(output, contract)
    assert json.loads(path.read_text()) == contract
    assert ensure_extraction_contract(output, contract) == path
    with pytest.raises(ValueError, match="different extraction contract"):
        ensure_extraction_contract(output, {**contract, "input_mode": "registered_dce0_repeat"})

    embedding_path = output / "P1" / "P1_T0.pt"
    embedding_path.parent.mkdir()
    torch.save(torch.ones(1152), embedding_path)
    assert validate_embedding(embedding_path).shape == (1152,)
    torch.save(torch.ones(8), embedding_path)
    with pytest.raises(ValueError, match="violates the Pillar contract"):
        validate_embedding(embedding_path)


@pytest.mark.parametrize("workers", [1, 2])
def test_volume_prefetch_preserves_task_order(workers):
    tasks = list(range(7))

    observed = list(_prefetched_volumes(tasks, lambda value: value * 3, workers))

    assert observed == [(value, value * 3) for value in tasks]


def test_mewm_validation_rejects_in_place_phase_manifest_change(tmp_path):
    patient_id = "ISPY2-101"
    source_dir = tmp_path / "registered" / patient_id / "T0"
    source_dir.mkdir(parents=True)
    phase_paths = []
    for phase in range(3):
        path = source_dir / f"{patient_id}_T0_dce_aqc_{phase}.nii.gz"
        path.touch()
        phase_paths.append(str(path.resolve()))
    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    replacement = replacement_dir / f"{patient_id}_T0_dce_aqc_2.nii.gz"
    replacement.touch()

    geometry = {"spacing_xyz": [1.0, 1.0, 1.0], "shape_zyx": [3, 4, 5]}
    meta_path = source_dir / "meta.json"
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
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "visit_id": f"{patient_id}:T0",
                "visit": "T0",
                "dce0_path": phase_paths[0],
                "meta_path": str(meta_path.resolve()),
                "fold": "train",
                "slice_spacing_mm": 1.0,
                "row_spacing_mm": 1.0,
                "column_spacing_mm": 1.0,
            }
        ]
    ).to_csv(bundle_dir / "visits.csv", index=False)
    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(json.dumps({"artifacts": {"visits": {"path": "visits.csv"}}}))
    manifest_path = tmp_path / "phases.csv"

    def write_manifest(paths):
        pd.DataFrame(
            [{"patient_id": patient_id, "visit": "T0", "dce_paths": ";".join(paths)}]
        ).to_csv(manifest_path, index=False)

    write_manifest(phase_paths)
    cohort_dir = tmp_path / "cohort"
    split_dir = cohort_dir / "splits"
    split_dir.mkdir(parents=True)
    metadata_path = cohort_dir / "metadata_enriched.csv"
    pd.DataFrame(
        [
            {
                "pid": patient_id,
                "pCR": 1,
                "registered_timepoints": "T0",
                "pre_T0": 0,
                "post_early_T0": 1,
                "post_late_T0": 2,
            }
        ]
    ).to_csv(metadata_path, index=False)
    (split_dir / "train_ids.txt").write_text(f"{patient_id}\n")
    (split_dir / "val_ids.txt").write_text("")
    (split_dir / "test_ids.txt").write_text("")

    adapter = {
        "bundle_json": str(bundle_path),
        "phase_manifest_csvs": [str(manifest_path)],
        "input_mode": "registered_three_phase",
        "embedding_dim": 1152,
        "model_revision": "main",
        "target_shape_hwd": [384, 384, 192],
        "target_spacing_zyx": [1.0, 1.0, 1.0],
    }
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    contract = {
        "schema_version": 1,
        "model": "YalaLab/Pillar0-BreastMRI",
        "model_revision": "main",
        "embedding_dim": 1152,
        "input_mode": "registered_three_phase",
        "cohort_dir": str(cohort_dir.resolve()),
        "metadata_csv": str(metadata_path.resolve()),
        "splits": {"train": [patient_id], "val": [], "test": []},
        "labels": {patient_id: 1},
        "bundle_json": str(bundle_path.resolve()),
        "phase_manifest_csvs": [str(manifest_path.resolve())],
        "target_spacing_zyx": [1.0, 1.0, 1.0],
        "target_shape_hwd": [384, 384, 192],
        "patients": 1,
        "expected_visits": 1,
        "phase_selection_counts": {"metadata_pre_early_late": 1},
        "visits": [
            {
                "patient_id": patient_id,
                "timepoint": 0,
                "phase_paths": phase_paths,
                "phase_selection": "metadata_pre_early_late",
                "registered_spacing_zyx": [1.0, 1.0, 1.0],
                "registered_shape_zyx": [3, 4, 5],
            }
        ],
    }
    (embedding_dir / "extraction_contract.json").write_text(json.dumps(contract))
    (embedding_dir / "extraction_summary.json").write_text(
        json.dumps(
            {
                "input_mode": "registered_three_phase",
                "model": "YalaLab/Pillar0-BreastMRI",
                "contract": "extraction_contract.json",
                "expected_visits": 1,
                "processed_visits": 1,
                "validated_existing_visits": 0,
                "embedding_dim": 1152,
                "complete": True,
            }
        )
    )
    cfg = {"mewm_adapter": adapter}

    _validate_mewm_embedding_store(cfg, embedding_dir, metadata_path, split_dir)

    invalid_metadata = pd.read_csv(metadata_path)
    invalid_metadata["pCR"] = invalid_metadata["pCR"].astype(float)
    invalid_metadata.loc[0, "pCR"] = 0.5
    invalid_metadata.to_csv(metadata_path, index=False)
    with pytest.raises(SystemExit, match="invalid pCR label"):
        _validate_mewm_embedding_store(cfg, embedding_dir, metadata_path, split_dir)
    invalid_metadata.loc[0, "pCR"] = 1
    invalid_metadata.to_csv(metadata_path, index=False)

    write_manifest([phase_paths[0], phase_paths[1], str(replacement.resolve())])
    with pytest.raises(SystemExit, match="does not match the experiment config"):
        _validate_mewm_embedding_store(cfg, embedding_dir, metadata_path, split_dir)


def test_clinical_prior_state_reconstructs_split_logits():
    rng = np.random.default_rng(17)
    data = {
        "train": _split(rng, 12, 8, 2),
        "val": _split(rng, 6, 8, 2),
        "test": _split(rng, 6, 8, 2),
    }

    logits, state = fit_prior(data, seed=5)

    coefficient = np.asarray(state["coef"])[0]
    intercept = float(state["intercept"][0])
    for split in ("train", "val", "test"):
        reconstructed = data[split]["clinical"] @ coefficient + intercept
        assert np.allclose(reconstructed, logits[split], atol=1e-6)


def test_experiment_cli_writes_resolved_context_and_prior(tmp_path, monkeypatch):
    embedding_dir = tmp_path / "embeddings"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_ids = {
        "train": [f"TR{index}" for index in range(6)],
        "val": ["VA0", "VA1"],
        "test": ["TE0", "TE1"],
    }
    rows = []
    rng = np.random.default_rng(23)
    for split, pids in split_ids.items():
        (split_dir / f"{split}_ids.txt").write_text("\n".join(pids) + "\n")
        for index, pid in enumerate(pids):
            patient_dir = embedding_dir / pid
            patient_dir.mkdir(parents=True)
            torch.save(torch.tensor(rng.normal(size=8), dtype=torch.float32), patient_dir / f"{pid}_T0.pt")
            rows.append({"pid": pid, "pCR": index % 2, "HR": index % 2})
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    config = {
        "downstream": {
            "proj_dim": 4,
            "hidden_dim": 4,
            "sig_dim": 4,
            "n_layers": 1,
            "n_heads": 2,
            "use_prior": True,
            "epochs": 1,
            "patience": 1,
            "train_loss_floor": 0,
            "batch_size": 3,
            "lr": 1e-3,
            "weight_decay": 0,
            "dropout": 0,
            "grad_clip": 1,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments.py",
            "--global-dir",
            str(embedding_dir),
            "--metadata-csv",
            str(metadata_path),
            "--splits-dir",
            str(split_dir),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
            "--device",
            "cpu",
        ],
    )

    main()

    resolved = yaml.safe_load((output_dir / "resolved_config_seed7.yaml").read_text())
    assert resolved["runtime"]["resolved_device"] == "cpu"
    checkpoint = torch.load(
        output_dir / "seed_7" / "global_tab_temporal" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["clinical_prior"]["n_features_in"] == 17
    assert checkpoint["run_context"]["metadata_csv"] == str(metadata_path.resolve())
    assert (output_dir / "preds_seed7.npz").is_file()


def test_declared_seed_aggregation_ignores_stale_artifacts(tmp_path, monkeypatch, capsys):
    output = tmp_path / "results"
    output.mkdir()
    np.savez(
        output / "preds_seed42.npz",
        **{
            "current::test_y": np.asarray([0, 1]),
            "current::test_prob": np.asarray([0.1, 0.9]),
        },
    )
    np.savez(
        output / "preds_seed99.npz",
        **{
            "stale::test_y": np.asarray([0, 1]),
            "stale::test_prob": np.asarray([0.9, 0.1]),
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["aggregate.py", "--dir", str(output), "--seeds", "42"]
    )

    aggregate_main()

    captured = capsys.readouterr().out
    assert "1 seeds" in captured
    assert "current" in captured
    assert "stale" not in captured


def test_reproduction_wrapper_passes_completed_seeds_to_aggregate(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mewm_adapter": {
                    "embeddings_dir": str(tmp_path / "embeddings"),
                    "output_dir": str(tmp_path / "cohort"),
                },
                "experiment": {
                    "seeds": [42, 43],
                    "output_dir": str(tmp_path / "results"),
                },
            }
        )
    )
    commands = []

    def record(command, **kwargs):
        commands.append(command)

    monkeypatch.setattr("scripts.run_mewm_reproduction.subprocess.run", record)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_mewm_reproduction.py", "--config", str(config_path), "--device", "cpu"],
    )

    reproduction_main()

    aggregate_command = commands[-1]
    seed_index = aggregate_command.index("--seeds")
    assert aggregate_command[seed_index + 1 :] == ["42", "43"]

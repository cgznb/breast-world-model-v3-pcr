import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mewm_data import (
    RegisteredManifestSource,
    patient_key,
    visit_index,
    write_adapted_cohort,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ACQUISITION = re.compile(r"_dce_aqc_(\d+)\.nii(?:\.gz)?$")
SPLITS = ("train", "val", "test")
PHASE_COLUMNS = ("pre", "post_early", "post_late")


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _path_list(value):
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def _phase_index(path):
    match = ACQUISITION.search(Path(path).name)
    if match is None:
        raise ValueError(f"DCE path has no acquisition index: {path}")
    return int(match.group(1))


def _indexed_paths(value, identity):
    indexed = {}
    for path in _path_list(value):
        index = _phase_index(path)
        if index in indexed:
            raise ValueError(f"duplicate DCE acquisition {index} for {identity}")
        indexed[index] = path
    if not indexed:
        raise ValueError(f"no DCE paths for {identity}")
    return indexed


def select_strict_phase_paths(indexed_paths, metadata_row, timepoint):
    """Select metadata-declared pre/early/late paths without any fallback."""
    desired = []
    for prefix in PHASE_COLUMNS:
        value = metadata_row.get(f"{prefix}_T{timepoint}", np.nan)
        if pd.isna(value):
            raise ValueError(f"missing {prefix}_T{timepoint} phase metadata")
        desired.append(int(value))
    if len(set(desired)) != 3:
        raise ValueError(f"pre/early/late phases are not distinct at T{timepoint}")
    missing = [index for index in desired if index not in indexed_paths]
    if missing:
        raise ValueError(f"metadata-selected DCE phases are absent at T{timepoint}: {missing}")
    return tuple(indexed_paths[index] for index in desired), tuple(desired)


def _manifest_index(path, name):
    frame = pd.read_csv(path)
    required = {"patient_id", "visit", "dce_paths", "meta_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")
    indexed = {}
    for row in frame.to_dict(orient="records"):
        key = (patient_key(row["patient_id"]), visit_index(row["visit"]))
        if key in indexed:
            raise ValueError(f"duplicate {name} manifest visit: {key}")
        indexed[key] = row
    return indexed


def build_locked_splits(metadata, test_ids, *, seed=2026, val_size=98):
    """Lock test IDs, then stratify the remaining patients into train/validation."""
    labels = {str(row.pid): int(row.pCR) for row in metadata.itertuples(index=False)}
    if any(label not in (0, 1) for label in labels.values()):
        raise ValueError("pCR labels must be binary")
    test_ids = [str(pid) for pid in test_ids]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("locked test split contains duplicate patient IDs")
    missing = sorted(set(test_ids) - set(labels))
    if missing:
        raise ValueError(f"locked test patient is absent from full cohort: {missing[0]}")
    remaining = sorted(set(labels) - set(test_ids), key=patient_key)
    train_ids, val_ids = train_test_split(
        remaining,
        test_size=int(val_size),
        random_state=int(seed),
        stratify=[labels[pid] for pid in remaining],
    )
    return {
        "train": sorted(train_ids, key=patient_key),
        "val": sorted(val_ids, key=patient_key),
        "test": sorted(test_ids, key=patient_key),
    }


def _split_summary(splits, labels):
    summary = {}
    for split, ids in splits.items():
        positives = sum(labels[pid] for pid in ids)
        summary[split] = {
            "patients": len(ids),
            "pcr_negative": len(ids) - positives,
            "pcr_positive": positives,
        }
    return summary


def _canonical_id(key):
    return f"ISPY2-{key}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the independent full978 cohort and locked-102 Table 2 splits."
    )
    parser.add_argument("--config", default="configs/mewm_ispy2_full978_locked102_table2.yaml")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="require every selected local NIfTI and validate its header",
    )
    parser.add_argument(
        "--audit-values",
        action="store_true",
        help="also read every selected NIfTI payload and require finite values",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.audit_values and not args.check_files:
        raise SystemExit("--audit-values requires --check-files")
    config_path = _repo_path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    adapter = config.get("mewm_adapter")
    full = config.get("full978_data")
    experiment = config.get("experiment")
    if not all(isinstance(value, dict) for value in (adapter, full, experiment)):
        raise SystemExit("config must contain mewm_adapter, full978_data, and experiment mappings")
    output_dir = _repo_path(adapter["output_dir"])

    metadata = pd.read_csv(_repo_path(adapter["metadata_csv"]), dtype={"pid": str})
    if "pid" not in metadata or "pCR" not in metadata:
        raise SystemExit("metadata must contain pid and pCR")
    metadata["_key"] = metadata["pid"].map(patient_key)
    if metadata["_key"].duplicated().any():
        raise SystemExit("metadata aliases one numeric patient ID more than once")
    metadata_by_key = {row["_key"]: row for _, row in metadata.iterrows()}

    local_rows = _manifest_index(_repo_path(full["local_registered_manifest"]), "local")
    remote_rows = _manifest_index(_repo_path(full["qingyuan_remote_manifest"]), "remote")
    localized_rows = _manifest_index(
        _repo_path(full["qingyuan_localized_manifest"]), "Qingyuan localized"
    )
    excluded = {patient_key(value) for value in full.get("excluded_patient_ids", [])}
    remote_root = str(Path(full["qingyuan_remote_root"]))
    local_root = str(_repo_path(full["qingyuan_local_root"]))

    source_by_patient = {}
    for source_name, rows in (("local_ispy2", local_rows), ("qingyuan", remote_rows)):
        for key, _ in rows:
            if key not in metadata_by_key or key in excluded:
                continue
            previous = source_by_patient.setdefault(key, source_name)
            if previous != source_name:
                raise SystemExit(f"patient occurs in both registration sources: {key}")
    selected_keys = set(metadata_by_key) - excluded
    absent = sorted(selected_keys - set(source_by_patient))
    if absent:
        raise SystemExit(f"full cohort patient has no registered source: {absent[0]}")

    visit_rows = []
    availability_rows = []
    transfer_rows = []
    visits_by_patient = {key: [] for key in selected_keys}
    for key in sorted(selected_keys, key=int):
        source_name = source_by_patient[key]
        source_rows = local_rows if source_name == "local_ispy2" else remote_rows
        metadata_row = metadata_by_key[key]
        for timepoint in range(4):
            identity = (key, timepoint)
            source_row = source_rows.get(identity)
            if source_row is None:
                availability_rows.append(
                    {
                        "patient_id": _canonical_id(key),
                        "visit": f"T{timepoint}",
                        "source": source_name,
                        "registered": False,
                        "selected_files_present": False,
                        "valid_mask": 0,
                    }
                )
                continue

            source_indexed = _indexed_paths(source_row["dce_paths"], identity)
            if source_name == "local_ispy2":
                chosen_indexed = {index: str(Path(path).expanduser().resolve())
                                  for index, path in source_indexed.items()}
                meta_path = str(Path(source_row["meta_path"]).expanduser().resolve())
                remote_indexed = {}
            else:
                localized = localized_rows.get(identity)
                if localized is None:
                    raise SystemExit(f"Qingyuan visit has no localized metadata row: {identity}")
                local_indexed = _indexed_paths(localized["dce_paths"], identity)
                remote_indexed = source_indexed
                chosen_indexed = {}
                for index, remote_path in source_indexed.items():
                    if index in local_indexed:
                        chosen_indexed[index] = str(Path(local_indexed[index]).expanduser().resolve())
                    else:
                        if not remote_path.startswith(remote_root + "/"):
                            raise SystemExit(f"remote DCE path is outside configured root: {remote_path}")
                        relative = remote_path[len(remote_root) + 1 :]
                        chosen_indexed[index] = str((Path(local_root) / relative).resolve())
                meta_path = str(Path(localized["meta_path"]).expanduser().resolve())

            chosen, phase_indices = select_strict_phase_paths(
                chosen_indexed, metadata_row, timepoint
            )
            selected_present = all(Path(path).is_file() for path in chosen)
            if source_name == "local_ispy2" and not selected_present:
                missing_path = next(path for path in chosen if not Path(path).is_file())
                raise SystemExit(f"local ISPY2 selected DCE file is missing: {missing_path}")
            if source_name == "qingyuan":
                for index, local_path in zip(phase_indices, chosen):
                    if Path(local_path).is_file():
                        continue
                    remote_path = remote_indexed[index]
                    relative = remote_path[len(remote_root) + 1 :]
                    transfer_rows.append(
                        {
                            "patient_id": _canonical_id(key),
                            "visit": f"T{timepoint}",
                            "phase_index": index,
                            "remote_relative_path": relative,
                            "remote_path": remote_path,
                            "local_path": local_path,
                        }
                    )
            visits_by_patient[key].append(timepoint)
            visit_rows.append(
                {
                    "patient_id": _canonical_id(key),
                    "visit": f"T{timepoint}",
                    "source": source_name,
                    "dce_paths": ";".join(chosen),
                    "meta_path": meta_path,
                    "phase_indices": ";".join(str(index) for index in phase_indices),
                    "phase_selection": "metadata_pre_early_late",
                    "selected_files_present": selected_present,
                }
            )
            availability_rows.append(
                {
                    "patient_id": _canonical_id(key),
                    "visit": f"T{timepoint}",
                    "source": source_name,
                    "registered": True,
                    "selected_files_present": selected_present,
                    "valid_mask": 1,
                }
            )

    adapted = metadata[metadata["_key"].isin(selected_keys)].copy()
    adapted["pid"] = adapted["_key"].map(_canonical_id)
    adapted["mewm_patient_id"] = adapted["pid"]
    adapted["registered_timepoints"] = adapted["_key"].map(
        lambda key: ";".join(f"T{timepoint}" for timepoint in visits_by_patient[key])
    )
    adapted["n_registered_timepoints"] = adapted["_key"].map(
        lambda key: len(visits_by_patient[key])
    )
    adapted = adapted.sort_values("_key", key=lambda values: values.astype(int))
    adapted = adapted.drop(columns="_key")

    split_json = json.loads(_repo_path(full["biflow_split_json"]).read_text())
    locked = split_json.get(str(full.get("biflow_test_split", "val")))
    if not isinstance(locked, list):
        raise SystemExit("BiFlow split JSON does not contain the configured locked test list")
    locked = [_canonical_id(patient_key(pid)) for pid in locked]
    splits = build_locked_splits(
        adapted,
        locked,
        seed=int(experiment.get("split_seed", 2026)),
        val_size=int(experiment.get("validation_size", 98)),
    )
    labels = {str(row.pid): int(row.pCR) for row in adapted.itertuples(index=False)}
    split_stats = _split_summary(splits, labels)
    split_for_pid = {pid: split for split, ids in splits.items() for pid in ids}

    visit_frame = pd.DataFrame(visit_rows).sort_values(
        ["patient_id", "visit"], key=lambda values: values.map(patient_key)
        if values.name == "patient_id" else values
    )
    visit_frame["fold"] = visit_frame["patient_id"].map(split_for_pid)
    availability = pd.DataFrame(availability_rows)
    transfer = pd.DataFrame(
        transfer_rows,
        columns=[
            "patient_id", "visit", "phase_index", "remote_relative_path",
            "remote_path", "local_path",
        ],
    )
    patient_rows = []
    source_counts = {}
    for row in adapted.itertuples(index=False):
        key = patient_key(row.pid)
        source = source_by_patient[key]
        source_counts[source] = source_counts.get(source, 0) + 1
        present = set(visits_by_patient[key])
        patient_rows.append(
            {
                "patient_id": row.pid,
                "label": int(row.pCR),
                "split": split_for_pid[row.pid],
                "source": source,
                "registered_timepoints": ";".join(f"T{t}" for t in sorted(present)),
                **{f"valid_T{t}": int(t in present) for t in range(4)},
            }
        )

    expected = full.get("expected_counts", {})
    observed = {
        "patients": len(adapted),
        "visits": len(visit_frame),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }
    for name, value in observed.items():
        if name in expected and int(expected[name]) != value:
            raise SystemExit(f"full978 {name} count mismatch: expected {expected[name]}, found {value}")
    if set(splits["test"]) != set(locked):
        raise SystemExit("test split is not exactly equal to the BiFlow locked patient set")

    missing_visits = set(zip(transfer.get("patient_id", []), transfer.get("visit", [])))
    summary = {
        "schema": "pillar_full978_locked102_table2_v1",
        "input_mode": adapter["input_mode"],
        "source_type": adapter["source_type"],
        "patients": len(adapted),
        "registered_visits": len(visit_frame),
        "phase_selection_counts": {"metadata_pre_early_late": len(visit_frame)},
        "selected_nifti_files": 3 * len(visit_frame),
        "source_patient_counts": source_counts,
        "split_source": "biflow_locked_test_plus_stratified_train_validation",
        "split_seed": int(experiment.get("split_seed", 2026)),
        "splits": split_stats,
        "biflow_test_exact_match": set(splits["test"]) == set(locked),
        "excluded": len(excluded),
        "missing_selected_files": len(transfer),
        "missing_selected_visits": len(missing_visits),
        "missing_selected_patients": len(set(transfer.get("patient_id", []))),
        "files_checked": False,
        "values_audited": False,
    }
    exclusions = pd.DataFrame(
        [
            {
                "patient_id": _canonical_id(key),
                "reason": "current_registered_source_missing",
            }
            for key in sorted(excluded, key=int)
        ]
    )
    preparation = {
        "schema": "pillar_full978_data_preparation_v1",
        "phase_selection_policy": "metadata_pre_early_late_no_fallback",
        "missing_visit_policy": "zero_embedding_invalid_mask_zero_elapsed_days",
        "test_policy": "exact_BiFlow_validation_fold_102",
        "train_validation_policy": "pCR_stratified_random_split",
        "split_seed": int(experiment.get("split_seed", 2026)),
        "note": (
            "This is a paper-method reproduction on a new locked split and registered-image "
            "source, not a numerical reproduction of the paper's original 784/99/99 split."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "registered_visits.csv", visit_frame)
    _atomic_csv(output_dir / "visit_availability.csv", availability)
    _atomic_csv(output_dir / "patient_manifest.csv", pd.DataFrame(patient_rows))
    _atomic_csv(output_dir / "exclusions.csv", exclusions)
    _atomic_csv(output_dir / "transfer_missing_selected_phases.csv", transfer)
    _atomic_text(
        output_dir / "rsync_files_from.txt",
        "\n".join(transfer.get("remote_relative_path", []).tolist())
        + ("\n" if len(transfer) else ""),
    )
    write_adapted_cohort(output_dir, adapted, splits, summary)
    _atomic_text(
        output_dir / "data_preparation.json",
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
    )

    if args.check_files:
        if len(transfer):
            raise SystemExit(
                f"{len(transfer)} selected files are still missing; run the resumable transfer"
            )
        source = RegisteredManifestSource(output_dir / "registered_visits.csv")
        adapted_by_key = {
            patient_key(row.pid): row for _, row in adapted.iterrows()
        }
        checked = 0
        for key, timepoints in visits_by_patient.items():
            found = source.available_timepoints(
                _canonical_id(key), adapted_by_key[key], check_files=True
            )
            if found != tuple(timepoints):
                raise SystemExit(f"registered header audit failed for {_canonical_id(key)}")
            if args.audit_values:
                for timepoint in timepoints:
                    members = source.phase_members(
                        _canonical_id(key), timepoint, adapted_by_key[key]
                    )
                    for path in members or ():
                        source.load(_canonical_id(key), timepoint, path)
                        checked += 1
        if args.audit_values and checked != 3 * len(visit_frame):
            raise SystemExit("full NIfTI payload audit did not cover every selected file")
        summary["files_checked"] = True
        summary["values_audited"] = bool(args.audit_values)
        write_adapted_cohort(output_dir, adapted, splits, summary)

    print(
        f"patients={len(adapted)} visits={len(visit_frame)} "
        f"splits={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}"
    )
    print(
        f"missing selected files={len(transfer)} across {len(missing_visits)} visits; "
        f"wrote {output_dir}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import math
import os
import pickle
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_ids
from src.mewm_data import (
    INPUT_MODES,
    build_registered_volume,
    create_registered_source,
    patient_key,
    registered_source_contract,
)
from src.pillar import global_embedding, load_pillar

MODEL_ID = "YalaLab/Pillar0-BreastMRI"
PILLAR_EMBEDDING_DIM = 1152
PILLAR_INPUT_SHAPE_HWD = (384, 384, 192)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_embedding(path, expected_dim=PILLAR_EMBEDDING_DIM):
    path = Path(path)
    try:
        embedding = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError):
        raise ValueError(f"existing embedding is unreadable: {path}") from None
    if (
        not isinstance(embedding, torch.Tensor)
        or embedding.ndim != 1
        or embedding.numel() != int(expected_dim)
        or not bool(torch.isfinite(embedding).all())
    ):
        raise ValueError(f"existing embedding violates the Pillar contract: {path}")
    return embedding


def ensure_extraction_contract(output_dir, contract):
    output_dir = Path(output_dir)
    contract_path = output_dir / "extraction_contract.json"
    if contract_path.is_file():
        try:
            existing = json.loads(contract_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError(f"embedding extraction contract is unreadable: {contract_path}") from None
        if existing != contract:
            raise ValueError(
                "embedding output directory belongs to a different extraction contract: "
                f"{output_dir}"
            )
        return contract_path
    if output_dir.exists() and next(output_dir.glob("*/*.pt"), None) is not None:
        raise ValueError(
            "embedding output contains unprovenanced .pt files; use a new output directory"
        )
    _atomic_json(contract_path, contract)
    return contract_path


def _registered_timepoints(value):
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    if not tokens or any(len(token) != 2 or token[0] != "T" for token in tokens):
        raise ValueError("registered_timepoints is invalid")
    try:
        timepoints = tuple(int(token[1]) for token in tokens)
    except ValueError:
        raise ValueError("registered_timepoints is invalid") from None
    if (
        len(timepoints) != len(set(timepoints))
        or any(timepoint not in range(4) for timepoint in timepoints)
    ):
        raise ValueError("registered_timepoints is invalid")
    return timepoints


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Pillar-0 embeddings from registered MeWM I-SPY2 images."
    )
    parser.add_argument("--config", default="configs/mewm_ispy2_registered.yaml")
    parser.add_argument("--input-mode", choices=INPUT_MODES)
    parser.add_argument("--cohort-dir")
    parser.add_argument("--embeddings-dir")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many visits")
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=2,
        help="number of bounded CPU volume-build workers (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct and validate volumes without loading gated Pillar-0 weights",
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="extract currently ready visits but leave the full contract incomplete",
    )
    return parser.parse_args()


def _device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _prefetched_volumes(tasks, build, workers):
    """Build volumes concurrently while keeping memory bounded to `workers` futures."""
    if workers == 1:
        for task in tasks:
            yield task, build(task)
        return

    pending = deque()
    task_iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.append((task, pool.submit(build, task)))

        while pending:
            task, future = pending.popleft()
            try:
                next_task = next(task_iterator)
            except StopIteration:
                next_task = None
            if next_task is not None:
                pending.append((next_task, pool.submit(build, next_task)))
            yield task, future.result()


def main():
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.prefetch_workers < 1:
        raise SystemExit("--prefetch-workers must be positive")
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    adapter = config.get("mewm_adapter")
    if not isinstance(adapter, dict):
        raise SystemExit("config must contain a mewm_adapter mapping")

    input_mode = args.input_mode or adapter["input_mode"]
    cohort_dir = Path(args.cohort_dir or adapter["output_dir"])
    embeddings_dir = args.embeddings_dir or adapter["embeddings_dir"]
    if args.input_mode and args.input_mode != adapter["input_mode"]:
        if not args.cohort_dir or not args.embeddings_dir:
            raise SystemExit(
                "an input-mode override also requires --cohort-dir and --embeddings-dir"
            )
    metadata_path = cohort_dir / "metadata_enriched.csv"
    splits_dir = cohort_dir / "splits"
    cohort_summary_path = cohort_dir / "cohort_summary.json"
    if not metadata_path.is_file() or not cohort_summary_path.is_file():
        raise SystemExit("adapted cohort is missing; run scripts/prepare_mewm_ispy2.py first")
    cohort_summary = json.loads(cohort_summary_path.read_text())
    if cohort_summary.get("input_mode") != input_mode:
        raise SystemExit(
            "adapted cohort input mode does not match this extraction; use separate directories"
        )
    metadata = pd.read_csv(metadata_path)
    by_key = {}
    for _, row in metadata.iterrows():
        key = patient_key(row["pid"])
        if key in by_key:
            raise SystemExit("adapted metadata contains duplicate patient identities")
        by_key[key] = row
    split_ids = {}
    for split in ("train", "val", "test"):
        split_ids[split] = load_ids(splits_dir / f"{split}_ids.txt")
    ids = [pid for split in ("train", "val", "test") for pid in split_ids[split]]
    id_keys = [patient_key(pid) for pid in ids]
    if len(id_keys) != len(set(id_keys)):
        raise SystemExit("adapted patient splits overlap or contain duplicates")
    if set(id_keys) != set(by_key):
        raise SystemExit("adapted metadata and patient splits do not contain the same cohort")
    if int(cohort_summary.get("patients", -1)) != len(ids):
        raise SystemExit("adapted cohort summary patient count is stale")

    source = create_registered_source(adapter, input_mode=input_mode)
    target_h, target_w, target_d = (int(x) for x in adapter["target_shape_hwd"])
    if (target_h, target_w, target_d) != PILLAR_INPUT_SHAPE_HWD:
        raise SystemExit(f"Pillar-0 requires target_shape_hwd={PILLAR_INPUT_SHAPE_HWD}")
    spacing = tuple(float(x) for x in adapter.get("target_spacing_zyx", (1, 1, 1)))
    if len(spacing) != 3 or any(not math.isfinite(x) or x <= 0 for x in spacing):
        raise SystemExit("target_spacing_zyx must contain three positive finite values")
    expected_dim = int(adapter.get("embedding_dim", PILLAR_EMBEDDING_DIM))
    if expected_dim != PILLAR_EMBEDDING_DIM:
        raise SystemExit(f"Pillar-0 requires embedding_dim={PILLAR_EMBEDDING_DIM}")
    model_revision = str(adapter.get("model_revision", "main"))

    tasks = []
    visit_contracts = []
    phase_selection_counts = {}
    unavailable_tasks = set()
    for pid in ids:
        row = by_key[patient_key(pid)]
        for timepoint in _registered_timepoints(row["registered_timepoints"]):
            members, policy = source.phase_selection(pid, timepoint, row)
            if members is None or policy is None:
                raise SystemExit(f"registered source is missing {pid} T{timepoint}")
            missing = [path for path in set(members) if not Path(path).is_file()]
            if missing:
                if not args.allow_missing_files:
                    raise SystemExit(f"registered source file is missing: {missing[0]}")
                unavailable_tasks.add((str(pid), int(timepoint)))
            visit = source.visits[(patient_key(pid), timepoint)]
            tasks.append((pid, timepoint, row))
            phase_selection_counts[policy] = phase_selection_counts.get(policy, 0) + 1
            visit_contracts.append(
                {
                    "patient_id": str(pid),
                    "timepoint": int(timepoint),
                    "phase_paths": list(members),
                    "phase_selection": policy,
                    "registered_spacing_zyx": list(visit.spacing_zyx),
                    "registered_shape_zyx": list(visit.shape_zyx),
                }
            )
    if not tasks:
        raise SystemExit("adapted cohort contains no registered visits")
    if cohort_summary.get("phase_selection_counts") != phase_selection_counts:
        raise SystemExit("adapted cohort phase-selection summary is stale")
    required_policy = adapter.get("required_phase_selection")
    if required_policy and set(phase_selection_counts) != {required_policy}:
        raise SystemExit("registered phase selection does not match the experiment config")

    output_dir = Path(embeddings_dir)
    device = _device(args.device)
    try:
        model = None if args.dry_run else load_pillar(device, revision=model_revision)
    except OSError as error:
        if "gated repo" in str(error).lower() or "403" in str(error):
            raise SystemExit(
                "Pillar-0 access is not approved for the logged-in Hugging Face account; "
                f"request access at https://huggingface.co/{MODEL_ID}"
            ) from None
        raise
    contract = {
        "schema_version": 1,
        "model": MODEL_ID,
        "model_revision": model_revision,
        "embedding_dim": expected_dim,
        "input_mode": input_mode,
        "cohort_dir": str(cohort_dir.expanduser().resolve()),
        "metadata_csv": str(metadata_path.expanduser().resolve()),
        "splits": split_ids,
        "labels": {str(pid): int(by_key[patient_key(pid)]["pCR"]) for pid in ids},
        **registered_source_contract(adapter),
        "target_spacing_zyx": list(spacing),
        "target_shape_hwd": [target_h, target_w, target_d],
        "patients": len(ids),
        "expected_visits": len(tasks),
        "phase_selection_counts": phase_selection_counts,
        "visits": visit_contracts,
    }
    contract_path = None
    if not args.dry_run:
        contract_path = ensure_extraction_contract(output_dir, contract)
    processed = skipped = 0
    extraction_tasks = []
    for pid, timepoint, row in tasks:
        if (str(pid), int(timepoint)) in unavailable_tasks:
            continue
        destination = output_dir / str(pid) / f"{pid}_T{timepoint}.pt"
        if not args.dry_run and destination.is_file():
            validate_embedding(destination, expected_dim)
            skipped += 1
            continue

        extraction_tasks.append((pid, timepoint, row, destination))
    if args.limit:
        extraction_tasks = extraction_tasks[: args.limit]

    def build(task):
        pid, timepoint, row, _ = task
        volume = build_registered_volume(
            source,
            pid,
            timepoint,
            row,
            target_spacing=spacing,
            target_depth=target_d,
            target_hw=target_h,
        )
        if volume is None:
            raise RuntimeError(f"registered volume disappeared for {pid} T{timepoint}")
        return volume

    iterator = _prefetched_volumes(extraction_tasks, build, args.prefetch_workers)
    for task, volume in tqdm(iterator, total=len(extraction_tasks), desc="registered Pillar"):
        pid, timepoint, row, destination = task
        if args.dry_run:
            print(
                f"{pid} T{timepoint}: shape={tuple(volume.shape)} "
                f"range=[{float(volume.min()):.4f}, {float(volume.max()):.4f}]"
            )
        else:
            embedding = global_embedding(model, volume.to(device))
            if embedding.ndim != 1 or embedding.numel() != expected_dim:
                raise RuntimeError("Pillar-0 returned an unexpected embedding shape")
            if not bool(torch.isfinite(embedding).all()):
                raise RuntimeError("Pillar-0 returned a non-finite embedding")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".tmp.{os.getpid()}")
            torch.save(embedding, temporary)
            os.replace(temporary, destination)
        processed += 1

    if args.dry_run:
        print(f"dry-run validated {processed} registered visit volume(s); wrote no embeddings")
        return
    completed = processed + skipped
    complete = completed == len(tasks)
    if args.limit == 0 and not complete and not args.allow_missing_files:
        raise RuntimeError("embedding extraction ended before every registered visit was validated")
    if args.allow_missing_files and unavailable_tasks:
        print(f"temporarily unavailable source visits={len(unavailable_tasks)}")
    summary = {
        "input_mode": input_mode,
        "model": MODEL_ID,
        "contract": str(contract_path.name),
        "expected_visits": len(tasks),
        "processed_visits": processed,
        "validated_existing_visits": skipped,
        "unavailable_source_visits": len(unavailable_tasks),
        "embedding_dim": expected_dim,
        "complete": complete,
    }
    _atomic_json(output_dir / "extraction_summary.json", summary)
    print(f"processed={processed} skipped={skipped} -> {output_dir}")


if __name__ == "__main__":
    main()

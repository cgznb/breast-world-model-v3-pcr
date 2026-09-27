import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_full978_table2 import aggregate_table2, depth_slug
from scripts.run_experiments import (
    _resolve_device,
    _validate_mewm_embedding_store,
    run_cell,
    save_cell_artifacts,
    set_seed,
)
from src.data import load_all_splits, load_ids
from src.metrics import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
CELL_NAME = "global | +tab+temporal"
CELL_SLUG = "global_tab_temporal"


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _add_depth_column(path, temporal_depth):
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame["temporal_depth"] = str(temporal_depth)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _is_complete(cell_dir, seed, depth, max_tp):
    required = [
        cell_dir / "best.pt",
        cell_dir / "summary.json",
        cell_dir / "val_predictions.csv",
        cell_dir / "test_predictions.csv",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        checkpoint = torch.load(required[0], map_location="cpu", weights_only=False)
        summary = json.loads(required[1].read_text())
        return (
            int(checkpoint.get("seed", -1)) == int(seed)
            and int(checkpoint.get("effective_config", {}).get("max_tp", -1)) == int(max_tp)
            and checkpoint.get("selection", {}).get("criterion") == "validation_auroc"
            and summary.get("temporal_depth") == str(depth)
            and all(
                "temporal_depth" in pd.read_csv(path, nrows=1).columns
                for path in required[2:]
            )
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only the TDN + 17-D clinical + fixed-prior cells for full978 Table 2."
    )
    parser.add_argument("--config", default="configs/mewm_ispy2_full978_locked102_table2.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--depths", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    adapter = config.get("mewm_adapter")
    experiment = config.get("experiment")
    if not isinstance(adapter, dict) or not isinstance(experiment, dict):
        raise SystemExit("config must contain mewm_adapter and experiment mappings")
    declared_seeds = experiment.get("seeds")
    declared_depths = experiment.get("temporal_depths")
    if not isinstance(declared_seeds, list) or not isinstance(declared_depths, list):
        raise SystemExit("experiment seeds and temporal_depths must be lists")
    seeds = args.seeds or ([declared_seeds[0]] if args.smoke else declared_seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds):
        raise SystemExit("seeds must be unique integers")
    unknown_seeds = set(seeds) - set(declared_seeds)
    if unknown_seeds and not args.smoke:
        raise SystemExit(f"formal seed is not declared in config: {min(unknown_seeds)}")
    wanted_depths = set(args.depths or [str(row["name"]) for row in declared_depths])
    depths = [row for row in declared_depths if str(row["name"]) in wanted_depths]
    if len(depths) != len(wanted_depths):
        raise SystemExit("requested temporal depth is not declared in config")

    cohort_dir = _repo_path(adapter["output_dir"])
    embeddings_dir = _repo_path(adapter["embeddings_dir"])
    output_dir = _repo_path(args.output_dir or experiment["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "smoke"
    metadata_csv = cohort_dir / "metadata_enriched.csv"
    splits_dir = cohort_dir / "splits"
    ids = {
        split: load_ids(splits_dir / f"{split}_ids.txt")
        for split in ("train", "val", "test")
    }
    summary = json.loads((cohort_dir / "cohort_summary.json").read_text())
    expected = {"train": 778, "val": 98, "test": 102}
    if (
        int(summary.get("patients", -1)) != 978
        or int(summary.get("registered_visits", -1)) != 3648
        or not summary.get("biflow_test_exact_match")
        or any(len(ids[split]) != count for split, count in expected.items())
    ):
        raise SystemExit("full978 cohort or locked split contract is not satisfied")
    _validate_mewm_embedding_store(config, embeddings_dir, metadata_csv, splits_dir)
    data = load_all_splits(
        embeddings_dir, metadata_csv, ids["train"], ids["val"], ids["test"]
    )
    if data["train"]["clinical"].shape[-1] != 17:
        raise SystemExit("Table 2 requires the 17-D clinical/treatment vector")

    device = _resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": "pillar_full978_locked102_table2_run_v1",
        "config": str(config_path),
        "cohort_dir": str(cohort_dir),
        "embeddings_dir": str(embeddings_dir),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "temporal_depths": depths,
        "cell": "TDN + 17-D clinical features + frozen clinical prior",
        "model_selection": "validation_auroc_only",
        "test_role": "final_evaluation_only",
        "paper_comparability": "method reproduction; new split, not numerical reproduction",
        "smoke": bool(args.smoke),
    }
    _atomic_text(
        output_dir / "run_manifest.json",
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
    )

    for depth in depths:
        name = str(depth["name"])
        max_tp = int(depth["max_tp"])
        if max_tp not in (1, 2, 3, 4):
            raise SystemExit(f"invalid max_tp for {name}: {max_tp}")
        depth_dir = output_dir / depth_slug(name)
        for seed in seeds:
            cell_dir = depth_dir / f"seed_{seed}" / CELL_SLUG
            if not args.force and _is_complete(cell_dir, seed, name, max_tp):
                print(f"skip complete {name} seed {seed}", flush=True)
                continue
            run_config = copy.deepcopy(config)
            run_config["downstream"]["max_tp"] = max_tp
            if args.smoke:
                run_config["downstream"]["epochs"] = 2
                run_config["downstream"]["patience"] = 2
                run_config["downstream"]["train_loss_floor"] = 0
            set_seed(seed)
            run_context = {
                "seed": int(seed),
                "temporal_depth": name,
                "max_tp": max_tp,
                "requested_device": args.device,
                "resolved_device": str(device),
                "config": str(config_path),
                "cohort_dir": str(cohort_dir),
                "embeddings_dir": str(embeddings_dir),
                "output_dir": str(depth_dir),
                "torch_version": str(torch.__version__),
                "test_role": "final_evaluation_only",
            }
            resolved = copy.deepcopy(run_config)
            resolved["runtime"] = run_context
            _atomic_text(
                depth_dir / f"seed_{seed}" / "resolved_config.yaml",
                yaml.safe_dump(resolved, sort_keys=True),
            )
            print(f"run {name} seed {seed}", flush=True)
            predictions = run_cell(
                run_config, data, "tdn", True, int(seed), device
            )
            predictions["_run_context"] = run_context
            for split in ("val", "test"):
                if not np.isfinite(predictions[f"{split}_prob"]).all():
                    raise RuntimeError(f"{name} seed {seed} produced non-finite probabilities")
            save_cell_artifacts(
                depth_dir,
                CELL_NAME,
                predictions,
                {"val": data["val"]["pids"], "test": data["test"]["pids"]},
                int(seed),
            )
            for split in ("val", "test"):
                _add_depth_column(cell_dir / f"{split}_predictions.csv", name)
            cell_summary = json.loads((cell_dir / "summary.json").read_text())
            cell_summary["temporal_depth"] = name
            cell_summary["max_tp"] = max_tp
            _atomic_text(
                cell_dir / "summary.json",
                json.dumps(cell_summary, indent=2, sort_keys=True) + "\n",
            )
            metrics = compute_metrics(predictions["test_y"], predictions["test_prob"])
            print(
                f"done {name} seed {seed}: AUROC={metrics['auroc']:.4f} "
                f"PR-AUC={metrics['prauc']:.4f}",
                flush=True,
            )

    if not args.no_aggregate:
        table = aggregate_table2(output_dir, cohort_dir, seeds, depths)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.data import load_ids
from src.metrics import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
CELL_SLUG = "global_tab_temporal"
METRIC_NAMES = {
    "auroc": "AUROC",
    "sens": "Sensitivity",
    "bacc": "Balanced Accuracy",
    "prauc": "PR-AUC",
    "npv": "NPV",
    "acc": "Accuracy",
    "spec": "Specificity",
    "prec": "Precision",
}
METRIC_COLUMNS = {
    "auroc": "auroc",
    "sens": "sensitivity",
    "bacc": "balanced_accuracy",
    "prauc": "pr_auc",
    "npv": "npv",
    "acc": "accuracy",
    "spec": "specificity",
    "prec": "precision",
}
PRIMARY_METRICS = ("auroc", "sens", "bacc", "prauc", "npv")
AUXILIARY_METRICS = ("acc", "spec", "prec")


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def depth_slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate_table2(output_dir, cohort_dir, seeds, depths):
    output_dir = Path(output_dir)
    cohort_dir = Path(cohort_dir)
    expected_ids = {
        split: load_ids(cohort_dir / "splits" / f"{split}_ids.txt")
        for split in ("val", "test")
    }
    metadata = pd.read_csv(cohort_dir / "metadata_enriched.csv", dtype={"pid": str})
    if {"pid", "pCR"} - set(metadata.columns) or metadata["pid"].duplicated().any():
        raise ValueError("cohort metadata has invalid patient IDs or labels")
    label_by_pid = {
        str(row.pid): int(row.pCR) for row in metadata.itertuples(index=False)
    }
    expected_labels = {}
    for split, ids in expected_ids.items():
        if any(pid not in label_by_pid for pid in ids):
            raise ValueError(f"{split} patient is absent from cohort metadata")
        expected_labels[split] = [label_by_pid[pid] for pid in ids]

    metric_rows = []
    prediction_frames = []
    for depth in depths:
        name = str(depth["name"])
        max_tp = int(depth["max_tp"])
        for seed in seeds:
            cell_dir = output_dir / depth_slug(name) / f"seed_{seed}" / CELL_SLUG
            checkpoint_path = cell_dir / "best.pt"
            summary_path = cell_dir / "summary.json"
            if not checkpoint_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"incomplete Table 2 run: {cell_dir}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            summary = json.loads(summary_path.read_text())
            if (
                int(checkpoint.get("seed", -1)) != int(seed)
                or int(checkpoint.get("effective_config", {}).get("max_tp", -1)) != max_tp
                or checkpoint.get("selection", {}).get("criterion") != "validation_auroc"
                or summary.get("temporal_depth") != name
            ):
                raise ValueError(f"Table 2 checkpoint contract mismatch: {cell_dir}")
            for split in ("val", "test"):
                path = cell_dir / f"{split}_predictions.csv"
                frame = pd.read_csv(path, dtype={"patient_id": str})
                required = {
                    "patient_id", "label", "probability", "seed", "split",
                    "temporal_depth",
                }
                if required - set(frame.columns):
                    raise ValueError(f"prediction columns are incomplete: {path}")
                if (
                    frame["patient_id"].tolist() != expected_ids[split]
                    or frame["patient_id"].duplicated().any()
                    or frame["label"].astype(int).tolist() != expected_labels[split]
                    or set(frame["seed"]) != {int(seed)}
                    or set(frame["split"]) != {split}
                    or set(frame["temporal_depth"]) != {name}
                    or not np.isfinite(frame["probability"]).all()
                    or not frame["probability"].between(0, 1).all()
                ):
                    raise ValueError(f"prediction contract mismatch: {path}")
                metrics = compute_metrics(frame["label"], frame["probability"], threshold=0.5)
                metric_rows.append(
                    {
                        "temporal_depth": name,
                        "max_tp": max_tp,
                        "seed": int(seed),
                        "split": split,
                        **metrics,
                    }
                )
                prediction_frames.append(frame)

    per_seed = pd.DataFrame(metric_rows)
    test = per_seed[per_seed["split"] == "test"]
    summary_rows = []
    json_rows = []
    ordered_metrics = PRIMARY_METRICS + AUXILIARY_METRICS
    for depth in depths:
        name = str(depth["name"])
        rows = test[test["temporal_depth"] == name]
        if len(rows) != len(seeds):
            raise ValueError(f"Table 2 depth {name} does not have every requested seed")
        csv_row = {
            "temporal_depth": name,
            "max_tp": int(depth["max_tp"]),
            "n_seeds": len(seeds),
        }
        metrics_json = {}
        for metric in ordered_metrics:
            values = rows[metric].to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            column = METRIC_COLUMNS[metric]
            csv_row[f"{column}_mean"] = mean
            csv_row[f"{column}_std"] = std
            csv_row[f"{column}_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
            metrics_json[METRIC_NAMES[metric]] = {"mean": mean, "std": std}
        summary_rows.append(csv_row)
        json_rows.append(
            {
                "temporal_depth": name,
                "max_tp": int(depth["max_tp"]),
                "n_seeds": len(seeds),
                "metrics": metrics_json,
            }
        )

    _atomic_csv(output_dir / "table2_metrics_per_seed.csv", per_seed)
    _atomic_csv(output_dir / "table2_summary.csv", pd.DataFrame(summary_rows))
    _atomic_csv(output_dir / "all_predictions.csv", pd.concat(prediction_frames, ignore_index=True))
    _atomic_json(
        output_dir / "table2_summary.json",
        {
            "schema": "pillar_full978_locked102_table2_results_v1",
            "test_patient_count": len(expected_ids["test"]),
            "test_ids_exactly_validated": True,
            "threshold": 0.5,
            "model_selection": "validation_auroc_only",
            "standard_deviation": "sample_ddof_1",
            "primary_metrics": [METRIC_NAMES[key] for key in PRIMARY_METRICS],
            "auxiliary_metrics": [METRIC_NAMES[key] for key in AUXILIARY_METRICS],
            "rows": json_rows,
        },
    )
    return pd.DataFrame(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and aggregate full978 Table 2 runs.")
    parser.add_argument("--config", default="configs/mewm_ispy2_full978_locked102_table2.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    with _repo_path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    adapter = config["mewm_adapter"]
    experiment = config["experiment"]
    output_dir = _repo_path(args.output_dir or experiment["output_dir"])
    seeds = args.seeds or experiment["seeds"]
    frame = aggregate_table2(
        output_dir,
        _repo_path(adapter["output_dir"]),
        seeds,
        experiment["temporal_depths"],
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

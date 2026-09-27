import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the declared multi-seed Pillar-TDN MeWM reproduction."
    )
    parser.add_argument("--config", default="configs/mewm_ispy2_registered.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--global-dir")
    parser.add_argument("--cohort-dir")
    parser.add_argument("--output-dir")
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

    seeds = args.seeds or experiment.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise SystemExit("experiment seeds must be a nonempty unique integer list")
    global_dir = _repo_path(args.global_dir or adapter["embeddings_dir"])
    cohort_dir = _repo_path(args.cohort_dir or adapter["output_dir"])
    metadata_csv = cohort_dir / "metadata_enriched.csv"
    splits_dir = cohort_dir / "splits"
    output_dir = _repo_path(args.output_dir or experiment["output_dir"])

    for seed in seeds:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_experiments.py"),
            "--global-dir",
            str(global_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--splits-dir",
            str(splits_dir),
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
        ]
        print(f"running Pillar-TDN seed {seed}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if not args.no_aggregate:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "aggregate.py"),
                "--dir",
                str(output_dir),
                "--seeds",
                *(str(seed) for seed in seeds),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()

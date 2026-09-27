import argparse
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume transfer of only the full978 metadata-selected missing DCE phases."
    )
    parser.add_argument("--config", default="configs/mewm_ispy2_full978_locked102_table2.yaml")
    parser.add_argument("--host", default="qingyuan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with _repo_path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    adapter = config["mewm_adapter"]
    full = config["full978_data"]
    cohort_dir = _repo_path(adapter["output_dir"])
    files_from = cohort_dir / "rsync_files_from.txt"
    if not files_from.is_file():
        raise SystemExit("missing rsync_files_from.txt; run prepare_full978_table2.py first")
    entries = [line.strip() for line in files_from.read_text().splitlines() if line.strip()]
    if len(entries) != len(set(entries)):
        raise SystemExit("rsync file list contains duplicate paths")
    if any(Path(entry).is_absolute() or ".." in Path(entry).parts for entry in entries):
        raise SystemExit("rsync file list contains an unsafe relative path")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    if args.reverse:
        entries = list(reversed(entries))
    direction = "reverse" if args.reverse else "forward"
    if args.num_shards > 1:
        entries = entries[args.shard_index :: args.num_shards]
        files_from = cohort_dir / (
            f"rsync_files_from.{direction}.shard_{args.shard_index}_of_{args.num_shards}.txt"
        )
        files_from.write_text(
            "\n".join(entries) + ("\n" if entries else "")
        )
        print(
            f"shard {args.shard_index + 1}/{args.num_shards}", flush=True
        )
    if not entries:
        print("all selected phases are already present")
        return

    remote_root = str(full["qingyuan_remote_root"]).rstrip("/") + "/"
    local_root = _repo_path(full["qingyuan_local_root"])
    local_root.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-av",
        "--partial-dir=.rsync-partial",
        "--ignore-existing",
        "--files-from",
        str(files_from),
    ]
    if args.dry_run:
        command.append("--dry-run")
    command.extend([f"{args.host}:{remote_root}", str(local_root) + "/"])
    print(f"transferring {len(entries)} selected phase files", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

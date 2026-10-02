"""Resume the four production stages sequentially with a single-run lock."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from responsewm.io import write_json
from responsewm.training import STAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in STAGES:
            command = [sys.executable, "-u", str(root / "joint.py"), "train",
                       "--manifest", str(Path(args.manifest).resolve()),
                       "--config", str(Path(args.config).resolve()),
                       "--output", str(output), "--stage", stage]
            if (output / stage / "last.pt").exists():
                command.append("--resume")
            write_json(output / "controller_status.json", {
                "status": "running", "stage": stage, "updated_unix": time.time(),
                "command": command})
            result = subprocess.run(command, cwd=root, check=False)
            if result.returncode:
                write_json(output / "controller_status.json", {
                    "status": "failed", "stage": stage, "returncode": result.returncode,
                    "updated_unix": time.time()})
                raise SystemExit(result.returncode)
        write_json(output / "controller_status.json", {
            "status": "completed", "updated_unix": time.time()})


if __name__ == "__main__":
    main()

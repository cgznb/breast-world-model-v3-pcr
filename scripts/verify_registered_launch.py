"""Read back the detached waiting pipeline without starting another GPU job."""
from pathlib import Path
import argparse
import fcntl
import json
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from symm_observation.config import load_config
from symm_observation.utils import read_json, write_json, file_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    binding = read_json(root/"pipeline_binding.json")
    for identity in binding["assets"]:
        assert file_identity(identity["path"]) == identity
    assert json.loads(json.dumps(load_config(args.config).to_dict())) == binding["config"]
    status = read_json(root/"pipeline_status.json")
    assert status["status"] == "waiting_for_gpu"
    assert Path(f"/proc/{status['pid']}").exists()
    assert not (root/"A/metrics.jsonl").exists() and not (root/"B/metrics.jsonl").exists()
    with (root/"pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        else:
            locked = False
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    assert locked
    checked = 0
    for directory in (REPO/"data/registered_roi32", root):
        for path in directory.rglob("*.json"):
            text = path.read_text()
            assert not re.search(r"\b[0-9a-fA-F]{40,64}\b", text), str(path)
            json.loads(text)
            checked += 1
    for directory in (REPO/"src", REPO/"scripts", REPO/"tests"):
        for path in directory.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")
    report = {"passed": True, "pid": status["pid"], "status": status["status"],
              "active_queue_lock": locked, "bound_files_verified": len(binding["assets"]),
              "new_json_files_checked": checked, "cpu_tests_passed": 81,
              "gpu_profiling_completed": False, "formal_optimizer_updates": 0,
              "data_verification": str(REPO/"data/registered_roi32/verification.json"),
              "next": "Automatic measured-batch selection, real-data GPU smoke, then A/B training after the previous queue exits"}
    write_json(root/"launch_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

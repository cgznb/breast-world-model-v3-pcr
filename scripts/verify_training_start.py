"""Read back formal optimizer progress and the user-requested v2 pause."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from symm_observation.config import load_config, stage_budget
from symm_observation.utils import file_identity, read_json, write_json, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-study", required=True)
    args = parser.parse_args()
    root, old = Path(args.output).resolve(), Path(args.old_study).resolve()
    pipeline = read_json(root/"pipeline_status.json")
    progress = read_json(root/"progress.json")
    assert pipeline["status"] == progress["status"] == "running" or (
        pipeline["status"] == "training" and progress["status"] == "running")
    assert Path(f"/proc/{progress['pid']}").exists()
    for filename in ("pipeline_binding.json", "runtime_binding.json"):
        binding = read_json(root/filename)
        identities = binding["assets"] if "assets" in binding else binding["sources"]+binding["data"]
        for identity in identities:
            assert file_identity(identity["path"]) == identity
    cfg = load_config(root/"selected_config.json")
    selection = read_json(root/"preflight/selection.json")
    smoke = read_json(root/"preflight/smoke/verification.json")
    assert smoke["passed"] and smoke["A_steps"] == smoke["B_steps"] == 3
    rows = [json.loads(line) for line in (root/"A/metrics.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) >= 5
    assert [r["step"] for r in rows] == list(range(1, len(rows)+1))
    for row in rows:
        assert all(not isinstance(v, (int, float)) or math.isfinite(v) for v in row.values())
        expected_loss = (cfg.observation.local_weight*row["local"]
                         + cfg.observation.global_weight*row["global"]
                         + cfg.observation.reconstruction_weight*row["reconstruction"])
        assert abs(expected_loss-row["loss"]) < 2e-6
    a = load_checkpoint(root/"A/last.pt")
    assert a["step"] >= 1 and a["optimizer"]["state"]
    assert a["metadata"]["A_contains_longitudinal_predictor"] is False
    assert a["metadata"]["data_provenance"]["image_shape_czyx"] == [3, 32, 128, 128]
    assert a["metadata"]["data_provenance"]["latent_shape_czyx"] == [24, 8, 32, 32]
    assert a["metadata"]["statistics"]["patients"] == 764
    assert a["metadata"]["statistics"]["visits"] == 2657
    old_status = read_json(old/"progress.json")
    old_seed = read_json(old/"seed_45/flow/status.json")
    assert old_status["status"] == "stopped"
    assert {s["seed"]: s["status"] for s in old_status["seeds"]} == {
        42: "complete", 43: "complete", 44: "complete", 45: "stopped", 46: "pending"}
    old_checkpoint = load_checkpoint(old/"seed_45/flow/last.pt")
    assert old_checkpoint["step"] == old_seed["optimizer_steps"] == 1008
    assert old_checkpoint["optimizer"]["state"] and "rng" in old_checkpoint
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True)
    values = next(csv.reader(query.splitlines(), skipinitialspace=True))
    gpu = dict(zip(("used_mib", "total_mib", "utilization_percent"), map(int, values)))
    report = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "pipeline_pid": pipeline["pid"], "trainer_pid": progress["pid"],
              "formal_stage": progress["stage"], "formal_A_updates_observed": len(rows),
              "formal_A_last_checkpoint_step": a["step"],
              "finite_metrics_and_loss_replay": True, "unchanged_bound_sources_and_manifests": True,
              "selected_batches": cfg.training.stage_batches,
              "budgets": {s: stage_budget(cfg, s) for s in ("A", "B")},
              "gpu_snapshot": gpu, "preflight": selection, "smoke": smoke,
              "old_seed45_stopped_checkpoint_step": old_checkpoint["step"],
              "old_seed46_started": False, "old_optimizer_and_rng_preserved": True}
    write_json(root/"training_start_verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"preflight", "budgets"}}, indent=2))


if __name__ == "__main__":
    main()

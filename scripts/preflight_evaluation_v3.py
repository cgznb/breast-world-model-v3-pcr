"""Exercise actual fixed-cohort evaluation; untrained engineering check only."""
from pathlib import Path
import argparse
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from responsewm.config import load_config
from responsewm.data_v2 import PatientTrajectoryStore
from responsewm.training_v3 import build_model
from responsewm.evaluation_v3 import validate
from responsewm.imaging_evaluation_v3 import evaluate_imaging
from responsewm.io import seed_all, read_json, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "statistics", "output"):
        p.add_argument("--"+name, required=True)
    args=p.parse_args()
    cfg=load_config(args.config)
    seed_all(cfg.training.seed, cfg.training.threads)
    store=PatientTrajectoryStore(args.manifest)
    store.set_statistics(read_json(args.statistics))
    model=build_model(cfg,store)
    store.fit_prior(model)
    model.to(cfg.training.device).eval().requires_grad_(False)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    if str(cfg.training.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    report=validate(model,store,cfg,"flow",full=True)
    report.update(engineering_only=True, trained_model=False, elapsed_seconds=time.monotonic()-start)
    write_json(root/"fixed_cohort.json",report)
    print("Fixed-cohort evaluation completed",report["patients"],flush=True)
    imaging=evaluate_imaging(model,store,cfg,root/"imaging")
    write_json(root/"summary.json", {"engineering_only":True,"trained_model":False,
        "fixed_cohort_patients":report["patients"],"decoded_imaging_patients":imaging["patients"],
        "elapsed_seconds":time.monotonic()-start,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "peak_reserved_bytes":torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
    print("Actual MRI decoding and roundtrip readout completed",imaging["patients"],flush=True)


if __name__ == "__main__":
    main()

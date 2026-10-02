"""Share the measured GPU budget across seeds without overlapping large stages."""
from contextlib import contextmanager
import fcntl
import gc
import os
from pathlib import Path
import time

from .io import read_json, write_json


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


@contextmanager
def stage_gpu_lease(stage, run, device):
    root = os.environ.get("RESPONSEWM_GPU_LEASE_DIR")
    if not root or not str(device).startswith("cuda"):
        yield
        return
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    state = root/"holders.json"
    pid, identity = os.getpid(), process_identity(os.getpid())
    units = 1 if stage == "representation" else 2

    def modify(acquire):
        with (root/"lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            holders = read_json(state) if state.exists() else []
            holders = [h for h in holders if process_identity(h["pid"]) == h["identity"]]
            if acquire:
                if sum(h["units"] for h in holders) + units > 2:
                    return False
                holders.append({"pid":pid,"identity":identity,"units":units,
                                "stage":stage,"run":str(run),"started_unix":time.time()})
            else:
                holders = [h for h in holders if (h["pid"],h["identity"]) != (pid,identity)]
            write_json(state, holders)
            return True

    print(f"Waiting for GPU stage lease: {stage}", flush=True)
    while not modify(True):
        time.sleep(1)
    print(f"Acquired GPU stage lease: {stage}", flush=True)
    try:
        yield
    finally:
        # Empty idle-worker caches before an exclusive B/C/D process takes over.
        import torch
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        modify(False)

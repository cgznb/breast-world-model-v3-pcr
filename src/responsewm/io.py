"""Atomic checkpoints and deterministic RNG bookkeeping, no unsafe pickle fallback."""
from __future__ import annotations
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import os
import random
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda:handle.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def save_checkpoint(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    torch.save(value,tmp)
    os.replace(tmp,path)


def load_checkpoint(path):
    # No weights_only=False fallback for downloaded/user-untrusted checkpoints.
    return torch.load(path,map_location="cpu",weights_only=True)


def seed_all(seed,threads=4,deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


def rng_state():
    npstate = np.random.get_state()
    return {"python":random.getstate(),"numpy":[npstate[0],npstate[1].tolist(),npstate[2],npstate[3],npstate[4]],
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0],np.asarray(a[1],dtype=np.uint32),a[2],a[3],a[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def autocast(device,precision):
    return torch.autocast(device_type="cuda",dtype=torch.bfloat16) if str(device).startswith("cuda") and precision == "bf16" else nullcontext()

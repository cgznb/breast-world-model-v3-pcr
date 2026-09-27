from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import random
import numpy as np
import torch
from torch import nn


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            h.update(part)
    return h.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def codec_file_id(path):
    value = file_identity(path)
    return "file:{path}:{size_bytes}:{mtime_ns}".format(**value)


def seed_all(seed, threads=2, deterministic=True):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def autocast_context(device, precision):
    if torch.device(device).type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def cpu_tree(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(cpu_tree(value), tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    # No arbitrary object deserialization. Old unsafe checkpoints are imported
    # through the explicitly documented codec conversion tool only.
    return torch.load(path, map_location=device, weights_only=True)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float):
    sd = source.state_dict()
    td = target.state_dict()
    if sd.keys() != td.keys():
        raise ValueError("EMA state dictionaries differ")
    for key, value in td.items():
        incoming = sd[key].to(value)
        if value.is_floating_point():
            value.lerp_(incoming, 1 - decay)
        else:
            value.copy_(incoming)


def finite_tensor(value, name="tensor"):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def zero_loss(reference):
    return reference.sum() * 0.0


def group_count(channels, preferred=32):
    return max(g for g in range(1, min(channels, preferred) + 1) if channels % g == 0)

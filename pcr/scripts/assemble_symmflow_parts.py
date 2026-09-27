"""Resume and assemble a lossless binary checkpoint transfer."""

import argparse
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1] / "results/ispy2_generator_comparison_20260911/symmflow_reference"
SIZE = 332689803
CHUNK = 41943040
parser = argparse.ArgumentParser()
parser.add_argument("--seed-prefix", action="store_true")
args = parser.parse_args()
destination = ROOT / "inference.pt"
if args.seed_prefix:
    if destination.stat().st_size < CHUNK:
        raise ValueError("Incomplete checkpoint prefix is shorter than one part")
    with destination.open("rb") as source, (ROOT / "inference.part00").open("wb") as target:
        remaining = CHUNK
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Unexpected end of transferred prefix")
            target.write(block)
            remaining -= len(block)
    print("Preserved first complete transfer part")
else:
    parts = [ROOT / f"inference.part{i:02d}" for i in range(8)]
    for i, part in enumerate(parts):
        expected = CHUNK if i < 7 else SIZE - 7 * CHUNK
        if part.stat().st_size != expected:
            raise ValueError(f"Transfer part {i} is incomplete")
    temporary = destination.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)
    if temporary.stat().st_size != SIZE:
        raise ValueError("Reassembled size differs from original export")
    import torch
    header = torch.load(temporary, map_location="cpu", weights_only=True)
    assert header["schema"] == "symmflow_tensor_exact_inference_export_v1"
    assert header["step"] == 100000
    assert all(torch.isfinite(x).all() for key in ("velocity_state", "condition_state")
               for x in header[key].values())
    os.replace(temporary, destination)
    print("Reassembled inference checkpoint; structure and finite tensors passed")

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_ids
from src.pillar import NiftiSource, build_volume, global_embedding, load_pillar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", required=True, help="BreastDCEDL_ISPY2 directory or .zip")
    ap.add_argument("--metadata-csv", required=True, help="enriched CSV with the DCE phase columns")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default="data/emb_global")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    source = NiftiSource(args.nifti)
    meta = pd.read_csv(args.metadata_csv)
    meta_by_pid = {str(r["pid"]): r for _, r in meta.iterrows()}
    ids = []
    for name in ("train", "val", "test"):
        ids += load_ids(os.path.join(args.splits_dir, f"{name}_ids.txt"))
    model = load_pillar(device)

    n_ok, n_skip, n_missing = 0, 0, 0
    for pid in tqdm(ids, desc="pillar"):
        row = meta_by_pid.get(str(pid))
        if row is None:
            continue
        for t in range(4):
            dst_dir = os.path.join(args.out, str(pid))
            dst = os.path.join(dst_dir, f"{pid}_T{t}.pt")
            if os.path.exists(dst):
                n_skip += 1
                continue
            volume = build_volume(source, pid, t, row)
            if volume is None:
                n_missing += 1
                continue
            emb = global_embedding(model, volume.to(device))
            os.makedirs(dst_dir, exist_ok=True)
            torch.save(emb, dst)
            n_ok += 1

    print(f"ok={n_ok} skip={n_skip} missing={n_missing} -> {args.out}")


if __name__ == "__main__":
    main()

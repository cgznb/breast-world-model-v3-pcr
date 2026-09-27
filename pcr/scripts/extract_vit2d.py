import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nifti import NiftiZip
from src.data import load_ids
from src.metrics import compute_metrics
from src.vit2d import (BaselineSliceDataset, build_vit, embed, extract_embeddings, slice_tensors)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_cache(nz, meta, pids):
    """Read each patient's baseline tumor-center slices ONCE (NIfTI reads are the bottleneck);
    reused every epoch for val and at the end for test."""
    cache = []
    for pid in pids:
        if not nz.has(pid, 0):
            continue
        row = meta[str(pid)]
        tens = slice_tensors(nz, row, pid, 0, augment=False)
        if tens:
            cache.append((torch.stack(tens), float(row["pCR"])))
    return cache


@torch.no_grad()
def eval_patients(model, cache, device):
    """Patient-level probability and label, averaged over the baseline tumor-center slices. The
    classifier is applied to the mean-pooled encoder feature, matching how the embeddings are
    extracted."""
    model.eval()
    ys, ps = [], []
    for x, y in cache:
        logits = model.classifier(embed(model, x.to(device)))
        ps.append(torch.softmax(logits, dim=1)[:, 1].mean().item())
        ys.append(y)
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="BreastDCEDL_ISPY2.zip (read directly)")
    ap.add_argument("--csv", required=True, help="enriched metadata CSV")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--out-emb", required=True, help="output dir for per-visit 768-D embeddings")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone (encoder) LR")
    ap.add_argument("--lr-head", type=float, default=1e-3, help="classifier head LR")
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="fraction of epochs for LR warmup")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nz = NiftiZip(args.zip)
    meta_df = pd.read_csv(args.csv)
    meta = {str(r["pid"]): r for _, r in meta_df.iterrows()}
    tr_ids = load_ids(os.path.join(args.splits_dir, "train_ids.txt"))
    va_ids = load_ids(os.path.join(args.splits_dir, "val_ids.txt"))
    te_ids = load_ids(os.path.join(args.splits_dir, "test_ids.txt"))

    train_ds = BaselineSliceDataset(nz, meta, tr_ids, augment=True)
    val_cache = build_eval_cache(nz, meta, va_ids)
    test_cache = build_eval_cache(nz, meta, te_ids)
    print(f"train slices={len(train_ds)} | val patients={len(val_cache)} | "
          f"test patients={len(test_cache)} | device={device}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = build_vit().to(device)
    n_pos = sum(1 for _, y, _ in train_ds.items if y == 1)
    n_neg = len(train_ds) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    # separate learning rates for the head and backbone, with warmup then cosine decay
    opt = torch.optim.AdamW([
        {"params": model.vit.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    from transformers import get_cosine_schedule_with_warmup
    warmup = max(1, int(args.warmup_frac * args.epochs))
    sched = get_cosine_schedule_with_warmup(opt, warmup, args.epochs)
    print(f"recipe: mean-pool head | lr backbone={args.lr} head={args.lr_head} | "
          f"warmup={warmup}/{args.epochs} epochs")

    best_auc, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.long().to(device)
            opt.zero_grad()
            loss = loss_fn(model.classifier(embed(model, x)), y)   # mean-pool + head
            loss.backward(); opt.step()
        sched.step()
        vy, vp = eval_patients(model, val_cache, device)
        vauc = roc_auc_score(vy, vp) if len(set(vy)) > 1 else 0.5
        print(f"epoch {ep:02d}  val_auroc={vauc:.3f}")
        if vauc > best_auc:
            best_auc, patience = vauc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    ty, tp = eval_patients(model, test_cache, device)
    m = compute_metrics(ty, tp)
    print("\n[ViT2D @T0 image-only] test: " + "  ".join(f"{k}={m[k]:.3f}" for k in m))

    os.makedirs(args.out_emb, exist_ok=True)
    n = extract_embeddings(model, nz, meta, tr_ids + va_ids + te_ids, args.out_emb, device=device)
    print(f"wrote {n} per-visit embeddings to {args.out_emb}")


if __name__ == "__main__":
    main()

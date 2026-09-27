import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
VIT_CKPT = "facebook/vit-mae-base"


# --------------------------------------------------------------------------- slices

def _minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype=np.float32)


def _safe_crop(a, cr, cc, size):
    """Crop a `size`x`size` window centred on (row cr, col cc) of the last two axes, with
    edge clamping + zero-pad if the volume is smaller than `size`."""
    H, W = a.shape[-2], a.shape[-1]
    half = size // 2
    r0 = int(np.clip(cr - half, 0, max(H - size, 0)))
    c0 = int(np.clip(cc - half, 0, max(W - size, 0)))
    crop = a[..., r0:r0 + size, c0:c0 + size]
    ph, pw = size - crop.shape[-2], size - crop.shape[-1]
    if ph > 0 or pw > 0:
        pad = [(0, 0)] * (crop.ndim - 2) + [(0, max(ph, 0)), (0, max(pw, 0))]
        crop = np.pad(crop, pad)
    return crop


def _ival(row, col, default):
    v = row.get(col, np.nan)
    return default if pd.isna(v) else int(v)


def slice_indices(row, t):
    """The tumor-center slice +/-2, taken from the bounding box columns in the metadata."""
    nz = _ival(row, f"n_z_T{t}", 0)
    f = max(_ival(row, f"mask_start_T{t}", 0), 0)
    l = min(_ival(row, f"mask_end_T{t}", nz or 10_000), nz or 10_000)
    if l <= f:
        l = f + 1
    idx = (f + l) // 2
    rng = list(range(max(idx - 2, f), min(idx + 2, l)))
    return rng or [idx]


def slice_tensors(nz_zip, row, pid, t, crop=224, augment=False):
    """List of [3,224,224] tensors — one per tumor-center slice."""
    rgbvol = nz_zip.rgb_volume(pid, t)                  # (3,H,W,Z)
    H, W = rgbvol.shape[1], rgbvol.shape[2]
    cr = (_ival(row, f"sraw_T{t}", H // 2) + _ival(row, f"eraw_T{t}", H // 2)) // 2
    cc = (_ival(row, f"scol_T{t}", W // 2) + _ival(row, f"ecol_T{t}", W // 2)) // 2
    Z = rgbvol.shape[-1]
    out = []
    for k in slice_indices(row, t):
        if k < 0 or k >= Z:
            continue
        img = np.stack([_minmax(rgbvol[c, :, :, k]) for c in range(3)], axis=0)  # (3,H,W)
        img = _safe_crop(img, cr, cc, crop)
        ten = torch.from_numpy(img.astype(np.float32))
        ten = F.interpolate(ten.unsqueeze(0), size=(224, 224), mode="bilinear",
                            align_corners=False).squeeze(0)
        ten = (ten - IMAGENET_MEAN) / IMAGENET_STD
        if augment:                                     # gentle flips only (no rot90)
            if np.random.rand() < 0.5:
                ten = torch.flip(ten, dims=[-1])
            if np.random.rand() < 0.5:
                ten = torch.flip(ten, dims=[-2])
        out.append(ten)
    return out


# --------------------------------------------------------------------------- dataset

class BaselineSliceDataset(torch.utils.data.Dataset):
    """Baseline-visit (T0) tumor-center slices, each labeled with the patient's pCR."""

    def __init__(self, nz_zip, meta_by_pid, pids, augment=False):
        self.nz, self.aug = nz_zip, augment
        self.items = []
        for pid in pids:
            if not nz_zip.has(pid, 0):
                continue
            row = meta_by_pid[str(pid)]
            for ten in slice_tensors(nz_zip, row, pid, 0, augment=False):
                self.items.append((ten, float(row["pCR"]), pid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ten, y, _ = self.items[i]
        if self.aug:                                    # gentle flips only (no rot90)
            if np.random.rand() < 0.5:
                ten = torch.flip(ten, dims=[-1])
            if np.random.rand() < 0.5:
                ten = torch.flip(ten, dims=[-2])
        return ten, torch.tensor(y, dtype=torch.float32)


# --------------------------------------------------------------------------- model

def build_vit(num_labels=2):
    return ViTForImageClassification.from_pretrained(VIT_CKPT, num_labels=num_labels)


def embed(model, pixel_values):
    """768-D embedding: the mean of the ViT patch tokens. The MAE checkpoint's CLS token is not
    trained to summarize the image, so the patch-token mean is used instead."""
    out = model.vit(pixel_values=pixel_values)
    return out.last_hidden_state[:, 1:, :].mean(dim=1)   # drop CLS (index 0), average the patches


# --------------------------------------------------------------------------- extract

@torch.no_grad()
def extract_embeddings(model, nz_zip, meta_by_pid, pids, out_dir, device="cpu"):
    """Freeze the trained ViT and write a 768-D embedding per (patient, visit) — the
    mean over the tumor-center slices — to out_dir/{pid}/{pid}_T{t}.pt (EmbStore format)."""
    model.eval()
    n = 0
    for pid in pids:
        row = meta_by_pid[str(pid)]
        for t in nz_zip.timepoints(pid):
            tens = slice_tensors(nz_zip, row, pid, t, augment=False)
            if not tens:
                continue
            x = torch.stack(tens).to(device)
            z = embed(model, x).mean(dim=0).cpu()       # mean over the +/-2 slices -> [768]
            d = os.path.join(out_dir, str(pid))
            os.makedirs(d, exist_ok=True)
            torch.save(z, os.path.join(d, f"{pid}_T{t}.pt"))
            n += 1
    return n

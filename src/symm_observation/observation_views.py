"""Two spatial crops from ONE observation and 3D relative patch coordinates."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from .observation_encoder import positions_3d


def latent_affine(record):
    """Return a representative latent-cell affine, with explicit index-space fallback.

    Relative transforms between crops of the SAME array are exact in its index
    coordinates; this is not a longitudinal registration transform.
    """
    g = record.get("geometry") or {}
    if "latent_to_lps" in g:
        a = torch.as_tensor(g["latent_to_lps"], dtype=torch.float64)
    elif all(k in g for k in ("spacing_xyz_mm", "origin_lps_mm", "direction_lps")):
        direction = torch.as_tensor(g["direction_lps"], dtype=torch.float64).reshape(3, 3)
        spacing = torch.as_tensor(g["spacing_xyz_mm"], dtype=torch.float64)
        # Latent storage is ZYX; representative cell-center convention for 4x VQ.
        xyz_linear = direction @ torch.diag(spacing)
        a = torch.eye(4, dtype=torch.float64)
        a[:3, :3] = xyz_linear[:, [2, 1, 0]] * 4
        a[:3, 3] = torch.as_tensor(g["origin_lps_mm"], dtype=torch.float64) + xyz_linear @ torch.full((3,), 1.5, dtype=torch.float64)
    else:
        a = torch.eye(4, dtype=torch.float64)
    if a.shape != (4, 4) or not torch.isfinite(a).all() or abs(float(torch.det(a[:3, :3]))) < 1e-12:
        raise ValueError("Invalid/singular observation affine")
    if not torch.allclose(a[3], torch.tensor([0., 0., 0., 1.], dtype=torch.float64)):
        raise ValueError("Invalid affine homogeneous row")
    return a


def patch_affine(base, origin, patch):
    local = torch.eye(4, dtype=torch.float64)
    local[:3, :3] = torch.diag(torch.as_tensor(patch, dtype=torch.float64))
    local[:3, 3] = torch.as_tensor(origin, dtype=torch.float64) + (torch.as_tensor(patch, dtype=torch.float64)-1)*.5
    return base @ local


def map_query_positions(source_affine, target_affine, positions):
    transform = torch.linalg.solve(source_affine.double(), target_affine.double()).to(positions.device)
    homogeneous = torch.cat((positions.double(), torch.ones_like(positions[..., :1]).double()), -1)
    return torch.einsum("bij,bnj->bni", transform, homogeneous)[..., :3].float()


def block_mask(valid, grid, ratio, blocks, rng):
    """Union of 3D boxes, capped below all-valid masking; invalid patches never targets."""
    good = np.flatnonzero(np.asarray(valid, bool))
    if len(good) < 2:
        raise ValueError("Fewer than two valid patches in observation crop")
    target = min(len(good)-1, max(1, round(ratio*len(good))))
    mask = np.zeros(grid, dtype=bool)
    side = max(.15, (ratio/max(1, blocks))**(1/3))
    for _ in range(blocks*12):
        ext = [max(1, min(n, round(n*side*rng.uniform(.65, 1.35)))) for n in grid]
        org = [int(rng.integers(n-e+1)) for n, e in zip(grid, ext)]
        proposal = mask.copy()
        proposal[tuple(slice(o, o+e) for o, e in zip(org, ext))] = True
        if (proposal.reshape(-1)[good]).sum() < len(good):
            mask = proposal
        if mask.reshape(-1)[good].sum() >= target:
            break
    mask = mask.reshape(-1) & np.asarray(valid, bool)
    if not mask.any():
        mask[good[int(rng.integers(len(good)))]] = True
    # With tiny grids boxes may under-shoot; retain spatial blocks rather than force exact random counts.
    return torch.from_numpy(mask)


def _slice(x, origin, shape):
    return x[(slice(None), *(slice(o, o+n) for o, n in zip(origin, shape)))].clone()


def make_view(item, cfg, rng, augmentation=None, previous_origin=None):
    raw = item["latent"]
    shape, patch = cfg.observation.crop, cfg.encoder.patch
    if any(n > total for n, total in zip(shape, raw.shape[1:])):
        raise ValueError("Observation crop exceeds volume: explicitly choose a smaller crop")
    grid = tuple(n//p for n, p in zip(shape, patch))
    limits = [(total-n)//p for total, n, p in zip(raw.shape[1:], shape, patch)]
    for attempt in range(64):
        origin = tuple(int(rng.integers(limit+1))*p for limit, p in zip(limits, patch))
        if previous_origin == origin and any(limits) and attempt < 32:
            continue
        valid = _slice(item["valid"], origin, shape).amin(0, keepdim=True)
        coverage = F.avg_pool3d(valid[None], patch, patch).flatten().numpy()
        good = coverage >= cfg.observation.minimum_patch_coverage
        if good.sum() >= 2:
            break
    else:
        raise ValueError("Cannot find an observation crop with sufficient valid support")
    hidden = block_mask(good, grid, cfg.observation.mask_ratio, cfg.observation.mask_blocks, rng)
    context_z = item["context_latent"] if augmentation is None else augmentation["context_latent"]
    action = item["action"] if augmentation is None else augmentation["action"]
    view = {"clean": _slice(raw, origin, shape), "context": _slice(context_z, origin, shape),
            "valid": valid, "good": torch.from_numpy(good), "hidden": hidden,
            "visible": torch.from_numpy(good) & ~hidden, "action": action,
            "affine": patch_affine(latent_affine(item["record"]), origin, patch), "origin": origin}
    for key in ("kinetics", "segmentation"):
        if key in item:
            view[key] = _slice(item[key], origin, shape)
    return view


def observation_batch(store, records, cfg, counter, device, *, validation=False):
    rng = np.random.default_rng(np.random.SeedSequence([cfg.training.seed, 7001 if validation else 1709, int(counter)]))
    a, b, phenotypes, domains = [], [], [], []
    for row in records:
        item = store.read(row, include_aux=True, strict_alignment=cfg.observation.require_phase_alignment)
        if cfg.observation.domain_enabled:
            bank = row.get("augmentations", [])
            if not bank:
                raise ValueError("domain_enabled requires an image-space augmentation bank for each selected visit")
            aa, ab = [store.read(row, augmentation=bank[int(rng.integers(len(bank)))]) for _ in range(2)]
        else:
            aa = ab = None
        av = make_view(item, cfg, rng, aa)
        bv = make_view(item, cfg, rng, ab, previous_origin=av["origin"])
        a.append(av); b.append(bv)
        phenotypes.append(row.get("phenotype_label", -1)); domains.append(row.get("domain_label", -1))

    def collate(views):
        keys = ("clean", "context", "valid", "good", "hidden", "visible", "action", "affine")
        out = {k: torch.stack([v[k] for v in views]).to(device) for k in keys}
        for key, channels in (("kinetics", 2), ("segmentation", 1)):
            present = [key in v for v in views]
            out[key+"_present"] = torch.tensor(present, device=device, dtype=torch.bool)
            out[key] = torch.stack([v.get(key, torch.zeros(channels, *cfg.observation.crop)) for v in views]).to(device)
        return out
    return {"a": collate(a), "b": collate(b), "phenotype": torch.tensor(phenotypes, device=device),
            "domain": torch.tensor(domains, device=device), "visit_ids": [r["visit_id"] for r in records]}

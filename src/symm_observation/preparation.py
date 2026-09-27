"""Image-backed, SAME-VISIT DCE readouts and domain augmentation caches.

No gamma/brightness operation is applied directly to VQ latent coordinates.
The bank explicitly passes transformed MRI through the matching frozen codec.
"""
from __future__ import annotations
from pathlib import Path
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from .codec import load_codec
from .data import VisitStore, array, resolve
from .utils import file_identity, codec_file_id, write_json


def image_transform(images, action, generator=None):
    """A proposed MRI nuisance transform: shared gain/offset, Gaussian noise/blur.

    The 4 scalars are (gain-1, offset, noise std, blur sigma), in the documented
    shared normalized image units. Not a calibrated scanner simulation.
    """
    if images.ndim != 5 or images.shape[1] != 3 or len(action) != 4:
        raise ValueError("Expected [B,3,D,H,W] and four nuisance parameters")
    gain_delta, offset, noise_std, sigma = [float(x) for x in action]
    if not all(math.isfinite(x) for x in action) or gain_delta <= -1 or min(noise_std, sigma) < 0:
        raise ValueError("Invalid image nuisance parameters")
    out = images*(1+gain_delta)+offset
    if sigma > 0:
        radius = max(1, math.ceil(3*sigma))
        coords = torch.arange(-radius, radius+1, device=out.device, dtype=out.dtype)
        kernel = torch.exp(-coords.square()/(2*sigma*sigma)); kernel /= kernel.sum()
        for axis in range(3):
            shape = [1, 1, 1]; shape[axis] = len(kernel)
            weight = kernel.reshape(1, 1, *shape).repeat(3, 1, 1, 1, 1)
            padding = [0]*6
            j = 2*(2-axis); padding[j:j+2] = [radius, radius]
            out = F.conv3d(F.pad(out, padding, mode="replicate"), weight, groups=3)
    if noise_std:
        out = out+noise_std*torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
    return out


def prepare_sidecars(manifest, codec_checkpoint, output, *, samples=2, device="cpu", seed=2026,
                     cache_tolerance=.03, trusted_legacy=False):
    store = VisitStore(manifest)
    output = Path(output).resolve()
    if (output/"visits.json").exists():
        raise FileExistsError("Use a fresh sidecar output directory")
    expected_id = codec_file_id(codec_checkpoint)
    if store.codec_id != expected_id:
        raise ValueError("Sidecar preparation requires the codec file identity declared in the manifest")
    if samples < 1:
        raise ValueError("At least one image augmentation is required")
    codec = load_codec(codec_checkpoint, device, trusted_legacy=trusted_legacy)
    result = copy.deepcopy(store.manifest)
    rng = np.random.default_rng(seed)
    report = []
    output.mkdir(parents=True, exist_ok=True)
    for ordinal, row in enumerate(result["visits"]):
        original = store.visits[ordinal]
        n = row.get("image_normalization", {})
        if (not row.get("image_path") or not row.get("phase_alignment_verified")
                or n.get("shared_across_phases") is not True or n.get("stored") != "normalized"):
            raise ValueError(f"{row['id']} needs same-visit images, verified phase alignment, and declared shared normalized units")
        if not math.isfinite(float(n.get("mean", float('nan')))) or not math.isfinite(float(n.get("std", float('nan')))) or n["std"] <= 0:
            raise ValueError("Document the shared image mean and positive std")
        image_path = resolve(store.base, row["image_path"])
        images = array(image_path, "images")[None].to(device)
        item = store.read(original)
        raw = item["latent"][None].to(device)
        with torch.no_grad():
            encoded = codec.encode(images).float()
        if encoded.shape != raw.shape:
            raise ValueError("Same-visit image and latent grids are not compatible with this codec")
        mismatch = float((encoded-raw).square().mean().sqrt()/(1+raw.square().mean().sqrt()))
        if mismatch > cache_tolerance:
            raise ValueError(f"{row['id']} image/codec does not reproduce its latent (relative RMS {mismatch:.4g})")
        # Two independent DCE signal differences, pooled to the ORIGINAL latent lattice.
        kinetics = torch.cat((images[:, 1:2]-images[:, 0:1], images[:, 2:3]-images[:, 1:2]), 1)
        kinetics = F.adaptive_avg_pool3d(kinetics, raw.shape[2:]).float()[0].cpu().numpy()
        stem = f"visit_{ordinal:06d}"
        kp = output / (stem+"_kinetics.npy")
        np.save(kp, kinetics)
        row["kinetic_path"] = str(kp)
        row["kinetic_provenance"] = "measured_same_visit_shared_normalization"
        row["augmentations"] = []
        for j in range(samples):
            action = [float(rng.uniform(-.15, .15)), float(rng.uniform(-.05, .05)),
                      float(rng.uniform(0, .03)), float(rng.uniform(0, .6))]
            generator = torch.Generator(device=device).manual_seed(seed+ordinal*1009+j)
            with torch.no_grad():
                altered = image_transform(images, action, generator)
                az = codec.encode(altered).float()[0].cpu().numpy()
            ap = output / (stem+f"_domain_{j:02d}.npy")
            np.save(ap, az)
            row["augmentations"].append({"latent_path": str(ap), "action": action,
                "transform_space": "image", "visit_id": row["visit_id"],
                "base_latent_identity": file_identity(resolve(store.base, original["latent_path"])), "codec_id": expected_id})
        for key in ("latent_path", "valid_path", "image_path", "segmentation_path"):
            if original.get(key):
                row[key] = str(resolve(store.base, original[key]))
        report.append({"visit_id": row["visit_id"], "image_identity": file_identity(image_path),
                       "base_encoding_relative_rms": mismatch, "augmentation_count": samples})
    write_json(output/"visits.json", result)
    VisitStore(output/"visits.json")
    write_json(output/"preparation_report.json", {"codec_id": expected_id, "records": report,
               "warning": "Augmentations are proposed nuisance perturbations, not validated scanner simulators"})
    return {"visits_manifest": str(output/"visits.json"), "visits": len(report)}

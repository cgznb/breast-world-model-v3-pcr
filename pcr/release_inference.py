"""Input-only export and inference for the historical 300/50 TDN classifiers."""

from pathlib import Path

import numpy as np
import torch

from scripts.run_full978_anti_overfit import _prior_from_state
from src.data import TABULAR_FEATURE_NAMES, _vec_from_tensor
from src.tdn import TDN
from src.temporal import canonicalize_temporal_prefix

SCHEMA = "roi32-pcr-inference-v1"


def export_bundle(checkpoint, output):
    # Original research checkpoints contain task metadata and require trusted input.
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema") != "registered_three_phase_roi32_pcr_300_50":
        raise ValueError("Expected a trusted 300/50 classifier checkpoint")
    task = source["task"]
    prior = source["clinical_prior"]
    if prior["feature_names"] != list(TABULAR_FEATURE_NAMES):
        raise ValueError("Clinical feature order differs from the trained model")
    bundle = {
        "schema": SCHEMA,
        "config": task["effective"],
        "classifier": {key: task[key] for key in ("candidate", "seed", "outer", "depth")},
        "clinical_prior": {key: prior[key] for key in ("coef", "intercept", "feature_names")},
        "model_state": source["model_state"],
        "requires_labels": False,
    }
    model = TDN({"downstream": bundle["config"]})
    model.load_state_dict(bundle["model_state"], strict=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return bundle


@torch.inference_mode()
def predict(bundles, embeddings, masks, clinical, days):
    """Average trajectory probabilities, then fold probabilities, for one seed."""
    torch.backends.mha.set_fastpath_enabled(False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    clinical = np.asarray(clinical, dtype=np.float32)
    days = np.asarray(days, dtype=np.float32)
    if embeddings.ndim == 3:
        embeddings = embeddings[None]
    if embeddings.ndim != 4 or embeddings.shape[2:] != (4, 1152):
        raise ValueError("Embeddings must have shape [draws,patients,4,1152]")
    draws, count = embeddings.shape[:2]
    if (not draws or not count or masks.shape != (count, 4)
            or days.shape != masks.shape or clinical.shape != (count, 17)
            or not np.isin(masks, [0, 1]).all() or not (masks[:, 0] == 1).all()
            or not all(np.isfinite(x).all() for x in (embeddings, masks, clinical, days))
            or (days < 0).any()):
        raise ValueError("Invalid pCR input shapes, masks, days or finite values")
    # EmbStore historically normalizes each visit before temporal masking.
    normalized = np.stack([
        _vec_from_tensor(torch.from_numpy(row), require_finite=True)
        for row in embeddings.reshape(-1, 1152)
    ]).reshape(embeddings.shape)
    fold_probabilities, group, folds = [], None, set()
    for path in bundles:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle.get("schema") != SCHEMA or bundle.get("requires_labels") is not False:
            raise ValueError("Unsupported input-only pCR bundle")
        identity = bundle["classifier"]
        current = tuple(identity[key] for key in ("candidate", "seed", "depth"))
        if (group is not None and current != group) or identity["outer"] in folds:
            raise ValueError("Use distinct folds of one classifier version, seed and window")
        group = current
        folds.add(identity["outer"])
        if bundle["clinical_prior"]["feature_names"] != list(TABULAR_FEATURE_NAMES):
            raise ValueError("Clinical feature order mismatch")
        model = TDN({"downstream": bundle["config"]}).eval()
        model.load_state_dict(bundle["model_state"], strict=True)
        prior = torch.from_numpy(_prior_from_state(clinical, bundle["clinical_prior"]))
        values = []
        for draw in normalized:
            z, mask, elapsed = canonicalize_temporal_prefix(draw, masks, days, identity["depth"])
            parts = []
            for start in range(0, count, 256):
                rows = slice(start, start + 256)
                logits = model(torch.from_numpy(z[rows]), torch.from_numpy(mask[rows]),
                               torch.from_numpy(clinical[rows]), days=torch.from_numpy(elapsed[rows]),
                               prior_logit=prior[rows])
                parts.append(logits.sigmoid().numpy())
            values.append(np.concatenate(parts))
        fold_probabilities.append(np.stack(values).mean(axis=0))
    if not fold_probabilities:
        raise ValueError("At least one classifier bundle is required")
    folds_array = np.stack(fold_probabilities)
    return {"probability": folds_array.mean(axis=0), "fold_probabilities": folds_array}

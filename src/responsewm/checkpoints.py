"""Strict V2 warm-start migration, without claiming whole-pipeline equivalence."""
from __future__ import annotations
import torch
from .io import load_checkpoint,digest


def migrate_v2(model,path):
    payload = load_checkpoint(path)
    state = payload.get("model")
    if not isinstance(state,dict):
        raise ValueError("Expected V2 checkpoint with a model state dictionary")
    report = {"source_sha256":digest(path),"source_format":"V2 model state_dict",
              "not_transferred":["old clinical encoder","old context adapter","deterministic future predictor","external Pillar/TDN weights"],
              "normalization":"Current fold training statistics are retained; representation recalibration is required",
              "pretraining_patient_overlap":"unverified; do not claim an independent end-to-end test without auditing exposure"}
    for name,destination,prefix in (("encoder",model.encoder,"encoder."),
                                    ("state_heads",model.state_heads,"heads."),
                                    ("image_backbone",model.velocity.image,"velocity.")):
        values = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if not values:
            if name == "state_heads":
                report[name] = "not present"
                continue
            if name == "image_backbone" and "target_encoder.latent_mean" in state:
                report[name] = "representation-only source: no image weights"
                continue
            raise ValueError(f"Missing required {name} weights in V2 source")
        current = destination.state_dict()
        if name == "encoder":
            # Changing a fold's fitted scaler is explicit, not silent target-data reuse.
            for key in ("latent_mean","latent_std"):
                values[key] = current[key].clone()
        if set(values) != set(current):
            raise ValueError(f"{name} parameter keys differ: missing={sorted(set(current)-set(values))[:8]}, unexpected={sorted(set(values)-set(current))[:8]}")
        mismatches = [k for k in values if values[k].shape != current[k].shape]
        if mismatches:
            raise ValueError(f"{name} incompatible dimensions at {mismatches[:8]}; no partial silent load")
        destination.load_state_dict(values,strict=True)
        report[name] = {"tensors":len(values),"strict":True}
    model.target_encoder.load_state_dict(model.encoder.state_dict(),strict=True)
    return report

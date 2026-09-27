"""Export tensor-exact inference weights without optimizer state or legacy digests."""

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, "/path/to/local/ispy2-symmflow3d/src")
from ispy2_symmflow.models.mewm_vqgan import MEWM_REGISTERED_VQGAN_SHA256
from ispy2_symmflow.training.provenance import require_latent_statistics_fingerprint

root = Path("/path/to/local/ispy2-symmflow3d/outputs/mewm_all_pairs_5090")
source = root / "symmflow/symmflow_best.pt"
header = torch.load(source, map_location="cpu", weights_only=False)
assert header["step"] == 100000
assert header["autoencoder_id"] == MEWM_REGISTERED_VQGAN_SHA256
require_latent_statistics_fingerprint(header["latent_statistics"])

velocity = {
    k.removeprefix("velocity."): v
    for k, v in header["model"].items() if k.startswith("velocity.")
}
assert set(header["ema"]["shadow"]) <= set(velocity)
velocity.update(header["ema"]["shadow"])
conditions = {
    k.removeprefix("conditions."): v
    for k, v in header["model"].items() if k.startswith("conditions.")
}
assert velocity and conditions
payload = {
    "schema": "symmflow_tensor_exact_inference_export_v1",
    "source_checkpoint": str(source), "source_bytes": source.stat().st_size,
    "source_mtime_ns": source.stat().st_mtime_ns, "step": header["step"],
    "velocity_state": velocity, "condition_state": conditions,
    "velocity_config": header["config"]["velocity"],
    "sigma_min": header["config"]["flow"]["sigma_min"],
    "feature_schema": header["feature_schema"],
    "latent_statistics": {k: header["latent_statistics"][k] for k in ("mean", "std")},
    "original_codec_binding_verified": True,
    "original_statistics_binding_verified": True,
    "velocity_weights": "EMA", "condition_weights": "checkpoint",
}
output = root / "comparison_export_20260911"
output.mkdir(exist_ok=True)
path = output / "inference.pt"
torch.save(payload, path)
check = torch.load(path, map_location="cpu", weights_only=True)
for key in ("velocity_state", "condition_state"):
    assert set(check[key]) == set(payload[key])
    assert all(torch.equal(check[key][k], v) for k, v in payload[key].items())
fields = ("pair_id", "patient_id", "earlier_stage", "later_stage", "delta_days",
          "interval_missing", "interval_source", "baseline_clinical", "treatment")
records = [json.loads(line) for line in (root / "imported_latents/pairs.jsonl").open()]
selected = [{k: r[k] for k in fields} for r in records if r["split"] == "val"]
(output / "conditions.json").write_text(json.dumps(selected, indent=2) + "\n")
print(json.dumps({"inference_bytes": path.stat().st_size, "validation_pairs": len(selected),
                  "tensor_exact_readback": True, "output": str(output)}))

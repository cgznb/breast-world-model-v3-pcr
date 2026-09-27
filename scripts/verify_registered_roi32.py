"""CPU audit of the actual dataset, codec, split and observation-only batches."""
from pathlib import Path
import argparse
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
import numpy as np
import torch
from symm_observation.config import load_config, stage_budget
from symm_observation.codec import load_codec
from symm_observation.data import VisitStore, PairStore, PatientSampler
from symm_observation.observation_views import observation_batch
from symm_observation.observation_encoder import ThreePhaseObservationEncoder
from symm_observation.training import validation_records
from symm_observation.utils import read_json, write_json, file_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = Path(args.data)
    cfg = load_config(args.config)
    a, b = VisitStore(data/"visits.json"), PairStore(data/"pairs.json")
    stats = read_json(data/"statistics.json")
    sampler = PatientSampler(a.records("train"), cfg.training.seed)
    rows = sampler.batch(0, 4)
    batch = observation_batch(a, rows, cfg, 0, "cpu")
    assert batch["a"]["clean"].shape == (4, 24, *cfg.observation.crop)
    assert batch["visit_ids"] == [r["visit_id"] for r in rows]
    assert set(batch) == {"a", "b", "phenotype", "domain", "visit_ids"}
    assert (batch["a"]["affine"] != batch["b"]["affine"]).any()
    assert (batch["phenotype"] == -1).all() and (batch["domain"] == -1).all()
    va = validation_records(a.records("val"), 102, True)
    vb = validation_records(b.records("val"), 102, True)
    assert len({r["patient_id"] for r in va}) == len({r["patient_id"] for r in vb}) == 102
    provenance = a.manifest["provenance"]
    codec_path = provenance["codec"]["path"]
    assert file_identity(codec_path) == provenance["codec"]
    codec = load_codec(codec_path)
    # Compare the new module's codec with the previously verified local decoder.
    sys.path.insert(0, str(REPO.parent/"symm-fm-world-v2/src"))
    from symm_world.codec import load_codec as load_previous_codec
    previous_codec = load_previous_codec(codec_path)
    raw = a.read(rows[0])["latent"][None]
    with torch.no_grad():
        actual, expected = codec.decode(raw), previous_codec.decode(raw)
    assert actual.shape == (1, 3, 32, 128, 128)
    assert torch.isfinite(actual).all()
    error = float((actual-expected).abs().max())
    assert error <= 1e-6
    fitted = a.fit_statistics()
    for key in ("mean", "std"):
        np.testing.assert_allclose(fitted[key], stats[key], rtol=0, atol=1e-12)
    report = {"passed": True, "dataset": read_json(data/"audit.json"),
              "different_same_visit_crops": True, "A_labels_absent": True,
              "validation_A_patients": len(va), "validation_B_patients": len(vb),
              "codec_shape": list(actual.shape), "codec_max_absolute_error": error,
              "statistics_replayed": True, "budgets": {s: stage_budget(cfg, s) for s in ("A", "B")}}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

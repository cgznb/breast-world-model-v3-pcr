"""Small synthetic tensors for program-path tests. NEVER patient training data."""
from pathlib import Path
import numpy as np
from .utils import write_json
from .data import PHASES, VISIT_SCHEMA, PAIR_SCHEMA


def make_synthetic(root, shape=(8, 8, 8), seed=111):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    visits, pairs = [], []
    for i in range(7):
        split = "train" if i < 4 else "val" if i < 6 else "test"
        base = rng.normal(size=(24, *shape)).astype(np.float32)
        patient_rows = []
        for t in range(2 if i != 3 else 1):
            latent = base*(1-.05*t)+.1*t+rng.normal(0, .02, base.shape).astype(np.float32)
            name = f"synthetic_{i:02d}_T{t}"
            path = root/(name+".npy"); np.save(path, latent)
            vp = root/(name+"_valid.npy"); np.save(vp, np.ones((3, *shape), np.float32))
            kp = root/(name+"_kinetics.npy")
            # Synthetic measured channels: explicit toy observation differences, not real MRI biomarkers.
            np.save(kp, np.stack((latent[1]-latent[0], latent[2]-latent[1])))
            sp = root/(name+"_seg.npy"); np.save(sp, (latent[0:1] > 1).astype(np.float32))
            row = {"id": name, "patient_id": f"synthetic_{i:02d}", "visit_id": name, "stage": f"T{t}",
                   "split": split, "latent_path": str(path), "valid_path": str(vp),
                   "geometry": {"latent_to_lps": np.diag([2., 2., 2., 1.]).tolist()},
                   "phase_alignment_verified": True, "source_available_grid": True,
                   "kinetic_path": str(kp), "kinetic_provenance": "measured_same_visit_shared_normalization",
                   "segmentation_path": str(sp), "phenotype_label": i % 2, "domain_label": (i//2) % 2}
            visits.append(row); patient_rows.append(row)
        if len(patient_rows) == 2:
            pairs.append({"id": f"pair_{i:02d}", "patient_id": patient_rows[0]["patient_id"], "split": split,
                          "source": patient_rows[0], "target": patient_rows[1],
                          "conditions": {"stage_i": "T0", "stage_j": "T1", "treatment_arm": "synthetic_plan",
                             "hr_status": "positive", "age": 40+i, "delta_days": 30., "interval_verified": True}})
    common = {"phase_order": PHASES, "latent_channels": 24, "codec_id": "synthetic_only_no_patient_codec",
              "provenance": {"synthetic": True, "not_patient_data": True}}
    write_json(root/"visits.json", {**common, "schema": VISIT_SCHEMA, "visits": visits})
    write_json(root/"pairs.json", {**common, "schema": PAIR_SCHEMA, "pairs": pairs})
    return {"visits": str(root/"visits.json"), "pairs": str(root/"pairs.json")}

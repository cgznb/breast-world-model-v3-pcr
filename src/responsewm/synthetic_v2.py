"""Deterministic public engineering fixtures, never clinical training evidence."""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np

from .data import PHASES
from .data_v2 import SCHEMA
from .io import write_json


def make_synthetic_v2(output, shape=(24, 4, 8, 8), patients=8, *, missing_labels=False):
    if patients < 6:
        raise ValueError("Synthetic smoke cohort needs at least six patients for disjoint splits")
    if len(shape) != 4 or shape[0] != 24 or any(not isinstance(v, int) or v < 1 for v in shape):
        raise ValueError("Expected [24,D,H,W] continuous latent fixture shape")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "patient_trajectories_v2.json").exists():
        raise FileExistsError("Preserving existing synthetic v2 cohort")
    rng = np.random.default_rng(321)
    records = []
    for index in range(patients):
        pid = f"synthetic_v2_{index:03d}"
        label = index % 2
        split = "train" if index < patients - 4 else ("val" if index < patients - 2 else "test")
        source = rng.normal(size=shape).astype(np.float32)
        visits = []
        for stage in range(4):
            if stage == 1 and index % 3 == 1:
                continue
            latent = (source * (1 - .1 * stage * (1 + label)) + rng.normal(0, .1, shape)).astype(np.float32)
            path = root / f"{pid}_T{stage}.npy"
            np.save(path, latent)
            visits.append({"stage": stage, "latent": str(path), "available_at": stage,
                           "anatomy_comparable": False})
        records.append({"patient_key": pid, "split": split,
                        "baseline": {"values": [40 + index, label, 1 - label, None],
                                     "known_at": [0, 0, 0, None]},
                        "visits": visits, "events": [], "interval_plans": [],
                        "target": {"pcr": None if missing_labels and index % 3 == 0 else label,
                                   "label_source": "Synthetic fixture outcome; not clinical pathology"}})
    manifest = {"schema": SCHEMA, "time_basis": "stage_index", "canonical_stages": [0, 1, 2, 3],
                "phase_order": PHASES, "clinical_features": ["age_at_screening", "hr_positive", "her2_positive", "mammaprint_binary"],
                "action_features": [], "latent_shape": list(shape),
                "vq_identity": "sha256:" + hashlib.sha256(b"responsewm_v2_synthetic_dummy_codec_no_real_weights").hexdigest(),
                "shared_grid_verified": True, "synthetic": True, "patients": records,
                "provenance": {"warning": "SYNTHETIC ENGINEERING FIXTURE; NO REAL MRI OR CLINICAL EVIDENCE",
                               "codec_policy": "Explicit deterministic dummy identity; no pretrained codec used",
                               "availability_policy": "Synthetic visits are available at their canonical stage"}}
    path = root / "patient_trajectories_v2.json"
    write_json(path, manifest)
    return path

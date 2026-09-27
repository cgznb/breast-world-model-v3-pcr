"""Frozen phase-specific channel moments and unique-visit ROI32 latent caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from .registered_roi32_data import check_disk, file_identity, read_json, verify_identity, write_json
from .registered_roi32_latents import latent_filename, save_latent
from .registered_three_phase_data import LATENT_SHAPE, PHASES, ThreePhaseCrops
from .registered_three_phase_model import SharedPhaseCodec


def fit_statistics(records, read_latent):
    training = [row for row in records if row["fold"] == "train"]
    ids = [row["visit_id"] for row in training]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Statistics require unique training visits")
    sums, squares, count = np.zeros(24), np.zeros(24), 0
    for row in training:
        latent = np.asarray(read_latent(row["visit_id"]), dtype=np.float64)
        if latent.shape != LATENT_SHAPE or not np.isfinite(latent).all():
            raise ValueError("Invalid three-phase training latent")
        flat = latent.reshape(24, -1)
        sums += flat.sum(axis=1)
        squares += np.square(flat).sum(axis=1)
        count += flat.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(0, squares / count - mean**2))
    if not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError("Degenerate three-phase latent channel")
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train", "fit_visit_ids": sorted(ids),
            "phase_order": list(PHASES), "element_count_per_channel": count,
            "scope": "unique_training_visits_separate_moments_for_each_phase_and_channel"}


def contract_for(config, baseline):
    from . import registered_three_phase_data as data
    from . import registered_three_phase_model as model

    return {"schema": config["schema"], "stage": "latents", "configuration": config,
            "inventory": file_identity(Path(config["output_dir"]) / "inventory.json"),
            "codec": file_identity(Path(baseline["output_dir"]) / "vq/best.pt"),
            "runtime_sources": [file_identity(path) for path in (__file__, data.__file__, model.__file__, vq.__file__)]}


@torch.no_grad()
def extract(config, baseline, manifest, device):
    if manifest["missing_sources"]:
        raise ValueError("Registered phases are missing; restore the listed source files before preparing the full cohort")
    contract = contract_for(config, baseline)
    if runtime.stage_complete(config, "latents", contract):
        return True
    root = Path(config["output_dir"]) / "latents"
    if (root / "binding.json").exists():
        if read_json(root / "binding.json") != contract:
            raise ValueError("Prepared latent binding changed")
    else:
        write_json(root / "binding.json", contract)
    check_disk(config, len(manifest["visits"]) * np.prod(LATENT_SHAPE) * 2 + 3 * 1024**3)
    codec, _ = vq.load_frozen(baseline, device)
    shared = SharedPhaseCodec(codec)
    original = {key: value.clone() for key, value in codec.quantizer.state_dict().items()}
    pending = [row for row in manifest["visits"] if not (root / "raw" / latent_filename(row["visit_id"])).is_file()]
    loader = runtime.evaluation_loader(ThreePhaseCrops(baseline, manifest, records=pending), 1,
                                        config["runtime"]["loader_workers"]) if pending else []
    completed = len(manifest["visits"]) - len(pending)
    for batch in loader:
        with runtime.autocast(device):
            latent = shared.encode(batch["image"].to(device)).float().cpu().numpy()
        stored = latent.astype(np.float16)
        if not np.isfinite(stored).all():
            raise FloatingPointError("Three-phase latent overflow in FP16 storage")
        save_latent(root / "raw" / latent_filename(batch["visit_id"][0]), stored[0])
        completed += 1
        if completed == 1 or completed % 25 == 0:
            runtime.log_event(config, "latents", "encoding", completed=completed, total=len(manifest["visits"]))
            runtime.periodic_guard(config, contract)
        if runtime.STOP_REQUESTED:
            return False
    if any(not torch.equal(value, codec.quantizer.state_dict()[key]) for key, value in original.items()):
        raise ValueError("Frozen encoding updated the shared codebook")
    identities = []
    for row in manifest["visits"]:
        path = root / "raw" / latent_filename(row["visit_id"])
        latent = np.load(path, allow_pickle=False)
        if latent.shape != LATENT_SHAPE or latent.dtype != np.float16 or not np.isfinite(latent).all():
            raise ValueError("Invalid prepared phase cache")
        identities.append(file_identity(path))
    statistics = fit_statistics(manifest["visits"], lambda visit: np.load(root / "raw" / latent_filename(visit), allow_pickle=False))
    write_json(root / "statistics.json", statistics)
    write_json(root / "cache_manifest.json", {"files": identities, "codec": contract["codec"]})
    runtime.stage_finished(config, "latents", contract, [root / "statistics.json", root / "cache_manifest.json"])
    return True


class PhaseLatentPairs(Dataset):
    def __init__(self, config, baseline, manifest, split):
        if not runtime.stage_complete(config, "latents", contract_for(config, baseline)):
            raise ValueError("Three-phase latents have not completed preparation")
        self.root = Path(config["output_dir"]) / "latents"
        self.records = [row for row in manifest["pairs"] if row["split"] == split]
        self.statistics = read_json(self.root / "statistics.json")
        expected = sorted(row["visit_id"] for row in manifest["visits"] if row["fold"] == "train")
        if (self.statistics["fit_split"] != "train" or self.statistics["fit_visit_ids"] != expected
                or self.statistics["phase_order"] != list(PHASES)):
            raise ValueError("Latent normalization used a different cohort or phase order")
        self.mean = np.array(self.statistics["mean"], dtype=np.float32).reshape(24, 1, 1, 1)
        self.std = np.array(self.statistics["std"], dtype=np.float32).reshape(24, 1, 1, 1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid frozen phase moments")
        for identity in read_json(self.root / "cache_manifest.json")["files"]:
            verify_identity(identity)

    def __len__(self):
        return len(self.records)

    def read(self, visit_id):
        raw = np.load(self.root / "raw" / latent_filename(visit_id), allow_pickle=False).astype(np.float32)
        if raw.shape != LATENT_SHAPE or not np.isfinite(raw).all():
            raise ValueError("Invalid three-phase cached latent")
        return torch.from_numpy((raw - self.mean) / self.std)

    def __getitem__(self, index):
        row = self.records[index]
        return {"record": row, "earlier_latent": self.read(row["earlier_visit_id"]), "later_latent": self.read(row["later_visit_id"])}

"""Audited, transactional v1 -> multistage-v2 warm starts, never implicit resume."""
from __future__ import annotations

from pathlib import Path
import torch

from .io import digest, load_checkpoint, stable_hash


class MigrationError(ValueError):
    """A rejected migration with a machine-readable report and unchanged model."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def migrate_checkpoint(model, source_path, metadata):
    """Reuse exact encoder/native weights and explicitly map semantic residuals.

    New history, conditions, assimilation, PCR, bridges, predictor deep heads,
    and stage-readiness flags retain their fresh destination initialization.
    Required encoder/image incompatibility aborts before changing any tensor.
    """
    path = Path(source_path)
    payload = load_checkpoint(path)
    if not isinstance(payload, dict) or payload.get("schema") != "responsewm_checkpoint_v1":
        raise ValueError("Warm start requires an explicit responsewm_checkpoint_v1 payload")
    source_config = payload.get("config", {})
    if source_config.get("schema") != "responsewm_v1" or model.cfg.schema != "responsewm_v2":
        raise ValueError("Migration requires a v1 source configuration and v2 destination")
    if source_config.get("network", {}).get("backend") != "native" or model.cfg.network.backend != "native":
        raise ValueError("This audited migration supports native -> native only")
    if source_config.get("network", {}).get("semantic_depth") != 4:
        raise ValueError("The audited semantic mapping requires four source blocks")
    source = payload.get("model")
    if not isinstance(source, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in source.items()):
        raise ValueError("Checkpoint model must be a tensor-only state dictionary")
    current = model.state_dict()
    statistics = metadata.get("statistics", {})
    normalized = {}
    for name in ("latent_mean", "latent_std"):
        if name not in statistics:
            raise ValueError("Migration requires current training-fold latent_mean and latent_std")
        target = current[f"encoder.{name}"]
        value = torch.as_tensor(statistics[name], dtype=target.dtype, device=target.device)
        if value.numel() != target.numel() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid current training-fold {name}")
        if name == "latent_std" and bool((value <= 0).any()):
            raise ValueError("Current training-fold latent_std must be positive")
        normalized[name] = value.reshape_as(target)
    source_meta = payload.get("metadata", {})
    old_contract, new_contract = source_meta.get("data_contract", {}), metadata.get("data_contract", {})
    for name in ("phase_order", "latent_shape", "vq_identity"):
        if name in old_contract and name in new_contract and old_contract[name] != new_contract[name]:
            raise ValueError(f"Incompatible source/destination {name}; migration cannot change latent coordinates")
    report = {
        "schema": "responsewm_migration_report_v2", "status": "pending", "source_path": str(path.resolve()),
        "source_sha256": digest(path), "source_schema": payload["schema"],
        "source_stage": payload.get("stage"), "source_step": payload.get("step"),
        "source_config_digest": payload.get("config_digest", stable_hash(source_config)),
        "source_manifest_digest": payload.get("manifest_digest", source_meta.get("manifest_digest")),
        "destination_config_digest": model.cfg.digest,
        "destination_manifest_digest": metadata.get("manifest_digest"),
        "data_exposure": {
            "source_declaration": source_meta.get("data_exposure", source_meta.get("pretraining_patient_overlap", "unverified")),
            "overlap_audited": False,
            "independent_test_claim": False,
            "note": "Warm-start weights retain source-data exposure; new normalization does not remove that exposure.",
        },
        "normalization": {
            "source": "destination metadata.statistics fitted on current training split",
            "statistics_digest": stable_hash({k: statistics[k] for k in normalized}),
            "source_normalization_ignored": True,
            "representation_recalibration_required": True,
            "teacher_policy": "copy migrated online encoder including current-fold statistics",
        },
        "semantic_mapping": {"layers": {str(i): i for i in range(4)},
                             "new_layers": [4, 5],
                             "modulation": {"old_0_to_5": "new_0_to_5_self_history",
                                            "old_6_to_8": "new_9_to_11_FFN",
                                            "new_6_to_8": "zero_action_residual"}},
        "bridge_initialization": "fresh destination parameters; no old bottleneck bridge transferred",
        "licenses": [
            {"component": "native/encoder local adaptations", "licenses": ["MIT", "BSD-3-Clause"],
             "source": "docs/SOURCES.md"},
            {"component": "DiT-derived semantic modulation", "license": "CC-BY-NC-4.0",
             "upstream_commit": "ed81ce2229091fd4ecc9a223645f95cf379d582b"},
            {"component": "upstream VQ coordinate provenance", "license": "CC-BY-NC-4.0",
             "note": "VQ weights are not transferred by this operation"},
        ],
        "modules": {}, "unused_source": [], "errors": [],
    }
    merged, consumed = dict(current), set()

    def record(key, status, **detail):
        module = key.split(".")[0]
        if key.startswith("velocity."):
            module = ".".join(key.split(".")[:2])
        module_report = report["modules"].setdefault(module, {"loaded": [], "new": [], "mismatch": [], "derived": []})
        module_report[status].append({"key": key, **detail})

    for key, target in current.items():
        if key.startswith("target_encoder."):
            continue
        if key in {"encoder.latent_mean", "encoder.latent_std"}:
            name = key.split(".")[-1]
            merged[key] = normalized[name]
            record(key, "derived", source="current_training_statistics")
            continue
        required = key.startswith("encoder.") or key.startswith("velocity.image.")
        semantic = key.startswith(("velocity.semantic_in.", "velocity.token_position", "velocity.time.",
                                   "velocity.semantic_out.", "velocity.spatial_projection."))
        block = key.startswith("velocity.blocks.") and int(key.split(".")[2]) < 4
        if not (required or semantic or block):
            record(key, "new", reason="new_v2_module_or_changed_semantics")
            continue
        if key not in source:
            record(key, "new", reason="absent_in_source")
            if required:
                report["errors"].append(f"Missing required source tensor: {key}")
            continue
        value = source[key]
        if block and ".modulation.1." in key:
            dim = model.cfg.encoder.dim
            expected = (9 * dim, *target.shape[1:])
            if tuple(value.shape) == expected and tuple(target.shape) == (12 * dim, *target.shape[1:]):
                expanded = torch.zeros_like(target)
                expanded[:6 * dim] = value[:6 * dim].to(expanded)
                expanded[9 * dim:] = value[6 * dim:].to(expanded)
                value = expanded
        if value.shape != target.shape or not bool(torch.isfinite(value).all()):
            record(key, "mismatch", source_shape=list(source[key].shape), destination_shape=list(target.shape),
                   reason="shape_or_nonfinite", policy="abort" if required else "retain_fresh")
            if required:
                report["errors"].append(f"Incompatible required source tensor: {key}")
            continue
        merged[key] = value
        consumed.add(key)
        record(key, "loaded", source_key=key, source_shape=list(source[key].shape),
               destination_shape=list(target.shape), mapped_modulation=block and ".modulation.1." in key)
    for key in current:
        if key.startswith("target_encoder."):
            online_key = "encoder." + key[len("target_encoder."):]
            merged[key] = merged[online_key]
            record(key, "derived", source=online_key)
    for key in sorted(set(source) - consumed):
        if key.endswith(("latent_mean", "latent_std")):
            reason = "current_training_statistics_take_precedence"
        elif key.startswith("target_encoder."):
            reason = "teacher_rebuilt_from_migrated_online_encoder"
        elif key in current:
            reason = "new_contract_or_reported_mismatch"
        else:
            reason = "no_semantically_equivalent_destination"
        report["unused_source"].append({"key": key, "shape": list(source[key].shape), "reason": reason})
    source_required = {k for k in source if k.startswith(("encoder.", "velocity.image."))}
    extra_required = sorted(source_required - set(current))
    if extra_required:
        report["errors"].append(f"Unexpected source encoder/image keys: {extra_required}")
    report["counts"] = {status: sum(len(module[status]) for module in report["modules"].values())
                        for status in ("loaded", "new", "mismatch", "derived")}
    report["counts"]["unused_source"] = len(report["unused_source"])
    if report["errors"]:
        report["status"] = "rejected"
        raise MigrationError("Required encoder/image migration failed; destination is unchanged", report)
    model.load_state_dict(merged, strict=True)
    report["status"] = "applied"
    report["validation"] = {"strict_destination_state_load": True,
                            "normalization_from_current_metadata": True,
                            "whole_model_numerical_equivalence_claim": False,
                            "optimizer_or_training_progress_resumed": False}
    return report

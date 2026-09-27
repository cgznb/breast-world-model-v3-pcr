from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml

from .contracts import (
    CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT,
    EPSILON_PREDICTION_TYPE,
    REGISTERED_LARGE_PAPER_ARCHITECTURE,
    X0_PREDICTION_TYPE,
)
from .latent_statistics import (
    LATENT_CHANNEL_STATISTICS_SCHEMA,
    StreamingChannelMoments,
    sha256_file,
    write_latent_statistics,
)


DEFAULT_X0_RUN_DIRECTORY = Path(
    "/path/to/research/MAM/MeWM-ISPY2/runs/"
    "registered_strict_a_mewm_x0_channelstd_v1"
)
_EXPECTED_LATENT_SHAPE = (8, 24, 64, 64)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unique_train_endpoint_visit_ids(loaded: Any) -> tuple[str, ...]:
    records = getattr(loaded, "records", None)
    if not isinstance(records, tuple):
        raise TypeError("loaded transition records must be an exact tuple")
    identifiers = {
        visit_id
        for record in records
        if record.fold == "train"
        for visit_id in (record.source_visit_id, record.target_visit_id)
    }
    if not identifiers:
        raise ValueError("training fold has no source or target visits")
    validation_identifiers = {
        visit_id
        for record in records
        if record.fold == "val"
        for visit_id in (record.source_visit_id, record.target_visit_id)
    }
    overlap = identifiers & validation_identifiers
    if overlap:
        raise ValueError("train and validation endpoint visit identities overlap")
    return tuple(sorted(identifiers))


@torch.no_grad()
def compute_train_latent_statistics(
    vqgan: torch.nn.Module,
    *,
    visit_ids: tuple[str, ...],
    visit_loader: Callable[[str], Any],
    vqgan_sha256: str,
    data_contract_sha256: str,
    device: torch.device,
    encode_batch_size: int,
) -> Any:
    if not visit_ids or visit_ids != tuple(sorted(set(visit_ids))):
        raise ValueError("latent statistics visit IDs must be sorted and unique")
    if not callable(visit_loader):
        raise TypeError("visit_loader must be callable")
    if type(encode_batch_size) is not int or encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be a positive integer")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    vqgan = vqgan.to(device).eval()
    vqgan.requires_grad_(False)
    moments = StreamingChannelMoments(channels=_EXPECTED_LATENT_SHAPE[0])
    for start in range(0, len(visit_ids), encode_batch_size):
        batch_ids = visit_ids[start : start + encode_batch_size]
        images = torch.stack(
            [visit_loader(visit_id).image.float() for visit_id in batch_ids]
        ).to(device=device)
        latent = vqgan.encode_continuous(images)
        if tuple(latent.shape[1:]) != _EXPECTED_LATENT_SHAPE:
            raise ValueError(
                "VQGAN continuous latent shape is not [8,24,64,64]"
            )
        moments.update(latent)
        del images, latent
    return moments.finalize(
        visit_ids=visit_ids,
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=data_contract_sha256,
        latent_shape_czyx=_EXPECTED_LATENT_SHAPE,
    )


def _x0_config_payload(
    source: dict[str, Any],
    *,
    output_directory: Path,
    statistics_path: Path,
    statistics_sha256: str,
    physical_batch_size: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    diffusion = payload.get("diffusion")
    runtime = payload.get("runtime")
    if not isinstance(diffusion, dict) or not isinstance(runtime, dict):
        raise ValueError("source config diffusion/runtime mappings are missing")
    if diffusion.get("denoiser_architecture") != REGISTERED_LARGE_PAPER_ARCHITECTURE:
        raise ValueError("x0 preparation requires registered-large paper diffusion")
    if diffusion.get("prediction_type") != EPSILON_PREDICTION_TYPE:
        raise ValueError("x0 preparation source config must be the epsilon baseline")
    diffusion["prediction_type"] = X0_PREDICTION_TYPE
    diffusion["latent_contract"] = CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT
    diffusion["x0_objective"] = "l2"
    diffusion["latent_statistics"] = {
        "path": str(statistics_path.resolve()),
        "sha256": statistics_sha256,
        "schema": LATENT_CHANNEL_STATISTICS_SCHEMA,
    }
    runtime["output_dir"] = str(output_directory.resolve())
    training = payload.setdefault("diffusion_training", {})
    if not isinstance(training, dict):
        raise ValueError("source config diffusion_training must be a mapping")
    training["batch_size"] = physical_batch_size
    return payload


def prepare_x0_latents(
    source_config_path: str | Path,
    vqgan_checkpoint_path: str | Path,
    output_directory: str | Path = DEFAULT_X0_RUN_DIRECTORY,
    *,
    device: str | torch.device | None = None,
    encode_batch_size: int = 1,
    autotune: Callable[[Path, Path, Path], tuple[int, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    from .config import load_experiment_config
    from .workflows import _load_data, _visit_loader, load_mri_vqgan

    source_config = load_experiment_config(source_config_path)
    if source_config.diffusion.prediction_type != EPSILON_PREDICTION_TYPE:
        raise ValueError("x0 preparation source must use epsilon prediction")
    loaded = _load_data(source_config)
    visit_ids = unique_train_endpoint_visit_ids(loaded)
    checkpoint = Path(vqgan_checkpoint_path).resolve()
    vqgan_sha256 = sha256_file(checkpoint)
    contract = source_config.diffusion.paper
    if contract is None or contract.vqgan_sha256 != vqgan_sha256:
        raise ValueError("x0 preparation VQGAN does not match the pinned source config")
    vqgan = load_mri_vqgan(
        checkpoint,
        expected_data_contract_sha256=loaded.data_contract_sha256,
        expected_data_backend=loaded.backend,
        expected_numeric_contract=contract.vqgan_numeric_contract,
    )
    resolved_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    statistics = compute_train_latent_statistics(
        vqgan,
        visit_ids=visit_ids,
        visit_loader=_visit_loader(source_config, loaded),
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=loaded.data_contract_sha256,
        device=resolved_device,
        encode_batch_size=encode_batch_size,
    )
    vqgan.to("cpu")
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    output = Path(output_directory).resolve()
    statistics_path = output / "latent_statistics.json"
    statistics_sha256 = write_latent_statistics(statistics_path, statistics)
    provisional_path = output / "resolved" / ".diffusion.preparing.yaml"
    final_config_path = output / "resolved" / "diffusion.yaml"
    source_payload = copy.deepcopy(source_config.raw)
    provisional_payload = _x0_config_payload(
        source_payload,
        output_directory=output,
        statistics_path=statistics_path,
        statistics_sha256=statistics_sha256,
        physical_batch_size=1,
    )
    _atomic_yaml_write(provisional_path, provisional_payload)
    try:
        if autotune is None:
            from .x0_batch_autotune import autotune_registered_x0_batch

            autotune = autotune_registered_x0_batch
        selected_batch_size, batch_report = autotune(
            provisional_path,
            checkpoint,
            output,
        )
        final_payload = _x0_config_payload(
            source_payload,
            output_directory=output,
            statistics_path=statistics_path,
            statistics_sha256=statistics_sha256,
            physical_batch_size=selected_batch_size,
        )
        _atomic_yaml_write(final_config_path, final_payload)
    finally:
        provisional_path.unlink(missing_ok=True)

    report = {
        "schema_version": "mewm_ispy2_x0_latent_preparation_v1",
        "source_config": str(Path(source_config_path).resolve()),
        "resolved_config": str(final_config_path),
        "latent_statistics": str(statistics_path),
        "latent_statistics_sha256": statistics_sha256,
        "vqgan_checkpoint": str(checkpoint),
        "vqgan_sha256": vqgan_sha256,
        "data_contract_sha256": loaded.data_contract_sha256,
        "visit_count": statistics.visit_count,
        "visit_ids_sha256": statistics.visit_ids_sha256,
        "latent_shape_czyx": list(statistics.latent_shape_czyx),
        "encode_batch_size": encode_batch_size,
        "physical_batch_size": selected_batch_size,
        "batch_autotune_report": batch_report,
    }
    _atomic_json_write(output / "latent_statistics_report.json", report)
    return report


__all__ = [
    "DEFAULT_X0_RUN_DIRECTORY",
    "compute_train_latent_statistics",
    "prepare_x0_latents",
    "unique_train_endpoint_visit_ids",
]

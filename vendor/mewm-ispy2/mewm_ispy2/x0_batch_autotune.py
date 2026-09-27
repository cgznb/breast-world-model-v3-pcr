from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .contracts import X0_PREDICTION_TYPE


DEFAULT_HEADROOM_FRACTION = 0.05


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


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if type(value) is dict:
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if type(value) is list:
        return [_move_to_device(item, device) for item in value]
    if type(value) is tuple:
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
    )


def select_largest_acceptable_batch(
    probes: list[dict[str, Any]],
) -> int:
    successes = [
        probe["batch_size"]
        for probe in probes
        if probe.get("status") == "success"
        and probe.get("within_headroom") is True
    ]
    if not successes:
        raise RuntimeError("no physical batch size completed within the VRAM headroom")
    return max(successes)


def _probe_batch(
    *,
    system: Any,
    optimizer: torch.optim.Optimizer,
    dataset: Any,
    collate: Any,
    batch_size: int,
    device: torch.device,
    total_memory: int,
    headroom_fraction: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    denoiser_calls = 0

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal denoiser_calls
        denoiser_calls += 1

    handle = system.model.denoiser.register_forward_hook(count_forward)
    try:
        items = [dataset[index % len(dataset)] for index in range(batch_size)]
        batch = _move_to_device(collate(items), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = system.model.training_objective(batch, augment_source=True)
        if denoiser_calls != 1:
            raise RuntimeError("x0 batch probe must execute exactly one denoiser forward")
        output.total_loss.backward()
        system.on_after_backward()
        parameters = [
            parameter
            for group in system.trainable_parameter_groups().values()
            for parameter in group
        ]
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("x0 batch probe gradient norm is nonfinite")
        optimizer.step()
        system.model.update_ema()
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        limit = int(total_memory * (1.0 - headroom_fraction))
        return {
            "batch_size": batch_size,
            "status": "success",
            "within_headroom": peak_reserved <= limit,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "headroom_limit_bytes": limit,
            "denoiser_forward_count": denoiser_calls,
            "gradient_norm_before_clip": gradient_norm,
            "wall_time_seconds": time.perf_counter() - started,
        }
    except BaseException as error:
        if not _is_cuda_oom(error):
            raise
        return {
            "batch_size": batch_size,
            "status": "oom",
            "within_headroom": False,
            "error_type": type(error).__name__,
            "wall_time_seconds": time.perf_counter() - started,
        }
    finally:
        handle.remove()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def autotune_registered_x0_batch(
    config_path: str | Path,
    vqgan_checkpoint_path: str | Path,
    output_directory: str | Path,
    *,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    maximum_batch_size: int = 256,
) -> tuple[int, dict[str, Any]]:
    from .config import load_experiment_config
    from .workflows import _build_paper_training_components, _load_data

    if not torch.cuda.is_available():
        raise RuntimeError("x0 physical batch autotune requires CUDA")
    if (
        type(headroom_fraction) is not float
        or not math.isfinite(headroom_fraction)
        or not 0.0 < headroom_fraction < 1.0
    ):
        raise ValueError("headroom_fraction must be a finite fraction")
    if type(maximum_batch_size) is not int or maximum_batch_size <= 0:
        raise ValueError("maximum_batch_size must be positive")
    config = load_experiment_config(config_path)
    if config.diffusion.prediction_type != X0_PREDICTION_TYPE:
        raise ValueError("batch autotune requires an x0 diffusion config")
    loaded = _load_data(config)
    torch.manual_seed(config.runtime.seed)
    torch.cuda.manual_seed_all(config.runtime.seed)
    output = Path(output_directory)
    system, train_loader, _val_loader, _callbacks = _build_paper_training_components(
        config,
        type("Args", (), {"vqgan_checkpoint": str(vqgan_checkpoint_path)})(),
        loaded,
        output / "batch_autotune_work",
    )
    device = torch.device("cuda")
    system = system.to(device).train()
    optimizer = system.configure_optimizers()
    dataset = train_loader.dataset
    collate = train_loader.collate_fn
    properties = torch.cuda.get_device_properties(device)
    total_memory = int(properties.total_memory)
    free_before, runtime_total = torch.cuda.mem_get_info(device)
    if int(runtime_total) != total_memory:
        total_memory = int(runtime_total)
    cap = min(maximum_batch_size, len(dataset))
    probes: list[dict[str, Any]] = []
    attempted: set[int] = set()
    largest_success = 0
    first_failure: int | None = None
    candidate = 1
    while candidate <= cap:
        probe = _probe_batch(
            system=system,
            optimizer=optimizer,
            dataset=dataset,
            collate=collate,
            batch_size=candidate,
            device=device,
            total_memory=total_memory,
            headroom_fraction=headroom_fraction,
        )
        probes.append(probe)
        attempted.add(candidate)
        if probe["status"] == "success" and probe["within_headroom"]:
            largest_success = candidate
            if candidate == cap:
                break
            candidate = min(candidate * 2, cap)
            if candidate in attempted:
                break
            continue
        first_failure = candidate
        break
    if largest_success == 0:
        selected = select_largest_acceptable_batch(probes)
    else:
        upper = (first_failure - 1) if first_failure is not None else cap
        for candidate in range(largest_success + 1, upper + 1):
            if candidate in attempted:
                continue
            probe = _probe_batch(
                system=system,
                optimizer=optimizer,
                dataset=dataset,
                collate=collate,
                batch_size=candidate,
                device=device,
                total_memory=total_memory,
                headroom_fraction=headroom_fraction,
            )
            probes.append(probe)
            attempted.add(candidate)
            if probe["status"] != "success" or not probe["within_headroom"]:
                break
        selected = select_largest_acceptable_batch(probes)
    selected_probe = next(
        probe for probe in probes if probe["batch_size"] == selected
    )
    report = {
        "schema_version": "mewm_ispy2_x0_batch_autotune_v1",
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": total_memory,
        "gpu_free_memory_before_bytes": int(free_before),
        "pytorch_cuda_alloc_conf": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            os.environ.get("PYTORCH_ALLOC_CONF"),
        ),
        "precision": "bf16-mixed",
        "optimizer": "Adam",
        "optimizer_step_per_probe": True,
        "gradient_clip_val": 1.0,
        "ema_decay": config.diffusion.ema_decay,
        "accumulate_grad_batches": 1,
        "headroom_fraction": headroom_fraction,
        "maximum_batch_size": cap,
        "selected_physical_batch_size": selected,
        "selected_peak_allocated_bytes": selected_probe["peak_allocated_bytes"],
        "selected_peak_reserved_bytes": selected_probe["peak_reserved_bytes"],
        "probes": sorted(probes, key=lambda value: value["batch_size"]),
    }
    _atomic_json_write(output / "batch_autotune.json", report)
    del optimizer, system
    torch.cuda.empty_cache()
    return selected, report


__all__ = [
    "DEFAULT_HEADROOM_FRACTION",
    "autotune_registered_x0_batch",
    "select_largest_acceptable_batch",
]

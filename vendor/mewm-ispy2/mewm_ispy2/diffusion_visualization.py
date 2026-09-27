from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import time
import traceback
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .backend import load_transition_records, sha256_file
from .cache import MOTFMROICache
from .checkpoint import CheckpointIdentity, load_diffusion_checkpoint
from .conditioning import (
    MEDGEMMA_MODEL_ID,
    MEDGEMMA_REVISION,
    ISPY2Conditioner,
    SharedMedGemmaTower,
)
from .config import load_experiment_config
from .contracts import PAPER_FAITHFUL_ARCHITECTURE, TransitionRecord
from .data import PreparedVisit
from .diffusion import ConditionalLatentDiffusion, DiffusionSample
from .paper_checkpoint import (
    PAPER_CHECKPOINT_SCHEMA_VERSION,
    PaperCheckpointIdentity,
)
from .vqgan_visualization import orthogonal_planes, tumor_centroid


_CHECKPOINT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")
_CT_CHECKPOINT_PATTERN = re.compile(r"step-(\d+)\.ckpt")
PLANE_NAMES = ("axial", "coronal", "sagittal")
FIGURE_SIZE = (12.0, 9.0)
FIGURE_DPI = 160
DEFAULT_CONFIG_PATH = Path("configs/ispy2_current_ct_aligned_weak_gan_deep_b1.yaml")
DEFAULT_VQGAN_CHECKPOINT_PATH = Path(
    "runs/current_dce0_ct_aligned_weak_gan_deep_b1/vqgan/checkpoints/composite/"
    "best-composite-42-50654.ckpt"
)
DEFAULT_DIFFUSION_CHECKPOINT_DIRECTORY = Path(
    "runs/current_dce0_ct_aligned_weak_gan_deep_b1/diffusion/checkpoints"
)
DEFAULT_OUTPUT_ROOT = Path(
    "runs/current_dce0_ct_aligned_weak_gan_deep_b1/diffusion/test_visualizations"
)


@dataclass(frozen=True)
class CheckpointSelection:
    path: Path
    epoch: int
    step: int


@dataclass(frozen=True)
class SelectedTestCase:
    record: TransitionRecord
    source: PreparedVisit


@dataclass(frozen=True)
class DiffusionTestExample:
    selected: SelectedTestCase
    target: PreparedVisit
    generated: torch.Tensor
    centroid_zyx: tuple[int, int, int]
    seed: int
    metrics: Mapping[str, float]
    latent_diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDiffusionModel:
    model: Any
    identity: CheckpointIdentity | PaperCheckpointIdentity
    global_step: int


def select_top_checkpoint(directory: str | Path) -> CheckpointSelection:
    checkpoint_directory = Path(directory)
    legacy_matches: list[CheckpointSelection] = []
    ct_matches: list[CheckpointSelection] = []
    for path in checkpoint_directory.iterdir():
        if not path.is_file() or path.name == "last.ckpt":
            continue
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            legacy_matches.append(
                CheckpointSelection(
                    path=path.resolve(),
                    epoch=int(match.group(1)),
                    step=int(match.group(2)),
                )
            )
            continue
        ct_match = _CT_CHECKPOINT_PATTERN.fullmatch(path.name)
        if ct_match is not None:
            ct_matches.append(
                CheckpointSelection(
                    path=path.resolve(), epoch=-1, step=int(ct_match.group(1))
                )
            )
    if legacy_matches:
        if len(legacy_matches) != 1:
            raise ValueError("checkpoint directory must contain exactly one top-1 checkpoint")
        return legacy_matches[0]
    if ct_matches:
        return max(ct_matches, key=lambda candidate: candidate.step)
    if not legacy_matches:
        raise ValueError("checkpoint directory must contain exactly one top-1 checkpoint")
    raise AssertionError("unreachable checkpoint selection state")


def select_test_cases(
    records: Sequence[TransitionRecord],
    visit_loader: Callable[[str], PreparedVisit],
) -> list[SelectedTestCase]:
    test_records = sorted(
        (record for record in records if record.fold == "test"),
        key=lambda record: record.transition_id,
    )
    selected: list[SelectedTestCase] = []
    for stage_id in (1, 2, 3):
        for record in test_records:
            if record.stage_id != stage_id:
                continue
            source = visit_loader(record.source_visit_id)
            if bool((source.mask > 0).any()):
                selected.append(SelectedTestCase(record=record, source=source))
                break
        else:
            raise ValueError(f"test split has no eligible transition for stage {stage_id}")
    return selected


def _validate_prepared_visit(prepared: PreparedVisit, name: str) -> None:
    expected_shape = (1, 128, 128, 128)
    if tuple(prepared.image.shape) != expected_shape:
        raise ValueError(f"{name} image must have shape [1,128,128,128]")
    if prepared.mask.shape != prepared.image.shape:
        raise ValueError(f"{name} image and mask shapes must match")
    if not torch.isfinite(prepared.image).all():
        raise ValueError(f"{name} image must contain only finite values")
    if not torch.isfinite(prepared.mask).all():
        raise ValueError(f"{name} mask must contain only finite values")


def _validate_generated(generated: torch.Tensor) -> None:
    if tuple(generated.shape) != (1, 1, 128, 128, 128):
        raise ValueError("generated tensor must have shape [1,1,128,128,128]")
    if not torch.isfinite(generated).all():
        raise ValueError("generated tensor must contain only finite values")


def _display_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    generated_display = ((generated.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    target_display = ((target.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    difference = generated_display - target_display
    mae = float(difference.abs().mean())
    mse = float(difference.square().mean())
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    return {"mae": mae, "mse": mse, "psnr": psnr}


def generate_test_example(
    selected: SelectedTestCase,
    *,
    target_loader: Callable[[str], PreparedVisit],
    model: Any,
    device: torch.device,
    seed: int,
) -> DiffusionTestExample:
    source = selected.source
    record = selected.record
    if source.visit_id != record.source_visit_id:
        raise ValueError("selected source visit ID does not match the transition")
    _validate_prepared_visit(source, "source")
    if not bool((source.mask > 0).any()):
        raise ValueError("source tumor mask is empty")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    source_dce0 = source.image.unsqueeze(0).to(device=device, dtype=torch.float32)
    source_mask = source.mask.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        sample_arguments = (
            source_dce0,
            source_mask,
            [record.action_text],
            [record.clinical_text],
            torch.tensor([float(record.delta_days)], device=device),
            torch.tensor([record.stage_id], dtype=torch.long, device=device),
        )
        if hasattr(model, "sample_with_diagnostics"):
            sample_result = model.sample_with_diagnostics(*sample_arguments)
        else:
            sample_result = model.sample(*sample_arguments)
    latent_diagnostics: dict[str, float] = {}
    if isinstance(sample_result, DiffusionSample):
        generated_device = sample_result.image
        latent_diagnostics = {
            "unique_code_count": float(sample_result.unique_code_count),
            "effective_code_count": float(sample_result.effective_code_count),
        }
    else:
        generated_device = sample_result
    if not isinstance(generated_device, torch.Tensor):
        raise ValueError("generated output must be a tensor")
    generated = generated_device.detach().to(device="cpu", dtype=torch.float32)
    _validate_generated(generated)

    target = target_loader(record.target_visit_id)
    if target.visit_id != record.target_visit_id:
        raise ValueError("loaded target visit ID does not match the transition")
    _validate_prepared_visit(target, "target")
    metrics = _display_metrics(generated[0], target.image)
    source_metrics = _display_metrics(source.image, target.image)
    metrics.update({f"source_{name}": value for name, value in source_metrics.items()})
    return DiffusionTestExample(
        selected=selected,
        target=target,
        generated=generated,
        centroid_zyx=tumor_centroid(source.mask),
        seed=seed,
        metrics=metrics,
        latent_diagnostics=latent_diagnostics,
    )


def diffusion_plane_arrays(
    example: DiffusionTestExample,
    plane_name: str,
) -> dict[str, np.ndarray]:
    if plane_name not in PLANE_NAMES:
        raise ValueError(f"unsupported plane: {plane_name}")
    source = example.selected.source
    target = example.target
    generated = example.generated
    if generated.ndim != 5 or generated.shape[:2] != (1, 1):
        raise ValueError("generated tensor must have shape [1,1,Z,Y,X]")
    generated_volume = generated[0]
    if source.image.shape != target.image.shape or source.image.shape != generated_volume.shape:
        raise ValueError("source, generated, and target shapes must match")
    if source.mask.shape != source.image.shape:
        raise ValueError("source image and mask shapes must match")
    for name, volume in (
        ("source", source.image),
        ("generated", generated_volume),
        ("target", target.image),
        ("mask", source.mask),
    ):
        if not torch.isfinite(volume).all():
            raise ValueError(f"{name} volume must contain only finite values")

    source_plane = orthogonal_planes(source.image, example.centroid_zyx)[plane_name]
    generated_plane = orthogonal_planes(generated_volume, example.centroid_zyx)[
        plane_name
    ]
    target_plane = orthogonal_planes(target.image, example.centroid_zyx)[plane_name]
    mask_plane = orthogonal_planes(source.mask, example.centroid_zyx)[plane_name]
    source_display = ((source_plane.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    generated_display = ((generated_plane.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    target_display = ((target_plane.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    arrays = {
        "source": source_display,
        "generated": generated_display,
        "target": target_display,
        "error": (generated_display - target_display).abs(),
        "mask": mask_plane,
    }
    return {
        name: values.detach().to(device="cpu", dtype=torch.float32).numpy()
        for name, values in arrays.items()
    }


def _render_plane_row(
    axes: np.ndarray,
    example: DiffusionTestExample,
    plane_name: str,
    *,
    show_titles: bool,
) -> None:
    arrays = diffusion_plane_arrays(example, plane_name)
    panels = (
        ("source", "gray"),
        ("generated", "gray"),
        ("target", "gray"),
        ("error", "magma"),
    )
    titles = ("Source DCE0", "Generated future", "Real target", "Absolute error")
    for index, (axis, (name, color_map)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        axis.imshow(
            arrays[name],
            cmap=color_map,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
        )
        if name == "source" and np.any(arrays["mask"] > 0):
            axis.contour(
                arrays["mask"],
                levels=[0.5],
                colors=["#00FFFF"],
                linewidths=1.0,
                origin="lower",
            )
        if show_titles:
            axis.set_title(titles[index], fontsize=10)
        axis.axis("off")
    axes[0].text(
        -0.08,
        0.5,
        plane_name.capitalize(),
        rotation=90,
        va="center",
        ha="center",
        transform=axes[0].transAxes,
        fontsize=10,
    )


def _save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=FIGURE_DPI)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def detail_filename(example: DiffusionTestExample) -> str:
    return f"case-stage-{example.selected.record.stage_id}.png"


def raw_tensor_filename(example: DiffusionTestExample) -> str:
    return f"case-stage-{example.selected.record.stage_id}-generated.pt"


def _checkpoint_label(checkpoint: CheckpointSelection | None) -> str:
    if checkpoint is None:
        return "checkpoint not specified"
    return f"checkpoint epoch {checkpoint.epoch}, step {checkpoint.step}"


def render_detail(
    example: DiffusionTestExample,
    output_path: str | Path,
    *,
    checkpoint: CheckpointSelection | None = None,
) -> Path:
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("detail output must be a PNG file")
    figure, axes = plt.subplots(
        3,
        4,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
        squeeze=False,
    )
    for row, plane_name in enumerate(PLANE_NAMES):
        _render_plane_row(
            axes[row], example, plane_name, show_titles=row == 0
        )
    record = example.selected.record
    figure.suptitle(
        f"Stage {record.stage_id} | {record.transition_type} | "
        f"interval {record.delta_days} days | {_checkpoint_label(checkpoint)} | "
        f"MAE {float(example.metrics['mae']):.5f}",
        fontsize=12,
    )
    _save_figure(figure, output)
    return output


def _ordered_examples(
    examples: Sequence[DiffusionTestExample],
) -> list[DiffusionTestExample]:
    ordered = sorted(examples, key=lambda example: example.selected.record.stage_id)
    if [example.selected.record.stage_id for example in ordered] != [1, 2, 3]:
        raise ValueError("overview requires exactly one example for each stage")
    return ordered


def render_overview(
    examples: Sequence[DiffusionTestExample],
    output_path: str | Path,
    *,
    checkpoint: CheckpointSelection | None = None,
) -> Path:
    ordered = _ordered_examples(examples)
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("overview output must be a PNG file")
    figure, axes = plt.subplots(
        3,
        4,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
        squeeze=False,
    )
    for row, example in enumerate(ordered):
        _render_plane_row(axes[row], example, "axial", show_titles=row == 0)
        record = example.selected.record
        axes[row, 0].text(
            0.0,
            1.08,
            f"Stage {record.stage_id} | {record.transition_type} | "
            f"{record.delta_days} days | MAE {float(example.metrics['mae']):.5f}",
            transform=axes[row, 0].transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
        )
    figure.suptitle(
        f"Diffusion test-set examples | {_checkpoint_label(checkpoint)}",
        fontsize=14,
    )
    _save_figure(figure, output)
    return output


def _validate_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} SHA256 is invalid")


def _case_manifest_row(example: DiffusionTestExample) -> dict[str, Any]:
    generated = example.generated
    source = example.selected.source.image
    target = example.target.image
    record = example.selected.record
    return {
        "transition_id": record.transition_id,
        "source_visit_id": record.source_visit_id,
        "target_visit_id": record.target_visit_id,
        "transition_type": record.transition_type,
        "stage_id": record.stage_id,
        "delta_days": record.delta_days,
        "seed": example.seed,
        "centroid_zyx": list(example.centroid_zyx),
        "source": {"min": float(source.min()), "max": float(source.max())},
        "generated": {
            "shape": list(generated.shape),
            "dtype": str(generated.dtype).removeprefix("torch."),
            "min": float(generated.min()),
            "max": float(generated.max()),
        },
        "target": {"min": float(target.min()), "max": float(target.max())},
        "display_metrics": {
            name: float(value) if math.isfinite(float(value)) else None
            for name, value in example.metrics.items()
        },
        "latent_diagnostics": {
            name: float(value) for name, value in example.latent_diagnostics.items()
        },
        "files": {
            "detail": detail_filename(example),
            "raw_generated": raw_tensor_filename(example),
        },
    }


def write_manifest(
    examples: Sequence[DiffusionTestExample],
    output_path: str | Path,
    *,
    config_path: Path,
    checkpoint: CheckpointSelection,
    checkpoint_sha256: str,
    checkpoint_identity: CheckpointIdentity | PaperCheckpointIdentity,
    checkpoint_global_step: int,
    vqgan_checkpoint_path: Path,
    vqgan_sha256: str,
    data_backend: str,
    data_contract_sha256: str,
) -> Path:
    ordered = _ordered_examples(examples)
    _validate_sha256(checkpoint_sha256, "diffusion checkpoint")
    _validate_sha256(vqgan_sha256, "VQGAN checkpoint")
    _validate_sha256(data_contract_sha256, "data contract")
    if checkpoint_identity.vqgan_sha256 != vqgan_sha256:
        raise ValueError("VQGAN identity does not match the diffusion checkpoint")
    if checkpoint_identity.data_contract_sha256 != data_contract_sha256:
        raise ValueError("data contract does not match the diffusion checkpoint")
    if checkpoint_identity.data_backend != data_backend:
        raise ValueError("data backend does not match the diffusion checkpoint")
    checkpoint_payload: dict[str, Any] = {
        "path": str(checkpoint.path.resolve()),
        "epoch": checkpoint.epoch,
        "step": checkpoint.step,
        "global_step": int(checkpoint_global_step),
        "sha256": checkpoint_sha256,
    }
    if type(checkpoint_identity) is PaperCheckpointIdentity:
        checkpoint_payload["schema_version"] = PAPER_CHECKPOINT_SCHEMA_VERSION
    payload = {
        "schema_version": "mewm_ispy2_diffusion_test_visualizations_v1",
        "config_path": str(config_path.resolve()),
        "checkpoint": checkpoint_payload,
        "checkpoint_identity": asdict(checkpoint_identity),
        "vqgan": {
            "path": str(vqgan_checkpoint_path.resolve()),
            "sha256": vqgan_sha256,
        },
        "data": {
            "backend": data_backend,
            "contract_sha256": data_contract_sha256,
            "fold": "test",
        },
        "sampling": {
            "sampler": checkpoint_identity.sampler,
            "timesteps": checkpoint_identity.timesteps,
            "latent_contract": checkpoint_identity.latent_contract,
            "denoiser_architecture": checkpoint_identity.denoiser_architecture,
            "denoiser_input_channels": checkpoint_identity.denoiser_input_channels,
            "semantic_channels": checkpoint_identity.semantic_channels,
            "prediction_type": checkpoint_identity.prediction_type,
            "denoiser": "ema",
            "batch_size": 1,
        },
        "selection_rule": (
            "stable-sort test transitions by transition_id and select the first "
            "nonempty source mask in each stage"
        ),
        "display_windows": {
            "mri": [0.0, 1.0],
            "absolute_error": [0.0, 1.0],
        },
        "files": {
            "overview": "overview.png",
            "details": [detail_filename(example) for example in ordered],
            "raw_generated": [raw_tensor_filename(example) for example in ordered],
            "run_log": "run.log",
        },
        "cases": [_case_manifest_row(example) for example in ordered],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _default_transitions_loader(config: Any) -> Any:
    return load_transition_records(
        config.data.bundle_json,
        config.data.phase_manifest_csv,
        backend=config.data.backend,
    )


def _default_visit_loader_factory(
    config: Any,
    loaded: Any,
) -> Callable[[str], PreparedVisit]:
    if config.data.roi_cache_dir is None:
        raise ValueError("diffusion visualization requires the locked MOTFM ROI cache")
    cache = MOTFMROICache(
        config.data.roi_cache_dir,
        bundle_json=config.data.bundle_json,
        output_shape_zyx=config.data.output_shape_zyx,
    )
    return lambda visit_id: cache.load(loaded.visits[visit_id])


def _default_model_loader(
    *,
    config: Any,
    loaded: Any,
    vqgan_checkpoint_path: Path,
    diffusion_checkpoint_path: Path,
    vqgan_sha256: str,
) -> LoadedDiffusionModel:
    from .workflows import (
        _build_diffusion_model,
        _diffusion_checkpoint_identity,
        load_paper_diffusion_for_inference,
        load_mri_vqgan,
    )

    if config.diffusion.denoiser_architecture == PAPER_FAITHFUL_ARCHITECTURE:
        diffusion, identity, global_step = load_paper_diffusion_for_inference(
            runtime=config.diffusion,
            loaded=loaded,
            vqgan_checkpoint_path=vqgan_checkpoint_path,
            diffusion_checkpoint_path=diffusion_checkpoint_path,
            vqgan_sha256=vqgan_sha256,
        )
        return LoadedDiffusionModel(diffusion, identity, global_step)

    vqgan = load_mri_vqgan(
        vqgan_checkpoint_path,
        expected_data_contract_sha256=loaded.data_contract_sha256,
        expected_data_backend=loaded.backend,
    )
    conditioner = ISPY2Conditioner(
        SharedMedGemmaTower.from_pretrained(local_files_only=True)
    )
    diffusion = _build_diffusion_model(vqgan, conditioner, config.diffusion)
    identity = _diffusion_checkpoint_identity(
        diffusion,
        vqgan_sha256=vqgan_sha256,
        loaded=loaded,
    )
    global_step = load_diffusion_checkpoint(
        diffusion,
        diffusion_checkpoint_path,
        identity,
    )
    return LoadedDiffusionModel(diffusion, identity, global_step)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(device_name)


def _save_raw_tensor(tensor: torch.Tensor, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        torch.save(tensor, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def generate_diffusion_test_visualizations(
    *,
    config_path: str | Path,
    vqgan_checkpoint_path: str | Path,
    diffusion_checkpoint_dir: str | Path,
    output_root: str | Path,
    device_name: str,
    seed: int,
    config_loader: Callable[[str | Path], Any] = load_experiment_config,
    transitions_loader: Callable[[Any], Any] = _default_transitions_loader,
    visit_loader_factory: Callable[
        [Any, Any], Callable[[str], PreparedVisit]
    ] = _default_visit_loader_factory,
    model_loader: Callable[..., LoadedDiffusionModel] = _default_model_loader,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> Path:
    config_file = Path(config_path)
    vqgan_file = Path(vqgan_checkpoint_path).resolve()
    checkpoint = select_top_checkpoint(diffusion_checkpoint_dir)
    output = Path(output_root) / checkpoint.path.stem
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"visualization output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    log = log_path.open("w", encoding="utf-8", buffering=1)

    try:
        started = time.monotonic()
        log.write(f"starting checkpoint {checkpoint.path}\n")
        config = config_loader(config_file)
        if config.data.roi_cache_dir is None:
            raise ValueError("diffusion visualization requires the locked MOTFM ROI cache")
        loaded = transitions_loader(config)
        if loaded.backend != config.data.backend:
            raise ValueError("loaded data backend does not match the config")
        _validate_sha256(loaded.data_contract_sha256, "data contract")
        visit_loader = visit_loader_factory(config, loaded)
        selected = select_test_cases(loaded.records, visit_loader)
        checkpoint_sha256 = file_hasher(checkpoint.path)
        vqgan_sha256 = file_hasher(vqgan_file)
        _validate_sha256(checkpoint_sha256, "diffusion checkpoint")
        _validate_sha256(vqgan_sha256, "VQGAN checkpoint")
        device = _resolve_device(device_name)
        loaded_model = model_loader(
            config=config,
            loaded=loaded,
            vqgan_checkpoint_path=vqgan_file,
            diffusion_checkpoint_path=checkpoint.path,
            vqgan_sha256=vqgan_sha256,
        )
        identity = loaded_model.identity
        if identity.vqgan_sha256 != vqgan_sha256:
            raise ValueError("loaded diffusion VQGAN identity does not match")
        if identity.data_contract_sha256 != loaded.data_contract_sha256:
            raise ValueError("loaded diffusion data contract does not match")
        if identity.data_backend != loaded.backend:
            raise ValueError("loaded diffusion data backend does not match")
        model = loaded_model.model.to(device).eval()
        log.write(
            f"validated identities on {device.type}; global_step={loaded_model.global_step}\n"
        )

        examples: list[DiffusionTestExample] = []
        for item in selected:
            stage_started = time.monotonic()
            case_seed = seed + item.record.stage_id - 1
            example = generate_test_example(
                item,
                target_loader=visit_loader,
                model=model,
                device=device,
                seed=case_seed,
            )
            examples.append(example)
            _save_raw_tensor(example.generated, output / raw_tensor_filename(example))
            render_detail(
                example,
                output / detail_filename(example),
                checkpoint=checkpoint,
            )
            log.write(
                f"completed stage {item.record.stage_id} in "
                f"{time.monotonic() - stage_started:.3f}s\n"
            )

        render_overview(examples, output / "overview.png", checkpoint=checkpoint)
        write_manifest(
            examples,
            output / "manifest.json",
            config_path=config_file,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity=identity,
            checkpoint_global_step=loaded_model.global_step,
            vqgan_checkpoint_path=vqgan_file,
            vqgan_sha256=vqgan_sha256,
            data_backend=loaded.backend,
            data_contract_sha256=loaded.data_contract_sha256,
        )
        log.write(f"completed all cases in {time.monotonic() - started:.3f}s\n")
        return output
    except Exception:
        traceback.print_exc(file=log)
        raise
    finally:
        log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render diffusion test-set examples"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--vqgan-checkpoint",
        type=Path,
        default=DEFAULT_VQGAN_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--diffusion-checkpoint-dir",
        type=Path,
        default=DEFAULT_DIFFUSION_CHECKPOINT_DIRECTORY,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cuda"
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = generate_diffusion_test_visualizations(
        config_path=args.config,
        vqgan_checkpoint_path=args.vqgan_checkpoint,
        diffusion_checkpoint_dir=args.diffusion_checkpoint_dir,
        output_root=args.output_root,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(output.resolve())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

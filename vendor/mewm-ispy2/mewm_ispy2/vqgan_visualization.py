from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .backend import load_transition_records, sha256_file
from .cache import MOTFMROICache
from .config import load_experiment_config
from .data import PreparedVisit
from .vqgan import VQGAN_NUMERIC_CONTRACT


SIZE_GROUPS = ("small", "medium", "large")
PLANE_NAMES = ("axial", "coronal", "sagittal")
DETAIL_FIGURE_SIZE = (12.0, 9.0)
OVERVIEW_FIGURE_SIZE = (20.0, 22.5)
FIGURE_DPI = 160
METRIC_FIELDS = (
    "visit_id",
    "size_group",
    "stratum_rank",
    "tumor_voxels",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "volume_mae",
    "tumor_mae",
    "non_tumor_mae",
    "psnr",
    "outside_range_fraction",
    "detail_filename",
)
DEFAULT_CONFIG_PATH = Path("configs/ispy2_current_ct_aligned_weak_gan.yaml")
DEFAULT_CHECKPOINT_PATH = Path(
    "runs/current_dce0_ct_aligned_weak_gan/vqgan/checkpoints/composite/"
    "best-composite-35-42408.ckpt"
)
DEFAULT_OUTPUT_PATH = Path(
    "runs/current_dce0_ct_aligned_weak_gan/vqgan/"
    "reconstruction_examples_epoch35"
)


@dataclass(frozen=True)
class SelectedVisit:
    visit_id: str
    size_group: str
    tumor_voxels: int
    stratum_rank: int


@dataclass(frozen=True)
class ReconstructionExample:
    selected: SelectedVisit
    image: torch.Tensor
    reconstruction: torch.Tensor
    mask: torch.Tensor
    centroid_zyx: tuple[int, int, int]
    metrics: Mapping[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render stratified MRI VQGAN reconstruction examples"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cuda"
    )
    return parser


def select_stratified_visits(sizes: Mapping[str, int]) -> list[SelectedVisit]:
    if len(sizes) < 6:
        raise ValueError("at least six visits are required")
    ordered: list[tuple[str, int]] = []
    for visit_id, value in sizes.items():
        tumor_voxels = int(value)
        if tumor_voxels <= 0:
            raise ValueError("tumor voxel counts must be positive")
        ordered.append((str(visit_id), tumor_voxels))
    ordered.sort(key=lambda item: (item[1], item[0]))

    selected: list[SelectedVisit] = []
    for size_group, indices in zip(
        SIZE_GROUPS, np.array_split(np.arange(len(ordered)), 3), strict=True
    ):
        if len(indices) < 2:
            raise ValueError("each tumor-size stratum must contain two visits")
        positions = (
            math.floor((len(indices) - 1) / 3),
            math.ceil(2 * (len(indices) - 1) / 3),
        )
        for position in positions:
            visit_id, tumor_voxels = ordered[int(indices[position])]
            selected.append(
                SelectedVisit(
                    visit_id=visit_id,
                    size_group=size_group,
                    tumor_voxels=tumor_voxels,
                    stratum_rank=position + 1,
                )
            )

    if len({item.visit_id for item in selected}) != 6:
        raise ValueError("stratified selection must contain six distinct visits")
    return selected


def _validate_volume(volume: torch.Tensor, name: str) -> None:
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1,Z,Y,X]")
    if any(size <= 0 for size in volume.shape[1:]):
        raise ValueError(f"{name} spatial dimensions must be positive")
    if not torch.isfinite(volume).all():
        raise ValueError(f"{name} must contain only finite values")


def tumor_centroid(mask: torch.Tensor) -> tuple[int, int, int]:
    _validate_volume(mask, "mask")
    coordinates = torch.nonzero(mask[0] > 0, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("tumor mask is empty")
    centroid = coordinates.float().mean(dim=0).round().to(torch.int64)
    return tuple(int(value) for value in centroid)


def orthogonal_planes(
    volume: torch.Tensor, centroid_zyx: tuple[int, int, int]
) -> dict[str, torch.Tensor]:
    _validate_volume(volume, "volume")
    if len(centroid_zyx) != 3:
        raise ValueError("centroid must contain z, y, and x coordinates")
    z, y, x = (int(value) for value in centroid_zyx)
    limits = volume.shape[1:]
    if not all(0 <= value < limit for value, limit in zip((z, y, x), limits)):
        raise ValueError("centroid lies outside the volume")
    return {
        "axial": volume[0, z],
        "coronal": volume[0, :, y, :],
        "sagittal": volume[0, :, :, x],
    }


def reconstruction_metrics(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    _validate_volume(image, "image")
    _validate_volume(reconstruction, "reconstruction")
    _validate_volume(mask, "mask")
    if reconstruction.shape != image.shape or mask.shape != image.shape:
        raise ValueError("image, reconstruction, and mask shapes must match")
    tumor = mask > 0
    if not tumor.any():
        raise ValueError("tumor mask is empty")
    non_tumor = ~tumor
    if not non_tumor.any():
        raise ValueError("tumor mask cannot fill the whole volume")

    image_zero_one = (image.float() + 1.0) / 2.0
    reconstruction_zero_one = (reconstruction.float() + 1.0) / 2.0
    absolute_error = (image_zero_one - reconstruction_zero_one).abs()
    mse = (image_zero_one - reconstruction_zero_one).square().mean()
    mse_value = float(mse)
    psnr = math.inf if mse_value == 0.0 else 10.0 * math.log10(1.0 / mse_value)
    return {
        "volume_mae": float(absolute_error.mean()),
        "tumor_mae": float(absolute_error[tumor].mean()),
        "non_tumor_mae": float(absolute_error[non_tumor].mean()),
        "psnr": psnr,
        "outside_range_fraction": float(
            ((reconstruction < -1.0) | (reconstruction > 1.0)).float().mean()
        ),
    }


def reconstruct_example(
    selected: SelectedVisit,
    prepared: PreparedVisit,
    model: torch.nn.Module,
    device: torch.device,
) -> ReconstructionExample:
    _validate_volume(prepared.image, "image")
    _validate_volume(prepared.mask, "mask")
    expected_shape = (1, 128, 128, 128)
    if tuple(prepared.image.shape) != expected_shape:
        raise ValueError("prepared image must have shape [1,128,128,128]")
    if prepared.mask.shape != prepared.image.shape:
        raise ValueError("prepared image and mask shapes must match")
    if prepared.visit_id != selected.visit_id:
        raise ValueError("selected and prepared visit IDs do not match")
    tumor_voxels = int((prepared.mask > 0).sum())
    if tumor_voxels == 0:
        raise ValueError("tumor mask is empty")
    if tumor_voxels != selected.tumor_voxels:
        raise ValueError("selected tumor voxel count does not match the mask")

    batch = prepared.image.unsqueeze(0).to(device=device, dtype=torch.float32)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        model_output = model(batch)
    reconstruction_batch = (
        model_output[0] if isinstance(model_output, tuple) else model_output
    )
    if reconstruction_batch.shape != batch.shape:
        raise ValueError("VQGAN reconstruction shape does not match its input")
    reconstruction = reconstruction_batch[0].detach().to(device="cpu", dtype=torch.float32)
    image = prepared.image.detach().to(device="cpu", dtype=torch.float32)
    mask = prepared.mask.detach().to(device="cpu")
    metrics = reconstruction_metrics(image, reconstruction, mask)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("reconstruction metrics must be finite")
    return ReconstructionExample(
        selected=selected,
        image=image,
        reconstruction=reconstruction,
        mask=mask,
        centroid_zyx=tumor_centroid(mask),
        metrics=metrics,
    )


def _plane_arrays(
    example: ReconstructionExample, plane_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_volume(example.image, "image")
    _validate_volume(example.reconstruction, "reconstruction")
    _validate_volume(example.mask, "mask")
    if (
        example.image.shape != example.reconstruction.shape
        or example.image.shape != example.mask.shape
    ):
        raise ValueError("example image, reconstruction, and mask shapes must match")
    image = orthogonal_planes(example.image, example.centroid_zyx)[plane_name]
    reconstruction = orthogonal_planes(
        example.reconstruction, example.centroid_zyx
    )[plane_name]
    mask = orthogonal_planes(example.mask, example.centroid_zyx)[plane_name]
    image_zero_one = ((image.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    reconstruction_zero_one = ((reconstruction.float() + 1.0) / 2.0).clamp(
        0.0, 1.0
    )
    error = ((image.float() - reconstruction.float()).abs() / 2.0)
    return tuple(
        value.detach().cpu().numpy()
        for value in (image_zero_one, reconstruction_zero_one, error, mask)
    )


def _render_plane_row(
    axes: np.ndarray,
    example: ReconstructionExample,
    plane_name: str,
    *,
    show_titles: bool,
) -> None:
    image, reconstruction, error, mask = _plane_arrays(example, plane_name)
    panels = (
        (image, "gray", 0.0, 1.0),
        (reconstruction, "gray", 0.0, 1.0),
        (error, "magma", 0.0, 0.25),
        (image, "gray", 0.0, 1.0),
    )
    titles = ("Original DCE0", "Reconstruction", "Absolute error", "Tumor contour")
    for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
        values, color_map, lower, upper = panel
        axis.imshow(values, cmap=color_map, vmin=lower, vmax=upper, origin="lower")
        if index == 3 and np.any(mask > 0):
            axis.contour(
                mask,
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


def detail_filename(example: ReconstructionExample) -> str:
    safe_visit_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", example.selected.visit_id)
    return f"{example.selected.size_group}-{safe_visit_id}.png"


def _save_figure(figure: plt.Figure, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=FIGURE_DPI)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def render_detail(example: ReconstructionExample, output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("detail output must be a PNG file")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        3,
        4,
        figsize=DETAIL_FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
        squeeze=False,
    )
    for row, plane_name in enumerate(PLANE_NAMES):
        _render_plane_row(
            axes[row], example, plane_name, show_titles=row == 0
        )
    figure.suptitle(
        f"{example.selected.visit_id} | {example.selected.size_group} | "
        f"tumor voxels={example.selected.tumor_voxels} | "
        f"MAE={float(example.metrics['volume_mae']):.5f}",
        fontsize=12,
    )
    _save_figure(figure, output)
    return output


def _grouped_examples(
    examples: Sequence[ReconstructionExample],
) -> dict[str, list[ReconstructionExample]]:
    grouped = {
        group: [example for example in examples if example.selected.size_group == group]
        for group in SIZE_GROUPS
    }
    if len(examples) != 6 or any(len(grouped[group]) != 2 for group in SIZE_GROUPS):
        raise ValueError("overview requires exactly two examples per size group")
    if len({example.selected.visit_id for example in examples}) != 6:
        raise ValueError("overview examples must contain six distinct visits")
    return grouped


def render_overview(
    examples: Sequence[ReconstructionExample], output_path: str | Path
) -> Path:
    grouped = _grouped_examples(examples)
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("overview output must be a PNG file")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        9,
        8,
        figsize=OVERVIEW_FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
        squeeze=False,
    )
    for group_index, size_group in enumerate(SIZE_GROUPS):
        base_row = group_index * 3
        for case_index, example in enumerate(grouped[size_group]):
            column_start = case_index * 4
            for plane_index, plane_name in enumerate(PLANE_NAMES):
                _render_plane_row(
                    axes[
                        base_row + plane_index,
                        column_start : column_start + 4,
                    ],
                    example,
                    plane_name,
                    show_titles=plane_index == 0,
                )
            axes[base_row, column_start].text(
                0.0,
                1.22,
                f"{example.selected.visit_id} | {size_group} | "
                f"voxels={example.selected.tumor_voxels} | "
                f"MAE={float(example.metrics['volume_mae']):.5f}",
                transform=axes[base_row, column_start].transAxes,
                fontsize=11,
                fontweight="bold",
                ha="left",
            )
    figure.suptitle(
        "Weak-GAN MRI VQGAN validation reconstructions",
        fontsize=16,
    )
    _save_figure(figure, output)
    return output


def _metric_row(example: ReconstructionExample) -> dict[str, str | int | float]:
    z, y, x = example.centroid_zyx
    return {
        "visit_id": example.selected.visit_id,
        "size_group": example.selected.size_group,
        "stratum_rank": example.selected.stratum_rank,
        "tumor_voxels": example.selected.tumor_voxels,
        "centroid_z": z,
        "centroid_y": y,
        "centroid_x": x,
        "volume_mae": float(example.metrics["volume_mae"]),
        "tumor_mae": float(example.metrics["tumor_mae"]),
        "non_tumor_mae": float(example.metrics["non_tumor_mae"]),
        "psnr": float(example.metrics["psnr"]),
        "outside_range_fraction": float(
            example.metrics["outside_range_fraction"]
        ),
        "detail_filename": detail_filename(example),
    }


def write_metrics_csv(
    examples: Sequence[ReconstructionExample], output_path: str | Path
) -> Path:
    _grouped_examples(examples)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(_metric_row(example) for example in examples)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_manifest(
    examples: Sequence[ReconstructionExample],
    output_path: str | Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> Path:
    _grouped_examples(examples)
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint SHA256 is invalid")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mewm_ispy2_vqgan_reconstruction_examples_v1",
        "config_path": str(config_path.resolve()),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
        },
        "fold": "val",
        "numeric_contract": VQGAN_NUMERIC_CONTRACT,
        "selection_rule": (
            "sort by (tumor_voxels, visit_id), split into three rank-balanced "
            "strata, select one-third and two-thirds positions"
        ),
        "display_windows": {
            "mri": [0.0, 1.0],
            "absolute_error": [0.0, 0.25],
        },
        "files": {
            "overview": "overview.png",
            "details": [detail_filename(example) for example in examples],
            "metrics": "metrics.csv",
        },
        "selected_visits": [_metric_row(example) for example in examples],
    }
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
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


def _default_visit_loader_factory(config: Any, loaded: Any) -> Callable[[str], PreparedVisit]:
    cache = MOTFMROICache(
        config.data.roi_cache_dir,
        bundle_json=config.data.bundle_json,
        output_shape_zyx=config.data.output_shape_zyx,
    )
    return lambda visit_id: cache.load(loaded.visits[visit_id])


def _default_model_loader(checkpoint_path: Path) -> torch.nn.Module:
    from .workflows import load_mri_vqgan

    return load_mri_vqgan(checkpoint_path)


def _validation_visit_ids(loaded: Any) -> list[str]:
    visit_ids = {
        visit_id
        for record in loaded.records
        if record.fold == "val"
        for visit_id in (record.source_visit_id, record.target_visit_id)
    }
    if not visit_ids:
        raise ValueError("validation fold contains no visits")
    return sorted(visit_ids)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(device_name)


def _scan_tumor_voxels(
    visit_ids: Sequence[str], visit_loader: Callable[[str], PreparedVisit]
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    expected_shape = (1, 128, 128, 128)
    for visit_id in visit_ids:
        prepared = visit_loader(visit_id)
        _validate_volume(prepared.image, "image")
        _validate_volume(prepared.mask, "mask")
        if tuple(prepared.image.shape) != expected_shape:
            raise ValueError("prepared image must have shape [1,128,128,128]")
        if prepared.mask.shape != prepared.image.shape:
            raise ValueError("prepared image and mask shapes must match")
        tumor_voxels = int((prepared.mask > 0).sum())
        if tumor_voxels == 0:
            raise ValueError(f"tumor mask is empty for {visit_id}")
        sizes[visit_id] = tumor_voxels
    return sizes


def generate_reconstruction_examples(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str,
    config_loader: Callable[[str | Path], Any] = load_experiment_config,
    transitions_loader: Callable[[Any], Any] = _default_transitions_loader,
    visit_loader_factory: Callable[
        [Any, Any], Callable[[str], PreparedVisit]
    ] = _default_visit_loader_factory,
    model_loader: Callable[[Path], torch.nn.Module] = _default_model_loader,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> list[ReconstructionExample]:
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    output = Path(output_dir)
    config = config_loader(config_file)
    if config.data.roi_cache_dir is None:
        raise ValueError("VQGAN visualization requires the locked MOTFM ROI cache")
    loaded = transitions_loader(config)
    visit_loader = visit_loader_factory(config, loaded)
    validation_ids = _validation_visit_ids(loaded)
    selected = select_stratified_visits(
        _scan_tumor_voxels(validation_ids, visit_loader)
    )

    device = _resolve_device(device_name)
    model = model_loader(checkpoint_file).to(device).eval()
    examples = [
        reconstruct_example(item, visit_loader(item.visit_id), model, device)
        for item in selected
    ]
    output.mkdir(parents=True, exist_ok=True)
    for example in examples:
        render_detail(example, output / detail_filename(example))
    render_overview(examples, output / "overview.png")
    write_metrics_csv(examples, output / "metrics.csv")
    write_manifest(
        examples,
        output / "manifest.json",
        config_path=config_file,
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=file_hasher(checkpoint_file),
    )
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = generate_reconstruction_examples(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected_visits": [
                    {
                        "visit_id": example.selected.visit_id,
                        "size_group": example.selected.size_group,
                        "tumor_voxels": example.selected.tumor_voxels,
                        "volume_mae": example.metrics["volume_mae"],
                    }
                    for example in examples
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

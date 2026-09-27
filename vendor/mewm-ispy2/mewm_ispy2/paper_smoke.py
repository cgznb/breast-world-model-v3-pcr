from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .ct_denoiser import DenoiserResult
from .paper_attenuation import MorphoGaussianAttenuator
from .paper_checkpoint import (
    PAPER_CHECKPOINT_SCHEMA_VERSION,
    PaperCheckpointIdentity,
    build_paper_checkpoint,
    load_paper_checkpoint,
)
from .paper_conditioning import PaperActionConditioner, PaperConditionOutput
from .paper_contracts import (
    PaperRuntimeContract,
    default_paper_runtime_contract,
    paper_contract_payload,
    paper_contract_sha256,
)
from .paper_diffusion import (
    DDIM_ETA0_SAMPLER,
    PaperDiffusionConfig,
    PaperDiffusionTrainingSystem,
    PaperFaithfulLatentDiffusion,
)
from .registered_large_checkpoint import (
    REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION,
    RegisteredLargeCheckpointIdentity,
    build_registered_large_checkpoint,
    load_registered_large_checkpoint,
)
from .registered_x0_checkpoint import (
    REGISTERED_X0_CHECKPOINT_SCHEMA_VERSION,
    RegisteredX0CheckpointIdentity,
    build_registered_x0_checkpoint,
    load_registered_x0_checkpoint,
)
from .contracts import X0_PREDICTION_TYPE


_ANCHOR_ACTION = "treatment arm Paclitaxel"
_NEGATIVE_ACTIONS = (
    "treatment arm Paclitaxel + Trastuzumab",
    "treatment arm T-DM1 + Pertuzumab",
)
_CLINICAL_TEXT = (
    "age at screening 53; HR 0; HER2 1; MP 0; "
    "menopausal status Above categories not applicable AND Age < 50"
)


class _SmokeCLIPTower(torch.nn.Module):
    hidden_size = 512

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if (
            isinstance(texts, (str, bytes))
            or not isinstance(texts, Sequence)
            or not texts
            or not all(isinstance(text, str) and text for text in texts)
        ):
            raise ValueError("smoke CLIP texts must be nonempty strings")
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = torch.tensor(tuple(digest), dtype=torch.float32)
            rows.append(values.repeat(self.hidden_size // values.numel()) / 127.5 - 1.0)
        return torch.stack(rows)


class _TinyQuantizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        values = torch.linspace(-2.0, 2.0, 16 * 8).reshape(16, 8)
        self.embeddings = torch.nn.Parameter(values)

    def forward(
        self, continuous: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        indices = torch.zeros(
            continuous.shape[0],
            *continuous.shape[2:],
            dtype=torch.long,
            device=continuous.device,
        )
        return continuous, {
            "indices": indices,
            "perplexity": continuous.new_ones(()),
        }


class _TinyVQGAN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = _TinyQuantizer()

    def encode_continuous(self, image: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool3d(image, kernel_size=4, stride=4)
        return pooled.repeat(1, 8, 1, 1, 1) * 3.0

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            latent.mean(dim=1, keepdim=True),
            scale_factor=4,
            mode="trilinear",
            align_corners=False,
        )


class _TinyPaperDenoiser(torch.nn.Module):
    architecture = "mewm_paper_faithful_v1"
    network_input_channels = 49
    spatial_semantic_channels = 32

    def __init__(self) -> None:
        super().__init__()
        self.noisy_scale = torch.nn.Parameter(torch.tensor(0.2))
        self.spatial_scale = torch.nn.Parameter(torch.tensor(0.1))
        self.global_projection = torch.nn.Linear(512, 8, bias=False)
        self.context_projection = torch.nn.Linear(512, 8, bias=False)
        self.calls = 0

    def forward(
        self,
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        condition: PaperConditionOutput,
        timesteps: torch.Tensor,
    ) -> DenoiserResult:
        del timesteps
        self.calls += 1
        batch = noisy_latent.shape[0]
        global_term = self.global_projection(condition.global_condition).reshape(
            batch, 8, 1, 1, 1
        )
        weights = condition.context_mask.to(condition.context_tokens.dtype)
        context = (
            condition.context_tokens * weights[:, :, None]
        ).sum(dim=1) / weights.sum(dim=1, keepdim=True)
        context_term = self.context_projection(context).reshape(
            batch, 8, 1, 1, 1
        )
        predicted = (
            noisy_latent * self.noisy_scale
            + spatial_condition[:, :8] * self.spatial_scale
            + 0.01 * global_term
            + 0.01 * context_term
        )
        semantic = noisy_latent.new_zeros(
            batch, 32, *noisy_latent.shape[-3:]
        )
        return DenoiserResult(
            predicted_noise=predicted,
            network_input=torch.cat(
                (noisy_latent, spatial_condition, semantic), dim=1
            ),
            spatial_semantic=semantic,
        )


def _tiny_paper_model(
    contract: PaperRuntimeContract,
) -> PaperFaithfulLatentDiffusion:
    return PaperFaithfulLatentDiffusion(
        _TinyVQGAN(),
        PaperActionConditioner(_SmokeCLIPTower(), contract=contract),
        _TinyPaperDenoiser(),
        MorphoGaussianAttenuator(contract),
        PaperDiffusionConfig(runtime=contract),
    )


def _tiny_identity(contract: PaperRuntimeContract) -> PaperCheckpointIdentity:
    return PaperCheckpointIdentity(
        denoiser_architecture="mewm_paper_faithful_v1",
        vqgan_sha256=contract.vqgan_sha256,
        data_contract_sha256=contract.data_contract_sha256,
        data_backend="current",
        denoiser_input_channels=49,
        semantic_channels=32,
        prediction_type="epsilon",
        timesteps=200,
        sampler="ancestral_ddpm",
        latent_contract="continuous_codebook_minmax_v1",
        ema_decay=0.995,
        paper_contract=paper_contract_payload(contract),
        paper_contract_sha256=paper_contract_sha256(contract),
    )


def _tiny_transition(
    value: float,
    *,
    patient_id: str,
    transition_id: str,
) -> dict[str, Any]:
    source = torch.full((1, 1, 8, 32, 32), value)
    target = torch.full_like(source, value + 0.2)
    mask = torch.zeros_like(source)
    mask[..., 2:6, 10:22, 10:22] = 1.0
    return {
        "model_inputs": {
            "source_dce0": source,
            "source_mask": mask,
            "action_text": [_ANCHOR_ACTION],
            "clinical_text": [_CLINICAL_TEXT],
            "delta_days": torch.tensor([34.0]),
            "stage_id": torch.tensor([1]),
        },
        "supervision": {"target_dce0": target},
        "metadata": [
            {
                "patient_id": patient_id,
                "fold": "train",
                "transition_type": "T0->T1",
                "transition_id": transition_id,
            }
        ],
    }


def _tiny_batch(*, valid: bool) -> dict[str, Any]:
    return {
        "anchor": _tiny_transition(
            0.1,
            patient_id="smoke-anchor",
            transition_id="smoke-anchor:T0->T1",
        ),
        "positive": (
            _tiny_transition(
                0.4,
                patient_id="smoke-positive",
                transition_id="smoke-positive:T0->T1",
            )
            if valid
            else None
        ),
        "negative_action_texts": (
            tuple((action,) for action in _NEGATIVE_ACTIONS)
            if valid
            else ((), ())
        ),
        "ccl_valid_mask": torch.tensor([valid]),
    }


def _gradient_norm(parameters: Sequence[torch.nn.Parameter], *, name: str) -> float:
    squared = 0.0
    seen = False
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"nonfinite smoke gradient: {name}")
        squared += float(gradient.detach().float().square().sum())
        seen = True
    value = math.sqrt(squared)
    if not seen or not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"missing or zero smoke gradient: {name}")
    return value


def _tiny_gradient_groups(
    model: PaperFaithfulLatentDiffusion,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    conditioner = model.conditioner
    return {
        "denoiser": tuple(model.denoiser.parameters()),
        "holistic_projection": tuple(conditioner.holistic_projection.parameters()),
        "component_projection": tuple(conditioner.component_projection.parameters()),
        "component_concepts": tuple(conditioner.component_concepts.parameters()),
    }


def _tiny_gradient_norms(model: PaperFaithfulLatentDiffusion) -> dict[str, float]:
    return {
        name: _gradient_norm(parameters, name=name)
        for name, parameters in _tiny_gradient_groups(model).items()
    }


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(destination: Path, report: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_tiny_paper_cpu_smoke(
    output_directory: str | Path, *, seed: int = 2026
) -> dict[str, Any]:
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    torch.manual_seed(seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    contract = default_paper_runtime_contract()
    model = _tiny_paper_model(contract).cpu()
    identity = _tiny_identity(contract)
    system = PaperDiffusionTrainingSystem(
        model,
        learning_rate=1e-4,
        checkpoint_identity=identity,
    )
    optimizer = system.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    valid_batch = _tiny_batch(valid=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are trying to `self\.log\(\)`.*",
        )
        loss = system.training_step(valid_batch, 0)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("tiny paper total loss is nonfinite")
    valid_call_count = model.denoiser.calls
    if valid_call_count != 5:
        raise RuntimeError("valid paper smoke must execute five denoiser calls")
    loss.backward()
    system.on_after_backward()
    gradient_norms = _tiny_gradient_norms(model)
    optimizer.step()
    model.update_ema()

    model.denoiser.calls = 0
    singleton_output = model.training_objective(
        _tiny_batch(valid=False),
        augment_source=False,
        random_seed=seed,
    )
    singleton_call_count = model.denoiser.calls
    if singleton_call_count != 1:
        raise RuntimeError("singleton paper smoke must execute one denoiser call")
    if not bool(torch.isfinite(singleton_output.total_loss)):
        raise FloatingPointError("tiny paper singleton loss is nonfinite")

    all_model_tensors_finite = all(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
        for tensor in model.state_dict().values()
    )
    if not all_model_tensors_finite:
        raise FloatingPointError("tiny paper model contains a nonfinite tensor")

    checkpoint_path = output / "smoke-step-0000001.ckpt"
    _atomic_torch_save(
        build_paper_checkpoint(model, identity, global_step=1),
        checkpoint_path,
    )
    reloaded = _tiny_paper_model(contract).cpu()
    if load_paper_checkpoint(reloaded, checkpoint_path, identity) != 1:
        raise RuntimeError("tiny paper checkpoint global step did not round-trip")
    anchor_inputs = valid_batch["anchor"]["model_inputs"]
    latent_shape = reloaded.spatial_condition(
        anchor_inputs["source_dce0"], anchor_inputs["source_mask"]
    ).shape
    sample = reloaded.sample(
        anchor_inputs["source_dce0"],
        anchor_inputs["source_mask"],
        anchor_inputs["action_text"],
        anchor_inputs["clinical_text"],
        anchor_inputs["delta_days"],
        anchor_inputs["stage_id"],
        noise=torch.randn(
            latent_shape[0],
            8,
            *latent_shape[-3:],
            generator=torch.Generator().manual_seed(seed),
        ),
    )
    sample_finite = bool(torch.isfinite(sample).all())
    if not sample_finite:
        raise FloatingPointError("tiny paper sample is nonfinite")
    return {
        "valid_denoiser_calls": valid_call_count,
        "singleton_denoiser_calls": singleton_call_count,
        "ema_updates": 1,
        "total_loss": float(loss.detach()),
        "singleton_loss": float(singleton_output.total_loss.detach()),
        "gradient_norms": gradient_norms,
        "all_model_tensors_finite": all_model_tensors_finite,
        "sample_finite": sample_finite,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }


def _report_mapping(report: Any) -> dict[str, Any]:
    if type(report) is dict:
        return report
    if is_dataclass(report) and not isinstance(report, type):
        return asdict(report)
    raise TypeError("paper smoke helper must return a dictionary or dataclass")


def _default_warm_start_components(
    config_path: str | Path, vqgan_checkpoint: str | Path
) -> tuple[PaperFaithfulLatentDiffusion, tuple[Any, Any]]:
    from .config import load_experiment_config
    from .workflows import _build_paper_training_components, _load_data

    config = load_experiment_config(config_path)
    loaded = _load_data(config)
    system, _train, _validation, _callbacks = _build_paper_training_components(
        config,
        SimpleNamespace(vqgan_checkpoint=vqgan_checkpoint),
        loaded,
        config.runtime.output_dir / "diffusion",
    )
    return system.model, (config, loaded)


def _default_warm_start_import(
    model: PaperFaithfulLatentDiffusion,
    context: tuple[Any, Any],
    output_directory: str | Path,
) -> Any:
    from .paper_warm_start import import_v4_paper_warm_start

    config, loaded = context
    contract = config.diffusion.paper
    if contract is None:
        raise ValueError("paper warm-start requires the paper config")
    return import_v4_paper_warm_start(
        model,
        contract.warm_start_path,
        expected_sha256=contract.warm_start_sha256,
        expected_vqgan_sha256=contract.vqgan_sha256,
        expected_data_contract_sha256=loaded.data_contract_sha256,
        output_directory=output_directory,
    )


def run_paper_warm_start_verification(
    config_path: str | Path,
    vqgan_checkpoint: str | Path,
    output_directory: str | Path,
    *,
    build_components: Callable[..., tuple[Any, Any]] | None = None,
    import_warm_start: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    builder = build_components or _default_warm_start_components
    importer = import_warm_start or _default_warm_start_import
    model, context = builder(config_path, vqgan_checkpoint)
    report = importer(model, context, output_directory)
    return _report_mapping(report)


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if type(value) is dict:
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if type(value) is list:
        return [_move_batch_to_device(item, device) for item in value]
    if type(value) is tuple:
        return tuple(_move_batch_to_device(item, device) for item in value)
    return value


def _production_gradient_groups(
    model: PaperFaithfulLatentDiffusion,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    denoiser_named = tuple(model.denoiser.named_parameters())
    conditioner_named = tuple(model.conditioner.named_parameters())
    return {
        "denoiser": tuple(parameter for _name, parameter in denoiser_named),
        "cross_attention": tuple(
            parameter
            for name, parameter in denoiser_named
            if "cross_attention" in name
        ),
        "action_projection": tuple(
            parameter
            for name, parameter in conditioner_named
            if name.startswith("holistic_projection")
            or name.startswith("component_projection")
        ),
        "component_concepts": tuple(
            parameter
            for name, parameter in conditioner_named
            if name.startswith("component_concepts")
        ),
    }


def _validate_formal_seed(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")


def _seed_formal_workflow(seed: int) -> None:
    _validate_formal_seed(seed)
    pl.seed_everything(seed, workers=True, verbose=False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _all_trainable_parameters(
    system: PaperDiffusionTrainingSystem,
) -> tuple[torch.nn.Parameter, ...]:
    groups = system.trainable_parameter_groups()
    parameters = tuple(
        parameter
        for group in groups.values()
        for parameter in group
    )
    if not parameters:
        raise RuntimeError("paper smoke has no trainable parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("paper smoke trainable parameter groups overlap")
    return parameters


def _validate_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
) -> None:
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("paper smoke optimizer must be Adam")
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if len({id(parameter) for parameter in optimizer_parameters}) != len(
        optimizer_parameters
    ):
        raise RuntimeError("paper smoke optimizer parameters overlap")
    if {id(parameter) for parameter in optimizer_parameters} != {
        id(parameter) for parameter in parameters
    }:
        raise RuntimeError(
            "paper smoke optimizer must contain all trainable parameters"
        )


def _global_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = 0.0
    seen = False
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite gradient in paper smoke")
        squared += float(gradient.detach().float().square().sum())
        seen = True
    value = math.sqrt(squared)
    if not seen or not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("paper smoke global gradient is missing or zero")
    return value


def _stable_valid_item(dataset: Any) -> tuple[Any, str]:
    records = getattr(dataset, "records", None)
    pair_index = getattr(dataset, "pair_index", None)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("paper smoke dataset records must be a sequence")
    if pair_index is None or not callable(getattr(pair_index, "select", None)):
        raise TypeError("paper smoke dataset must expose a CCL pair index")
    candidates: list[tuple[str, int]] = []
    for index, record in enumerate(records):
        transition_id = getattr(record, "transition_id", None)
        if type(transition_id) is not str or not transition_id:
            raise ValueError("paper smoke dataset has an invalid transition ID")
        candidates.append((transition_id, index))
    for transition_id, index in sorted(candidates):
        if pair_index.select(transition_id, epoch=0).valid:
            return dataset[index], transition_id
    raise RuntimeError("paper smoke dataset has no valid CCL tuple")


def _stable_anchor_items(dataset: Any, batch_size: int) -> tuple[list[Any], str]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("paper smoke physical batch size must be positive")
    records = getattr(dataset, "records", None)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("paper smoke dataset records must be a sequence")
    if len(records) < batch_size:
        raise ValueError("paper smoke dataset is smaller than the physical batch")
    ordered = sorted(
        (
            getattr(record, "transition_id", None),
            index,
        )
        for index, record in enumerate(records)
    )
    if any(
        type(transition_id) is not str or not transition_id
        for transition_id, _ in ordered
    ):
        raise ValueError("paper smoke dataset has an invalid transition ID")
    selected = ordered[:batch_size]
    return [dataset[index] for _transition_id, index in selected], selected[0][0]


def _run_paper_one_step_core(
    system: PaperDiffusionTrainingSystem,
    dataset: Any,
    collate: Callable[[list[Any]], dict[str, Any]],
    *,
    device: torch.device,
    seed: int,
    output_directory: str | Path,
    autocast_factory: Callable[[], Any],
    autocast_dtype_name: str,
    gradient_group_provider: Callable[
        [PaperFaithfulLatentDiffusion],
        dict[str, Sequence[torch.nn.Parameter]],
    ],
    optimizer: torch.optim.Optimizer | None = None,
    physical_batch_size: int = 1,
    validation_dataset: Any | None = None,
) -> dict[str, Any]:
    if not isinstance(system, PaperDiffusionTrainingSystem):
        raise TypeError("paper smoke system must be a PaperDiffusionTrainingSystem")
    if not isinstance(device, torch.device):
        raise TypeError("paper smoke device must be a torch.device")
    _validate_formal_seed(seed)
    if not callable(collate):
        raise TypeError("paper smoke collate must be callable")
    if not callable(autocast_factory):
        raise TypeError("paper smoke autocast factory must be callable")
    if type(autocast_dtype_name) is not str or not autocast_dtype_name:
        raise TypeError("paper smoke autocast dtype name must be a nonempty string")
    if not callable(gradient_group_provider):
        raise TypeError("paper smoke gradient group provider must be callable")
    identity = system.checkpoint_identity
    if type(identity) is PaperCheckpointIdentity:
        checkpoint_builder = build_paper_checkpoint
        checkpoint_loader = load_paper_checkpoint
        checkpoint_schema_version = PAPER_CHECKPOINT_SCHEMA_VERSION
    elif type(identity) is RegisteredLargeCheckpointIdentity:
        checkpoint_builder = build_registered_large_checkpoint
        checkpoint_loader = load_registered_large_checkpoint
        checkpoint_schema_version = REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION
    elif type(identity) is RegisteredX0CheckpointIdentity:
        checkpoint_builder = build_registered_x0_checkpoint
        checkpoint_loader = load_registered_x0_checkpoint
        checkpoint_schema_version = REGISTERED_X0_CHECKPOINT_SCHEMA_VERSION
    else:
        raise TypeError(
            "unsupported paper smoke checkpoint identity: "
            f"{type(identity).__name__}"
        )

    is_x0 = system.model.config.prediction_type == X0_PREDICTION_TYPE
    if is_x0:
        if type(identity) is not RegisteredX0CheckpointIdentity:
            raise TypeError("x0 paper smoke requires a v7 checkpoint identity")
        if physical_batch_size != identity.physical_batch_size:
            raise ValueError("x0 paper smoke batch does not match the v7 identity")
        items, selected_transition_id = _stable_anchor_items(
            dataset, physical_batch_size
        )
    else:
        if physical_batch_size != 1:
            raise ValueError("epsilon paper smoke retains its one-item probe")
        item, selected_transition_id = _stable_valid_item(dataset)
        items = [item]
    output = Path(output_directory)
    batch = collate(items)
    batch = _move_batch_to_device(batch, device)
    system = system.to(device)
    system.train()
    model = system.model
    trainable_parameters = _all_trainable_parameters(system)
    if optimizer is None:
        optimizer = system.configure_optimizers()
    _validate_optimizer_parameters(optimizer, trainable_parameters)
    optimizer.zero_grad(set_to_none=True)

    anchor_target = batch["anchor"]["supervision"]["target_dce0"]
    with torch.no_grad():
        target_latent = model.target_latent(anchor_target)
    generator = torch.Generator(device=device).manual_seed(seed)
    base_timestep = seed % model.config.timesteps
    contract = model.config.runtime
    ccl_span = contract.ccl_timestep_max - contract.ccl_timestep_min + 1
    ccl_timestep = contract.ccl_timestep_min + seed % ccl_span
    attenuation_level = contract.attenuation_levels[
        seed % len(contract.attenuation_levels)
    ].level
    base_noise = torch.randn(
        target_latent.shape,
        dtype=target_latent.dtype,
        device=device,
        generator=generator,
    )
    ccl_noise = (
        None
        if is_x0
        else torch.randn(
            target_latent.shape,
            dtype=target_latent.dtype,
            device=device,
            generator=generator,
        )
    )
    denoiser_call_count = 0

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal denoiser_call_count
        denoiser_call_count += 1

    handle = model.denoiser.register_forward_hook(count_forward)
    try:
        with autocast_factory():
            objective_kwargs = {
                "augment_source": True,
                "base_noise": base_noise,
                "base_timesteps": torch.full(
                    (target_latent.shape[0],),
                    base_timestep,
                    device=device,
                    dtype=torch.long,
                ),
                "attenuation_level": attenuation_level,
            }
            if not is_x0:
                objective_kwargs.update(
                    {
                        "ccl_noise": ccl_noise,
                        "ccl_timesteps": torch.tensor(
                            [ccl_timestep], device=device
                        ),
                    }
                )
            result = model.training_objective(batch, **objective_kwargs)
        expected_calls = 1 if is_x0 else 5
        if denoiser_call_count != expected_calls:
            if is_x0:
                raise RuntimeError(
                    "paper smoke must execute exactly one denoiser call"
                )
            raise RuntimeError("paper smoke must execute exactly five denoiser calls")
        result.total_loss.backward()
        system.on_after_backward()
        gradient_groups = gradient_group_provider(model)
        if type(gradient_groups) is not dict or not gradient_groups:
            raise TypeError("paper smoke gradient groups must be a nonempty dictionary")
        gradient_norms = {
            name: _gradient_norm(tuple(parameters), name=name)
            for name, parameters in gradient_groups.items()
        }
        global_preclip_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        )
        if (
            not math.isfinite(global_preclip_gradient_norm)
            or global_preclip_gradient_norm <= 0.0
        ):
            raise FloatingPointError("paper smoke preclip gradient norm is invalid")
        global_postclip_gradient_norm = _global_gradient_norm(trainable_parameters)
        optimizer.step()
        model.update_ema()
    finally:
        handle.remove()

    checkpoint_path = output / "smoke-step-0000001.ckpt"
    checkpoint_payload = checkpoint_builder(
        model,
        identity,
        global_step=1,
    )
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    if checkpoint_loader(model, checkpoint_path, identity) != 1:
        raise RuntimeError(
            "paper smoke checkpoint failed validation: "
            f"{checkpoint_schema_version}"
        )
    anchor_metadata = batch["anchor"]["metadata"][0]
    report = {
        "selected_transition_id": selected_transition_id,
        "anchor_transition_id": anchor_metadata["transition_id"],
        "physical_batch_size": physical_batch_size,
        "base_timestep": base_timestep,
        "attenuation_level": attenuation_level,
        "total_loss": float(result.total_loss.detach()),
        "gradient_norms": gradient_norms,
        "global_preclip_gradient_norm": global_preclip_gradient_norm,
        "global_postclip_gradient_norm": global_postclip_gradient_norm,
        "gradient_clip_val": 1.0,
        "denoiser_call_count": denoiser_call_count,
        "ema_updates": 1,
        "autocast_device_type": device.type,
        "autocast_dtype": autocast_dtype_name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_global_step": 1,
        "checkpoint_schema_version": checkpoint_schema_version,
    }
    if is_x0:
        latent_shape = tuple(target_latent.shape[1:])
        report.update(
            {
                "x0_mse": float(result.x0_mse.detach()),
                "x0_mae": float(result.x0_mae.detach()),
            }
        )
        optimizer.zero_grad(set_to_none=True)
        del checkpoint_payload, batch, anchor_target, target_latent, base_noise, result
        torch.cuda.empty_cache()
        validation_source = validation_dataset or dataset
        validation_items, validation_transition_id = _stable_anchor_items(
            validation_source, 1
        )
        validation_batch = _move_batch_to_device(collate(validation_items), device)
        inputs = validation_batch["anchor"]["model_inputs"]
        sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
        sample_noise = torch.randn(
            (1, *latent_shape),
            device=device,
            dtype=inputs["source_dce0"].dtype,
            generator=sample_generator,
        )
        model.eval()
        sample = model.sample_with_diagnostics(
            inputs["source_dce0"],
            inputs["source_mask"],
            inputs["action_text"],
            inputs["clinical_text"],
            inputs["delta_days"],
            inputs["stage_id"],
            sampler=DDIM_ETA0_SAMPLER,
            sampling_steps=4,
            noise=sample_noise,
        )
        if sample.image.shape != inputs["source_dce0"].shape:
            raise RuntimeError("x0 DDIM smoke decoded image shape is invalid")
        if not bool(torch.isfinite(sample.image).all()) or not bool(
            torch.any(sample.image != 0)
        ):
            raise FloatingPointError("x0 DDIM smoke decoded image is empty or nonfinite")
        report["ddim_smoke"] = {
            "validation_transition_id": validation_transition_id,
            "sampling_steps": 4,
            "image_shape": list(sample.image.shape),
            "image_min": float(sample.image.min()),
            "image_max": float(sample.image.max()),
            "image_nonzero_voxels": int(torch.count_nonzero(sample.image)),
            "unique_code_count": sample.unique_code_count,
            "effective_code_count": sample.effective_code_count,
        }
    else:
        positive_metadata = batch["positive"]["metadata"][0]
        report.update(
            {
                "positive_transition_id": positive_metadata["transition_id"],
                "ccl_timestep": ccl_timestep,
                "epsilon_mse": float(result.epsilon_mse.detach()),
                "ccl_loss": float(result.ccl_loss.detach()),
                "positive_similarity": float(result.positive_similarity.detach()),
                "negative_similarity": float(result.negative_similarity.detach()),
            }
        )
    return report


def run_paper_gpu_smoke(
    config_path: str | Path,
    vqgan_checkpoint: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 2026,
    device: str | torch.device | None = None,
    autocast_factory: Callable[[], Any] | None = None,
    autocast_dtype_name: str | None = None,
    gradient_group_provider: Callable[
        [PaperFaithfulLatentDiffusion],
        dict[str, Sequence[torch.nn.Parameter]],
    ] = _production_gradient_groups,
) -> dict[str, Any]:
    _seed_formal_workflow(seed)
    from .backend import sha256_file
    from .config import load_experiment_config
    from .paper_warm_start import import_v4_paper_warm_start
    from .workflows import _build_paper_training_components, _load_data

    resolved_device = torch.device("cuda" if device is None else device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("paper GPU smoke requires CUDA")
    if device is None and resolved_device.type != "cuda":
        raise RuntimeError("paper GPU smoke production path requires CUDA")
    if autocast_factory is None:
        if resolved_device.type != "cuda":
            raise ValueError("CPU paper smoke requires an injected autocast factory")
        autocast_factory = lambda: torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
        resolved_autocast_dtype = "bfloat16"
    else:
        if type(autocast_dtype_name) is not str or not autocast_dtype_name:
            raise ValueError(
                "injected paper smoke autocast requires its dtype name"
            )
        resolved_autocast_dtype = autocast_dtype_name

    config = load_experiment_config(config_path)
    loaded = _load_data(config)
    output = Path(output_directory)
    system, train_loader, validation_loader, _callbacks = _build_paper_training_components(
        config,
        SimpleNamespace(vqgan_checkpoint=vqgan_checkpoint),
        loaded,
        output,
    )
    contract = config.diffusion.paper
    if contract is None:
        raise ValueError("paper GPU smoke requires a paper runtime contract")
    initialization_method = getattr(contract, "initialization_method", None)
    if initialization_method is None:
        import_v4_paper_warm_start(
            system.model,
            contract.warm_start_path,
            expected_sha256=contract.warm_start_sha256,
            expected_vqgan_sha256=contract.vqgan_sha256,
            expected_data_contract_sha256=loaded.data_contract_sha256,
            output_directory=output / "warm_start",
        )
    elif initialization_method == "random":
        if (
            contract.warm_start_path is not None
            or contract.warm_start_sha256 is not None
        ):
            raise ValueError(
                "random paper initialization must not define a warm-start source"
            )
    else:
        raise ValueError(
            f"unsupported paper initialization method: {initialization_method!r}"
        )
    dataset = train_loader.dataset
    collate = getattr(train_loader, "collate_fn", None)
    if collate is None:
        collate = lambda items: items[0]

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
        torch.cuda.synchronize(resolved_device)
    started = time.perf_counter()
    report = _run_paper_one_step_core(
        system,
        dataset,
        collate,
        device=resolved_device,
        seed=seed,
        output_directory=output,
        autocast_factory=autocast_factory,
        autocast_dtype_name=resolved_autocast_dtype,
        gradient_group_provider=gradient_group_provider,
        physical_batch_size=(
            config.diffusion_training.batch_size
            if getattr(config.diffusion, "prediction_type", None)
            == X0_PREDICTION_TYPE
            else 1
        ),
        validation_dataset=getattr(validation_loader, "dataset", None),
    )
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            resolved_device
        )
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            resolved_device
        )
    report["wall_time_seconds"] = time.perf_counter() - started
    report["checkpoint_sha256"] = sha256_file(Path(report["checkpoint_path"]))
    _atomic_json_write(Path(output_directory) / "paper_gpu_smoke.json", report)
    return report


__all__ = [
    "_SmokeCLIPTower",
    "run_paper_gpu_smoke",
    "run_paper_warm_start_verification",
    "run_tiny_paper_cpu_smoke",
]

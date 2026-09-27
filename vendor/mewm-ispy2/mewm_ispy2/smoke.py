from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .conditioning import ISPY2Conditioner
from .contracts import CT_DENOISER_ARCHITECTURE, FILM_DENOISER_ARCHITECTURE
from .diffusion import ConditionalLatentDiffusion, DiffusionConfig, build_denoiser
from .inference import publish_prediction
from .segmenter import segment_generated_samples
from .vqgan import MRILevelVQGAN, VQGANConfig


class _SmokeTextTower(torch.nn.Module):
    hidden_size = 4

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.tensor(
            [[len(text), sum(map(ord, text)) % 29, text.count(" "), 1] for text in texts],
            dtype=torch.float32,
        )


def run_tiny_cpu_smoke(
    output_directory: str | Path,
    *,
    seed: int = 2026,
    denoiser_architecture: str = FILM_DENOISER_ARCHITECTURE,
) -> dict[str, object]:
    torch.manual_seed(seed)
    vqgan = MRILevelVQGAN(
        VQGANConfig(
            hidden_channels=4,
            embedding_dim=8,
            n_codes=16,
            num_groups=2,
            nearest_chunk_size=64,
        )
    )
    conditioner = ISPY2Conditioner(_SmokeTextTower(), text_hidden_size=4)
    is_ct = denoiser_architecture == CT_DENOISER_ARCHITECTURE
    denoiser = build_denoiser(denoiser_architecture, smoke=True)
    diffusion = ConditionalLatentDiffusion(
        vqgan,
        conditioner,
        denoiser,
        DiffusionConfig(
            denoiser_architecture=denoiser_architecture,
            timesteps=4,
            ema_decay=0.995,
            denoiser_input_channels=49 if is_ct else 17,
            semantic_channels=32 if is_ct else 0,
        ),
    ).cpu()
    image_size = 32 if is_ct else 16
    latent_size = image_size // 4
    source = torch.rand(1, 1, image_size, image_size, image_size)
    source_mask = torch.zeros_like(source)
    lower = image_size // 2 - 2
    upper = image_size // 2 + 2
    source_mask[:, :, lower:upper, lower:upper, lower:upper] = 1
    target = torch.rand_like(source)
    diffusion.eval()
    loss, details = diffusion.training_loss(
        source,
        source_mask,
        target,
        ["treatment arm smoke"],
        ["age at screening 53; HR 0; HER2 0; MP 1; menopausal status Above categories not applicable AND Age < 50"],
        torch.tensor([30.0]),
        torch.tensor([1]),
        noise=torch.randn(1, 8, latent_size, latent_size, latent_size),
        timesteps=torch.tensor([2]),
    )
    repeat = 8
    samples = diffusion.sample(
        source.repeat(repeat, 1, 1, 1, 1),
        source_mask.repeat(repeat, 1, 1, 1, 1),
        ["treatment arm smoke"] * repeat,
        ["age at screening 53; HR 0; HER2 0; MP 1; menopausal status Above categories not applicable AND Age < 50"] * repeat,
        torch.full((repeat,), 30.0),
        torch.ones(repeat, dtype=torch.long),
        noise=torch.randn(repeat, 8, latent_size, latent_size, latent_size),
    )
    segmenter = torch.nn.Conv3d(1, 1, 1)
    result = segment_generated_samples(segmenter, samples)
    publish_prediction(
        output_directory,
        result,
        metadata={
            "transition_id": "SMOKE:T0->T1",
            "source_phase_index": 0,
            "source_n_times": 4,
            "source_image_sha256": "0" * 64,
            "data_backend": "smoke",
            "seed": seed,
        },
    )
    return {
        "diffusion_loss": float(loss.detach()),
        "denoiser_architecture": denoiser_architecture,
        "denoiser_input_shape": list(details["denoiser_input"].shape),
        "sample_shape": list(samples.shape),
        "sample_value_range": [float(samples.min()), float(samples.max())],
        "output_directory": str(Path(output_directory).resolve()),
    }

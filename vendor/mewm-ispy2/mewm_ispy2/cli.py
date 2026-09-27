from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .contracts import PAPER_FAITHFUL_ARCHITECTURE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mewm-ispy2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate locked data and model inputs")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--prepare-sample", action="store_true")
    preflight.add_argument("--ct-checkpoint")
    preflight.add_argument("--vqgan-checkpoint")

    for command in ("train-vqgan", "train-diffusion", "train-segmenter"):
        stage = subparsers.add_parser(command)
        stage.add_argument("--config", required=True)
        stage.add_argument("--max-epochs", type=int, required=True)
        stage.add_argument("--max-steps", type=int, default=-1)
        stage.add_argument("--devices", default="auto")
        stage.add_argument("--resume")
        if command == "train-vqgan":
            initialization = stage.add_mutually_exclusive_group()
            initialization.add_argument("--ct-checkpoint")
            initialization.add_argument("--init-mri-checkpoint")
        elif command == "train-diffusion":
            stage.add_argument("--vqgan-checkpoint", required=True)
            stage.add_argument("--warm-start-checkpoint")

    inference = subparsers.add_parser("infer-source")
    inference.add_argument("--config", required=True)
    inference.add_argument("--transition-id", required=True)
    inference.add_argument("--vqgan-checkpoint", required=True)
    inference.add_argument("--diffusion-checkpoint", required=True)
    inference.add_argument("--segmenter-checkpoint", required=True)
    inference.add_argument("--output-dir", required=True)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--prediction-dir", required=True)
    evaluation.add_argument("--output-dir", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--seed", type=int, default=2026)
    smoke.add_argument(
        "--denoiser-architecture",
        choices=(
            "film_unet3d_v1",
            "ct_unet3d_spatial32_v1",
            PAPER_FAITHFUL_ARCHITECTURE,
        ),
        default="film_unet3d_v1",
    )

    warm_start = subparsers.add_parser("paper-warm-start")
    warm_start.add_argument("--config", required=True)
    warm_start.add_argument("--vqgan-checkpoint", required=True)
    warm_start.add_argument("--output-dir", required=True)

    gpu_smoke = subparsers.add_parser("paper-gpu-smoke")
    gpu_smoke.add_argument("--config", required=True)
    gpu_smoke.add_argument("--vqgan-checkpoint", required=True)
    gpu_smoke.add_argument("--output-dir", required=True)
    gpu_smoke.add_argument("--seed", type=int, default=2026)

    prepare_x0 = subparsers.add_parser("prepare-x0-latents")
    prepare_x0.add_argument("--config", required=True)
    prepare_x0.add_argument("--vqgan-checkpoint", required=True)
    prepare_x0.add_argument(
        "--output-dir",
        default=(
            "/path/to/research/MAM/MeWM-ISPY2/runs/"
            "registered_strict_a_mewm_x0_channelstd_v1"
        ),
    )
    prepare_x0.add_argument("--device")
    prepare_x0.add_argument("--encode-batch-size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.command == "preflight":
        from .preflight import CT_DEFAULT_PATH, run_preflight

        report = run_preflight(
            args.config,
            prepare_sample=args.prepare_sample,
            ct_checkpoint=args.ct_checkpoint or CT_DEFAULT_PATH,
            vqgan_checkpoint=args.vqgan_checkpoint,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    if args.command == "smoke":
        if args.denoiser_architecture == PAPER_FAITHFUL_ARCHITECTURE:
            from .paper_smoke import run_tiny_paper_cpu_smoke

            report = run_tiny_paper_cpu_smoke(
                Path(args.output_dir),
                seed=args.seed,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        from .smoke import run_tiny_cpu_smoke

        report = run_tiny_cpu_smoke(
            Path(args.output_dir),
            seed=args.seed,
            denoiser_architecture=args.denoiser_architecture,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "paper-warm-start":
        from .paper_smoke import run_paper_warm_start_verification

        report = run_paper_warm_start_verification(
            args.config,
            args.vqgan_checkpoint,
            Path(args.output_dir),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "paper-gpu-smoke":
        from .paper_smoke import run_paper_gpu_smoke

        report = run_paper_gpu_smoke(
            args.config,
            args.vqgan_checkpoint,
            Path(args.output_dir),
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-x0-latents":
        from .x0_preparation import prepare_x0_latents

        report = prepare_x0_latents(
            args.config,
            args.vqgan_checkpoint,
            args.output_dir,
            device=args.device,
            encode_batch_size=args.encode_batch_size,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command.startswith("train-"):
        from .workflows import run_training

        run_training(args.command.removeprefix("train-"), args)
        return 0
    if args.command == "infer-source":
        from .workflows import run_source_inference

        run_source_inference(args)
        return 0
    if args.command == "evaluate":
        from .workflows import run_evaluation

        run_evaluation(args)
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

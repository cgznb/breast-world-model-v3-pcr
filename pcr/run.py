"""Portable entry points for the retained ROI32 pCR workflow."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Export an input-only bundle from a trusted local checkpoint")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    predict = sub.add_parser("predict", help="Predict from existing real/generated Pillar embeddings")
    predict.add_argument("--bundles", nargs="+", required=True)
    predict.add_argument("--inputs", required=True)
    predict.add_argument("--output", required=True)
    encode = sub.add_parser("encode", help="Encode decoded ROI32 images with a local frozen Pillar")
    encode.add_argument("--images", required=True)
    encode.add_argument("--foreground", required=True)
    encode.add_argument("--normalization", required=True)
    encode.add_argument("--pillar", required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    os.umask(0o077)
    from release_inference import export_bundle, predict as predict_pcr

    if args.command == "export":
        export_bundle(args.checkpoint, args.output)
        print("Exported input-only classifier bundle")
    elif args.command == "predict":
        with np.load(args.inputs, allow_pickle=False) as inputs:
            required = {"embeddings", "masks", "clinical", "days"}
            if set(inputs.files) != required:
                raise ValueError("Inputs must contain embeddings, masks, clinical and days only")
            result = predict_pcr(args.bundles, **{key: inputs[key] for key in required})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as handle:
            np.savez_compressed(handle, **result)
        print(f"Predicted {len(result['probability'])} records")
    else:
        from transformers import AutoModel, PreTrainedModel
        from src.pillar import _ensure_transformers_remote_code_compatibility
        from src.registered_three_phase_pcr import build_volume
        from scripts.run_first_post_pcr import pillar_forward

        _ensure_transformers_remote_code_compatibility(PreTrainedModel)
        model = AutoModel.from_pretrained(args.pillar, trust_remote_code=True,
                                         local_files_only=True, low_cpu_mem_usage=False)
        model = model.to(args.device).float().requires_grad_(False).eval()
        with np.load(args.images, allow_pickle=False) as data:
            images = data["images"]
        foreground = np.load(args.foreground, allow_pickle=False)
        normalization = json.loads(Path(args.normalization).read_text())
        if images.ndim != 5 or images.shape[1:] != (3, 32, 128, 128):
            raise ValueError("Expected images [draws,3,32,128,128]")
        features = []
        with torch.inference_mode():
            for image in images:
                volume, _ = build_volume(image, foreground, normalization)
                features.append(pillar_forward(model, volume[None].to(args.device))[0].numpy())
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as handle:
            np.save(handle, np.stack(features), allow_pickle=False)
        print(f"Encoded {len(features)} image draws")


if __name__ == "__main__":
    main()

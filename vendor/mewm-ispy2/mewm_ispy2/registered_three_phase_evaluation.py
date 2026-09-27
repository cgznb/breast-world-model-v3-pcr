"""Source-only three-phase forecasts with image and enhancement baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from .perceptual import UncheckedLPIPSLoss
from .registered_roi32_data import file_identity, read_json, save_npz, write_json
from .registered_roi32_evaluation import gallery, image_metrics, plot_case, write_csv
from .registered_roi32_fm import bridge
from .registered_three_phase_audit import enhancement_metrics, grouped_metrics
from .registered_three_phase_data import PHASES, ThreePhaseCrops
from .registered_three_phase_latents import PhaseLatentPairs
from .registered_three_phase_model import SharedPhaseCodec
from .registered_three_phase_training import contract_for, load_selected


@torch.no_grad()
def evaluate(config, baseline, manifest, device):
    module = bridge(config)
    root = Path(config["output_dir"]) / "fm_evaluation"
    model = load_selected(config, baseline, manifest, device)
    codec, _ = vq.load_frozen(baseline, device)
    codec = SharedPhaseCodec(codec)
    contract = {**contract_for(config, baseline), "stage": "fm_evaluation",
                "selected_model": file_identity(Path(config["output_dir"]) / "fm/best-endpoint.pt"),
                "evaluation_source": file_identity(__file__)}
    if runtime.stage_complete(config, "fm_evaluation", contract):
        return True
    if (root / "binding.json").exists() and read_json(root / "binding.json") != contract:
        raise ValueError("Forecast evaluation contract changed")
    write_json(root / "binding.json", contract)
    pairs = PhaseLatentPairs(config, baseline, manifest, "val")
    crops = ThreePhaseCrops(baseline, manifest, split="val")
    crop_index = {row["visit_id"]: index for index, row in enumerate(crops.records)}
    perceptual = UncheckedLPIPSLoss.vgg().to(device).eval().requires_grad_(False)
    rows, enhancement, figures = [], [], []
    for index, record in enumerate(pairs.records):
        name = record["pair_id"].replace(":", "_").replace("->", "_to_")
        case = root / "cases" / (name + ".json")
        if case.exists():
            saved = read_json(case)
            rows.extend(saved["rows"])
            enhancement.extend(saved["enhancement"])
            figures.extend(saved["figures"])
            continue
        # Target arrays are first opened after generation has returned.
        source = pairs.read(record["earlier_visit_id"])[None].to(device)
        batch = module.collate_pairs([{"record": record, "earlier_latent": source[0], "later_latent": source[0]}])
        generator = torch.Generator(device=device).manual_seed(config["fm"]["validation_seed"] + index)
        predictions = []
        for _ in range(config["fm"]["final_samples"]):
            noise = torch.randn(source.shape, generator=generator, device=device)
            with runtime.autocast(device):
                predicted = model.sample(source, batch["conditions"], noise, steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"])
                predictions.append(codec.decode(predicted, pairs.statistics))
        target_item = crops[crop_index[record["later_visit_id"]]]
        source_item = crops[crop_index[record["earlier_visit_id"]]]
        target = target_item["image"][None].to(device)
        mask = target_item["mask"][None].to(device)
        valid = target_item["valid"][None].to(device)
        target_latent = pairs.read(record["later_visit_id"])[None].to(device)
        with runtime.autocast(device):
            reconstruction = codec.decode(target_latent, pairs.statistics)
        mean = torch.stack(predictions).mean(0)
        comparisons = [("sample", sample, value) for sample, value in enumerate(predictions)]
        comparisons += [("sample_mean", -1, mean), ("source_copy", -1, source_item["image"][None].to(device)),
                        ("vq_target", -1, reconstruction)]
        case_rows, case_enhancement = [], []
        for method, sample, predicted in comparisons:
            common = {"pair_id": record["pair_id"], "patient_id": record["patient_id"], "method": method,
                      "sample": sample, "transition": record["earlier_stage"] + "->" + record["later_stage"]}
            for phase_index, phase in enumerate(PHASES):
                metrics = image_metrics(predicted[:, phase_index:phase_index + 1], target[:, phase_index:phase_index + 1], mask,
                                        valid[:, phase_index:phase_index + 1], perceptual, record["later_visit_id"])
                case_rows.append({**common, "phase": phase, **metrics})
            case_enhancement += [{**common, **value} for value in enhancement_metrics(predicted[0], target[0], mask[0, 0], valid[0])]
        case_figures = []
        if index < 8:
            for phase_index, phase in enumerate(PHASES):
                figure = root / (name + "_" + phase + ".png")
                title = record["pair_id"] + " | " + phase
                plot_case(figure, title, {"Target": target[0, phase_index].cpu().numpy(),
                                          "Forecast mean": mean[0, phase_index].float().cpu().numpy()}, mask[0, 0].cpu().numpy())
                case_figures.append((figure.name, title))
            report = read_json(crops.records[crop_index[record["earlier_visit_id"]]]["crop_report"]["path"])
            values = torch.stack(predictions)[:, 0].float().cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise ValueError("Saved forecast overflows FP16")
            save_npz(root / "selected_samples" / (name + ".npz"), samples=values,
                     mean=mean[0].float().cpu().numpy(), affine_lps=np.asarray(report["crop_affine_lps"]))
        write_json(case, {"rows": case_rows, "enhancement": case_enhancement, "figures": case_figures})
        rows.extend(case_rows)
        enhancement.extend(case_enhancement)
        figures.extend(case_figures)
        runtime.log_event(config, "fm_evaluation", "evaluating", completed=index + 1, total=len(pairs))
        runtime.periodic_guard(config, contract)
        if runtime.STOP_REQUESTED:
            return False
    metrics = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum")
    write_csv(root / "images.csv", rows)
    write_csv(root / "enhancement.csv", enhancement)
    write_json(root / "summary.json", {"split": "val", "pairs": len(pairs),
               "phase_metrics": grouped_metrics(rows, ["method", "phase"], metrics),
               "transition_metrics": grouped_metrics(rows, ["method", "phase", "transition"], metrics),
               "enhancement_metrics": grouped_metrics(enhancement, ["method", "enhancement", "region"],
                                                      ("mae", "relative_mae", "signed_bias", "spatial_correlation"))})
    gallery(root / "index.html", "Registered three-phase ROI32 | target and forecast mean", figures)
    runtime.stage_finished(config, "fm_evaluation", contract, [root / name for name in ("summary.json", "images.csv", "enhancement.csv", "index.html")])
    return True

"""Frozen DCE0 VQ transfer audit, including interphase enhancement fidelity."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from .perceptual import UncheckedLPIPSLoss
from .registered_roi32_data import file_identity, read_json, save_npz, verify_identity, visit_filename, write_json
from .registered_roi32_evaluation import gallery, image_metrics, write_csv
from .registered_three_phase_data import PHASES, ThreePhaseCrops


def enhancement_metrics(predicted, target, mask, valid):
    if predicted.shape != target.shape or target.shape[0] != 3:
        raise ValueError("Enhancement metrics require three ordered phases")
    results = []
    for index, name in ((1, "first_post_minus_pre"), (2, "late_minus_pre")):
        real, generated = target[index] - target[0], predicted[index] - predicted[0]
        support = valid[index] & valid[0]
        for region, selected in (("foreground", support), ("t0_roi", support & mask)):
            if not selected.any():
                raise ValueError("No common support for enhancement evaluation")
            expected, actual = real[selected].double(), generated[selected].double()
            error = actual - expected
            signal = float(expected.abs().mean())
            x, y = expected - expected.mean(), actual - actual.mean()
            denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
            results.append({"enhancement": name, "region": region, "mae": float(error.abs().mean()),
                            "signed_bias": float(error.mean()), "target_absolute_signal": signal,
                            "relative_mae": float(error.abs().mean()) / signal if signal > 1e-8 else None,
                            "spatial_correlation": float((x * y).sum()) / denominator if denominator > 1e-8 else None,
                            "voxels": int(selected.sum())})
    return results


def grouped_metrics(rows, keys, metrics):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, values in sorted(buckets.items(), key=lambda item: str(item[0])):
        by_patient = defaultdict(list)
        for row in values:
            by_patient[row["patient_id"]].append(row)
        result = {**dict(zip(keys, key, strict=True)), "visits": len(values), "patients": len(by_patient)}
        for metric in metrics:
            observations = [v[metric] for v in values if v[metric] is not None]
            patient_means = [np.mean([v[metric] for v in visits if v[metric] is not None])
                             for visits in by_patient.values() if any(v[metric] is not None for v in visits)]
            result[metric] = float(np.mean(patient_means)) if patient_means else None
            result[metric + "_visit_mean"] = float(np.mean(observations)) if observations else None
        output.append(result)
    return output


def plot_phases(path, source, reconstruction, mask, visit_id):
    path = Path(path)
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    foreground = source[source != 0]
    lower, upper = np.percentile(foreground, (1, 99))
    if upper <= lower:
        upper = lower + 1
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    figure.suptitle(f"{visit_id} | frozen DCE0 VQ | registered ROI32", fontsize=12)
    for index, phase in enumerate(PHASES):
        for column, (label, data) in enumerate((("Real", source), ("Reconstruction", reconstruction),
                                               ("Absolute error", np.abs(source - reconstruction)))):
            panel = axes[index, column]
            panel.imshow(data[index, z], cmap="magma" if column == 2 else "gray",
                         vmin=0 if column == 2 else lower, vmax=0.5 if column == 2 else upper)
            if mask[z].any():
                panel.contour(mask[z], levels=[0.5], colors=["lime"], linewidths=0.6)
            panel.set_title(f"{phase}: {label}", fontsize=10)
            panel.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def audit_contract(config, baseline, manifest, codec_identity):
    settings = config["three_phase"]
    selected = [r for r in manifest["visits"] if r["fold"] == settings["audit"]["split"]]
    return {"schema": config["schema"], "codec": codec_identity, "phase_order": list(PHASES),
            "split": settings["audit"]["split"], "precision": "bf16_cuda_or_fp32_cpu",
            "normalization": file_identity(Path(baseline["output_dir"]) / "data/normalization.json"),
            "sources": manifest["sources"], "selection": [r["visit_id"] for r in selected],
            "runtime_sources": [file_identity(path) for path in (__file__, vq.__file__)],
            "interpretation": "development_transfer_diagnostic_not_an_independent_test_or_finetuning_comparison"}


def write_report(root, summary):
    lines = ["# Registered ROI32: frozen single-phase VQ on three phases", "",
             f"Validation: {summary['patients']} patients, {summary['visits']} visits, three phases per visit.", "",
             "One fixed T0 crop and the frozen training-DCE0 mean/std are shared by all phases. "
             "Late phase uses the explicit clinical phase table. No training or optimizer updates occur in this audit.", "",
             "Patient-macro reconstruction metrics (visits are averaged within each patient first):", "",
             "| Phase | Foreground L1 | T0-region L1 | SSIM [-1,3] | LPIPS sum |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["phase_metrics"]:
        lines.append(f"| {row['phase']} | {row['valid_l1']:.6f} | {row['t0_roi_l1']:.6f} | "
                     f"{row['ssim_clamped_neg1_pos3']:.6f} | {row['lpips_three_slices_sum']:.6f} |")
    lines += ["", "Patient-macro enhancement fidelity, in the same frozen intensity units:", "",
              "| Enhancement | Region | MAE | Absolute target signal | Relative MAE | Correlation |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["enhancement_metrics"]:
        values = ["NA" if row[key] is None else f"{row[key]:.6f}" for key in
                  ("mae", "target_absolute_signal", "relative_mae", "spatial_correlation")]
        lines.append(f"| {row['enhancement']} | {row['region']} | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation", "",
              "Different phases have different signal and texture, so a larger raw L1 alone does not prove that "
              "retraining is necessary. Compare T0-region errors, LPIPS, enhancement attenuation and the saved images. "
              "Codebook occupancy alone is not an acceptance criterion.", "",
              "This audit measures direct transfer only. It cannot establish that a separately trained or mixed-phase "
              "codec is better without a matched comparison. A useful next comparison keeps this frozen baseline and "
              "fine-tunes a copy on equally sampled training pre/first-post/late images; retain precontrast examples "
              "to measure and limit forgetting. The shared single-channel architecture need not change.", "",
              "The existing validation cohort is development data used for model selection. T0 contours are model "
              "predictions and may be clipped; T0-region scores are not follow-up lesion segmentation accuracy. "
              "SSIM uses clipping, whereas L1 and enhancement errors use unclipped images.", "",
              "[Reconstruction examples](index.html)", ""]
    (root / "report.md").write_text("\n".join(lines))


@torch.no_grad()
def evaluate(config, baseline, manifest, device):
    root = Path(config["output_dir"]) / "vq_audit"
    codec, identity = vq.load_frozen(baseline, device)
    contract = audit_contract(config, baseline, manifest, identity)
    contract["device_type"] = device.type
    if runtime.stage_complete(config, "vq_audit", contract):
        return read_json(root / "summary.json")
    if (root / "binding.json").exists():
        if read_json(root / "binding.json") != contract:
            raise ValueError("VQ audit contract changed")
    else:
        write_json(root / "binding.json", contract)
    original_codebook = {k: value.clone() for k, value in codec.quantizer.state_dict().items()}
    dataset = ThreePhaseCrops(baseline, manifest, split=contract["split"])
    selected = []
    for stage in ("T0", "T1", "T2", "T3"):
        for clipped in (False, True):
            candidates = [r for r in dataset.records if r["visit"] == stage
                          and read_json(r["crop_report"]["path"])["largest_clipped"] == clipped]
            if candidates:
                selected.append(min(candidates, key=lambda row: row["visit_id"])["visit_id"])
    examples = set(selected[:config["three_phase"]["audit"]["examples"]])
    pending, rows, enhancements, figures = [], [], [], []
    counts = np.zeros((3, config["vq"]["model"]["n_codes"]), dtype=np.int64)
    for record in dataset.records:
        path = root / "cases" / visit_filename(record["visit_id"]).replace(".npz", ".json")
        if not path.exists():
            pending.append(record)
            continue
        saved = read_json(path)
        if saved["sources"] != [record[k] for k in ("phase_sources", "metadata_source", "crop_report")]:
            raise ValueError("Audit source changed during resume")
        verify_identity(saved["code_counts"])
        with np.load(saved["code_counts"]["path"], allow_pickle=False) as arrays:
            counts += arrays["counts"]
        rows.extend(saved["phase_rows"])
        enhancements.extend(saved["enhancement_rows"])
        if saved["figure"]:
            verify_identity(saved["figure"])
            figures.append((Path(saved["figure"]["path"]).name, record["visit_id"]))
    perceptual = UncheckedLPIPSLoss.vgg().to(device).eval().requires_grad_(False)
    loader = runtime.evaluation_loader(ThreePhaseCrops(baseline, manifest, records=pending), 1,
                                        config["runtime"]["loader_workers"]) if pending else []
    by_id = {row["visit_id"]: row for row in dataset.records}
    completed = len(dataset) - len(pending)
    for batch in loader:
        visit_id = batch["visit_id"][0]
        record = by_id[visit_id]
        image = batch["image"][0, :, None].to(device)
        with runtime.autocast(device):
            latent = codec.encode_continuous(image)
            quantized, information = codec.quantizer(latent)
            prediction = codec.decode(quantized).float()
        mask = batch["mask"].to(device)
        valid = batch["valid"][0, :, None].to(device)
        phase_rows, code_counts = [], []
        common = {"visit_id": visit_id, "patient_id": record["patient_id"], "visit": record["visit"],
                  "t0_largest_clipped": bool(batch["largest_clipped"][0])}
        for phase_index, phase in enumerate(PHASES):
            metrics = image_metrics(prediction[phase_index:phase_index + 1], image[phase_index:phase_index + 1], mask,
                                    valid[phase_index:phase_index + 1], perceptual, visit_id)
            metrics["quantization_mse"] = float((latent[phase_index].float() - quantized[phase_index].float()).square().mean())
            phase_rows.append({**common, "phase": phase, **metrics})
            code_counts.append(torch.bincount(information["indices"][phase_index].flatten(), minlength=counts.shape[1]).cpu().numpy())
        case_enhancement = [{**common, **value} for value in enhancement_metrics(prediction[:, 0], image[:, 0], mask[0, 0], valid[:, 0])]
        name = visit_filename(visit_id)
        counts_path = root / "code_counts" / name
        case_counts = np.stack(code_counts)
        save_npz(counts_path, counts=case_counts)
        figure = None
        if visit_id in examples:
            path = root / name.replace(".npz", ".png")
            plot_phases(path, image[:, 0].float().cpu().numpy(), prediction[:, 0].cpu().numpy(), mask[0, 0].cpu().numpy(), visit_id)
            figure = file_identity(path)
            figures.append((path.name, visit_id))
        write_json(root / "cases" / name.replace(".npz", ".json"),
                   {"sources": [record[k] for k in ("phase_sources", "metadata_source", "crop_report")],
                    "phase_rows": phase_rows, "enhancement_rows": case_enhancement,
                    "code_counts": file_identity(counts_path), "figure": figure})
        counts += case_counts
        rows.extend(phase_rows)
        enhancements.extend(case_enhancement)
        completed += 1
        if completed == 1 or completed % 10 == 0 or completed == len(dataset):
            runtime.log_event(config, "vq_audit", "evaluating", completed=completed, total=len(dataset))
            runtime.periodic_guard(config, contract)
        if runtime.STOP_REQUESTED:
            return None
    if any(not torch.equal(value, codec.quantizer.state_dict()[key]) for key, value in original_codebook.items()):
        raise RuntimeError("Frozen reconstruction changed the VQ codebook")
    phase_metrics = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum", "quantization_mse")
    enhancement_fields = ("mae", "signed_bias", "target_absolute_signal", "relative_mae", "spatial_correlation")
    occupancy = []
    for index, phase in enumerate(PHASES):
        probabilities = counts[index][counts[index] > 0] / counts[index].sum()
        occupancy.append({"phase": phase, "active_codes": len(probabilities),
                          "perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum()))})
    summary = {"status": "completed", "split": contract["split"], "visits": len(dataset),
               "patients": len({row["patient_id"] for row in dataset.records}), "codec": identity,
               "codebook_unchanged": True, "phase_order": list(PHASES),
               "phase_metrics": grouped_metrics(rows, ["phase"], phase_metrics),
               "phase_by_visit": grouped_metrics(rows, ["phase", "visit"], phase_metrics),
               "phase_by_crop_coverage": grouped_metrics(rows, ["phase", "t0_largest_clipped"], phase_metrics),
               "enhancement_metrics": grouped_metrics(enhancements, ["enhancement", "region"], enhancement_fields),
               "codebook": occupancy, "fine_tuning_comparison_performed": False}
    write_csv(root / "phases.csv", rows)
    write_csv(root / "enhancement.csv", enhancements)
    write_json(root / "summary.json", summary)
    gallery(root / "index.html", "Registered three-phase ROI32 | frozen DCE0 VQ", sorted(figures))
    write_report(root, summary)
    runtime.stage_finished(config, "vq_audit", contract, [root / name for name in
                           ("summary.json", "phases.csv", "enhancement.csv", "index.html", "report.md")], visits=len(dataset))
    return summary

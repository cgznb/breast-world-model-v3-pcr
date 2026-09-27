"""Image-space metrics, paired uncertainty, and figures for I-SPY2 generators."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import minimum_filter
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_ispy2_generators import (
    ROOT, OUT, BF, context, read_prediction, trajectories, _atomic_csv,
    _atomic_json, _atomic_torch_save, model_root, MODELS,
)
from mewm_ispy2.ispy2_biflow_cohort_evaluation import comparison_metrics, region_masks, SSIM_WIN_SIZE
from mewm_ispy2.source_bridge_evaluation import change_metrics

LABELS = {"biflow": "BiFlow (97,104)", "symmflow": "SymmFlow (100,000)",
          "bridge025": "Bridge-0.25", "bridge050": "Bridge-0.50",
          "source_copy": "Source copy", "target_vq": "Target VQ reconstruction", "real": "Real"}
METRICS = ("mae_unclipped", "rmse_unclipped", "psnr_windowed_db", "ssim_3d_windowed")
CHANGE_METRICS = ("change_cosine", "predicted_change_mae", "true_change_mae")
REGIONS = {"common_foreground", "target_tumor", "tumor_union_neighborhood"}


def model_label(model):
    path = model_root(model) / "checkpoint_provenance.json"
    if model.startswith("bridge") and path.exists():
        provenance = json.loads(path.read_text())
        suffix = "; interim" if provenance["intermediate_checkpoint"] else ""
        return f"{LABELS[model]} ({provenance['optimizer_step']:,}{suffix})"
    return LABELS[model]


def pending_label(model):
    if model.startswith("bridge") and not (BF / f"runs/source_bridge_pilot_20260910/{model}/checkpoints/best.ckpt").exists():
        return "No trained checkpoint"
    return "Pending aligned evaluation"


def bridge_training_observation():
    commands = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            commands.append(path.read_bytes().decode().split("\0"))
        except (OSError, UnicodeDecodeError):
            continue
    rows = []
    for model in ("bridge025", "bridge050"):
        directory = BF / f"runs/source_bridge_pilot_20260910/{model}"
        config = str(BF / f"configs/source_bridge_pilot_{model}.yaml")
        active = any("mewm_ispy2.source_bridge_workflow" in command and "train" in command
                     and config in command for command in commands)
        progress_path = directory / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        completed_path = directory / "training_result.json"
        completed = completed_path.exists() and json.loads(completed_path.read_text()).get("status") == "completed"
        state = ("Training" if active else "Completed" if completed else
                 "No live training process" if progress else "Not started")
        rows.append({"model": model, "state": state,
                     "last_logged_step": progress.get("optimizer_step"),
                     "trained_checkpoint_exists": (directory / "checkpoints/best.ckpt").exists()})
    value = {"observed_utc": datetime.now(timezone.utc).isoformat(), "models": rows}
    _atomic_json(OUT / "bridge_training_observation.json", value)
    return value


def comparison_ssim_centers(foreground):
    # A separable minimum filter equals erosion by the full cubic SSIM window.
    return minimum_filter(np.asarray(foreground, dtype=bool), size=SSIM_WIN_SIZE,
                          mode="constant", cval=0)


def reusable_metrics(previous, sources):
    required = {"model", "region", *METRICS, *CHANGE_METRICS,
                "voxel_count", "ssim_center_count"}
    if not required <= set(previous):
        return previous.iloc[:0]
    valid = [model for model in sources
             if len(previous[previous.model == model]) == len(REGIONS)
             and set(previous.loc[previous.model == model, "region"]) == REGIONS]
    return previous[previous.model.isin(valid)].copy()


def vq_path(route):
    return OUT / "controls/target_vq" / route.patient_id / f"T{route.target_stage}.pt"


def controls(device):
    from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config
    from mewm_ispy2.ispy2_dce0_world_latents import ISPY2DCE0ContinuousLatentCache
    config, _, routes, _, _, _, loader = context()
    base = load_ispy2_biflow_config(config["biflow"]["config"])
    latents = ISPY2DCE0ContinuousLatentCache(base.base.data.continuous_root)
    codec, decode = trajectories._load_vqgan(config, device)
    for i, route in enumerate(routes["direct_t0"], 1):
        destination = vq_path(route)
        if destination.exists():
            value = torch.load(destination, weights_only=True)
            if value.shape != (1, 96, 256, 256) or not torch.isfinite(value).all():
                raise ValueError("Invalid target reconstruction cache")
            continue
        with torch.inference_mode():
            if route.target_visit_id in latents.visit_ids:
                z = latents.denormalize(latents.load(route.target_visit_id))[None, None].to(device)
            else:
                target, _, _ = loader.load(route.patient_id, route.target_stage)
                z = codec.encode_continuous(target[:1][None].float().to(device))[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = decode(codec, z)[0, 0].cpu().half()
        _atomic_torch_save(destination, image)
        if i % 25 == 0 or i == 296:
            print(f"Target VQ reconstructions {i}/296", flush=True)


def lesion_mask(loaded, roi_cache, patient_id, stage):
    visit = loaded.visits.get(f"{patient_id}:T{stage}")
    if visit is None:
        return np.zeros((96, 256, 256), dtype=bool)
    return roi_cache.load(visit).mask[0].bool().numpy()


def images(models, strategy, workers):
    directory = OUT / "image_metrics" / strategy
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".metrics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _images(models, strategy, workers)


def _images(models, strategy, workers):
    _, _, routes, _, loaded, cache, loader = context()
    selected = routes[strategy]
    metric_root = OUT / "image_metrics" / strategy

    def one(route):
        sources = ["source_copy", "target_vq", *models]
        path = metric_root / "shards" / route.patient_id / f"T{route.target_stage}.csv"
        previous = pd.DataFrame()
        if path.exists():
            previous = reusable_metrics(pd.read_csv(path), sources)
            if not previous.empty and set(previous.model) == set(sources):
                return previous
        source, sf, _ = loader.load(route.patient_id, route.source_stage)
        target, tf, _ = loader.load(route.patient_id, route.target_stage)
        regions = region_masks(sf.numpy(), tf.numpy(),
            lesion_mask(loaded, cache, route.patient_id, route.source_stage),
            lesion_mask(loaded, cache, route.patient_id, route.target_stage))
        if not regions["common_foreground"].any():
            raise ValueError("Empty common foreground")
        centers = comparison_ssim_centers(regions["common_foreground"])
        rows = previous.to_dict("records")
        completed = set(previous.model) if not previous.empty else set()
        for model in sources:
            if model in completed:
                continue
            if model == "source_copy":
                prediction = source[:1]
            elif model == "target_vq":
                prediction = torch.load(vq_path(route), weights_only=True)
            else:
                prediction = read_prediction(model, route)
            metrics = comparison_metrics(prediction, target[:1], regions, centers)
            for region, values in metrics.items():
                changes = change_metrics(prediction[0].float(), source[0].float(), target[0].float(), regions[region])
                rows.append(dict(patient_id=route.patient_id, target_stage=route.target_stage,
                                 source_stage=route.source_stage, delta_days=route.delta_days,
                                 model=model, region=region, **values, **changes))
        frame = pd.DataFrame(rows)
        _atomic_csv(path, frame)
        return frame

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(one, selected), 1):
            frames.append(frame)
            if i % 20 == 0 or i == len(selected):
                print(f"Image metrics {strategy}: {i}/{len(selected)}", flush=True)
    endpoint = pd.concat(frames, ignore_index=True)
    if endpoint.duplicated(["patient_id", "target_stage", "model", "region"]).any():
        raise ValueError("Duplicated image comparison endpoint")
    _atomic_csv(metric_root / "endpoint_metrics.csv", endpoint)
    patient = endpoint.groupby(["model", "region", "patient_id"])[list(METRICS+CHANGE_METRICS)].mean().reset_index()
    _atomic_csv(metric_root / "patient_metrics.csv", patient)
    summary = []
    for values, group in patient.groupby(["model", "region"], sort=False):
        row = dict(model=values[0], region=values[1], patients=len(group), endpoints=296)
        for metric in METRICS+CHANGE_METRICS:
            v = group[metric].dropna().to_numpy()
            row[f"{metric}_n"] = len(v)
            row[f"{metric}_mean"] = np.mean(v) if len(v) else np.nan
            row[f"{metric}_std"] = np.std(v, ddof=1) if len(v) > 1 else np.nan
            if len(v):
                rng = np.random.default_rng(20260911)
                means = np.mean(v[rng.integers(0, len(v), size=(2000, len(v)))], axis=1)
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(means, [.025, .975])
        summary.append(row)
    _atomic_csv(metric_root / "summary.csv", pd.DataFrame(summary))
    stage = endpoint.groupby(["model", "region", "target_stage", "patient_id"])[list(METRICS)].mean().reset_index()
    stage = stage.groupby(["model", "region", "target_stage"])[list(METRICS)].agg(["mean", "std", "count"])
    stage.columns = ["_".join(column) for column in stage.columns]
    _atomic_csv(metric_root / "by_target_stage.csv", stage.reset_index())


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def gray(array, low, high, size=240):
    scaled = np.clip((array.astype(np.float32) - low) / (high-low), 0, 1)
    return Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).convert("RGB").resize((size, size))


def wrap_label(label, face, width):
    lines = []
    for paragraph in label.splitlines():
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if current and face.getlength(trial) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    return "\n".join(lines)


def figures(models, strategy, scope=MODELS):
    _, ids, routes, _, _, _, loader = context()
    output = OUT / "figures" / strategy
    output.mkdir(parents=True, exist_ok=True)
    complete = sorted(pid for pid in ids if sum(r.patient_id == pid for r in routes[strategy]) == 3)
    chosen = sorted(np.random.default_rng(20260911).choice(complete, 3, replace=False).tolist())
    _atomic_json(output / "display_selection.json", {
        "patients": chosen, "selection": "seeded random from patients with all three future visits",
        "seed": 20260911, "slice": "axial z=48 in shared T0 crop", "target_metrics_used": False,
        "window": "source T0 foreground p0.5-p99.5, shared by all columns and future visits",
    })
    columns = ["Source T0", "Real target", "Target VQ", *[model_label(m) for m in scope]]
    tile, gap, label_width, top = 240, 10, 140, 85
    width = label_width + (tile+gap)*len(columns) + gap
    paths = []
    for pid in chosen:
        rows = sorted([r for r in routes[strategy] if r.patient_id == pid], key=lambda r:r.target_stage)
        canvas = Image.new("RGB", (width, top + len(rows)*(tile+55)+30), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), f"I-SPY2 | {pid} | {strategy} | Euler-20, candidate 0", fill="black", font=font(23, True))
        for j, label in enumerate(columns):
            face = font(15, True)
            draw.multiline_text((label_width+j*(tile+gap), 45), wrap_label(label, face, tile), fill="black", font=face, spacing=2)
        original, foreground, _ = loader.load(pid, 0)
        source = original[0].float().numpy()
        low, high = np.percentile(source[foreground.numpy()], [.5, 99.5])
        if high <= low:
            raise ValueError("Degenerate source display window")
        for i, route in enumerate(rows):
            y = top+i*(tile+55)
            draw.text((14, y+88), f"T0 -> T{route.target_stage}", fill="black", font=font(18, True))
            draw.text((14, y+116), f"{route.delta_days} days", fill="#333333", font=font(16))
            target, _, _ = loader.load(pid, route.target_stage)
            arrays = [source, target[0].float().numpy(), torch.load(vq_path(route), weights_only=True)[0].float().numpy()]
            arrays += [read_prediction(m, route)[0].float().numpy() if m in models else None for m in scope]
            for j, array in enumerate(arrays):
                x = label_width+j*(tile+gap)
                if array is not None:
                    canvas.paste(gray(array[48], low, high), (x, y))
                else:
                    draw.rectangle((x,y,x+tile,y+tile), fill="#eeeeee")
                    draw.multiline_text((x+15,y+100), wrap_label(pending_label(scope[j-3]), font(16), tile-30),
                                        fill="#666666", font=font(16), spacing=4)
            draw.text((label_width,y+tile+9), f"Shared intensity window [{low:.3f}, {high:.3f}]; standardized DCE0; axial z=48", fill="#444444", font=font(15))
        path = output / f"{pid}.png"
        canvas.save(path)
        paths.append(path)
    return paths


def pcr_uncertainty():
    frame = pd.read_csv(OUT / "pcr/predictions.csv", dtype={"patient_id": str})
    ensemble = frame.groupby(["model", "strategy", "temporal_depth", "patient_id"]).agg(
        label=("label", "first"), probability=("probability", "mean")
    ).reset_index()
    rows = []
    for (depth, model, strategy), group in ensemble.groupby(["temporal_depth", "model", "strategy"]):
        if model == "real":
            continue
        group = group.sort_values("patient_id")
        y = group.label.to_numpy(); p = group.probability.to_numpy()
        rng = np.random.default_rng(20260911)
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = np.concatenate([rng.choice(positive, (2000,len(positive))),
                                rng.choice(negative, (2000,len(negative)))], axis=1)
        for reference in ("real", "biflow"):
            if model == reference:
                continue
            ref = ensemble[(ensemble.temporal_depth == depth) & (ensemble.model == reference)]
            if reference != "real":
                ref = ref[ref.strategy == strategy]
            ref = ref.set_index("patient_id").loc[group.patient_id]
            if not np.array_equal(y, ref.label.to_numpy()):
                raise ValueError("Paired bootstrap cohort differs")
            q = ref.probability.to_numpy()
            for name, fn in (("auroc", roc_auc_score), ("prauc", average_precision_score)):
                differences = [fn(y[ix],p[ix])-fn(y[ix],q[ix]) for ix in draws]
                low, high = np.quantile(differences, [.025,.975])
                rows.append(dict(temporal_depth=depth, model=model, strategy=strategy,
                                 reference=reference, metric=name, delta=fn(y,p)-fn(y,q),
                                 ci_low=low, ci_high=high, bootstrap_samples=2000,
                                 estimator="ten_seed_mean_probability_predictor"))
    _atomic_csv(OUT / "pcr/paired_bootstrap.csv", pd.DataFrame(rows))


def pcr_plot(summary, models, strategy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depths = ("T0", "T0-T1", "T0-T2", "T0-T3")
    colors = {"real": "#333333", "biflow": "#2878a5", "symmflow": "#c05245",
              "bridge025": "#438b61", "bridge050": "#b78525"}
    frame = summary[summary.threshold_policy == "development_oof_bacc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("auroc", "prauc"), ("AUROC", "PR-AUC")):
        for model in ("real", *models):
            group = frame[(frame.model == model) & (frame.strategy == ("real" if model == "real" else strategy))]
            group = group.set_index("temporal_depth").loc[list(depths)]
            axis.errorbar(np.arange(4), group[f"{metric}_mean"], yerr=group[f"{metric}_std"],
                          label=model_label(model), color=colors[model], marker="o", markersize=4,
                          linewidth=1.6, capsize=3, linestyle="--" if model == "real" else "-")
        axis.set_xticks(np.arange(4), depths)
        axis.set_title(title)
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Frozen pCR | 102 patients | 10 seeds | mean +/- seed SD", fontsize=12)
    directory = OUT / "figures" / strategy
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"pcr_comparison.{suffix}", dpi=180)
    plt.close(fig)


def report(models, strategy, scope=MODELS):
    image_summary = pd.read_csv(OUT / "image_metrics" / strategy / "summary.csv")
    pcr = pd.read_csv(OUT / "pcr/summary.csv")
    pcr_plot(pcr, models, strategy)
    training = bridge_training_observation() if any(m.startswith("bridge") for m in scope) else None
    selection = json.loads((OUT / "figures" / strategy / "display_selection.json").read_text())
    lines = ["# I-SPY2 Generator Comparison", "", "Date: 2026-09-11", "",
        "Scope: " + ", ".join(model_label(m) for m in scope) + ".", "",
        "BiFlow uses the 97,104-update checkpoint; SymmFlow uses the completed 100,000-update EMA weights.",
        "Checkpoint labels identify the frozen weights used for this evaluation.",
        "Training budgets, conditions, and checkpoint criteria differ. This is a descriptive comparison of available systems.", "",
        "| Model | Aligned evaluation |", "|---|---|"]
    for model in scope:
        lines.append(f"| {model_label(model)} | {'Complete' if model in models else pending_label(model)} |")
    if training is not None:
        lines += ["", f"Bridge training observed at {training['observed_utc']}:", "",
                  "| Model | Training state | Last logged updates |", "|---|---|---:|"]
        for row in training["models"]:
            if row["model"] not in scope:
                continue
            step = f"{row['last_logged_step']:,} / 40,000" if row["last_logged_step"] is not None else "None"
            lines.append(f"| {LABELS[row['model']]} | {row['state']} | {step} |")
    lines += ["",
        "## Generated Images", "",
        "All models use the same 102 patients and 296 future targets, direct from real T0, Euler-20, one fixed candidate.",
        "No target-selected best-of-K samples or sample-mean smoothing is used. All columns share the source-derived intensity window.", ""]
    for pid in selection["patients"]:
        lines += [f"![{pid}](figures/{strategy}/{pid}.png)", ""]
    lines += ["## Image Metrics", "", "Patient macro means; unclipped standardized DCE0 for MAE/RMSE, the existing fixed window for PSNR/3D SSIM.",
              "Metric CSVs retain patient-bootstrap intervals, per-patient values, target-tumor and tumor-neighborhood regions, and stage-specific results.", "",
              "| Model | Foreground MAE | RMSE | PSNR (dB) | 3D SSIM |", "|---|---:|---:|---:|---:|"]
    selected = image_summary[image_summary.region == "common_foreground"].set_index("model")
    for model in ["source_copy", "target_vq", *scope]:
        if model in selected.index:
            row = selected.loc[model]
            values = " | ".join(f"{row[m+'_mean']:.5f}" for m in METRICS)
        else:
            values = " | ".join(["Pending"]*len(METRICS))
        lines.append(f"| {model_label(model)} | {values} |")
    lines += ["", "## Frozen pCR Evaluation", "",
        "Only future DCE0 is replaced. Real future early/late DCE, the existing clinical variables, temporal metadata, and missing-visit policy are retained.",
        "These are not predictions from T0 alone or fully generated future examinations.", "",
        "T0/T0-T1 use five-fold best200; T0-T2/T0-T3 use the previous optimized five-fold layerwise AdamW policies without residual L2.",
        "Models and development OOF thresholds are frozen. No downstream training or threshold tuning was performed.",
        "Values below are ten-seed mean AUROC / PR-AUC. The CSV includes seed SDs and all eight metrics at development OOF and fixed-0.5 thresholds.", "",
        f"![Frozen pCR comparison](figures/{strategy}/pcr_comparison.png)", "",
        "| Input | T0 | T0-T1 | T0-T2 | T0-T3 |", "|---|---:|---:|---:|---:|"]
    pcr = pcr[pcr.threshold_policy == "development_oof_bacc"]
    for model in ["real", *scope]:
        selected = pcr[(pcr.model == model) & (pcr.strategy == ("real" if model == "real" else strategy))].set_index("temporal_depth")
        values = ([f"{selected.loc[d,'auroc_mean']:.5f} / {selected.loc[d,'prauc_mean']:.5f}" for d in ("T0","T0-T1","T0-T2","T0-T3")]
                  if not selected.empty else ["Shared real T0", "Pending", "Pending", "Pending"])
        lines.append("| " + model_label(model) + " | " + " | ".join(values) + " |")
    lines += ["",
        "Seed SD measures pCR-model training variability, not patient uncertainty. `pcr/paired_bootstrap.csv` provides stratified paired patient-bootstrap intervals for the ten-seed mean-probability predictor.", "",
        "The 102 patients were used for generator validation and have been inspected repeatedly in downstream experiments. Results are descriptive, not an untouched external test.", "",
        "## Artifacts", "",
        "- `pcr/predictions.csv`: all patient predictions.",
        "- `pcr/fold_predictions.csv`: new generators' individual fold predictions.",
        "- `pcr/metrics_per_seed.csv`, `pcr/summary.csv`, `pcr/paired_bootstrap.csv`.",
        f"- `image_metrics/{strategy}/endpoint_metrics.csv`, `patient_metrics.csv`, `summary.csv`, `by_target_stage.csv`.",
        "- `symmflow/direct_t0/`: the 296 new decoded volumes, Pillar embeddings, and completion records.",
        "- `symmflow/checkpoint_provenance.json`, `symmflow/sampling_audit.json`, and `pcr/audit.json`.",
        "- BiFlow decoded volumes are reused from the existing `MeWM-ISPY2-DCE0-BiFlowNet-RF/runs/ispy2_dce0_biflow_original_resume_no_early_stop_v2/generated_trajectories/full978_locked102_euler20_seed22026/` output.",
        "- BiFlow pCR predictions are reused from `results/mewm_ispy2_full978_locked102_final_hybrid_policy/`; the new comparison directory contains only a one-case BiFlow embedding parity check.", ""]
    path = OUT / "report.md"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("controls", "images", "figures", "uncertainty", "report"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["biflow", "symmflow", "bridge025"])
    parser.add_argument("--scope", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--strategy", choices=("direct_t0", "rollout_generated_dce0_real_ser"), default="direct_t0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not set(args.models) <= set(args.scope):
        parser.error("Evaluated models must be included in the report scope")
    torch.set_num_threads(2)
    if args.stage == "controls":
        controls(torch.device(args.device))
    elif args.stage == "images":
        images(args.models, args.strategy, args.workers)
    elif args.stage == "figures":
        figures(args.models, args.strategy, args.scope)
    elif args.stage == "uncertainty":
        pcr_uncertainty()
    else:
        print(report(args.models, args.strategy, args.scope))


if __name__ == "__main__":
    main()

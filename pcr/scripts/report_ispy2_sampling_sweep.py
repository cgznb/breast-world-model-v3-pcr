"""Publish matched image metrics, frozen pCR results, and twenty patient sheets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import ispy2_sampling_sweep as sweep

OUT = sweep.OUT
POLICY_LABELS = {"direct_t0": "From real T0", "adjacent_real": "From previous real visit",
                 sweep.ROLLOUT: "From generated previous DCE0"}
MODEL_LABELS = {"biflow": "BiFlow", "symmflow": "SymmFlow", "real": "Real images"}
COLORS = {"biflow": "#187568", "symmflow": "#bc4c42"}
REGION_LABELS = {"common_foreground": "Common foreground", "target_tumor": "Target tumor",
                 "tumor_union_neighborhood": "Tumor union neighborhood"}


def image_plot_name(region):
    return "image_step_comparison" if region == "common_foreground" else f"image_{region}_step_comparison"


def region_coverage():
    endpoints = pd.read_csv(OUT / "comparison/image_endpoints.csv",
                            usecols=["patient_id", "target_stage", "region", "voxel_count"])
    endpoints = endpoints.drop_duplicates(["patient_id", "target_stage", "region"])
    return {region: dict(total_targets=len(frame), valid_targets=int(frame.voxel_count.gt(0).sum()),
                         patients=int(frame.loc[frame.voxel_count.gt(0), "patient_id"].nunique()))
            for region, frame in endpoints.groupby("region")}


def aggregate(steps):
    image_frames, pcr_frames, predictions = [], [], []
    for model in sweep.MODELS:
        for count in steps:
            root = sweep.setting_root(model, count)
            for name, state in (("audit", "passed"), ("complete", "completed")):
                record = json.loads((root / f"{name}.json").read_text())
                if record.get("status") != state or record.get("model") != model or record.get("steps") != count:
                    raise ValueError(f"Unverified setting: {root}")
            image_frames.append(pd.read_csv(root / "image_metrics/endpoint_metrics.csv"))
            pcr_frames.append(pd.read_csv(root / "pcr/summary.csv").assign(steps=count, setting_model=model))
            predictions.append(pd.read_csv(root / "pcr/predictions.csv").assign(steps=count, setting_model=model))
    endpoints = pd.concat(image_frames, ignore_index=True)
    supports = endpoints.groupby(["patient_id", "target_stage", "region"])[["voxel_count", "ssim_center_count"]]
    if not supports.nunique().eq(1).all().all():
        raise ValueError("Image evaluation supports differ across models or steps")
    if len(endpoints) != len(steps) * 2 * 2664:
        raise ValueError("Incomplete image comparison")
    probabilities = pd.concat(predictions, ignore_index=True)
    real = probabilities[probabilities.model == "real"]
    real_groups = real.groupby(["temporal_depth", "seed", "patient_id"])
    if ((real_groups.probability.max() - real_groups.probability.min()).max() > 1e-8
        or not real_groups.label.nunique().eq(1).all()):
        raise ValueError("Frozen real predictions differ across sampling settings")
    initial = probabilities[probabilities.temporal_depth == "T0"]
    if initial.groupby(["seed", "patient_id"]).probability.apply(lambda x: x.max() - x.min()).max() > 1e-8:
        raise ValueError("T0 pCR should be independent of the future generator")
    probabilities = pd.concat([
        probabilities[probabilities.model != "real"],
        real.drop_duplicates(["temporal_depth", "seed", "patient_id"]).assign(steps=0),
    ], ignore_index=True).drop(columns="setting_model")
    if (probabilities.duplicated(["model", "steps", "strategy", "temporal_depth", "seed", "patient_id"]).any()
        or len(probabilities) != (len(steps) * 2 * 3 + 1) * 4080):
        raise ValueError("Combined pCR prediction inventory differs")
    pcr = pd.concat(pcr_frames, ignore_index=True)
    pcr = pcr[pcr.threshold_policy == "development_oof_bacc"].copy()
    real_summary = pcr[pcr.model == "real"].drop_duplicates("temporal_depth").assign(steps=0)
    pcr = pd.concat([pcr[pcr.model != "real"], real_summary], ignore_index=True).drop(columns="setting_model")
    columns = list(sweep.METRIC_COLUMNS)
    patient = endpoints.groupby(["model", "steps", "strategy", "region", "patient_id"])[columns].mean().reset_index()
    image_summary = patient.groupby(["model", "steps", "strategy", "region"])[columns].agg(["mean", "std", "count"])
    image_summary.columns = ["_".join(c) for c in image_summary.columns]
    image_summary = image_summary.reset_index()
    stage_summary = endpoints.groupby(["model", "steps", "strategy", "region", "target_stage"])[columns].agg(["mean", "std", "count"])
    stage_summary.columns = ["_".join(c) for c in stage_summary.columns]
    directory = OUT / "comparison"
    for name, frame in (("image_endpoints", endpoints), ("image_patients", patient),
                         ("image_summary", image_summary), ("image_by_target_stage", stage_summary.reset_index()),
                         ("pcr_summary", pcr), ("pcr_predictions", probabilities)):
        sweep.base._atomic_csv(directory / f"{name}.csv", frame)
    return image_summary, pcr


def figure_directory(steps):
    return OUT / "figures" / (f"euler_{steps[0]}" if len(steps) == 1 else "all_steps")


def write_case_pdf(pages, path):
    temporary = path.with_suffix(".tmp.pdf")
    with PdfPages(temporary, metadata={"Title": "I-SPY2 BiFlow and SymmFlow comparison"}) as pdf:
        for canvas in pages:
            figure = plt.figure(figsize=(canvas.width / 150, canvas.height / 150))
            axes = figure.add_axes([0, 0, 1, 1])
            axes.imshow(canvas, interpolation="none")
            axes.set_axis_off()
            pdf.savefig(figure, dpi=150)
            plt.close(figure)
    temporary.replace(path)


def figures(steps):
    protocol = json.loads((OUT / "protocol.json").read_text())
    patients = protocol["display_patients"]
    if len(patients) != 20 or len(set(patients)) != 20:
        raise ValueError("Exactly twenty distinct display patients are required")
    destination = figure_directory(steps)
    marker = destination / "display_audit.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("steps") == steps and previous.get("patients") == patients:
            if all((destination / f"{pid}.png").exists() for pid in patients) and (destination / "comparison_20_cases.pdf").exists():
                if previous.get("pdf_encoding") != "lossless":
                    pages = []
                    for pid in patients:
                        with Image.open(destination / f"{pid}.png") as source:
                            pages.append(source.convert("RGB"))
                    write_case_pdf(pages, destination / "comparison_20_cases.pdf")
                    sweep.base._atomic_json(marker, dict(previous, pdf_encoding="lossless"))
                return destination
    _, _, routes, seeds, _, _, loader = sweep.base.context()
    indexed = {policy: {r.target_key: r for r in routes[policy]} for policy in sweep.POLICIES}
    destination.mkdir(parents=True, exist_ok=True)
    columns = [(None, None, "Real T0"), (None, None, "Real target")]
    columns += [(model, count, f"{MODEL_LABELS[model]}\nEuler-{count}")
                for model in sweep.MODELS for count in steps]
    tile, gap, left, top, row_height = 224, 10, 188, 132, 252
    width, height = left + len(columns) * (tile + gap) + 12, top + 9 * row_height + 60
    pages, thumbnails, low_contrast = [], [], []
    for number, pid in enumerate(patients, 1):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"I-SPY2 | {pid} | Case {number:02}/20", fill="#141c1b", font=sweep.metrics.font(26, True))
        draw.text((16, 49), "DCE0 | Euler | One candidate | Matched target seeds", fill="#414847", font=sweep.metrics.font(17))
        for column, (model, _, label) in enumerate(columns):
            draw.multiline_text((left + column * (tile + gap), 82), label,
                                fill=COLORS.get(model, "#141c1b"), font=sweep.metrics.font(18, True), spacing=3)
        original, foreground, _ = loader.load(pid, 0)
        low, high = np.percentile(original[0].float().numpy()[foreground.numpy()], [.5, 99.5])
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("Invalid shared display window")
        real = {0: original[0]}
        for target in (1, 2, 3):
            real[target] = loader.load(pid, target)[0][0]
        for policy_index, policy in enumerate(sweep.POLICIES):
            for target in (1, 2, 3):
                item = indexed[policy][(pid, target)]
                y = top + (policy_index * 3 + target - 1) * row_height
                if target == 1:
                    draw.line((10, y - 6, width - 12, y - 6), fill="#c8cfcc", width=2)
                source_label = f"T{item.source_stage}" + (" generated" if item.source_dce0 == "generated" else " real")
                text = f"{POLICY_LABELS[policy]}\n\n{source_label}\nto T{target}\n{item.delta_days} days"
                draw.multiline_text((14, y + 20), sweep.metrics.wrap_label(text, sweep.metrics.font(16), left - 28),
                                    fill="#242a28", font=sweep.metrics.font(16), spacing=4)
                values = [real[0], real[target]]
                values += [sweep.prediction(model, count, item, seeds[item.target_key])[0]
                           for model in sweep.MODELS for count in steps]
                for column, value in enumerate(values):
                    panel = sweep.metrics.gray(value[protocol["display_slice"]].float().numpy(), low, high, size=tile)
                    if np.asarray(panel).std() < 1:
                        if column < 2:
                            raise ValueError(f"Blank real MRI display panel: {pid}, T{target}, column {column}")
                        low_contrast.append(dict(patient_id=pid, strategy=policy, target_stage=target,
                                                 model=columns[column][0], steps=columns[column][1]))
                        draw.text((left + column * (tile + gap), y + tile + 3), "Low display contrast",
                                  fill="#9a302a", font=sweep.metrics.font(14))
                    canvas.paste(panel, (left + column * (tile + gap), y))
        footer = f"Axial z=48 | Shared real-T0 window [{low:.3f}, {high:.3f}] | BiFlow: DCE0 + real SER; SymmFlow: DCE0"
        draw.text((16, height - 44), footer, fill="#414847", font=sweep.metrics.font(15))
        canvas.save(destination / f"{pid}.png")
        pages.append(canvas)
        thumb = canvas.copy()
        thumb.thumbnail((370, 370))
        thumbnails.append((pid, thumb))
        print(f"Euler {steps}: patient sheet {number}/20", flush=True)
    write_case_pdf(pages, destination / "comparison_20_cases.pdf")
    contact = Image.new("RGB", (1560, 2050), "#e9eeeb")
    contact_draw = ImageDraw.Draw(contact)
    for index, (pid, thumb) in enumerate(thumbnails):
        x, y = 10 + (index % 4) * 390, 12 + (index // 4) * 408
        contact.paste(thumb, (x + (370 - thumb.width) // 2, y))
        contact_draw.text((x, y + 377), f"{index + 1:02} | {pid}", fill="#202724", font=sweep.metrics.font(17))
    contact.save(destination / "contact_sheet.jpg", quality=92)
    sweep.base._atomic_json(marker, dict(status="passed", steps=steps, patients=patients, sheets=20,
        panels_per_sheet=9 * len(columns), dimensions=[width, height], slice=48,
        common_window=True, pdf_encoding="lossless", real_panel_check="passed", low_contrast_generated_panels=low_contrast,
        selection_uses_outcomes_or_errors=False))
    return destination


def plots(image_summary, pcr, steps):
    destination = OUT / "comparison"
    groups = [(region, ("mae_unclipped", "ssim_3d_windowed")) for region in REGION_LABELS]
    groups.append(("pcr", ("auroc", "prauc")))
    for kind, metrics in groups:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row", layout="constrained")
        if kind != "pcr":
            fig.suptitle(REGION_LABELS[kind], fontsize=13)
        for column, policy in enumerate(sweep.POLICIES):
            for row, metric in enumerate(metrics):
                ax = axes[row, column]
                for model in sweep.MODELS:
                    frame = (image_summary[(image_summary.region == kind)] if kind != "pcr" else
                             pcr[pcr.temporal_depth == "T0-T3"])
                    frame = frame[(frame.model == model) & (frame.strategy == policy)].sort_values("steps")
                    ax.plot(frame.steps, frame[f"{metric}_mean"], "o-" if model == "biflow" else "s--", color=COLORS[model],
                            label=MODEL_LABELS[model], linewidth=2, markersize=5)
                if kind == "pcr":
                    reference = pcr[(pcr.model == "real") & (pcr.temporal_depth == "T0-T3")].iloc[0]
                    ax.axhline(reference[f"{metric}_mean"], linestyle="--", color="#545e59", label="Real images")
                ax.set_xticks(steps)
                ax.set_xlabel("Euler steps per transition")
                ax.set_ylabel({"mae_unclipped": "MAE (lower is better)", "ssim_3d_windowed": "3D SSIM",
                               "auroc": "T0-T3 pCR AUROC", "prauc": "T0-T3 pCR PR-AUC"}[metric])
                ax.grid(axis="y", alpha=.22)
                ax.spines[["right", "top"]].set_visible(False)
                if row == 0:
                    ax.set_title(POLICY_LABELS[policy], fontsize=11)
                if column == 0 and row == 0:
                    ax.legend(fontsize=9)
        name = "pcr_step_comparison" if kind == "pcr" else image_plot_name(kind)
        fig.savefig(destination / f"{name}.png", dpi=160)
        fig.savefig(destination / f"{name}.pdf")
        plt.close(fig)


def numeric_table(frame, columns, labels):
    rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(labels)) + " |"]
    for record in frame[columns].itertuples(index=False, name=None):
        values = [f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value) for value in record]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def report(image_summary, pcr, steps, destination):
    partial = set(steps) != set(sweep.STEPS)
    coverage = region_coverage()
    state = f"Partial: Euler {steps}; remaining settings are pending." if partial else "Complete: Euler 2, 10, 20 and 50."
    relative = destination.relative_to(OUT)
    sections = ["# I-SPY2: BiFlow vs SymmFlow sampling comparison", "", state, "",
        f"[20-page comparison PDF]({relative}/comparison_20_cases.pdf) | [Patient gallery](index.html) | [20-case overview]({relative}/contact_sheet.jpg)", "",
        "## Evaluation", "",
        "- Same 102 patients, 296 future targets per source policy, one candidate, shared target seeds starting at 22026.",
        "- Euler steps are per transition. A complete three-transition rollout uses 6/30/60/150 velocity evaluations for 2/10/20/50 steps respectively.",
        "- BiFlow uses the original 97,104-update checkpoint; SymmFlow uses the 100,000-update EMA checkpoint. No generator or pCR retraining.",
        "- BiFlow consumes DCE0 and real SER; SymmFlow consumes DCE0. The architecture comparison therefore also differs in conditioning.",
        "- Direct uses real T0; adjacent uses the previous available real visit; rollout feeds generated previous DCE0 back, retaining real previous foreground (and real SER for BiFlow). Missing visits are skipped.",
        "- Image means average valid targets within each patient before averaging 102 patients. All settings share real-T0/target foreground and tumor masks; empty regions are undefined and omitted from regional means. Change metrics use real T0 as origin. MAE/RMSE use standardized unclipped intensities; PSNR/3D SSIM use the existing fixed window.",
        "- pCR replaces future DCE0 only; future early/late DCE remain real. These are hybrid-input evaluations, not predictions using T0 alone. T0 is always real. Effective contiguous visit counts are 102/101/97/94.",
        "- pCR uses the previously frozen four-depth policy: ten seeds, five folds each, 200 checkpoint replays per setting. Tables report ten-seed mean AUROC/PR-AUC and the development-selected threshold. CSVs retain seed variability and other classification metrics.",
        "- Twenty complete-visit patients were selected without outcomes or image errors: previous three random cases plus seventeen seeded random additions. All panels use axial z=48 and the same real-T0 display window within a patient.",
        "- BiFlow Euler-20 was replayed with batch size one and matched routes/seeds. Historical BiFlow previous-real outputs have different seeds and five routing differences and are excluded. SymmFlow Euler-20 reuses its verified previous outputs.",
        "- This repeatedly inspected generator-validation cohort supports descriptive comparisons. Point-estimate differences are not evidence of statistical significance or an independently validated optimal step count.", "",
        "## Image metrics", "",
        "Tumor-region masks are real evaluation references. Regional metrics and whole-foreground metrics can favor different sampling steps.", ""]
    for region, label in REGION_LABELS.items():
        image = image_summary[image_summary.region == region]
        counts = coverage[region]
        sections += [f"### {label}", "",
            f"Nonempty evaluation masks: {counts['valid_targets']}/{counts['total_targets']} future targets, {counts['patients']} patients.", "",
            f"![{label}](comparison/{image_plot_name(region)}.png)", ""]
        for policy in sweep.POLICIES:
            sections += [f"#### {POLICY_LABELS[policy]}", "", numeric_table(
                image[image.strategy == policy].sort_values(["model", "steps"]).assign(model=lambda f: f.model.map(MODEL_LABELS)),
                ["model", "steps", "mae_unclipped_mean", "rmse_unclipped_mean", "psnr_windowed_db_mean", "ssim_3d_windowed_mean"],
                ["Model", "Steps", "MAE", "RMSE", "PSNR (dB)", "3D SSIM"]), ""]
    sections += ["## Frozen pCR evaluation", "", "![pCR metrics](comparison/pcr_step_comparison.png)", ""]
    for depth in sweep.base.DEPTHS:
        table = pcr[pcr.temporal_depth == depth].sort_values(["strategy", "model", "steps"]).assign(
            model=lambda f: f.model.map(MODEL_LABELS),
            strategy=lambda f: f.strategy.map({**POLICY_LABELS, "real": "Real images"}))
        sections += [f"### {depth}", "", numeric_table(table,
            ["model", "strategy", "steps", "auroc_mean", "prauc_mean", "bacc_mean"],
            ["Model", "Source policy", "Steps (0 = real)", "AUROC", "PR-AUC", "Balanced accuracy"]), ""]
    sections += ["## Full artifacts", "", "- `comparison/image_summary.csv`: all three regions, patient means and standard deviations.",
        "- `comparison/image_by_target_stage.csv`: separate T1, T2 and T3 metrics, including support counts.",
        "- `comparison/image_endpoints.csv` and `image_patients.csv`: paired endpoint and patient metrics.",
        "- `comparison/pcr_summary.csv` and `pcr_predictions.csv`: all depths and patient probabilities, with one shared real baseline (steps=0). Each setting also retains per-seed metrics and fold predictions.",
        "- All new endpoint latents and all embeddings are retained. Full decoded volumes are retained for the 20 selected patients; other newly owned decoded intermediates are pruned only after the full setting audit. Previous experiment files remain in their original locations.", ""]
    (OUT / "report.md").write_text("\n".join(sections))


def gallery(image_summary, pcr, steps, destination):
    protocol = json.loads((OUT / "protocol.json").read_text())
    relative = str(destination.relative_to(OUT))
    options = "".join(f'<option value="{html.escape(pid)}">{index:02} / 20 | {html.escape(pid)}</option>'
                      for index, pid in enumerate(protocol["display_patients"], 1))
    step_options = f'<option value="{relative}">All available steps</option>'
    step_options += "".join(f'<option value="figures/euler_{count}">Euler-{count}</option>' for count in steps)
    policy_options = "".join(f'<option value="{key}">{label}</option>' for key, label in POLICY_LABELS.items())
    region_options = "".join(f'<option value="{key}">{label}</option>' for key, label in REGION_LABELS.items())
    depth_options = "".join(f'<option value="{depth}" {"selected" if depth == "T0-T3" else ""}>{depth}</option>' for depth in sweep.base.DEPTHS)
    records = image_summary.to_dict("records")
    payload = json.dumps(dict(images=records, pcr=pcr.to_dict("records"), coverage=region_coverage()), allow_nan=False)
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>I-SPY2 | BiFlow and SymmFlow</title><style>
*{box-sizing:border-box}body{margin:0;background:#fafcfb;color:#202a25;font:15px system-ui,sans-serif;letter-spacing:0}
header,main{max-width:1500px;margin:auto;padding:20px 24px}header{border-bottom:1px solid #cbd5d0}
h1{font-size:24px;margin:0 0 12px}h2{font-size:19px;margin:0}a{color:#167365}nav{display:flex;gap:22px;flex-wrap:wrap}
section{padding:22px 0;border-bottom:1px solid #cbd5d0}.toolbar{display:flex;align-items:end;gap:20px;flex-wrap:wrap;margin:16px 0}
label{display:grid;gap:6px;font-size:13px;font-weight:600}select{font:inherit;padding:9px 12px;max-width:100%;border:1px solid #a5b7ac;border-radius:4px;background:white}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:11px 12px;border-bottom:1px solid #dde4e0;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:#56655b}td:first-child{font-weight:650}.biflow{color:#187568}.symmflow{color:#bc4c42}
#case-image{display:block;width:100%;height:auto;background:white}figure{margin:0}figcaption,p{color:#536359;font-size:13px;line-height:1.6}
.plot{width:100%;height:auto}footer{padding:20px 0;font-size:13px}@media(max-width:600px){header,main{padding:16px}h1{font-size:21px}.toolbar{gap:12px}label{max-width:100%}th,td{padding:9px}}
</style></head><body><header><h1>I-SPY2 | BiFlow and SymmFlow</h1><p>__STATE__</p>
<nav><a href="#cases">20 patient comparisons</a><a href="#metrics">Metrics</a><a href="report.md">Research report</a><a id="pdf-link" href="__DIRECTORY__/comparison_20_cases.pdf">20-page PDF</a></nav></header>
<main><section id="cases"><h2>DCE0 generation</h2><div class="toolbar">
<label>Patient<select id="patient">__PATIENTS__</select></label><label>Sampling<select id="sampling">__STEPS__</select></label>
<a id="image-link" href="__DIRECTORY__/__FIRST__.png">Full-resolution PNG</a></div>
<figure><img id="case-image" width="__IMAGE_WIDTH__" height="2460" src="__DIRECTORY__/__FIRST__.png" alt="Matched MRI comparison"><figcaption>Axial z=48. Identical real-T0 intensity window within each patient. Real T0, real target, BiFlow and SymmFlow.</figcaption></figure></section>
<section id="metrics"><h2>Generation and pCR metrics</h2><div class="toolbar"><label>Source policy<select id="policy">__POLICIES__</select></label>
<label>Image region<select id="region">__REGIONS__</select></label>
<label>pCR temporal depth<select id="depth">__DEPTHS__</select></label></div>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Euler steps</th><th>MAE</th><th>3D SSIM</th><th>PSNR (dB)</th><th>pCR AUROC</th><th>pCR PR-AUC</th></tr></thead><tbody id="metrics-body"></tbody></table></div>
<p id="region-coverage"></p>
<p>pCR: ten-seed means, five frozen folds per seed; future early/late phases remain real. Descriptive comparisons on a previously inspected cohort.</p>
<a id="image-plot-link" href="comparison/image_step_comparison.png"><img id="image-plot" class="plot" src="comparison/image_step_comparison.png" alt="Image metrics versus Euler steps"></a>
<a href="comparison/pcr_step_comparison.png"><img class="plot" src="comparison/pcr_step_comparison.png" alt="T0-T3 pCR metrics versus Euler steps"></a></section>
<footer><a href="comparison/image_summary.csv">Image CSV</a> &middot; <a href="comparison/pcr_summary.csv">pCR CSV</a> &middot; <a href="__DIRECTORY__/contact_sheet.jpg">20-case overview</a></footer></main>
<script>const data=__DATA__;const el=id=>document.getElementById(id);const number=x=>Number(x).toFixed(5);
function showCase(){const dir=el('sampling').value;const path=dir+'/'+el('patient').value+'.png';el('case-image').src=path;el('case-image').alt=el('patient').value+' MRI comparison';el('image-link').href=path;el('pdf-link').href=dir+'/comparison_20_cases.pdf'}
function showMetrics(){const policy=el('policy').value,depth=el('depth').value,region=el('region').value;const records=data.images.filter(x=>x.strategy===policy&&x.region===region).sort((a,b)=>a.model.localeCompare(b.model)||a.steps-b.steps);el('metrics-body').replaceChildren();
const plot='comparison/'+(region==='common_foreground'?'image':'image_'+region)+'_step_comparison.png';el('image-plot').src=plot;el('image-plot-link').href=plot;el('image-plot').alt=el('region').selectedOptions[0].text+' metrics versus Euler steps';
const coverage=data.coverage[region];el('region-coverage').textContent=coverage.patients+' patient means | '+coverage.valid_targets+'/'+coverage.total_targets+' future targets with nonempty real evaluation masks. Empty regions are omitted from regional means.';
for(const x of records){const p=data.pcr.find(y=>y.model===x.model&&y.strategy===policy&&y.steps===x.steps&&y.temporal_depth===depth);const tr=document.createElement('tr');for(const v of [x.model==='biflow'?'BiFlow':'SymmFlow',x.steps,number(x.mae_unclipped_mean),number(x.ssim_3d_windowed_mean),number(x.psnr_windowed_db_mean),number(p.auroc_mean),number(p.prauc_mean)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.firstChild.className=x.model;el('metrics-body').append(tr)}
const p=data.pcr.find(x=>x.model==='real'&&x.temporal_depth===depth);const tr=document.createElement('tr');for(const v of ['Real images','-','-','-','-',number(p.auroc_mean),number(p.prauc_mean)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}el('metrics-body').append(tr)}
for(const id of ['patient','sampling'])el(id).addEventListener('change',showCase);for(const id of ['policy','region','depth'])el(id).addEventListener('change',showMetrics);showMetrics();</script></body></html>
"""
    pending = sorted(set(sweep.STEPS) - set(steps))
    state = "Completed: Euler " + ", ".join(map(str, steps)) + " | 102 patients | 296 targets per policy."
    if pending:
        state += " Pending: Euler " + ", ".join(map(str, pending)) + "."
    for key, value in {"__IMAGE_WIDTH__": str(200 + (2 + 2 * len(steps)) * 234),
                       "__STATE__": state, "__DIRECTORY__": relative, "__PATIENTS__": options, "__STEPS__": step_options,
                       "__FIRST__": protocol["display_patients"][0], "__POLICIES__": policy_options, "__REGIONS__": region_options,
                       "__DEPTHS__": depth_options, "__DATA__": payload}.items():
        document = document.replace(key, value)
    if (OUT / "tumor_review/publication.json").exists():
        (OUT / "overview.html").write_text(document)
        from scripts.ispy2_tumor_review import publish_index
        publish_index()
    else:
        (OUT / "index.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", choices=sweep.STEPS, required=True)
    args = parser.parse_args()
    steps = sorted(set(args.steps))
    torch.set_num_threads(2)
    image, pcr = aggregate(steps)
    for count in steps:
        figures([count])
    destination = figures(steps)
    plots(image, pcr, steps)
    report(image, pcr, steps, destination)
    gallery(image, pcr, steps, destination)
    sweep.base._atomic_json(OUT / "comparison/audit.json", dict(
        status="passed", scope="complete" if set(steps) == set(sweep.STEPS) else "partial",
        updated_utc=datetime.now(timezone.utc).isoformat(), steps=steps, models=list(sweep.MODELS),
        policies=list(sweep.POLICIES), patients=102, targets_per_policy=296, display_patients=20,
        shared_image_supports=True, real_pcr_replay_shared=True, t0_predictions_shared=True,
        image_summary_rows=len(image), pcr_summary_rows=len(pcr)))


if __name__ == "__main__":
    main()

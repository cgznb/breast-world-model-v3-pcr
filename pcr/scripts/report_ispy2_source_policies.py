"""Compare SymmFlow source policies on fixed targets, masks and frozen pCR models."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import compare_ispy2_generators as generation
from scripts import report_ispy2_generators as reporting
from scripts.extract_pillar_mewm import validate_embedding
from src.biflow_trajectories import validate_locked102_routes

OUT = generation.OUT / "source_policies"
POLICIES = generation.STRATEGIES
ROLLOUT = "rollout_generated_dce0_real_ser"
LABELS = {
    "real_t0_copy": "Real T0 copy",
    "previous_real_copy": "Real previous-visit copy",
    "target_vq": "Target VQ reconstruction",
    "direct_t0": "SymmFlow: real T0",
    "adjacent_real": "SymmFlow: real previous visit",
    ROLLOUT: "SymmFlow: generated previous DCE0",
}
ARMS = tuple(LABELS)
SCHEMA = "symmflow_source_policies_fixed_t0_regions_v1"


def route_index(routes):
    indexed = {policy: {r.target_key: r for r in routes[policy]} for policy in POLICIES}
    expected = set(indexed["direct_t0"])
    if any(set(values) != expected or len(values) != len(routes[policy])
           for policy, values in indexed.items()):
        raise ValueError("Source policies must have identical unique patient-target pairs")
    return indexed


def metric_group_complete(frame):
    return (len(frame) == len(ARMS) * len(reporting.REGIONS)
            and "schema" in frame and set(frame.schema) == {SCHEMA}
            and set(frame.model) == set(ARMS)
            and not frame.duplicated(["model", "region"]).any()
            and all(set(g.region) == reporting.REGIONS for _, g in frame.groupby("model")))


def images(workers):
    directory = OUT / "image_metrics"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".metrics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _images(directory, workers)


def _images(directory, workers):
    _, _, routes, _, loaded, cache, loader = generation.context()
    indexed = route_index(routes)
    original_path = generation.OUT / "image_metrics/direct_t0/endpoint_metrics.csv"
    original = pd.read_csv(original_path, dtype={"patient_id": str})
    reuse = {key: group for key, group in original.groupby(["patient_id", "target_stage"])}

    def one(direct):
        path = directory / "shards" / direct.patient_id / f"T{direct.target_stage}.csv"
        if path.exists():
            previous = pd.read_csv(path, dtype={"patient_id": str})
            if metric_group_complete(previous):
                return previous
            raise ValueError(f"Incompatible metric shard: {path}")
        adjacent = indexed["adjacent_real"][direct.target_key]
        initial = reuse[direct.target_key]
        rows = []
        aliases = {"source_copy": "real_t0_copy", "target_vq": "target_vq", "symmflow": "direct_t0"}
        for source_model, arm in aliases.items():
            group = initial[initial.model == source_model]
            if len(group) != 3 or set(group.region) != reporting.REGIONS:
                raise ValueError("Direct-T0 metric reference is incomplete")
            for row in group.to_dict("records"):
                rows.append(dict(row, model=arm, schema=SCHEMA, metric_reference_stage=0,
                                 generation_source_stage=0))
        if adjacent.source_stage == 0:
            for old, new in (("real_t0_copy", "previous_real_copy"),
                             ("direct_t0", "adjacent_real"), ("direct_t0", ROLLOUT)):
                rows.extend(dict(row, model=new) for row in tuple(rows) if row["model"] == old)
        else:
            # A common T0/target mask and T0 change origin make all policies comparable.
            source, sf, _ = loader.load(direct.patient_id, 0)
            target, tf, _ = loader.load(direct.patient_id, direct.target_stage)
            masks = reporting.region_masks(
                sf.numpy(), tf.numpy(),
                reporting.lesion_mask(loaded, cache, direct.patient_id, 0),
                reporting.lesion_mask(loaded, cache, direct.patient_id, direct.target_stage),
            )
            if not masks["common_foreground"].any():
                raise ValueError("Empty common real-T0/target foreground")
            centers = reporting.comparison_ssim_centers(masks["common_foreground"])
            real_previous, _, _ = loader.load(direct.patient_id, adjacent.source_stage)
            for arm in ("previous_real_copy", "adjacent_real", ROLLOUT):
                prediction = (real_previous[:1] if arm == "previous_real_copy" else
                              generation.read_prediction("symmflow", indexed[arm][direct.target_key]))
                metrics = reporting.comparison_metrics(prediction, target[:1], masks, centers)
                for region, values in metrics.items():
                    changes = reporting.change_metrics(prediction[0].float(), source[0].float(),
                                                       target[0].float(), masks[region])
                    rows.append(dict(patient_id=direct.patient_id, target_stage=direct.target_stage,
                                     source_stage=0, delta_days=direct.delta_days, model=arm, region=region,
                                     schema=SCHEMA, metric_reference_stage=0,
                                     generation_source_stage=adjacent.source_stage, **values, **changes))
        frame = pd.DataFrame(rows)
        if not metric_group_complete(frame):
            raise ValueError("Incomplete source-policy metric group")
        generation._atomic_csv(path, frame)
        return frame

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, frame in enumerate(pool.map(one, routes["direct_t0"]), 1):
            frames.append(frame)
            if index % 20 == 0 or index == 296:
                print(f"Matched source-policy image metrics: {index}/296", flush=True)
    endpoint = pd.concat(frames, ignore_index=True)
    generation._atomic_csv(directory / "endpoint_metrics.csv", endpoint)
    metrics = list(reporting.METRICS + reporting.CHANGE_METRICS)
    patient = endpoint.groupby(["model", "region", "patient_id"])[metrics].mean().reset_index()
    generation._atomic_csv(directory / "patient_metrics.csv", patient)
    summary = patient.groupby(["model", "region"])[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    generation._atomic_csv(directory / "summary.csv", summary.reset_index())
    stages = endpoint.groupby(["model", "region", "target_stage"])[metrics].agg(["mean", "std", "count"])
    stages.columns = ["_".join(column) for column in stages.columns]
    generation._atomic_csv(directory / "by_target_stage.csv", stages.reset_index())
    generation._atomic_json(directory / "audit.json", {
        "schema": SCHEMA, "patients": patient.patient_id.nunique(), "targets": 296,
        "rows": len(endpoint), "arms": list(ARMS), "regions_shared_across_policies": True,
        "foreground": "intersection of real T0 and real target foreground",
        "tumor_neighborhood": "real T0/target tumor union neighborhood",
        "change_reference": "real T0 for every arm", "reused_direct_metrics": str(original_path),
    })


def figures():
    _, _, routes, _, _, _, loader = generation.context()
    indexed = route_index(routes)
    selection = json.loads((generation.OUT / "figures/direct_t0/display_selection.json").read_text())
    destination = OUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    generation._atomic_json(destination / "display_selection.json", dict(
        selection, source="unchanged direct-T0 case selection", columns_include_actual_rollout_source=True))
    columns = ["Real T0", "Real previous visit", "Rollout input DCE0",
               "Real target", "From real T0", "From real previous", "From generated previous"]
    tile, gap, left, top, rowheight = 192, 10, 180, 90, 250
    width = left + len(columns) * (tile + gap) + gap
    for pid in selection["patients"]:
        selected = [r for r in routes["direct_t0"] if r.patient_id == pid]
        canvas = Image.new("RGB", (width, top + rowheight * len(selected) + 20), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), f"I-SPY2 | {pid} | SymmFlow 100k EMA | Euler-20 | One candidate", fill="black", font=reporting.font(22, True))
        for index, label in enumerate(columns):
            draw.multiline_text((left + index * (tile + gap), 48),
                                reporting.wrap_label(label, reporting.font(15, True), tile),
                                fill="black", font=reporting.font(15, True), spacing=3)
        real_t0, f0, _ = loader.load(pid, 0)
        low, high = np.percentile(real_t0[0].float().numpy()[f0.numpy()], [.5, 99.5])
        for index, direct in enumerate(selected):
            y = top + rowheight * index
            adjacent = indexed["adjacent_real"][direct.target_key]
            previous, foreground, _ = loader.load(pid, adjacent.source_stage)
            feedback = previous
            if adjacent.source_stage > 0:
                prior = indexed[ROLLOUT][(pid, adjacent.source_stage)]
                feedback = generation.compose_rollout_source_mri(
                    generation.read_prediction("symmflow", prior), previous, foreground)
            target, _, _ = loader.load(pid, direct.target_stage)
            images_to_show = [real_t0[0], previous[0], feedback[0], target[0]]
            images_to_show += [generation.read_prediction("symmflow", indexed[p][direct.target_key])[0]
                               for p in POLICIES]
            draw.multiline_text((12, y + 48), f"Target T{direct.target_stage}\nPrevious T{adjacent.source_stage}\n\nT0 gap: {direct.delta_days} d\nStep gap: {adjacent.delta_days} d",
                                fill="black", font=reporting.font(16), spacing=6)
            for column, value in enumerate(images_to_show):
                canvas.paste(reporting.gray(value[48].float().numpy(), low, high, size=tile),
                             (left + column * (tile + gap), y))
            draw.text((left, y + tile + 12), f"Axial z=48; shared real-T0 window [{low:.3f}, {high:.3f}]",
                      fill="#444444", font=reporting.font(15))
        canvas.save(destination / f"{pid}.png")


def validate_prediction_relations(frame):
    keys = ["temporal_depth", "seed", "patient_id"]
    direct = frame[(frame.model == "symmflow") & (frame.strategy == "direct_t0")].set_index(keys).sort_index()
    for policy in ("adjacent_real", ROLLOUT):
        other = frame[(frame.model == "symmflow") & (frame.strategy == policy)].set_index(keys).sort_index()
        if not other.index.equals(direct.index) or not np.array_equal(other.label, direct.label):
            raise ValueError("pCR source-policy cohorts differ")
        shared = direct.index.get_level_values("temporal_depth").isin(["T0", "T0-T1"])
        if not np.array_equal(other.probability[shared], direct.probability[shared]):
            raise ValueError("T0/T0-T1 predictions must be identical across source policies")


def audit():
    _, ids, routes, seeds, _, _, _ = generation.context()
    validate_locked102_routes(ids, {p: routes[p] for p in ("direct_t0", ROLLOUT)})
    indexed = route_index(routes)
    shared, counts = {}, {}
    for policy in POLICIES:
        identical = 0
        for route in routes[policy]:
            image = generation.read_prediction("symmflow", route, expected_seed=seeds[route.target_key])
            payload = torch.load(generation.decoded_path("symmflow", route), weights_only=True)
            if payload["checkpoint_step"] != 100000:
                raise ValueError("SymmFlow checkpoint differs across source policies")
            embedding = generation.model_root("symmflow") / policy / "embeddings" / route.patient_id / f"{route.patient_id}_T{route.target_stage}.pt"
            if validate_embedding(embedding).dtype != torch.float32:
                raise ValueError("Pillar embedding dtype differs")
            if policy != "direct_t0" and route.source_stage == 0:
                direct = indexed["direct_t0"][route.target_key]
                if not torch.equal(image, generation.read_prediction("symmflow", direct)):
                    raise ValueError("Shared T0-source image differs")
                direct_embedding = generation.model_root("symmflow") / "direct_t0/embeddings" / route.patient_id / embedding.name
                if not torch.equal(torch.load(embedding, weights_only=True), torch.load(direct_embedding, weights_only=True)):
                    raise ValueError("Shared T0-source embedding differs")
                identical += 1
        shared[policy] = identical
        counts[policy] = dict(Counter(r.transition_type for r in routes[policy]))
        if len(routes[policy]) != 296 or (policy != "direct_t0" and identical != 102):
            raise ValueError("Source-policy route count differs")
    predictions = pd.read_csv(OUT / "pcr/predictions.csv", dtype={"patient_id": str})
    if predictions.duplicated(["model", "strategy", "temporal_depth", "seed", "patient_id"]).any():
        raise ValueError("Duplicated pCR patient predictions")
    for _, group in predictions.groupby(["model", "strategy", "temporal_depth", "seed"]):
        if set(group.patient_id) != set(ids) or group.label.sum() != 32:
            raise ValueError("pCR patient inventory or labels differ")
    validate_prediction_relations(predictions)
    direct_old = pd.read_csv(generation.OUT / "pcr/predictions.csv", dtype={"patient_id": str})
    keys = ["model", "strategy", "temporal_depth", "seed", "patient_id"]
    old = direct_old.set_index(keys).sort_index()
    new = predictions.set_index(keys).loc[old.index]
    if not np.array_equal(old.label, new.label):
        raise ValueError("Previously completed pCR labels changed")
    if not np.allclose(old.probability, new.probability, rtol=0, atol=1e-5):
        raise ValueError("Previously completed direct-T0 pCR replay changed")
    endpoint = pd.read_csv(OUT / "image_metrics/endpoint_metrics.csv", dtype={"patient_id": str})
    if (len(endpoint) != 296 * len(ARMS) * len(reporting.REGIONS)
        or set(endpoint.patient_id) != set(ids)
        or endpoint.duplicated(["model", "region", "patient_id", "target_stage"]).any()):
        raise ValueError("Image metric inventory differs")
    supports = endpoint.groupby(["patient_id", "target_stage", "region"])[
        ["voxel_count", "ssim_center_count"]].nunique()
    if not supports.eq(1).all().all():
        raise ValueError("Image metric regions differ across source policies")
    foreground = endpoint[endpoint.region == "common_foreground"]
    if not np.isfinite(foreground[list(reporting.METRICS)]).all().all():
        raise ValueError("Nonfinite common-foreground image metrics")
    generation._atomic_json(OUT / "audit.json", {
        "status": "passed", "updated_utc": datetime.now(timezone.utc).isoformat(),
        "patients": len(ids), "targets_per_policy": 296, "policies": list(POLICIES),
        "transitions": counts, "shared_real_T0_tensors_and_embeddings": shared,
        "solver": "euler", "steps": 20, "candidate_count": 1, "checkpoint_step": 100000,
        "target_seeds_match": True, "finite_images_and_embeddings": True,
        "pcr_T0_and_T0_T1_identical": True, "previous_direct_pcr_replayed": True,
        "image_metric_rows": len(endpoint), "image_metric_region_counts_identical": True,
        "symmflow_reads_SER": False, "rollout_real_source_foreground_retained": True,
        "future_real_early_late_DCE_retained": True, "new_pcr_training": False,
    })


def pcr_plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = summary[summary.threshold_policy == "development_oof_bacc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    colors = {"real": "#333333", "direct_t0": "#2878a5", "adjacent_real": "#34845d", ROLLOUT: "#c05245"}
    for axis, metric, title in zip(axes, ("auroc", "prauc"), ("AUROC", "PR-AUC")):
        for policy in ("real", *POLICIES):
            model = "real" if policy == "real" else "symmflow"
            rows = frame[(frame.model == model) & (frame.strategy == policy)].set_index("temporal_depth").loc[list(generation.DEPTHS)]
            axis.errorbar(range(4), rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"],
                          color=colors[policy], label="Real" if policy == "real" else LABELS[policy],
                          marker="o", markersize=4, capsize=3, linewidth=1.6)
        axis.set_xticks(range(4), generation.DEPTHS)
        axis.set_title(title)
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Frozen pCR | 102 patients | 10 seeds | Mean +/- seed SD")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"pcr_source_policies.{suffix}", dpi=180)
    plt.close(fig)


def report():
    audit_record = json.loads((OUT / "audit.json").read_text())
    if audit_record["status"] != "passed":
        raise ValueError("Final report requires a passing artifact audit")
    image_summary = pd.read_csv(OUT / "image_metrics/summary.csv")
    pcr = pd.read_csv(OUT / "pcr/summary.csv")
    pcr_plot(pcr)
    chosen = json.loads((OUT / "figures/display_selection.json").read_text())["patients"]
    lines = ["# I-SPY2 SymmFlow Source-Policy Comparison", "", "Date: 2026-09-11", "",
        "All three SymmFlow policies completed for 102 patients and 296 future DCE0 targets. The original direct-T0 report is retained at [../report.md](../report.md).", "",
        "| Policy | Source for each future target |", "|---|---|",
        "| Direct T0 | Real T0 independently generates each available T1/T2/T3 |",
        "| Previous real | Real previous available visit generates the next available visit |",
        "| Generated rollout | Real T0 starts the chain; generated previous DCE0 is encoded and fed into the next step |", "",
        "Every transition uses Euler-20, one candidate, the same target seed schedule (base 22026), and the same 100k EMA checkpoint. A complete T0-to-T3 rollout has three 20-step calls; no best-of-K selection is used.",
        "Missing visits are skipped: real-previous and rollout have 101 T0->T1, 1 T0->T2, 97 T1->T2, 2 T1->T3, and 95 T2->T3 routes. Their 102 real-T0-source outputs and embeddings are exactly reused from direct T0.", "",
        "SymmFlow conditions on source DCE0 and structured clinical/treatment/stage/interval variables; it does not consume SER. The historical rollout directory name retains `real_ser` for compatibility with BiFlow. The real previous-visit foreground is retained when masking generated DCE0 for feedback, so this is not a completely observation-free future trajectory.", "",
        "## Same-Case Images", "",
        "The cases, axial slice and real-T0 intensity window are unchanged from the direct report. The third column shows the actual masked DCE0 fed back during rollout; the last three columns are the three generated targets.", ""]
    for pid in chosen:
        lines += [f"![Source policies](figures/{pid}.png)", ""]
    lines += ["## Image Metrics", "",
        "Patient macro means. Every arm uses identical real-T0/target foreground and tumor-region masks. Change metrics use real T0 as their common origin, not each arm's immediate source. MAE/RMSE use unclipped standardized images; PSNR/3D SSIM use the existing fixed window.", "",
        "| Input | Foreground MAE | RMSE | PSNR (dB) | 3D SSIM |", "|---|---:|---:|---:|---:|"]
    foreground = image_summary[image_summary.region == "common_foreground"].set_index("model")
    for arm in ARMS:
        row = foreground.loc[arm]
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{row[m+'_mean']:.5f}" for m in reporting.METRICS) + " |")
    lines += ["", "The previous-real copy baseline receives a newer real examination. Direct generation and rollout do not receive its real DCE0, so information availability differs by design. Per-stage and regional CSVs are retained.", "",
        "## Frozen pCR", "",
        "Only future DCE0 is replaced. Real future early/late DCE, clinical/time metadata and contiguous-prefix masks remain; effective visits are 102/101/97/94. This is a hybrid-input substitution study, not pCR prediction from T0 alone.",
        "The previous final policy is frozen: five-fold best200 at T0/T0-T1, layerwise AdamW without residual L2 at T0-T2/T0-T3, seeds 42-51, and fixed development OOF thresholds. No pCR training or threshold tuning was performed.",
        "The 200 checkpoint real-input replays passed; all source policies are identical at T0/T0-T1. Values are ten-seed mean AUROC / PR-AUC; CSVs retain seed SD and all eight metrics.", "",
        "![pCR source policies](figures/pcr_source_policies.png)", "",
        "| Input | T0 | T0-T1 | T0-T2 | T0-T3 |", "|---|---:|---:|---:|---:|"]
    selected = pcr[pcr.threshold_policy == "development_oof_bacc"]
    groups = [("real", "real", "Real")] + [("symmflow", p, LABELS[p]) for p in POLICIES]
    groups += [("biflow", "direct_t0", "BiFlow: real T0 reference"),
               ("biflow", ROLLOUT, "BiFlow: rollout reference"),
               ("biflow", "adjacent_real_historical", "BiFlow: previous real, historical protocol")]
    for model, policy, label in groups:
        rows = selected[(selected.model == model) & (selected.strategy == policy)].set_index("temporal_depth")
        values = [f"{rows.loc[d, 'auroc_mean']:.5f} / {rows.loc[d, 'prauc_mean']:.5f}" for d in generation.DEPTHS]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    t3 = selected[(selected.model == "symmflow") & (selected.temporal_depth == "T0-T3")].set_index("strategy")
    real_gain = t3.loc["adjacent_real", "auroc_mean"] - t3.loc["direct_t0", "auroc_mean"]
    rollout_gain = t3.loc[ROLLOUT, "auroc_mean"] - t3.loc["direct_t0", "auroc_mean"]
    lines += ["", f"At T0-T3, previous-real and generated-rollout change mean AUROC by {real_gain:+.5f} and {rollout_gain:+.5f} versus direct T0. These are observed point-estimate differences, not established significance. The previous-real policy receives a newer real DCE0 examination. Its image MAE is nevertheless higher than direct T0, and generated rollout has the highest image MAE of the three; image reconstruction accuracy does not rank these policies the same way as frozen pCR."]
    lines += ["", "BiFlow direct/rollout references use the same target seeds. Historical BiFlow previous-real predictions use Euler-20 with base seeds 2026/12026 and latest-earlier-Strict-A sources for supplemental targets. Five of its 296 source stages differ from the new previous-available policy: one T2 uses T0 instead of T1, and four T3 targets use T1 instead of T2. They are historical context, not a matched architecture comparison.",
        "Seed SD describes pCR training variability, not patient uncertainty or statistical significance. These generator-validation patients have been inspected repeatedly. Point estimates are descriptive and must not select thresholds, checkpoints or a winning method.", "",
        "## Artifacts", "", "- `audit.json`: all policies' identities, shared seeds, source policies, image/embedding readback and pCR equality checks.",
        "- `image_metrics/`: endpoint, patient, stage and aggregate image results with common masks.",
        "- `pcr/`: patient/fold predictions, eight metrics, summaries and frozen-real replay audit.",
        "- `figures/`: unchanged selected cases with actual rollout inputs and PNG/PDF pCR curves.",
        "- `../symmflow/adjacent_real/` and `../symmflow/rollout_generated_dce0_real_ser/`: 296 decoded targets and embeddings each.", ""]
    path = OUT / "report.md"
    path.write_text("\n".join(lines))
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("images", "figures", "audit", "report"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "images":
        images(args.workers)
    else:
        {"figures": figures, "audit": audit, "report": report}[args.stage]()


if __name__ == "__main__":
    main()

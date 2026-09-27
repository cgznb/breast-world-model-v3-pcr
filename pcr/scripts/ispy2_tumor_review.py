"""Build a truth-selected lesion review with the actual generator inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import ispy2_sampling_sweep as sweep

OUT = sweep.OUT / "tumor_review"
SCHEMA = "ispy2_actual_input_tumor_review_v1"


def load_json(path):
    return json.loads(path.read_text())


def masks_for_patient(loaded, cache, patient):
    visits = [loaded.visits.get(f"{patient}:T{stage}") for stage in range(4)]
    if any(visit is None for visit in visits):
        return None
    return np.stack([cache.load(visit).mask[0].bool().numpy() for visit in visits])


def lesion_plane(source, target):
    source_area, target_area = source.sum(axis=(1, 2)), target.sum(axis=(1, 2))
    eligible = (source_area >= 20) & (target_area >= 10)
    if not eligible.any():
        return None
    difference = np.logical_xor(source, target).sum(axis=(1, 2))
    return int(np.argmax(np.where(eligible, difference, -1)))


def square_crop(masks, slices, margin=12):
    union = masks[:, slices].any(axis=(0, 1))
    yy, xx = np.where(union)
    if not len(xx):
        raise ValueError("No tumor support on the selected slices")
    side = min(256, max(64, int(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)) + 2*margin))
    x = int(np.clip((int(xx.min())+int(xx.max())+1-side)//2, 0, 256-side))
    y = int(np.clip((int(yy.min())+int(yy.max())+1-side)//2, 0, 256-side))
    return [x, y, side, side]


def candidate(patient, masks, voxel_ml):
    volumes = masks.sum(axis=(1, 2, 3)).astype(float) * voxel_ml
    choices = []
    for target in range(1, 4):
        source = target-1
        if volumes[source] < 1 or volumes[target] <= 0:
            continue
        relative = (volumes[target]-volumes[source]) / volumes[source]
        plane = lesion_plane(masks[source], masks[target])
        if abs(relative) >= .35 and abs(volumes[target]-volumes[source]) >= 1 and plane is not None:
            choices.append((abs(volumes[target]-volumes[source]), target, relative, plane))
    if not choices:
        return None
    difference, target, relative, plane = max(choices)
    slices = sorted(set([int(np.clip(plane+offset, 0, 95)) for offset in (-4, -2, 0, 2, 4)] +
                        [int(mask.sum(axis=(1, 2)).argmax()) for mask in masks]))
    return dict(patient_id=patient, preferred_target=target, selected_source=target-1,
                volume_ml=volumes.tolist(), absolute_change_ml=float(difference),
                relative_change=float(relative), direction="decrease" if relative < 0 else "increase",
                central_slice=plane, slices=slices, crop=square_crop(masks, slices),
                source_area_pixels=int(masks[target-1, plane].sum()),
                target_area_pixels=int(masks[target, plane].sum()))


def select_cases(count=20):
    config, ids, routes, _, loaded, cache, _ = sweep.base.context()
    complete = {pid for pid in ids if sum(r.patient_id == pid for r in routes["direct_t0"]) == 3}
    voxel_ml = float(np.prod(config["biflow"]["strict_spacing_xyz"])) / 1000
    candidates, census = [], []
    for index, patient in enumerate(ids, 1):
        if patient not in complete:
            census.append(dict(patient_id=patient, eligible=False, reason="incomplete longitudinal visits"))
            continue
        masks = masks_for_patient(loaded, cache, patient)
        if masks is None:
            census.append(dict(patient_id=patient, eligible=False, reason="unavailable locked regional annotation"))
            continue
        item = candidate(patient, masks, voxel_ml)
        if item:
            candidates.append(item)
        census.append(dict(patient_id=patient, eligible=item is not None,
                           reason="candidate" if item else "no qualifying nonempty adjacent change",
                           **{f"T{stage}_mask_ml": float(masks[stage].sum()*voxel_ml) for stage in range(4)}))
        if index % 10 == 0:
            print(f"Truth-mask census {index}/{len(ids)}", flush=True)
    candidates.sort(key=lambda item: (-item["absolute_change_ml"], item["patient_id"]))
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} cases satisfy the documented selection criteria")
    selected = candidates[:count]
    protocol = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        cohort_patients=len(ids), candidate_patients=len(candidates), selected_patients=count,
        selection="largest absolute adjacent real-mask volume changes among complete four-visit cases",
        minimum_source_mask_ml=1, minimum_absolute_change_ml=1, minimum_relative_change=.35,
        nonempty_target_mask_required=True, model_errors_used_for_selection=False,
        pcr_outcomes_used_for_selection=False, masks_are="registered real annotation references, not generated segmentations",
        voxel_spacing_xyz=config["biflow"]["strict_spacing_xyz"],
        slice_selection="largest real source/target mask XOR area with both masks visible; nearby and each visit's largest-tumor planes retained",
        cases=selected)
    sweep.base._atomic_csv(OUT / "truth_census.csv", pd.DataFrame(census))
    sweep.base._atomic_json(OUT / "selection.json", protocol)
    print(f"Selected {len(selected)} of {len(candidates)} qualifying real-mask change cases", flush=True)


def shared_window(arrays, foreground, crop):
    x, y, width, height = crop
    selected = arrays[..., y:y+height, x:x+width]
    valid = foreground[..., y:y+height, x:x+width]
    values = selected[valid]
    low, high = np.percentile(values.astype(np.float32), [1, 99])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid shared real-image window")
    return [float(low), float(high)]


def save_atlas(path, slices, window):
    low, high = window
    scaled = np.rint(np.clip((slices.astype(np.float32)-low)/(high-low), 0, 1)*255).astype(np.uint8)
    image = Image.fromarray(scaled.reshape(-1, 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_mask_atlas(path, masks):
    rgba = np.zeros((*masks.shape, 4), dtype=np.uint8)
    for index, mask in enumerate(masks):
        boundary = mask & ~binary_erosion(mask)
        rgba[index, boundary] = [255, 209, 79, 255]
    Image.fromarray(rgba.reshape(-1, 256, 4)).save(path)


def prepare_real_cases():
    protocol = load_json(OUT / "selection.json")
    _, _, _, _, loaded, cache, loader = sweep.base.context()
    for index, case in enumerate(protocol["cases"], 1):
        patient, slices = case["patient_id"], case["slices"]
        directory = OUT / "cases" / patient
        directory.mkdir(parents=True, exist_ok=True)
        masks = masks_for_patient(loaded, cache, patient)
        real, foreground = [], []
        for stage in range(4):
            value, fg, _ = loader.load(patient, stage)
            real.append(value[:, slices].numpy())
            foreground.append(fg[slices].numpy())
        real, foreground = np.stack(real), np.stack(foreground)
        windows = {"dce0": shared_window(real[:, 0], foreground, case["crop"]),
                   "ser": shared_window(real[:, 1], foreground, case["crop"])}
        np.savez_compressed(directory / "real_slices.npz", dce0=real[:, 0], ser=real[:, 1],
                            foreground=foreground, masks=masks[:, slices])
        for stage in range(4):
            save_atlas(directory / f"real_T{stage}_dce0.png", real[stage, 0], windows["dce0"])
            save_atlas(directory / f"real_T{stage}_ser.png", real[stage, 1], windows["ser"])
            save_mask_atlas(directory / f"real_T{stage}_mask.png", masks[stage, slices])
        sweep.base._atomic_json(directory / "case.json", dict(case, windows=windows))
        print(f"Real lesion references {index}/{len(protocol['cases'])}", flush=True)


class RetainedDecoder:
    def __init__(self, model, config, device):
        self.model, self.device = model, device
        from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config
        base = load_ispy2_biflow_config(config["biflow"]["config"])
        if model == "biflow":
            from mewm_ispy2.ispy2_dce0_world_latents import ISPY2DCE0ContinuousLatentCache
            self.latents = ISPY2DCE0ContinuousLatentCache(base.base.data.continuous_root)
            self.codec, self.decode = sweep.base.trajectories._load_vqgan(config, device)
        else:
            sys.path.insert(0, "/path/to/local/ispy2-symmflow3d/src")
            from ispy2_symmflow.models.mewm_vqgan import load_mewm_vqgan_codec
            header = torch.load(sweep.base.OUT / "symmflow_reference/inference.pt", map_location="cpu", weights_only=True)
            statistics = header["latent_statistics"]
            self.codec = load_mewm_vqgan_codec(base.base.data.vqgan_checkpoint,
                latent_mean=statistics["mean"], latent_std=statistics["std"]).to(device).eval()

    @torch.inference_mode()
    def restore(self, steps, route, seed):
        latent = sweep.read_latent(self.model, steps, route, seed).float().to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.model == "biflow":
                value = self.decode(self.codec, self.latents.denormalize(latent))[0, 0]
            else:
                value = self.codec.decode(latent, denormalize=True)[0]
        value = value.detach().cpu().half().reshape(1, 96, 256, 256)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite restored image")
        return value


def actual_source(route, real, foreground, generated):
    if route.source_dce0 == "generated":
        return sweep.base.compose_rollout_source_mri(generated[route.source_stage], real, foreground)
    return real


def local_change_metrics(prediction, actual_input, real_source, real_target, region):
    if not region.any():
        return dict(roi_voxels=0, **{key: None for key in ("model_mae", "actual_input_copy_mae",
            "real_source_copy_mae", "predicted_change_l1", "true_change_l1", "change_cosine")})
    def values(tensor):
        return tensor.float().numpy()[region].astype(np.float64)
    p, a, s, t = map(values, (prediction, actual_input, real_source, real_target))
    pdiff, tdiff = p-s, t-s
    denominator = np.linalg.norm(pdiff)*np.linalg.norm(tdiff)
    return dict(roi_voxels=int(region.sum()), model_mae=float(np.abs(p-t).mean()),
                actual_input_copy_mae=float(np.abs(a-t).mean()), real_source_copy_mae=float(np.abs(s-t).mean()),
                predicted_change_l1=float(np.abs(pdiff).mean()), true_change_l1=float(np.abs(tdiff).mean()),
                change_cosine=float(np.dot(pdiff, tdiff)/denominator) if denominator > 0 else None)


def export_model(model, device):
    protocol = load_json(OUT / "selection.json")
    config, _, routes, seeds, loaded, cache, loader = sweep.base.context()
    decoder = RetainedDecoder(model, config, device)
    old_patient = load_json(sweep.OUT / "protocol.json")["display_patients"][0]
    sample = next(r for r in routes["direct_t0"] if r.patient_id == old_patient)
    restored = decoder.restore(2, sample, seeds[sample.target_key])
    original = sweep.prediction(model, 2, sample, seeds[sample.target_key])
    if not torch.equal(restored, original):
        raise ValueError(f"Retained-latent decode differs from saved image: {(restored.float()-original).abs().max().item()}")
    sweep.base._atomic_json(OUT / f"{model}_decoder_audit.json", dict(status="passed", model=model,
        saved_latent_decode_exact=True, maximum_absolute_difference=0, new_sampling=False))
    for case_index, entry in enumerate(protocol["cases"], 1):
        patient = entry["patient_id"]
        directory = OUT / "cases" / patient
        case = load_json(directory / "case.json")
        marker = directory / f"{model}_complete.json"
        if marker.exists() and load_json(marker).get("status") == "completed":
            print(f"{model}: reuse completed case {case_index}/{len(protocol['cases'])}", flush=True)
            continue
        masks = masks_for_patient(loaded, cache, patient)
        real = [loader.load(patient, stage)[:2] for stage in range(4)]
        regions = {}
        for policy in sweep.POLICIES:
            for route in (r for r in routes[policy] if r.patient_id == patient):
                edge = (route.source_stage, route.target_stage)
                if edge not in regions:
                    region = binary_dilation(masks[edge[0]] | masks[edge[1]], iterations=4)
                    region &= real[edge[0]][1].numpy() & real[edge[1]][1].numpy()
                    regions[edge] = region
        rows = []
        for steps in sweep.STEPS:
            predictions = {}
            for policy in sweep.POLICIES:
                previous = {}
                for route in (r for r in routes[policy] if r.patient_id == patient):
                    key = (policy, route.target_stage)
                    if policy != "direct_t0" and route.source_stage == 0:
                        value = predictions[("direct_t0", route.target_stage)]
                    elif sweep.artifact(model, steps, route, "decoded").exists():
                        value = sweep.prediction(model, steps, route, seeds[route.target_key])
                    else:
                        value = decoder.restore(steps, route, seeds[route.target_key])
                    predictions[key] = value
                    source, foreground = real[route.source_stage]
                    actual = actual_source(route, source, foreground, previous)
                    target, target_foreground = real[route.target_stage]
                    region = regions[(route.source_stage, route.target_stage)]
                    stem = f"{model}_e{steps}_{policy}_T{route.target_stage}"
                    output_slices = value[0, case["slices"]].numpy()
                    input_slices = actual[0, case["slices"]].numpy()
                    save_atlas(directory / f"{stem}_output.png", output_slices, case["windows"]["dce0"])
                    save_atlas(directory / f"{stem}_input.png", input_slices, case["windows"]["dce0"])
                    np.savez_compressed(directory / f"{stem}.npz", input=input_slices, output=output_slices)
                    rows.append(dict(patient_id=patient, model=model, steps=steps, strategy=policy,
                        source_stage=route.source_stage, target_stage=route.target_stage,
                        source_dce0=route.source_dce0, delta_days=route.delta_days,
                        source_ser="real" if model == "biflow" else "not used",
                        **local_change_metrics(value[0], actual[0], source[0], target[0], region)))
                    previous[route.target_stage] = value
        sweep.base._atomic_csv(directory / f"{model}_change_metrics.csv", pd.DataFrame(rows))
        sweep.base._atomic_json(marker, dict(status="completed", schema=SCHEMA, model=model, rows=len(rows)))
        print(f"{model}: actual inputs and lesion outputs {case_index}/{len(protocol['cases'])}", flush=True)
    sweep.base._atomic_json(OUT / f"{model}_export.json", dict(status="completed", model=model, cases=len(protocol["cases"]),
        steps=list(sweep.STEPS), new_sampling=False, decoded_from_existing_latents=True))


def collect_cases():
    protocol = load_json(OUT / "selection.json")
    cases = []
    for entry in protocol["cases"]:
        directory = OUT / "cases" / entry["patient_id"]
        case = load_json(directory / "case.json")
        frames = [pd.read_csv(directory / f"{model}_change_metrics.csv") for model in sweep.MODELS]
        metrics = pd.concat(frames, ignore_index=True)
        if len(metrics) != 72 or metrics.duplicated(["model", "steps", "strategy", "target_stage"]).any():
            raise ValueError("Incomplete lesion comparison inventory")
        case["metrics"] = metrics.astype(object).where(pd.notna(metrics), None).to_dict("records")
        cases.append(case)
    return cases


def validate_assets(cases):
    checked = 0
    for case in cases:
        directory = OUT / "cases" / case["patient_id"]
        with np.load(directory / "real_slices.npz") as payload:
            real = {key: payload[key] for key in payload.files}
        for record in case["metrics"]:
            stem = f"{record['model']}_e{record['steps']}_{record['strategy']}_T{record['target_stage']}"
            with np.load(directory / f"{stem}.npz") as payload:
                actual, output = payload["input"], payload["output"]
            expected_shape = (len(case["slices"]), 256, 256)
            assert actual.shape == output.shape == expected_shape
            assert np.isfinite(actual).all() and np.isfinite(output).all()
            source = record["source_stage"]
            if record["source_dce0"] == "real":
                expected_input = real["dce0"][source]
            else:
                previous_stem = f"{record['model']}_e{record['steps']}_{record['strategy']}_T{source}"
                with np.load(directory / f"{previous_stem}.npz") as previous:
                    expected_input = np.where(real["foreground"][source], previous["output"], np.float16(0))
            np.testing.assert_array_equal(actual, expected_input)
            low, high = case["windows"]["dce0"]
            for name, values in (("input", actual), ("output", output)):
                expected_pixels = np.rint(np.clip((values.astype(np.float32)-low)/(high-low), 0, 1)*255).astype(np.uint8)
                with Image.open(directory / f"{stem}_{name}.png") as atlas:
                    np.testing.assert_array_equal(np.asarray(atlas), expected_pixels.reshape(-1, 256))
            checked += 1
        print(f"Actual-input and pixel audit: {case['patient_id']}", flush=True)
    sweep.base._atomic_json(OUT / "asset_audit.json", dict(status="passed", schema=SCHEMA,
        cases=len(cases), comparisons=checked, png_atlases=checked*2,
        real_input_equals_real_source=True, rollout_input_equals_masked_same_model_step_previous_output=True,
        all_png_pixels_match_saved_view_arrays=True, contours_on_generated_images=False,
        model_errors_used_for_selection=False, new_sampling=False))


def atlas_panel(directory, stem, case, mask=None, tile=224):
    x, y, width, height = case["crop"]
    offset = case["slices"].index(case["central_slice"])*256
    box = (x, offset+y, x+width, offset+y+height)
    with Image.open(directory / f"{stem}.png") as image:
        panel = image.convert("RGB").crop(box)
    if mask:
        with Image.open(directory / f"{mask}.png") as image:
            overlay = image.crop(box)
            panel.paste(overlay, (0, 0), overlay)
    return panel.resize((tile, tile), Image.Resampling.NEAREST)


def patient_sheet(case, policy):
    directory = OUT / "cases" / case["patient_id"]
    target = case["preferred_target"]
    records = {(r["model"], r["steps"]): r for r in case["metrics"]
               if r["strategy"] == policy and r["target_stage"] == target}
    source = records[("biflow", 20)]["source_stage"]
    canvas = Image.new("RGB", (1510, 1900), "white")
    draw = ImageDraw.Draw(canvas)
    font = sweep.metrics.font
    draw.text((20, 14), f"{case['patient_id']} | T{source} to T{target} | {policy}", fill="#202a25", font=font(22, True))
    draw.text((20, 49), f"Selected real-mask interval T{case['selected_source']}-T{target}: {case['relative_change']*100:.1f}% | Axial z={case['central_slice']}",
              fill="#536359", font=font(17))
    volume_text = " | ".join(f"T{stage}: {volume:.2f} mL" for stage, volume in enumerate(case["volume_ml"]))
    draw.text((20, 79), f"Real annotation volumes in registered crop: {volume_text}", fill="#536359", font=font(16))
    for index, (stage, channel, label) in enumerate(((source, "dce0", "Real source DCE0"),
        (source, "ser", "Real source SER / BiFlow input"), (target, "dce0", "Real target DCE0"),
        (target, "ser", "Real target SER / reference"))):
        x = 20+index*372
        draw.text((x, 110), f"{label} T{stage}", fill="#202a25", font=font(16))
        canvas.paste(atlas_panel(directory, f"real_T{stage}_{channel}", case, f"real_T{stage}_mask"), (x, 138))
    draw.text((20, 382), "Yellow contours are real annotations, drawn only on real images. No predicted tumor segmentation is available.", fill="#536359", font=font(16))
    for row_index, steps in enumerate(sweep.STEPS):
        y = 422+row_index*348
        draw.line((20, y, 1490, y), fill="#cbd5d0", width=2)
        draw.text((20, y+9), f"Euler-{steps}", fill="#202a25", font=font(19, True))
        for model_index, model in enumerate(sweep.MODELS):
            record = records[(model, steps)]
            x = 20+model_index*756
            draw.text((x+130, y+9), "BiFlow" if model == "biflow" else "SymmFlow", font=font(19, True),
                      fill="#187568" if model == "biflow" else "#bc4c42")
            stem = f"{model}_e{steps}_{policy}_T{target}"
            actual_is_real = record["source_dce0"] == "real"
            columns = [(f"Actual input: {'real' if actual_is_real else 'generated'} T{source}", stem+"_input",
                        f"real_T{source}_mask" if actual_is_real else None),
                       (f"Generated target T{target}", stem+"_output", None),
                       (f"Real target T{target}", f"real_T{target}_dce0", f"real_T{target}_mask")]
            for column, (label, path, mask) in enumerate(columns):
                draw.text((x+column*244, y+40), label, fill="#202a25", font=font(15))
                canvas.paste(atlas_panel(directory, path, case, mask), (x+column*244, y+66))
            mae, baseline = record["model_mae"], record["actual_input_copy_mae"]
            text = "No nonempty regional evaluation mask" if mae is None else f"3D ROI MAE {mae:.5f} | Actual-input copy MAE {baseline:.5f}"
            draw.text((x, y+300), text, fill="#536359", font=font(15))
    draw.text((20, 1837), "One shared real-ROI DCE0 intensity window; the same crop and plane for all images. Models generate DCE0 only.", fill="#536359", font=font(16))
    draw.text((20, 1868), "Selected by real annotation change. These illustrative cases do not establish segmentation accuracy or modify cohort metrics.", fill="#536359", font=font(15))
    return canvas


def sheets(cases):
    from scripts.report_ispy2_sampling_sweep import write_case_pdf
    for policy in sweep.POLICIES:
        directory = OUT / "sheets" / policy
        directory.mkdir(parents=True, exist_ok=True)
        pages = []
        for index, case in enumerate(cases, 1):
            image = patient_sheet(case, policy)
            image.save(directory / f"{case['patient_id']}.png")
            pages.append(image)
            print(f"{policy}: lesion sheet {index}/{len(cases)}", flush=True)
        write_case_pdf(pages, OUT / f"{policy}_20_cases.pdf")
    sweep.base._atomic_json(OUT / "sheets_audit.json", dict(status="passed", policies=list(sweep.POLICIES),
        cases=len(cases), pages_per_pdf=20, actual_inputs_visible=True,
        dimensions=[1510, 1900], pdf_encoding="lossless", default_plane="truth-selected lesion slice"))


def publish_index(cases=None):
    from scripts.report_ispy2_sampling_sweep import region_coverage
    cases = collect_cases() if cases is None else cases
    for model in sweep.MODELS:
        assert load_json(OUT / f"{model}_export.json")["status"] == "completed"
    cohort = dict(images=pd.read_csv(sweep.OUT / "comparison/image_summary.csv").to_dict("records"),
                  pcr=pd.read_csv(sweep.OUT / "comparison/pcr_summary.csv").to_dict("records"),
                  coverage=region_coverage())
    default = next(case["patient_id"] for case in cases if case["preferred_target"] == 3)
    payload = dict(schema=SCHEMA, cases=cases, default_patient=default, cohort=cohort)
    sweep.base._atomic_json(OUT / "viewer_data.json", payload)
    template = (sweep.ROOT / "scripts/templates/ispy2_tumor_review.html").read_text()
    document = template.replace("__STUDY_DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/"))
    temporary = sweep.OUT / "index.tumor.tmp.html"
    temporary.write_text(document)
    previous = sweep.OUT / "overview.html"
    if not previous.exists():
        shutil.copyfile(sweep.OUT / "index.html", previous)
    temporary.replace(sweep.OUT / "index.html")


def publish():
    cases = collect_cases()
    validate_assets(cases)
    frames = [pd.DataFrame(case["metrics"]) for case in cases]
    sweep.base._atomic_csv(OUT / "focused_change_metrics.csv", pd.concat(frames, ignore_index=True))
    sheets(cases)
    publish_index(cases)
    sweep.base._atomic_json(OUT / "publication.json", dict(status="published", schema=SCHEMA, cases=len(cases),
        updated_utc=datetime.now(timezone.utc).isoformat(), entrypoint=str(sweep.OUT / "index.html"),
        original_overview=str(sweep.OUT / "overview.html"), global_cohort_metrics_changed=False,
        actual_inputs_visible=True, generated_contours=False, new_sampling=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "real", "export", "publish"))
    parser.add_argument("--model", choices=sweep.MODELS, default="biflow")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-gib", type=float, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "select":
        select_cases()
    elif args.stage == "real":
        prepare_real_cases()
    elif args.stage == "publish":
        publish()
    else:
        device = torch.device(args.device)
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/torch.cuda.get_device_properties(device).total_memory, device)
        export_model(args.model, device)


if __name__ == "__main__":
    main()

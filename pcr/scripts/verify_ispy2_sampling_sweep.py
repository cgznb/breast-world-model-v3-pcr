"""Verify published sweep tables, lossless case PDFs, and the static gallery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ispy2_biflow_symmflow_steps_20260911"
STEPS = [2, 10, 20, 50]
MODELS = ["biflow", "symmflow"]
POLICIES = ["direct_t0", "adjacent_real", "rollout_generated_dce0_real_ser"]
DEPTHS = ["T0", "T0-T1", "T0-T2", "T0-T3"]
REGIONS = ["common_foreground", "target_tumor", "tumor_union_neighborhood"]


def read(path):
    return json.loads(path.read_text())


def record(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                                         **values), indent=2) + "\n")
    temporary.replace(path)


def verify_tables():
    audit = read(OUT / "comparison/audit.json")
    assert audit["status"] == "passed" and audit["scope"] == "complete"
    assert audit["steps"] == STEPS
    assert read(OUT / "status.json")["state"] == "completed"
    for model in MODELS:
        for steps in STEPS:
            directory = OUT / model / f"euler_{steps}"
            for name, status in (("audit", "passed"), ("complete", "completed")):
                item = read(directory / f"{name}.json")
                assert (item["model"], item["steps"], item["status"]) == (model, steps, status)
            pcr_audit = read(directory / "pcr/audit.json")
            assert pcr_audit["real_frozen_replay"] == "passed"
            assert pcr_audit["effective_contiguous_visits_T0_T1_T2_T3"] == [102, 101, 97, 94]
            assert pcr_audit["future_DCE0_only_replaced"] and not pcr_audit["new_training"]

    predictions = pd.read_csv(OUT / "comparison/pcr_predictions.csv")
    summary = pd.read_csv(OUT / "comparison/pcr_summary.csv")
    keys = ["model", "steps", "strategy", "temporal_depth"]
    assert len(predictions) == 102000 and len(summary) == 100
    assert not predictions.duplicated(keys + ["seed", "patient_id"]).any()
    assert predictions.probability.between(0, 1).all()
    assert predictions.groupby("patient_id").label.nunique().eq(1).all()
    real = predictions[predictions.model == "real"]
    assert len(real) == 4080 and real.steps.eq(0).all()
    scores = []
    for values, group in predictions.groupby(keys + ["seed"]):
        assert len(group) == group.patient_id.nunique() == 102
        assert group.label.sum() == 32
        scores.append(dict(zip(keys + ["seed"], values),
                           auroc=roc_auc_score(group.label, group.probability),
                           prauc=average_precision_score(group.label, group.probability)))
    computed = pd.DataFrame(scores).groupby(keys)[["auroc", "prauc"]].agg(["mean", "std"])
    computed.columns = ["_".join(column) for column in computed.columns]
    published = summary.set_index(keys).loc[computed.index]
    np.testing.assert_allclose(computed, published[computed.columns], atol=1e-12, rtol=0)
    initial = predictions[predictions.temporal_depth == "T0"]
    assert initial.groupby(["seed", "patient_id"]).probability.nunique().eq(1).all()
    t1 = predictions[(predictions.model != "real") & (predictions.temporal_depth == "T0-T1")]
    assert t1.groupby(["model", "steps", "seed", "patient_id"]).probability.nunique().eq(1).all()

    endpoints = pd.read_csv(OUT / "comparison/image_endpoints.csv")
    image_summary = pd.read_csv(OUT / "comparison/image_summary.csv")
    image_keys = ["model", "steps", "strategy", "region"]
    assert len(endpoints) == 21312 and len(image_summary) == 72
    assert not endpoints.duplicated(image_keys + ["patient_id", "target_stage"]).any()
    support = endpoints.groupby(["patient_id", "target_stage", "region"])
    assert support[["voxel_count", "ssim_center_count"]].nunique().eq(1).all().all()
    metrics = ["mae_unclipped", "rmse_unclipped", "psnr_windowed_db", "ssim_3d_windowed",
               "change_cosine", "predicted_change_mae", "true_change_mae"]
    nonempty = endpoints.voxel_count.gt(0)
    assert np.isfinite(endpoints.loc[nonempty, metrics].to_numpy()).all()
    assert endpoints.loc[~nonempty, metrics].isna().all().all()
    patient = endpoints.groupby(image_keys + ["patient_id"])[metrics].mean()
    computed_image = patient.groupby(image_keys).agg(["mean", "std", "count"])
    computed_image.columns = ["_".join(column) for column in computed_image.columns]
    published_image = image_summary.set_index(image_keys).loc[computed_image.index]
    np.testing.assert_allclose(computed_image, published_image[computed_image.columns], atol=1e-12, rtol=0)
    record(OUT / "comparison/publication_table_audit.json", status="passed", settings=8,
           pcr_predictions=len(predictions), real_predictions=len(real), pcr_summaries=len(summary),
           image_endpoints=len(endpoints), image_summaries=len(image_summary),
           empty_region_rows=int((~nonempty).sum()), nonempty_region_metrics_finite=True,
           pcr_auroc_prauc_recomputed=True, patient_macro_image_metrics_recomputed=True,
           shared_t0_predictions=True, shared_t1_source_policy_predictions=True)
    print("Tables: 102,000 predictions, image summaries, and pCR AUROC/PR-AUC verified", flush=True)


def verify_pdfs():
    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    patients = read(OUT / "protocol.json")["display_patients"]
    for name, steps in (("euler_50", [50]), ("all_steps", STEPS)):
        directory = OUT / "figures" / name
        audit = read(directory / "display_audit.json")
        assert audit["steps"] == steps and audit["patients"] == patients
        reader = PdfReader(directory / "comparison_20_cases.pdf")
        assert len(reader.pages) == len(patients) == 20
        for index, (page, patient) in enumerate(zip(reader.pages, patients), 1):
            operations = ContentStream(page.get_contents(), reader).operations
            draws = [(position, operands[0]) for position, (operands, op) in enumerate(operations) if op == b"Do"]
            assert len(draws) == 1
            position, reference = draws[0]
            transform = [operands for operands, op in operations[:position] if op == b"cm"][-1]
            assert transform[0] > 0 and transform[3] < 0
            assert transform[1] == transform[2] == 0
            resource = page["/Resources"]["/XObject"][reference].get_object()
            assert resource["/Filter"] == "/FlateDecode"
            # Matplotlib shares image resources and draws each image with a negative Y scale.
            embedded = page.images[str(reference)].image.convert("RGB")
            with Image.open(directory / f"{patient}.png") as original:
                assert list(original.size) == audit["dimensions"]
                np.testing.assert_array_equal(np.asarray(embedded)[::-1], np.asarray(original.convert("RGB")))
            print(f"PDF {name}: exact pixels {index}/20", flush=True)
        record(directory / "pdf_fidelity_audit.json", status="passed", steps=steps, pages=20,
               encoding="lossless_FlateDecode", pdf_pixels_equal_png_after_draw_transform=True)


def verify_browser():
    from playwright.sync_api import sync_playwright

    patients = read(OUT / "protocol.json")["display_patients"]
    image_metrics = pd.read_csv(OUT / "comparison/image_summary.csv")
    pcr_metrics = pd.read_csv(OUT / "comparison/pcr_summary.csv").set_index(
        ["model", "steps", "strategy", "temporal_depth"])
    directory = OUT / "comparison/browser_verification"
    directory.mkdir(parents=True, exist_ok=True)
    options = ["figures/all_steps"] + [f"figures/euler_{steps}" for steps in STEPS]
    errors, failed_requests, counts = [], [], {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        for name, viewport in (("desktop", {"width": 1440, "height": 1000}),
                               ("mobile", {"width": 390, "height": 844})):
            page = browser.new_page(viewport=viewport, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: failed_requests.append(request.url))
            gallery_path = OUT / ("overview.html" if (OUT / "tumor_review/publication.json").exists() else "index.html")
            page.goto(gallery_path.as_uri(), wait_until="load")
            assert page.locator("#patient option").evaluate_all("xs => xs.map(x => x.value)") == patients
            assert page.locator("#sampling option").evaluate_all("xs => xs.map(x => x.value)") == options
            assert "Pending" not in page.locator("header").inner_text()
            assert page.locator("#region option").evaluate_all("xs => xs.map(x => x.value)") == REGIONS
            for region in REGIONS:
                page.select_option("#region", region)
                page.locator("#image-plot").evaluate("async image => {await image.decode()}")
                for policy in POLICIES:
                    page.select_option("#policy", policy)
                    for depth in DEPTHS:
                        page.select_option("#depth", depth)
                        assert page.locator("#metrics-body tr").count() == 9
                        assert not any(word in page.locator("#metrics-body").inner_text()
                                       for word in ("NaN", "undefined", "Infinity"))
                        cells = page.locator("#metrics-body tr").evaluate_all(
                            "xs => xs.map(row => [...row.cells].map(cell => cell.textContent))")
                        expected = image_metrics[(image_metrics.strategy == policy) &
                                                 (image_metrics.region == region)].sort_values(["model", "steps"])
                        for actual, item in zip(cells[:-1], expected.itertuples()):
                            assert actual[:2] == ["BiFlow" if item.model == "biflow" else "SymmFlow", str(item.steps)]
                            pcr_row = pcr_metrics.loc[(item.model, item.steps, policy, depth)]
                            np.testing.assert_allclose([float(value) for value in actual[2:]],
                                [item.mae_unclipped_mean, item.ssim_3d_windowed_mean, item.psnr_windowed_db_mean,
                                 pcr_row.auroc_mean, pcr_row.prauc_mean], atol=5.0001e-6, rtol=0)
                        reference = pcr_metrics.loc[("real", 0, "real", depth)]
                        assert cells[-1][:5] == ["Real images", "-", "-", "-", "-"]
                        np.testing.assert_allclose([float(value) for value in cells[-1][5:]],
                            [reference.auroc_mean, reference.prauc_mean], atol=5.0001e-6, rtol=0)
            case_count = 0
            for option in options:
                page.select_option("#sampling", option)
                for patient in patients:
                    page.select_option("#patient", patient)
                    page.locator("#case-image").evaluate("async image => {await image.decode()}")
                    size = page.locator("#case-image").evaluate("""image => ({
                        width:image.naturalWidth, height:image.naturalHeight,
                        displayedWidth:image.getBoundingClientRect().width,
                        displayedHeight:image.getBoundingClientRect().height})""")
                    expected_width = 2540 if option.endswith("all_steps") else 1136
                    assert (size["width"], size["height"]) == (expected_width, 2460)
                    assert abs(size["displayedWidth"] / size["displayedHeight"] - expected_width / 2460) < .001
                    assert page.locator("#image-link").get_attribute("href") == f"{option}/{patient}.png"
                    assert page.locator("#pdf-link").get_attribute("href") == f"{option}/comparison_20_cases.pdf"
                    case_count += 1
            page.select_option("#sampling", options[0])
            page.select_option("#patient", patients[0])
            page.select_option("#policy", "adjacent_real")
            page.select_option("#region", "common_foreground")
            page.select_option("#depth", "T0-T3")
            page.locator("#case-image").evaluate("async image => {await image.decode()}")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert page.locator("img").evaluate_all("xs => xs.every(x => x.complete && x.naturalWidth > 0)")
            for link in page.locator("a[href]").evaluate_all("xs => xs.map(x => x.href)"):
                parsed = urlparse(link)
                assert parsed.scheme == "file" and Path(unquote(parsed.path)).is_file()
            page.screenshot(path=str(directory / f"{name}.png"), full_page=True)
            counts[name] = dict(case_views=case_count, region_policy_depth_filters=36, metric_rows_per_filter=9,
                                page_overflow=False, image_aspect_ratios="passed", local_links="passed")
            page.close()
        browser.close()
    assert not errors and not failed_requests, (errors, failed_requests)
    record(directory / "audit.json", status="passed", steps=STEPS, patients=20,
           viewports=counts, javascript_errors=errors, failed_requests=failed_requests)
    print("Gallery: desktop/mobile, 200 case views, and 72 metric filter combinations verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=("tables", "pdf", "browser"),
                        default=["tables", "pdf", "browser"])
    args = parser.parse_args()
    actions = {"tables": verify_tables, "pdf": verify_pdfs, "browser": verify_browser}
    for check in args.checks:
        actions[check]()


if __name__ == "__main__":
    main()

"""Verify the actual-input lesion gallery, its controls, and lossless PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from PIL import Image

from verify_ispy2_sampling_sweep import DEPTHS, MODELS, OUT, POLICIES, REGIONS, STEPS, record

REVIEW = OUT / "tumor_review"


def read(path):
    return json.loads(path.read_text())


def verify_pdfs():
    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    patients = [case["patient_id"] for case in read(REVIEW / "selection.json")["cases"]]
    for policy in POLICIES:
        reader = PdfReader(REVIEW / f"{policy}_20_cases.pdf")
        assert len(reader.pages) == len(patients) == 20
        for page, patient in zip(reader.pages, patients):
            operations = ContentStream(page.get_contents(), reader).operations
            draws = [(position, operands[0]) for position, (operands, op) in enumerate(operations) if op == b"Do"]
            assert len(draws) == 1
            position, reference = draws[0]
            transform = [operands for operands, op in operations[:position] if op == b"cm"][-1]
            assert transform[0] > 0 and transform[3] < 0 and transform[1] == transform[2] == 0
            resource = page["/Resources"]["/XObject"][reference].get_object()
            assert resource["/Filter"] == "/FlateDecode"
            embedded = page.images[str(reference)].image.convert("RGB")
            with Image.open(REVIEW / "sheets" / policy / f"{patient}.png") as original:
                assert original.size == (1510, 1900)
                np.testing.assert_array_equal(np.asarray(embedded)[::-1], np.asarray(original))
        print(f"{policy}: all 20 PDF pages match PNG pixels", flush=True)
    record(REVIEW / "pdf_fidelity_audit.json", status="passed", policies=POLICIES,
           pages=60, encoding="lossless_FlateDecode", exact_png_pixels=True)


def ready(page):
    page.wait_for_selector('body[data-render-ready="true"]', timeout=60000)
    assert not page.locator("#error").inner_text()


def check_view(page, case, policy, target, counts):
    ready(page)
    canvases = page.locator("#cases canvas").evaluate_all("""xs => xs.map(c => {
        const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
        let low=255, high=0;
        for(let i=0;i<pixels.length;i+=4){low=Math.min(low,pixels[i]);high=Math.max(high,pixels[i])}
        const box=c.getBoundingClientRect();
        return {image:c.dataset.image,mask:c.dataset.mask,z:Number(c.dataset.z),
                ready:c.dataset.ready,low,high,width:box.width,height:box.height};
    })""")
    assert len(canvases) == 4+len(counts)*6
    assert all(c["ready"] == "true" and c["high"] > c["low"] for c in canvases)
    assert all(abs(c["width"]-c["height"]) < 1 for c in canvases)
    assert all(c["z"] in case["slices"] for c in canvases)
    source = 0 if policy == "direct_t0" else target-1
    generated = policy == "rollout_generated_dce0_real_ser" and source > 0
    assert f"T{source} to T{target}" in page.locator("#case-interval").inner_text()
    for count in counts:
        for model in MODELS:
            panel = page.locator(f'.step-row[data-steps="{count}"] .model.{model}')
            labels = panel.locator("figcaption").all_text_contents()
            assert labels == [f"Actual input: {'generated' if generated else 'real'} T{source}",
                              f"Generated target T{target}", f"Real target T{target}"]
            images = panel.locator("canvas").evaluate_all("xs=>xs.map(c=>({image:c.dataset.image,mask:c.dataset.mask}))")
            stem = f"tumor_review/cases/{case['patient_id']}/{model}_e{count}_{policy}_T{target}"
            assert images[0]["image"] == stem+"_input.png"
            assert images[1] == dict(image=stem+"_output.png", mask="")
            assert bool(images[0]["mask"]) == (not generated)
            row = next(r for r in case["metrics"] if (r["model"], r["steps"], r["strategy"], r["target_stage"]) ==
                       (model, count, policy, target))
            stats = panel.locator(".model-metrics").inner_text()
            for key in ("model_mae", "actual_input_copy_mae", "change_cosine"):
                assert ("N/A" if row[key] is None else f"{row[key]:.5f}") in stats
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.evaluate("imageCache.size <= 64")


def verify_metrics(page, data):
    for region in REGIONS:
        page.select_option("#region", region)
        page.locator("#image-plot").evaluate("async image=>{await image.decode()}")
        for policy in POLICIES:
            page.select_option("#policy", policy)
            for depth in DEPTHS:
                page.select_option("#depth", depth)
                cells = page.locator("#metrics-body tr").evaluate_all(
                    "xs=>xs.map(row=>[...row.cells].map(cell=>cell.textContent))")
                assert len(cells) == 9
                expected = sorted((r for r in data["cohort"]["images"] if r["strategy"] == policy and r["region"] == region),
                                  key=lambda r: (r["model"], r["steps"]))
                for actual, item in zip(cells[:-1], expected):
                    pcr = next(r for r in data["cohort"]["pcr"] if
                               (r["model"], r["steps"], r["strategy"], r["temporal_depth"]) ==
                               (item["model"], item["steps"], policy, depth))
                    values = [item["mae_unclipped_mean"], item["ssim_3d_windowed_mean"], item["psnr_windowed_db_mean"],
                              pcr["auroc_mean"], pcr["prauc_mean"]]
                    np.testing.assert_allclose([float(v) for v in actual[2:]], values, atol=5.0001e-6, rtol=0)
                real = next(r for r in data["cohort"]["pcr"] if r["model"] == "real" and r["temporal_depth"] == depth)
                np.testing.assert_allclose([float(v) for v in cells[-1][-2:]],
                                           [real["auroc_mean"], real["prauc_mean"]], atol=5.0001e-6, rtol=0)


def verify_browser(base_url):
    from playwright.sync_api import sync_playwright

    data = read(REVIEW / "viewer_data.json")
    for key, filename in (("images", "image_summary.csv"), ("pcr", "pcr_summary.csv")):
        pd.testing.assert_frame_equal(pd.DataFrame(data["cohort"][key]), pd.read_csv(OUT / "comparison" / filename), check_like=True)
    directory = REVIEW / "browser_verification"
    directory.mkdir(exist_ok=True)
    errors, failed, bad_http, counts = [], [], [], {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        for name, viewport in (("desktop", dict(width=1440, height=1000)),
                               ("mobile", dict(width=390, height=844))):
            page = browser.new_page(viewport=viewport, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: failed.append(request.url))
            page.on("response", lambda response: bad_http.append(response.url) if response.status >= 400 else None)
            page.goto(base_url+"/index.html", wait_until="load")
            ready(page)
            assert page.locator("#case-patient").input_value() == data["default_patient"]
            assert page.locator("#case-policy").input_value() == "adjacent_real"
            assert page.locator("#case-target").input_value() == "3"
            assert page.locator("#case-patient option").count() == 20
            case_views = 0
            page.select_option("#case-patient", data["default_patient"])
            page.select_option("#case-policy", "adjacent_real")
            case = next(c for c in data["cases"] if c["patient_id"] == data["default_patient"])
            for value in [str(s) for s in STEPS]+["all"]:
                page.select_option("#case-steps", value)
                check_view(page, case, "adjacent_real", 3, STEPS if value == "all" else [int(value)])
            before = page.locator("#slice").input_value()
            page.locator("#slice-next").click()
            ready(page)
            assert page.locator("#slice").input_value() != before
            page.locator("#slice-prev").click()
            ready(page)
            assert page.locator("#slice").input_value() == before
            image_before = page.locator(".step-row canvas").first.evaluate("c=>c.toDataURL()")
            page.locator("#show-mask").uncheck()
            ready(page)
            assert image_before != page.locator(".step-row canvas").first.evaluate("c=>c.toDataURL()")
            roi_pixels = page.locator(".step-row canvas").first.evaluate("c=>c.toDataURL()")
            page.locator('input[name="field"][value="full"]').check(force=True)
            ready(page)
            assert roi_pixels != page.locator(".step-row canvas").first.evaluate("c=>c.toDataURL()")
            page.locator('input[name="field"][value="roi"]').check(force=True)
            page.locator("#show-mask").check()
            page.locator("#timeline summary").click()
            page.wait_for_function("document.querySelectorAll('#timeline-images canvas').length===8")
            ready(page)
            assert page.locator("#timeline-images canvas").count() == 8
            page.locator("#timeline").screenshot(path=str(directory / f"{name}_timeline.png"))
            page.locator("#timeline summary").click()
            page.wait_for_function("document.querySelectorAll('#timeline-images canvas').length===0")
            ready(page)
            with page.expect_download() as download:
                page.locator("#download-png").click()
            path = directory / f"{name}_download.png"
            download.value.save_as(path)
            with Image.open(path) as image:
                assert image.size == (1680, 1764) and np.asarray(image).std() > 10
            verify_metrics(page, data)
            page.select_option("#region", "common_foreground")
            page.select_option("#policy", "adjacent_real")
            page.locator("#image-plot").evaluate("async image=>{await image.decode()}")
            for link in page.locator("a[href]").evaluate_all("xs=>xs.map(a=>a.href)"):
                parsed = urlparse(link)
                assert parsed.netloc == urlparse(base_url).netloc
                path = OUT / unquote(parsed.path).lstrip("/")
                assert path.is_file(), str(path)
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(path=str(directory / f"{name}_top.png"))
            page.locator(".step-row").first.screenshot(path=str(directory / f"{name}_comparison.png"))
            page.locator("#real-reference summary").click()
            page.locator("#truth-reference").screenshot(path=str(directory / f"{name}_truth.png"))
            page.locator("#real-reference summary").click()
            for case in data["cases"]:
                page.select_option("#case-patient", case["patient_id"])
                for policy in POLICIES:
                    page.select_option("#case-policy", policy)
                    for target in range(1, 4):
                        page.select_option("#case-target", str(target))
                        check_view(page, case, policy, target, STEPS)
                        case_views += 1
                print(f"{name}: verified case {case_views//9}/20, all stages/policies/steps", flush=True)
            counts[name] = dict(case_views=case_views, step_controls=5, metric_filter_combinations=36,
                                actual_input_labels=True, generated_contours=False, canvas_pixels_nonblank=True,
                                slice_and_roi_controls=True, png_download=True, page_overflow=False)
            page.close()
        browser.close()
    assert not errors and not failed and not bad_http, (errors, failed, bad_http)
    record(directory / "audit.json", status="passed", cases=20, viewports=counts,
           cohort_tables_equal_original_csv=True, javascript_errors=errors,
           failed_requests=failed, http_errors=bad_http)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=("pdf", "browser"), default=["pdf", "browser"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    for check in args.checks:
        if check == "pdf":
            verify_pdfs()
        else:
            verify_browser(args.base_url.rstrip("/"))


if __name__ == "__main__":
    main()

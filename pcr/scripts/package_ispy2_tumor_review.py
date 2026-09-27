"""Package the completed lesion review for ordinary offline browser opening."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ispy2_biflow_symmflow_steps_20260911"
DESTINATION = ROOT.parents[1] / "visualizations/ispy2_tumor_comparison_offline_20260914"


class StudyParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_study = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "study-data":
            self.in_study = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_study = False

    def handle_data(self, data):
        if self.in_study:
            self.parts.append(data)


def replace_once(document, old, new):
    if document.count(old) != 1:
        raise ValueError(f"Published viewer no longer matches the expected export hook: {old}")
    return document.replace(old, new)


def build(destination):
    destination = destination.resolve()
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError(f"Offline export already exists: {destination}")
    review = SOURCE / "tumor_review"
    publication = json.loads((review / "publication.json").read_text())
    if publication["status"] != "published":
        raise ValueError("Only the completed tumor review can be exported")
    document = (SOURCE / "index.html").read_text()
    parser = StudyParser()
    parser.feed(document)
    study = json.loads("".join(parser.parts))
    if len(study["cases"]) != 20:
        raise ValueError("The viewer must include all twenty selected cases")
    document = replace_once(document, "image.src=url",
                            "window.ispy2Offline.image(url).then(source=>{image.src=source},reject)")
    document = replace_once(document, '<script src="tumor_review/lucide.min.js"></script>',
                            '<script src="offline_assets/loader.js"></script>\n'
                            '<script src="tumor_review/lucide.min.js"></script>')
    document = replace_once(document, '<a href="overview.html">Original 20-case overview</a>',
                            '<a href="figures/all_steps/comparison_20_cases.pdf">Original 20-case overview PDF</a>')
    verified = StudyParser()
    verified.feed(document)
    if json.loads("".join(verified.parts)) != study:
        raise ValueError("Offline conversion changed scientific results")
    destination.mkdir(parents=True)
    assets = destination / "offline_assets"
    assets.mkdir()
    (destination / "index.html").write_text(document)
    copied = []

    def copy(source, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source.read_bytes() != target.read_bytes():
            raise ValueError(f"Offline copy differs: {relative}")
        copied.append(dict(path=str(relative), bytes=target.stat().st_size))

    copy(ROOT / "scripts/templates/ispy2_tumor_offline.js", Path("offline_assets/loader.js"))
    files = [Path("report.md"), Path("tumor_review/lucide.min.js"),
             Path("tumor_review/selection.json"), Path("tumor_review/truth_census.csv"),
             Path("tumor_review/focused_change_metrics.csv"),
             Path("comparison/image_summary.csv"), Path("comparison/pcr_summary.csv"),
             Path("figures/all_steps/comparison_20_cases.pdf"), Path("figures/all_steps/contact_sheet.jpg")]
    files.extend(p.relative_to(SOURCE) for p in sorted((SOURCE / "comparison").glob("*step_comparison.*")))
    files.extend(p.relative_to(SOURCE) for p in sorted(review.glob("*_20_cases.pdf")))
    for relative in files:
        copy(SOURCE / relative, relative)
    copy(ROOT / "docs/ispy2_tumor_review_20260912.md", Path("TUMOR_REVIEW_PROTOCOL.md"))

    inventories = []
    for case in study["cases"]:
        patient = case["patient_id"]
        if not re.fullmatch(r"ISPY2-\d+", patient):
            raise ValueError("Invalid case identifier")
        directory = review / "cases" / patient
        names = {f"real_T{stage}_{channel}.png" for stage in range(4) for channel in ("dce0", "ser", "mask")}
        for row in case["metrics"]:
            stem = f"{row['model']}_e{row['steps']}_{row['strategy']}_T{row['target_stage']}"
            names.update(f"{stem}_{kind}.png" for kind in ("input", "output"))
        if len(names) != 156:
            raise ValueError("Incomplete input/output/reference inventory")
        # Identical PNG bytes share an entry; no pixel conversion or resampling.
        unique, lookup, mapping = [], {}, {}
        for name in sorted(names):
            raw = (directory / name).read_bytes()
            if raw not in lookup:
                lookup[raw] = len(unique)
                unique.append("data:image/png;base64," + base64.b64encode(raw).decode("ascii"))
            mapping[name] = lookup[raw]
            if base64.b64decode(unique[mapping[name]].partition(",")[2], validate=True) != raw:
                raise ValueError("Embedded PNG bytes differ from the original")
        payload = json.dumps([patient, unique, mapping], separators=(",", ":"), allow_nan=False)
        path = assets / f"{patient}.js"
        path.write_text("window.ispy2Offline.register(..." + payload + ");\n")
        inventories.append(dict(images=len(names), unique_images=len(unique), script_bytes=path.stat().st_size))
        print(f"Offline cases: {len(inventories)}/20, {len(names)} images / {len(unique)} distinct PNGs", flush=True)

    (destination / "OPEN_LOCALLY.txt").write_text(
        "I-SPY2 BiFlow / SymmFlow offline tumor comparison\n\n"
        "Extract the entire ZIP on your computer, then double-click index.html.\n"
        "Use Chrome, Edge or Firefox. Keep the extracted folders together.\n"
        "No Python, server, port forwarding or Internet connection is required.\n"
        "Open the extracted index.html, not the preview inside the ZIP.\n\n"
        "Contains all 20 tumor-change cases, Euler 2/10/20/50, all three source\n"
        "policies, T1/T2/T3, actual inputs, lesion slices, real SER, full-cohort\n"
        "generation/pCR tables and local PDFs. PNG export works offline.\n"
        "The older random-case overview is included as its combined 20-page PDF.\n"
        "The generator outputs and numerical results are unchanged. Protocol\n"
        "documents describe the original server computation paths.\n"
    )
    manifest = dict(format="ispy2_tumor_review_offline_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    entrypoint="index.html", requires_server=False, cases=20, steps=[2, 10, 20, 50],
                    image_references=sum(i["images"] for i in inventories),
                    distinct_pngs=sum(i["unique_images"] for i in inventories),
                    embedded_png_bytes_equal_source=True, study_data_unchanged=True,
                    original_overview="figures/all_steps/comparison_20_cases.pdf", copied_files=copied)
    (destination / "offline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(destination.name) / path.relative_to(destination))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Offline ZIP integrity check failed")
    print(json.dumps(dict(directory=str(destination), archive=str(archive),
                          archive_bytes=archive.stat().st_size, image_references=manifest["image_references"],
                          distinct_pngs=manifest["distinct_pngs"])), flush=True)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", type=Path, default=DESTINATION)
    build(arguments.parse_args().output)

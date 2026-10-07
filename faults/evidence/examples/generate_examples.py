"""Generate a local gallery for every evidence-degradation task family."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from faults.evidence.modalities import (
    _event_selected,
    apply_ct_evidence,
    apply_ehr_data_quality_evidence,
    apply_ehr_event_evidence,
    apply_xray_evidence,
)
from faults.evidence.prepare_clinical_recipes import DEFAULT_TOPICS, load_topics
from faults.evidence.recipes import (
    EVIDENCE_SCENARIOS,
    apply_evidence_recipe,
    load_evidence_recipe,
)

# Keep scenario order consistent in every visual and tabular example.
SCENARIOS = (
    "control",
    "evidence-explicit-missing",
    "evidence-natural-missing",
    "evidence-low-quality",
)

# Resolve repository paths independently of the caller's current directory.
SCRIPT_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_ROOT.parents[2]
OUTPUT_ROOT = SCRIPT_ROOT / "generated"


# Recreate the output directory so obsolete example artifacts cannot linger.
def _reset_output() -> None:
    if OUTPUT_ROOT.is_dir():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)


# Write CSV rows with a stable field order for readable diffs and examples.
def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Write deterministic gzip CSV fixtures used by the production table adapters.
def _write_gzip_csv(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with __import__("io").TextIOWrapper(
                compressed,
                newline="",
                encoding="utf-8",
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


# Read a gzip CSV into dictionaries without retaining an open file handle.
def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# Draw one four-column comparison with a compact title above each panel.
def _save_panel_grid(
    images: list[Image.Image],
    labels: list[str],
    path: Path,
    *,
    footer: str,
) -> None:
    normalized = [image.convert("RGB") for image in images]
    width = max(image.width for image in normalized)
    height = max(image.height for image in normalized)
    header_height = 42
    footer_height = 36
    canvas = Image.new(
        "RGB",
        (width * len(normalized), height + header_height + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(normalized, labels, strict=True)):
        resized = image.resize((width, height))
        left = index * width
        canvas.paste(resized, (left, header_height))
        draw.text((left + 12, 14), label, fill="black")
        if index:
            draw.line((left, 0, left, height + header_height), fill=(180, 180, 180))
    draw.text((12, height + header_height + 12), footer, fill=(60, 60, 60))
    canvas.save(path)


# Convert one CT slice to a lung-window RGB image and expose NaNs in magenta.
def _ct_slice_image(volume_path: Path) -> Image.Image:
    image = nib.load(str(volume_path))
    z_index = image.shape[2] // 2
    array = np.asanyarray(image.dataobj[:, :, z_index]).astype(np.float32)
    missing = ~np.isfinite(array)
    clipped = np.clip(np.nan_to_num(array, nan=-1000.0), -1000.0, 500.0)
    pixels = np.round((clipped + 1000.0) / 1500.0 * 255.0).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    rgb[missing] = np.array([255, 0, 255], dtype=np.uint8)
    return Image.fromarray(np.rot90(rgb), mode="RGB").resize((420, 420))


# Derive a compact local volume from a real cached CT without publishing the NIfTI.
def _stage_ct_fixture(path: Path) -> str:
    candidates = sorted(
        (BENCHMARK_ROOT / "assets/ct_abnormality/assets/raw_cache").glob(
            "valid_*.nii.gz"
        )
    )
    if not candidates:
        raise RuntimeError("No cached CT volume is available for the example gallery")
    source = nib.load(str(candidates[0]))
    sampled = np.asanyarray(source.dataobj[::4, ::4, ::2]).astype(np.float32)
    nib.save(nib.Nifti1Image(sampled, np.eye(4)), str(path))
    return candidates[0].name


# Apply the production CT adapter to four matched copies and render the selected slab.
def _generate_ct_example(work_root: Path) -> dict[str, object]:
    family_root = OUTPUT_ROOT / "ct_abnormality"
    family_root.mkdir()
    pristine = work_root / "ct-pristine.nii.gz"
    source_name = _stage_ct_fixture(pristine)
    images: list[Image.Image] = []
    details: dict[str, object] = {}
    for scenario in SCENARIOS:
        target = work_root / f"ct-{scenario}.nii.gz"
        shutil.copyfile(pristine, target)
        if scenario == "control":
            application = apply_ct_evidence(
                target,
                condition="control",
                scenario="evidence-explicit-missing",
            )
        else:
            application = apply_ct_evidence(
                target,
                condition="error",
                scenario=scenario,
            )
        images.append(_ct_slice_image(target))
        details[scenario] = application.details
    figure = family_root / "ct_comparison.png"
    _save_panel_grid(
        images,
        ["Control", "Explicit missing", "Natural missing", "Low quality"],
        figure,
        footer="Real cached CT, downsampled locally; NaN voxels are visualized in magenta.",
    )
    return {
        "source_kind": "real-local-ct-derived-fixture",
        "source": source_name,
        "figure": str(figure.relative_to(OUTPUT_ROOT)),
        "details": details,
    }


# Create a privacy-safe radiograph-like fixture when gated MIMIC images are absent.
def _synthetic_xray(path: Path) -> None:
    size = 640
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    chest = np.exp(-((x / 0.82) ** 8 + (y / 0.95) ** 8))
    left_lung = np.exp(-(((x + 0.34) / 0.27) ** 2 + ((y + 0.02) / 0.62) ** 2))
    right_lung = np.exp(-(((x - 0.34) / 0.27) ** 2 + ((y + 0.02) / 0.62) ** 2))
    spine = np.exp(-((x / 0.055) ** 2 + ((y - 0.05) / 0.75) ** 2))
    heart = np.exp(-(((x + 0.05) / 0.30) ** 2 + ((y - 0.30) / 0.30) ** 2))
    pixels = 25 + 155 * chest - 85 * (left_lung + right_lung) + 55 * spine + 35 * heart
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for offset in range(80, 540, 42):
        draw.arc((95, offset - 75, 545, offset + 115), 205, 335, fill=(155, 155, 155), width=3)
    image.save(path, format="JPEG", quality=94)


# Apply the production X-ray adapter to a staged latest-study directory.
def _generate_xray_example(work_root: Path) -> dict[str, object]:
    family_root = OUTPUT_ROOT / "xray_report_correction"
    family_root.mkdir()
    pristine = work_root / "xray-pristine.jpg"
    _synthetic_xray(pristine)
    images: list[Image.Image] = []
    details: dict[str, object] = {}
    for scenario in SCENARIOS:
        patient_root = work_root / f"xray-{scenario}" / "patient"
        study_root = patient_root / "study_1"
        study_root.mkdir(parents=True)
        target = study_root / "view_ap.jpg"
        shutil.copyfile(pristine, target)
        if scenario == "control":
            application = apply_xray_evidence(
                patient_root,
                condition="control",
                scenario="evidence-explicit-missing",
            )
        else:
            application = apply_xray_evidence(
                patient_root,
                condition="error",
                scenario=scenario,
            )
        with Image.open(target) as image:
            images.append(image.convert("RGB").resize((420, 420)))
        details[scenario] = application.details
    figure = family_root / "xray_comparison.png"
    _save_panel_grid(
        images,
        ["Control", "Explicit missing", "Natural missing", "Low quality"],
        figure,
        footer="Synthetic radiograph fixture; transformations use the production X-ray adapter.",
    )
    return {
        "source_kind": "synthetic-privacy-safe-fixture",
        "figure": str(figure.relative_to(OUTPUT_ROOT)),
        "details": details,
    }


# Create a deterministic histology-like mosaic with four analysis-tile regions.
def _synthetic_pathology() -> Image.Image:
    rng = np.random.default_rng(20261001)
    size = 768
    base = np.full((size, size, 3), [244, 190, 208], dtype=np.uint8)
    texture = rng.normal(0, 10, base.shape[:2])
    base = np.clip(base.astype(np.int16) + texture[:, :, None].astype(np.int16), 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(2200):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        radius = int(rng.integers(2, 7))
        color = (int(rng.integers(65, 115)), 35, int(rng.integers(90, 145)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.line((size // 2, 0, size // 2, size), fill=(255, 255, 255), width=2)
    draw.line((0, size // 2, size, size // 2), fill=(255, 255, 255), width=2)
    return image


# Mirror the OpenSlide delivery transform on one scale-reduced selected tile.
def _pathology_variant(source: Image.Image, scenario: str) -> Image.Image:
    image = source.copy()
    box = (image.width // 2, 0, image.width, image.height // 2)
    if scenario == "evidence-explicit-missing":
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, fill=(255, 0, 255))
        draw.line((box[0], box[1], box[2], box[3]), fill="black", width=6)
        draw.line((box[0], box[3], box[2], box[1]), fill="black", width=6)
    elif scenario == "evidence-natural-missing":
        ImageDraw.Draw(image).rectangle(box, fill="white")
    elif scenario == "evidence-low-quality":
        tile = image.crop(box)
        small = tile.resize((max(1, tile.width // 8), max(1, tile.height // 8)))
        image.paste(small.resize(tile.size), box)
    return image


# Render pathology delivery behavior without reading multi-gigabyte gated slides.
def _generate_pathology_example() -> dict[str, object]:
    family_root = OUTPUT_ROOT / "tumor_area_selection_pathology"
    family_root.mkdir()
    control = _synthetic_pathology()
    images = [control] + [_pathology_variant(control, scenario) for scenario in SCENARIOS[1:]]
    figure = family_root / "pathology_comparison.png"
    _save_panel_grid(
        [image.resize((420, 420)) for image in images],
        ["Control", "Explicit missing", "Natural missing", "Low quality"],
        figure,
        footer="Synthetic 2x2 tile mosaic; the upper-right tile is the selected analysis tile.",
    )
    return {
        "source_kind": "synthetic-scale-reduced-tile-fixture",
        "figure": str(figure.relative_to(OUTPUT_ROOT)),
        "details": {
            "adapter": "pathology-openslide-analysis-tile-delivery-v1",
            "selected_tile": "upper-right for visualization",
        },
    }


# Save all four clinical text renderings from one real benchmark topic and recipe.
def _generate_clinical_example() -> dict[str, object]:
    family_root = OUTPUT_ROOT / "clinical_trial_matching"
    family_root.mkdir()
    task = "clinical_trial_matching_task_29"
    source = load_topics(DEFAULT_TOPICS)[29].encode("utf-8")
    recipe_path = BENCHMARK_ROOT / "faults/evidence/recipes" / f"{task}.json"
    recipe = load_evidence_recipe(recipe_path, require_approved=False)
    (family_root / "control.txt").write_bytes(source)
    details: dict[str, object] = {"control": "byte-identical"}
    for scenario in EVIDENCE_SCENARIOS:
        output, metadata = apply_evidence_recipe(
            source,
            recipe,
            scenario=scenario,
            condition="error",
        )
        (family_root / f"{scenario}.txt").write_bytes(output)
        details[scenario] = metadata
    return {
        "source_kind": "real-benchmark-topic",
        "task": task,
        "files": [str(path.relative_to(OUTPUT_ROOT)) for path in sorted(family_root.glob("*.txt"))],
        "details": details,
    }


# Stage three labelled rows so every data-quality cluster is selected deterministically.
def _quality_fixture(root: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    rows = [
        {"_row_id": "r1", "subject_id": "p1", "valuenum": "675.3", "valueuom": "mg/dL"},
        {"_row_id": "r2", "subject_id": "p2", "valuenum": "70.5", "valueuom": "mg/dL"},
        {"_row_id": "r3", "subject_id": "p3", "valuenum": "629.8", "valueuom": "mg/dL"},
        {"_row_id": "r4", "subject_id": "p4", "valuenum": "4.2", "valueuom": "mg/dL"},
    ]
    csv_root = root / "csv"
    table = csv_root / "labevents.csv.gz"
    fields = ["_row_id", "subject_id", "valuenum", "valueuom"]
    _write_gzip_csv(table, rows, fields)
    labels = root / "labels.csv"
    label_rows = [
        {
            "table": "labevents",
            "row_id": row["_row_id"],
            "field": "valuenum",
            "cluster_id": f"cluster-{index}",
        }
        for index, row in enumerate(rows[:3], start=1)
    ]
    _write_csv(labels, label_rows, ["table", "row_id", "field", "cluster_id"])
    return csv_root, labels, rows


# Apply the production data-quality adapter and export plain CSV comparisons.
def _generate_ehr_quality_example(work_root: Path) -> dict[str, object]:
    family_root = OUTPUT_ROOT / "ehr_data_quality"
    family_root.mkdir()
    details: dict[str, object] = {}
    files: list[str] = []
    for scenario in SCENARIOS:
        fixture_root = work_root / f"quality-{scenario}"
        csv_root, labels, _ = _quality_fixture(fixture_root)
        if scenario == "control":
            application = apply_ehr_data_quality_evidence(
                csv_root,
                labels,
                task_name="example-quality-task",
                condition="control",
                scenario="evidence-explicit-missing",
            )
        else:
            application = apply_ehr_data_quality_evidence(
                csv_root,
                labels,
                task_name="example-quality-task",
                condition="error",
                scenario=scenario,
            )
        rows = _read_gzip_csv(csv_root / "labevents.csv.gz")
        output = family_root / f"{scenario}.csv"
        fields = list(rows[0]) if rows else ["_row_id", "subject_id", "valuenum", "valueuom"]
        _write_csv(output, rows, fields)
        files.append(str(output.relative_to(OUTPUT_ROOT)))
        details[scenario] = application.details
    return {
        "source_kind": "synthetic-privacy-safe-table-fixture",
        "files": files,
        "details": details,
    }


# Create enough deterministic events to expose the production ten-percent sample.
def _event_fixture(root: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    fields = ["patient_id", "start", "code", "value", "unit", "visit_id"]
    rows = [
        {
            "patient_id": "patient-demo",
            "start": f"2026-01-{index % 28 + 1:02d}T08:00:00",
            "code": f"LAB/{index:03d}",
            "value": f"{12.345 + index * 0.137:.3f}",
            "unit": "mg/dL",
            "visit_id": f"visit-{index // 20}",
        }
        for index in range(120)
    ]
    events = root / "events.csv"
    examples = root / "test_examples.csv"
    root.mkdir(parents=True)
    _write_csv(events, rows, fields)
    _write_csv(examples, [{"patient_id": "patient-demo"}], ["patient_id"])
    return events, examples, rows


# Apply the production event adapter and retain a compact selected-row excerpt.
def _generate_ehr_event_example(work_root: Path) -> dict[str, object]:
    family_root = OUTPUT_ROOT / "ehr_event_modelling"
    family_root.mkdir()
    task_name = "example-event-task"
    details: dict[str, object] = {}
    files: list[str] = []
    selected_originals: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        fixture_root = work_root / f"event-{scenario}"
        events, examples, original_rows = _event_fixture(fixture_root)
        selected_originals = [
            row for row in original_rows if _event_selected(task_name, row, {"patient-demo"})
        ]
        if scenario == "control":
            application = apply_ehr_event_evidence(
                events,
                examples,
                task_name=task_name,
                condition="control",
                scenario="evidence-explicit-missing",
            )
            excerpt = selected_originals[:4]
        else:
            application = apply_ehr_event_evidence(
                events,
                examples,
                task_name=task_name,
                condition="error",
                scenario=scenario,
            )
            with events.open(newline="", encoding="utf-8") as stream:
                transformed = list(csv.DictReader(stream))
            # Match the same records without relying on code, which explicit missing clears.
            selected_keys = {
                (row["patient_id"], row["start"], row["visit_id"])
                for row in selected_originals[:4]
            }
            excerpt = [
                row
                for row in transformed
                if (row["patient_id"], row["start"], row["visit_id"])
                in selected_keys
            ]
        output = family_root / f"{scenario}.csv"
        fields = list(excerpt[0]) if excerpt else ["note"]
        if excerpt:
            _write_csv(output, excerpt, fields)
        else:
            _write_csv(output, [{"note": "selected records were deleted"}], fields)
        files.append(str(output.relative_to(OUTPUT_ROOT)))
        details[scenario] = application.details
    return {
        "source_kind": "synthetic-privacy-safe-event-fixture",
        "files": files,
        "selected_example_count": min(4, len(selected_originals)),
        "details": details,
    }


# Materialize the embedded EHR-to-MEDS admission-record contract as small CSVs.
def _generate_ehr_to_meds_example() -> dict[str, object]:
    family_root = OUTPUT_ROOT / "ehr_to_meds"
    family_root.mkdir()
    fields = [
        "subject_id",
        "hadm_id",
        "insurance",
        "language",
        "marital_status",
        "race",
    ]
    original = [
        {
            "subject_id": "1001",
            "hadm_id": "5001",
            "insurance": "Medicare",
            "language": "English",
            "marital_status": "Married",
            "race": "Asian",
        },
        {
            "subject_id": "1002",
            "hadm_id": "5002",
            "insurance": "Private",
            "language": "Spanish",
            "marital_status": "Single",
            "race": "White",
        },
    ]
    files: list[str] = []
    for scenario in SCENARIOS:
        rows = [dict(row) for row in original]
        output_fields = list(fields)
        if scenario == "evidence-explicit-missing":
            output_fields.append("_evidence_status")
            for field in ("insurance", "language", "marital_status", "race"):
                rows[0][field] = "[VALUE REDACTED]"
            rows[0]["_evidence_status"] = "ADMISSION_DEMOGRAPHICS:MISSING"
            rows[1]["_evidence_status"] = ""
        elif scenario == "evidence-natural-missing":
            rows = rows[1:]
        elif scenario == "evidence-low-quality":
            output_fields.append("_evidence_status")
            for field in ("insurance", "language", "marital_status", "race"):
                rows[0][field] = "UNKNOWN"
            rows[0]["_evidence_status"] = "ADMISSION_DEMOGRAPHICS:UNVERIFIED_SOURCE"
            rows[1]["_evidence_status"] = ""
        output = family_root / f"{scenario}.csv"
        _write_csv(output, rows, output_fields)
        files.append(str(output.relative_to(OUTPUT_ROOT)))
    return {
        "source_kind": "synthetic-privacy-safe-admission-fixture",
        "files": files,
        "details": {
            "adapter": "ehr-to-meds-admission-record-v1",
            "selected_record_count": 1,
        },
    }


# Build one Markdown index that links every generated artifact and states provenance.
def _write_readme(manifest: dict[str, object]) -> None:
    content = """# Evidence degradation example gallery

This directory contains deterministic examples for all seven HealthAgentBench
task families. `Control` preserves the evidence, `explicit` exposes a visible
missingness marker, `natural` removes evidence without an error label, and
`low-quality` retains degraded or unverified evidence.

## Image families

### CT abnormality

![CT evidence comparison](ct_abnormality/ct_comparison.png)

The figure uses a locally cached real CT that is downsampled before applying the
production CT adapter. Magenta in the explicit panel visualizes NaN voxels.

### X-ray report correction

![X-ray evidence comparison](xray_report_correction/xray_comparison.png)

MIMIC-CXR pixels are not cached locally, so the input is a privacy-safe synthetic
radiograph. All three error panels are produced by the production X-ray adapter.

### Pathology tumor-area selection

![Pathology evidence comparison](tumor_area_selection_pathology/pathology_comparison.png)

The host environment lacks OpenSlide, so this is a scale-reduced synthetic 2x2
tile mosaic. The selected upper-right tile receives the same crosshatch, white
suppression, or 8x pixelation used by the runtime delivery shim.

## Text and table families

- `clinical_trial_matching/`: real topic 29 rendered by the curated recipe.
- `ehr_data_quality/`: selected field blanking, record deletion, and source status.
- `ehr_event_modelling/`: selected event blanking, deletion, and numeric degradation.
- `ehr_to_meds/`: admission demographics redaction, record deletion, and `UNKNOWN` values.

Every clinical or table directory contains one file per condition. The complete
machine-readable provenance and transform metadata is in `manifest.json`.

## Reproduce

From the `HealthAgentBench` checkout:

```bash
.venv/bin/python -m faults.evidence.examples.generate_examples
```
"""
    (OUTPUT_ROOT / "README.md").write_text(content, encoding="utf-8")
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# Generate every family in one temporary workspace and retain only review artifacts.
def main() -> int:
    _reset_output()
    with tempfile.TemporaryDirectory(prefix="healthagent-evidence-examples-") as temporary:
        work_root = Path(temporary)
        manifest = {
            "clinical_trial_matching": _generate_clinical_example(),
            "ct_abnormality": _generate_ct_example(work_root),
            "xray_report_correction": _generate_xray_example(work_root),
            "tumor_area_selection_pathology": _generate_pathology_example(),
            "ehr_data_quality": _generate_ehr_quality_example(work_root),
            "ehr_event_modelling": _generate_ehr_event_example(work_root),
            "ehr_to_meds": _generate_ehr_to_meds_example(),
        }
    _write_readme(manifest)
    print(f"[evidence-examples] wrote {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

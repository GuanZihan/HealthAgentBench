"""Deterministic evidence-degradation adapters for non-text task families."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .recipes import EVIDENCE_SCENARIOS

EvidenceCondition = Literal["control", "error"]


class ModalityEvidenceError(RuntimeError):
    """Raised when a modality-specific degradation cannot be established."""


@dataclass(frozen=True)
class EvidenceApplication:
    """Describe a deterministic transform and any Agent-only environment."""

    details: dict[str, object]
    agent_environment: dict[str, str]


# Reject unsupported condition/scenario values before touching trial data.
def _validate(condition: str, scenario: str) -> None:
    if condition not in {"control", "error"}:
        raise ModalityEvidenceError(f"Unsupported evidence condition: {condition}")
    if scenario not in EVIDENCE_SCENARIOS:
        raise ModalityEvidenceError(f"Unsupported evidence scenario: {scenario}")


# Hash only non-sensitive selection metadata for experiment manifests.
def _selection_digest(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Replace a file atomically while retaining its original permission bits.
def _replace_file(source: Path, temporary: Path) -> None:
    mode = source.stat().st_mode & 0o7777
    temporary.chmod(mode)
    temporary.replace(source)


# Apply one central-slab degradation while keeping the NIfTI readable.
def apply_ct_evidence(
    target: Path,
    *,
    condition: EvidenceCondition,
    scenario: str,
) -> EvidenceApplication:
    _validate(condition, scenario)

    # Import the image stack lazily so table-only families need no image packages.
    import nibabel as nib
    import numpy as np

    image = nib.load(str(target))
    shape = tuple(int(value) for value in image.shape)
    if len(shape) < 3 or min(shape[:3]) < 8:
        raise ModalityEvidenceError(f"CT volume has unsupported shape: {shape}")
    slab_width = max(1, int(round(shape[2] * 0.20)))
    slab_start = (shape[2] - slab_width) // 2
    slab_end = slab_start + slab_width
    details: dict[str, object] = {
        "adapter": "ct-nifti-central-axial-slab-v1",
        "source_shape": list(shape),
        "selected_axis": 2,
        "selected_start": slab_start,
        "selected_end": slab_end,
        "selected_fraction": slab_width / shape[2],
        "control_byte_identical": True,
    }
    if condition == "control":
        return EvidenceApplication(details=details, agent_environment={})

    # Materialize one float32 copy so NaN and intensity transforms are explicit.
    data = np.asanyarray(image.dataobj).astype(np.float32)
    selected = data[:, :, slab_start:slab_end]
    if scenario == "evidence-explicit-missing":
        selected[...] = np.nan
        details["transform"] = "replace-selected-voxels-with-nan"
    elif scenario == "evidence-natural-missing":
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise ModalityEvidenceError("CT volume contains no finite voxels")
        background = float(np.percentile(finite, 1.0))
        selected[...] = background
        details["transform"] = "silently-suppress-selected-slab-to-background"
        details["background_percentile"] = 1.0
    else:
        coarse = selected[::4, ::4, ::2]
        pixelated = np.repeat(
            np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1), 2, axis=2
        )
        selected[...] = pixelated[
            : selected.shape[0], : selected.shape[1], : selected.shape[2]
        ]
        selected[...] = np.round(selected / 100.0) * 100.0
        details["transform"] = "pixelate-4x4x2-and-quantize-100-hu"

    # Preserve geometry and metadata while changing only the selected voxel evidence.
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(data, image.affine, header=header)
    temporary = target.with_name(f".{target.name}.evidence.nii.gz")
    nib.save(output, str(temporary))
    _replace_file(target, temporary)
    return EvidenceApplication(details=details, agent_environment={})


# Resolve only images belonging to the chronologically latest X-ray study.
def _target_xray_images(patient_root: Path) -> list[Path]:
    studies = sorted(path for path in patient_root.glob("study_*") if path.is_dir())
    if not studies:
        raise ModalityEvidenceError(f"No X-ray study directories found: {patient_root}")
    images = sorted(studies[-1].glob("view_*.jpg"))
    if not images:
        raise ModalityEvidenceError(
            f"Target X-ray study contains no JPEG views: {studies[-1]}"
        )
    return images


# Apply a matched central thoracic ROI transform to every target-study view.
def apply_xray_evidence(
    patient_root: Path,
    *,
    condition: EvidenceCondition,
    scenario: str,
) -> EvidenceApplication:
    _validate(condition, scenario)

    # Import Pillow lazily because only this family needs raster editing.
    from PIL import Image, ImageDraw, ImageFilter, ImageStat

    images = _target_xray_images(patient_root)
    selections: list[str] = []
    view_details: list[dict[str, object]] = []
    for image_path in images:
        with Image.open(image_path) as source:
            width, height = source.size
            left = int(round(width * 0.20))
            upper = int(round(height * 0.20))
            right = int(round(width * 0.80))
            lower = int(round(height * 0.85))
            box = (left, upper, right, lower)
            selections.append(f"{image_path.name}:{left}:{upper}:{right}:{lower}")
            view_details.append(
                {"view": image_path.name, "size": [width, height], "roi": list(box)}
            )
            if condition == "control":
                continue

            # Work in RGB so all three conditions produce portable JPEG files.
            image = source.convert("RGB")
            roi = image.crop(box)
            if scenario == "evidence-explicit-missing":
                draw = ImageDraw.Draw(image)
                draw.rectangle(
                    box,
                    fill=(127, 127, 127),
                    outline=(255, 0, 255),
                    width=max(4, width // 256),
                )
                draw.line(
                    (left, upper, right, lower),
                    fill=(255, 0, 255),
                    width=max(4, width // 256),
                )
                draw.line(
                    (left, lower, right, upper),
                    fill=(255, 0, 255),
                    width=max(4, width // 256),
                )
                draw.text(
                    (left + 12, upper + 12), "EVIDENCE REDACTED", fill=(255, 255, 255)
                )
            elif scenario == "evidence-natural-missing":
                median = int(ImageStat.Stat(roi.convert("L")).median[0])
                image.paste((median, median, median), box)
            else:
                blurred = roi.filter(
                    ImageFilter.GaussianBlur(radius=max(4.0, width / 160.0))
                )
                small_size = (max(1, roi.width // 8), max(1, roi.height // 8))
                blurred = blurred.resize(small_size).resize(roi.size)
                image.paste(blurred, box)

            # Replace the private staged JPEG without touching the shared cache.
            temporary = image_path.with_name(f".{image_path.name}.evidence.jpg")
            image.save(temporary, format="JPEG", quality=88, optimize=False)
            _replace_file(image_path, temporary)

    details = {
        "adapter": "xray-target-study-thoracic-roi-v1",
        "target_view_count": len(images),
        "selection_sha256": _selection_digest(selections),
        "views": view_details,
        "transform": {
            "evidence-explicit-missing": "labelled-gray-redaction-roi",
            "evidence-natural-missing": "unlabelled-median-suppression-roi",
            "evidence-low-quality": "blur-and-8x-pixelation-roi",
        }[scenario],
        "control_byte_identical": True,
    }
    return EvidenceApplication(details=details, agent_environment={})


# Read the verifier-only labels to choose task-relevant data-quality evidence.
def _quality_targets(labels_path: Path, task_name: str) -> dict[str, dict[str, str]]:
    with labels_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ModalityEvidenceError(f"Data-quality labels are empty: {labels_path}")
    clusters = sorted(
        {str(row["cluster_id"]) for row in rows},
        key=lambda value: hashlib.sha256(f"{task_name}|{value}".encode()).hexdigest(),
    )[:3]
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row["cluster_id"]) not in clusters:
            continue
        key = f"{row['table']}|{row['row_id']}"
        selected[key] = {"table": str(row["table"]), "field": str(row["field"])}
    if not selected:
        raise ModalityEvidenceError(
            "No task-relevant data-quality records were selected"
        )
    return selected


# Rewrite one gzip CSV while preserving its schema and private row identifiers.
def _rewrite_quality_table(
    path: Path,
    *,
    selected: dict[str, dict[str, str]],
    condition: EvidenceCondition,
    scenario: str,
) -> int:
    temporary = path.with_name(f".{path.name}.evidence.gz")
    changed = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "_row_id" not in reader.fieldnames:
            raise ModalityEvidenceError(f"Table lacks _row_id: {path}")
        fieldnames = list(reader.fieldnames)
        if (
            scenario != "evidence-natural-missing"
            and "_evidence_status" not in fieldnames
        ):
            fieldnames.append("_evidence_status")
        # Fix gzip mtime and filename so replicated error inputs hash identically.
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                mtime=0,
            ) as compressed_output:
                with io.TextIOWrapper(
                    compressed_output,
                    newline="",
                    encoding="utf-8",
                ) as output:
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in reader:
                        key = f"{path.name.removesuffix('.csv.gz')}|{row['_row_id']}"
                        selection = selected.get(key)
                        if selection is None:
                            if "_evidence_status" in fieldnames:
                                row["_evidence_status"] = ""
                            writer.writerow(row)
                            continue
                        changed += 1
                        if condition == "control":
                            writer.writerow(row)
                        elif scenario == "evidence-natural-missing":
                            continue
                        elif scenario == "evidence-explicit-missing":
                            field = selection["field"]
                            if field in row:
                                row[field] = ""
                            row["_evidence_status"] = f"{field}:MISSING"
                            writer.writerow(row)
                        else:
                            row["_evidence_status"] = (
                                f"{selection['field']}:UNVERIFIED_SOURCE"
                            )
                            writer.writerow(row)
    if condition == "error":
        _replace_file(path, temporary)
    else:
        temporary.unlink()
    return changed


# Degrade three deterministic verifier-relevant clusters at record/field level.
def apply_ehr_data_quality_evidence(
    csv_root: Path,
    labels_path: Path,
    *,
    task_name: str,
    condition: EvidenceCondition,
    scenario: str,
) -> EvidenceApplication:
    _validate(condition, scenario)
    selected = _quality_targets(labels_path, task_name)
    selected_tables = sorted({value["table"] for value in selected.values()})
    changed = 0
    for table in selected_tables:
        path = csv_root / f"{table}.csv.gz"
        if not path.is_file():
            raise ModalityEvidenceError(f"Selected EHR table is missing: {path}")
        changed += _rewrite_quality_table(
            path,
            selected=selected,
            condition=condition,
            scenario=scenario,
        )
    if changed != len(selected):
        raise ModalityEvidenceError(
            f"Selected {len(selected)} EHR rows but matched {changed} staged rows"
        )
    details = {
        "adapter": "ehr-data-quality-verifier-cluster-v1",
        "selection_unit": "record",
        "selected_record_count": len(selected),
        "selected_cluster_count": 3,
        "selected_tables": selected_tables,
        "selection_sha256": _selection_digest(list(selected)),
        "transform": {
            "evidence-explicit-missing": "blank-field-plus-missing-status",
            "evidence-natural-missing": "delete-record",
            "evidence-low-quality": "retain-value-plus-unverified-source-status",
        }[scenario],
        "control_byte_identical": True,
    }
    return EvidenceApplication(details=details, agent_environment={})


# Select a reproducible ten-percent subset without storing clinical row content.
def _event_selected(
    task_name: str, row: dict[str, str], test_patients: set[str]
) -> bool:
    patient_id = str(row.get("patient_id", ""))
    if patient_id not in test_patients:
        return False
    key = "|".join(
        (
            task_name,
            patient_id,
            str(row.get("start", "")),
            str(row.get("code", "")),
            str(row.get("visit_id", "")),
        )
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:2], "big") < 6554


# Reduce numeric precision without fabricating a more specific measurement.
def _round_numeric(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError:
        return value
    if not math.isfinite(numeric) or numeric == 0:
        return value
    digits = 1 - int(math.floor(math.log10(abs(numeric))))
    return f"{round(numeric, digits):g}"


# Stream the large event log so setup memory is bounded on Rivanna nodes.
def apply_ehr_event_evidence(
    events_path: Path,
    test_examples_path: Path,
    *,
    task_name: str,
    condition: EvidenceCondition,
    scenario: str,
) -> EvidenceApplication:
    _validate(condition, scenario)
    with test_examples_path.open(newline="", encoding="utf-8") as stream:
        test_patients = {str(row["patient_id"]) for row in csv.DictReader(stream)}
    if not test_patients:
        raise ModalityEvidenceError(f"No test patients found: {test_examples_path}")

    temporary = events_path.with_name(f".{events_path.name}.evidence")
    selected_count = 0
    with events_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ModalityEvidenceError(f"Event log has no header: {events_path}")
        # Scan controls without writing a disposable multi-gigabyte copy.
        if condition == "control":
            selected_count = sum(
                _event_selected(task_name, row, test_patients) for row in reader
            )
        else:
            fieldnames = list(reader.fieldnames)
            if (
                scenario != "evidence-natural-missing"
                and "_evidence_status" not in fieldnames
            ):
                fieldnames.append("_evidence_status")
            with temporary.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    selected = _event_selected(task_name, row, test_patients)
                    if selected:
                        selected_count += 1
                    if selected:
                        if scenario == "evidence-natural-missing":
                            continue
                        if scenario == "evidence-explicit-missing":
                            for field in ("code", "value", "unit"):
                                if field in row:
                                    row[field] = ""
                            row["_evidence_status"] = "CLINICAL_EVENT:MISSING"
                        else:
                            if "value" in row:
                                row["value"] = _round_numeric(row["value"])
                            if "unit" in row:
                                row["unit"] = ""
                            row["_evidence_status"] = "CLINICAL_EVENT:UNVERIFIED_SOURCE"
                    elif "_evidence_status" in fieldnames:
                        row["_evidence_status"] = ""
                    writer.writerow(row)
    if selected_count == 0:
        temporary.unlink(missing_ok=True)
        raise ModalityEvidenceError("Deterministic event selection produced no rows")
    if condition == "error":
        _replace_file(events_path, temporary)
    details = {
        "adapter": "ehr-event-test-history-hash-sample-v1",
        "selection_unit": "record",
        "selection_population": "test-patient-preprediction-events",
        "selection_probability": 0.10,
        "selected_record_count": selected_count,
        "test_patient_count": len(test_patients),
        "selection_key": "sha256(task,patient_id,start,code,visit_id)",
        "transform": {
            "evidence-explicit-missing": "blank-code-value-unit-plus-missing-status",
            "evidence-natural-missing": "delete-record",
            "evidence-low-quality": "round-value-drop-unit-plus-unverified-status",
        }[scenario],
        "control_byte_identical": True,
    }
    return EvidenceApplication(details=details, agent_environment={})


# Install a no-op/control or degrading/error OpenSlide delivery shim.
def prepare_pathology_evidence(
    runtime: Path,
    *,
    task_name: str,
    condition: EvidenceCondition,
    scenario: str,
    shim_source: Path,
) -> EvidenceApplication:
    _validate(condition, scenario)
    shim_root = runtime / "workspace/.healthagent_evidence"
    shim_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(shim_source, shim_root / "sitecustomize.py")
    bin_root = shim_root / "bin"
    bin_root.mkdir()
    write_png = bin_root / "openslide-write-png"
    write_png.write_text(
        """#!/usr/bin/python3
# Route the standard OpenSlide CLI through the matched Python delivery shim.
import sys
import openslide

if len(sys.argv) != 8:
    print("usage: openslide-write-png SLIDE X Y LEVEL WIDTH HEIGHT OUTPUT", file=sys.stderr)
    raise SystemExit(2)
slide = openslide.OpenSlide(sys.argv[1])
region = slide.read_region(
    (int(sys.argv[2]), int(sys.argv[3])),
    int(sys.argv[4]),
    (int(sys.argv[5]), int(sys.argv[6])),
)
region.save(sys.argv[7], format="PNG")
""",
        encoding="utf-8",
    )
    write_png.chmod(0o755)
    environment = {
        "PYTHONPATH": "/workspace/.healthagent_evidence",
        "PATH": (
            "/workspace/.healthagent_evidence/bin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "HEALTHAGENT_EVIDENCE_TASK": task_name,
        "HEALTHAGENT_EVIDENCE_CONDITION": condition,
        "HEALTHAGENT_EVIDENCE_SCENARIO": scenario,
        "HEALTHAGENT_EVIDENCE_EVENTS": "/logs/artifacts/evidence-events.jsonl",
    }
    details = {
        "adapter": "pathology-openslide-analysis-tile-delivery-v1",
        "selection_unit": "4096x4096-level0-analysis-tile",
        "selection_probability": 0.20,
        "selection_key": "deterministic-task-seeded-grid-hash",
        "delivery_paths": [
            "python-openslide-read_region",
            "openslide-write-png",
        ],
        "transform": {
            "evidence-explicit-missing": "magenta-crosshatch-selected-tiles",
            "evidence-natural-missing": "white-background-selected-tiles",
            "evidence-low-quality": "8x-pixelation-selected-tiles",
        }[scenario],
        "control_byte_identical": True,
        "raw_wsi_unchanged": True,
    }
    return EvidenceApplication(details=details, agent_environment=environment)


# Dispatch one family to its modality-specific deterministic adapter.
def apply_modality_evidence(
    *,
    family: str,
    runtime: Path,
    target: Path,
    task_name: str,
    task_dir: Path,
    condition: EvidenceCondition,
    scenario: str,
    shim_source: Path,
) -> EvidenceApplication:
    if family == "ct_abnormality":
        return apply_ct_evidence(target, condition=condition, scenario=scenario)
    if family == "xray_report_correction":
        return apply_xray_evidence(
            runtime / "data/patient",
            condition=condition,
            scenario=scenario,
        )
    if family == "ehr_data_quality":
        return apply_ehr_data_quality_evidence(
            runtime / "workspace/data/csv",
            task_dir / "tests/labels.csv",
            task_name=task_name,
            condition=condition,
            scenario=scenario,
        )
    if family == "ehr_event_modelling":
        return apply_ehr_event_evidence(
            runtime / "workspace/data/events.csv",
            runtime / "workspace/data/test_examples.csv",
            task_name=task_name,
            condition=condition,
            scenario=scenario,
        )
    if family == "tumor_area_selection":
        return prepare_pathology_evidence(
            runtime,
            task_name=task_name,
            condition=condition,
            scenario=scenario,
            shim_source=shim_source,
        )
    raise ModalityEvidenceError(f"No modality evidence adapter for family: {family}")

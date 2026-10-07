"""Export matched control/low-quality CT views from completed experiment jobs."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from faults.evidence.modalities import apply_ct_evidence

# Resolve stable repository-relative defaults without depending on the shell cwd.
SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_JOBS_ROOT = BENCHMARK_ROOT / "jobs/gpt6-luna-rep1"
DEFAULT_CACHE_ROOT = BENCHMARK_ROOT / "assets/ct_abnormality/assets/raw_cache"
DEFAULT_OUTPUT_ROOT = (
    BENCHMARK_ROOT / "faults/evidence/exports/gpt6-luna-rep1/ct-low-quality"
)

# Match the production CT adapter and the earlier CT task visualization window.
WINDOW_MIN_HU = -1000.0
WINDOW_MAX_HU = 500.0
PANEL_SIZE = 512
HEADER_HEIGHT = 46
FOOTER_HEIGHT = 38
BACKGROUND = (18, 21, 27)
TEXT = (235, 239, 245)
MUTED_TEXT = (176, 185, 199)
ACCENT = (244, 114, 182)


@dataclass(frozen=True)
class LowQualityJob:
    """Describe one low-quality injection recorded by the experiment harness."""

    task: str
    sample: str
    job_directory: Path
    setup_path: Path
    setup: dict[str, Any]


@dataclass(frozen=True)
class PlanePair:
    """Store matched control and transformed arrays for one anatomical plane."""

    name: str
    control: np.ndarray
    low_quality: np.ndarray
    width_spacing: float
    height_spacing: float
    slice_index: int


# Parse command-line paths while retaining reproducible repository defaults.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export control versus low-quality views for CT experiment jobs."
    )
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--export-volumes",
        action="store_true",
        help="Copy control NIfTI files and reconstruct complete low-quality inputs.",
    )
    return parser.parse_args()


# Discover one fault-setup record per low-quality error job in stable task order.
def discover_jobs(jobs_root: Path) -> list[LowQualityJob]:
    records: list[LowQualityJob] = []
    pattern = "ct_abnormality_*-evidence-low-quality-*-error"
    for job_directory in sorted(jobs_root.glob(pattern)):
        setup_paths = sorted(job_directory.glob("*/fault-setup.json"))
        if len(setup_paths) != 1:
            raise RuntimeError(
                f"Expected one fault-setup.json below {job_directory}, "
                f"found {len(setup_paths)}"
            )
        setup_path = setup_paths[0]
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if setup.get("scenario") != "evidence-low-quality":
            raise RuntimeError(f"Unexpected scenario in {setup_path}")
        if setup.get("condition") != "error":
            raise RuntimeError(f"Unexpected condition in {setup_path}")
        task = str(setup["task"])
        sample = task.removeprefix("ct_abnormality_")
        records.append(
            LowQualityJob(
                task=task,
                sample=sample,
                job_directory=job_directory,
                setup_path=setup_path,
                setup=setup,
            )
        )
    if not records:
        raise RuntimeError(f"No low-quality CT error jobs found below {jobs_root}")
    return records


# Reproduce the production adapter for only the three displayed central planes.
def build_plane_pairs(
    data: np.ndarray,
    voxel_sizes: tuple[float, float, float],
    *,
    slab_start: int,
    slab_end: int,
) -> list[PlanePair]:
    x_size, y_size, z_size = data.shape[:3]
    x_center, y_center, z_center = x_size // 2, y_size // 2, z_size // 2

    # Transform the central axial plane with the adapter's 4x4x2 sampling rule.
    sampled_z = slab_start + ((z_center - slab_start) // 2) * 2
    axial_control = data[:, :, z_center].copy()
    axial_coarse = data[::4, ::4, sampled_z]
    axial_low = np.repeat(np.repeat(axial_coarse, 4, axis=0), 4, axis=1)
    axial_low = axial_low[:x_size, :y_size]
    axial_low = np.round(axial_low / 100.0) * 100.0

    # Transform only the selected z slab in the central coronal plane.
    coronal_control = data[:, y_center, :].copy()
    coronal_low = coronal_control.copy()
    sampled_y = (y_center // 4) * 4
    coronal_coarse = data[::4, sampled_y, slab_start:slab_end:2]
    coronal_slab = np.repeat(
        np.repeat(coronal_coarse, 4, axis=0), 2, axis=1
    )
    coronal_slab = coronal_slab[:x_size, : slab_end - slab_start]
    coronal_low[:, slab_start:slab_end] = (
        np.round(coronal_slab / 100.0) * 100.0
    )

    # Transform only the selected z slab in the central sagittal plane.
    sagittal_control = data[x_center, :, :].copy()
    sagittal_low = sagittal_control.copy()
    sampled_x = (x_center // 4) * 4
    sagittal_coarse = data[sampled_x, ::4, slab_start:slab_end:2]
    sagittal_slab = np.repeat(
        np.repeat(sagittal_coarse, 4, axis=0), 2, axis=1
    )
    sagittal_slab = sagittal_slab[:y_size, : slab_end - slab_start]
    sagittal_low[:, slab_start:slab_end] = (
        np.round(sagittal_slab / 100.0) * 100.0
    )

    # Record display spacings after rotating each native array plane by 90 degrees.
    return [
        PlanePair(
            name="Axial",
            control=axial_control,
            low_quality=axial_low,
            width_spacing=voxel_sizes[0],
            height_spacing=voxel_sizes[1],
            slice_index=z_center,
        ),
        PlanePair(
            name="Coronal",
            control=coronal_control,
            low_quality=coronal_low,
            width_spacing=voxel_sizes[0],
            height_spacing=voxel_sizes[2],
            slice_index=y_center,
        ),
        PlanePair(
            name="Sagittal",
            control=sagittal_control,
            low_quality=sagittal_low,
            width_spacing=voxel_sizes[1],
            height_spacing=voxel_sizes[2],
            slice_index=x_center,
        ),
    ]


# Convert an HU plane into a lung-window image with physical aspect preserved.
def render_plane(
    array: np.ndarray,
    *,
    width_spacing: float,
    height_spacing: float,
    missing_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    missing = ~np.isfinite(array)
    clipped = np.clip(
        np.nan_to_num(array, nan=WINDOW_MIN_HU), WINDOW_MIN_HU, WINDOW_MAX_HU
    )
    pixels = np.round(
        (clipped - WINDOW_MIN_HU) / (WINDOW_MAX_HU - WINDOW_MIN_HU) * 255.0
    ).astype(np.uint8)
    rotated = np.rot90(pixels)
    image = Image.fromarray(rotated, mode="L").convert("RGB")

    # Fit the plane into a square using voxel dimensions rather than array dimensions.
    physical_width = image.width * width_spacing
    physical_height = image.height * height_spacing
    scale = min(PANEL_SIZE / physical_width, PANEL_SIZE / physical_height)
    target_width = max(1, int(round(physical_width * scale)))
    target_height = max(1, int(round(physical_height * scale)))
    resized = image.resize(
        (target_width, target_height), resample=Image.Resampling.NEAREST
    )
    if missing_color is not None and np.any(missing):
        missing_image = Image.fromarray(
            np.rot90(missing).astype(np.uint8) * 255, mode="L"
        )
        missing_resized = missing_image.resize(
            (target_width, target_height), resample=Image.Resampling.NEAREST
        )
        overlay = Image.new("RGB", resized.size, missing_color)
        resized.paste(overlay, mask=missing_resized)
    panel = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    left = (PANEL_SIZE - target_width) // 2
    upper = (PANEL_SIZE - target_height) // 2
    panel.paste(resized, (left, upper))
    return panel


# Add a compact caption outside the medical image pixels.
def caption_panel(image: Image.Image, title: str, footer: str) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (PANEL_SIZE, PANEL_SIZE + HEADER_HEIGHT + FOOTER_HEIGHT),
        BACKGROUND,
    )
    canvas.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 15), title, fill=TEXT, font=font)
    draw.text(
        (12, PANEL_SIZE + HEADER_HEIGHT + 12),
        footer,
        fill=MUTED_TEXT,
        font=font,
    )
    return canvas


# Join equal-sized panels with narrow separators for direct visual comparison.
def tile_panels(panels: list[Image.Image], columns: int) -> Image.Image:
    if not panels or len(panels) % columns:
        raise ValueError("Panel count must be a positive multiple of columns")
    rows = len(panels) // columns
    gap = 3
    width = columns * panels[0].width + (columns - 1) * gap
    height = rows * panels[0].height + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), ACCENT)
    for index, panel in enumerate(panels):
        x = (index % columns) * (panel.width + gap)
        y = (index // columns) * (panel.height + gap)
        canvas.paste(panel, (x, y))
    return canvas


# Hash one file without reading it all into memory.
def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Hash the decompressed NIfTI stream to ignore irrelevant gzip wrapper metadata.
def hash_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Copy the exact control volume and reconstruct the production low-quality volume.
def export_input_volumes(
    job: LowQualityJob,
    *,
    source_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    control_root = output_root / "exact_inputs/control"
    low_quality_root = output_root / "exact_inputs/low_quality"
    control_root.mkdir(parents=True, exist_ok=True)
    low_quality_root.mkdir(parents=True, exist_ok=True)
    control_path = control_root / source_path.name
    low_quality_path = low_quality_root / source_path.name

    # Preserve the original compressed input bytes as the standalone control file.
    shutil.copy2(source_path, control_path)
    expected_before = dict(job.setup["primary_input"]["before"])
    control_sha256 = hash_file(control_path)
    if control_sha256 != expected_before["sha256"]:
        raise RuntimeError(f"Control input hash mismatch for {job.sample}")

    # Run the same production adapter that modified /workspace/data/scan.nii.gz.
    shutil.copy2(source_path, low_quality_path)
    apply_ct_evidence(
        low_quality_path,
        condition="error",
        scenario="evidence-low-quality",
    )
    expected_after = dict(job.setup["primary_input"]["after"])
    content_sha256 = hash_gzip_content(low_quality_path)
    if content_sha256 != expected_after["gzip_content_sha256"]:
        raise RuntimeError(f"Low-quality NIfTI content mismatch for {job.sample}")

    # Record both semantic content and wrapper hashes for later provenance checks.
    return {
        "control": str(control_path.relative_to(output_root)),
        "control_sha256": control_sha256,
        "low_quality": str(low_quality_path.relative_to(output_root)),
        "low_quality_sha256": hash_file(low_quality_path),
        "low_quality_gzip_content_sha256": content_sha256,
        "recorded_low_quality_sha256": expected_after["sha256"],
    }


# Render all comparison files for one job and return its public manifest entry.
def export_job(
    job: LowQualityJob,
    *,
    cache_root: Path,
    output_root: Path,
    export_volumes: bool,
) -> dict[str, Any]:
    source_path = cache_root / f"{job.sample}.nii.gz"
    if not source_path.is_file():
        raise FileNotFoundError(f"Cached source CT not found: {source_path}")

    # Load one sample as float32 and confirm it matches the recorded injection shape.
    image = nib.load(str(source_path))
    source_shape = tuple(int(value) for value in image.shape[:3])
    evidence = dict(job.setup["evidence"])
    recorded_shape = tuple(int(value) for value in evidence["source_shape"][:3])
    if source_shape != recorded_shape:
        raise RuntimeError(
            f"Shape mismatch for {job.sample}: cache={source_shape}, "
            f"job={recorded_shape}"
        )
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    slab_start = int(evidence["selected_start"])
    slab_end = int(evidence["selected_end"])
    pairs = build_plane_pairs(
        data,
        voxel_sizes,
        slab_start=slab_start,
        slab_end=slab_end,
    )

    # Write one overview and one larger comparison per anatomical plane.
    sample_root = output_root / job.sample
    sample_root.mkdir(parents=True, exist_ok=True)
    overview_panels: list[Image.Image] = []
    files: dict[str, str] = {}
    standalone_control_root = output_root / "standalone_png/control"
    standalone_low_quality_root = output_root / "standalone_png/low_quality"
    standalone_control_root.mkdir(parents=True, exist_ok=True)
    standalone_low_quality_root.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        control_image = render_plane(
            pair.control,
            width_spacing=pair.width_spacing,
            height_spacing=pair.height_spacing,
        )
        low_quality_image = render_plane(
            pair.low_quality,
            width_spacing=pair.width_spacing,
            height_spacing=pair.height_spacing,
        )

        # Save unlabelled standalone renders before composing any captions or borders.
        control_standalone = (
            standalone_control_root / f"{job.sample}_{pair.name.lower()}.png"
        )
        low_quality_standalone = (
            standalone_low_quality_root / f"{job.sample}_{pair.name.lower()}.png"
        )
        control_image.save(control_standalone, format="PNG", optimize=True)
        low_quality_image.save(low_quality_standalone, format="PNG", optimize=True)
        files[f"standalone_control_{pair.name.lower()}"] = str(
            control_standalone.relative_to(output_root)
        )
        files[f"standalone_low_quality_{pair.name.lower()}"] = str(
            low_quality_standalone.relative_to(output_root)
        )
        control_panel = caption_panel(
            control_image,
            f"{pair.name} | control",
            f"central index={pair.slice_index}",
        )
        low_quality_panel = caption_panel(
            low_quality_image,
            f"{pair.name} | low quality",
            f"pixelate 4x4x2 + quantize 100 HU; z={slab_start}:{slab_end}",
        )
        comparison = tile_panels([control_panel, low_quality_panel], columns=2)
        plane_path = sample_root / f"{pair.name.lower()}_control_vs_low_quality.png"
        comparison.save(plane_path, format="PNG", optimize=True)
        files[pair.name.lower()] = str(plane_path.relative_to(output_root))
        overview_panels.extend([control_panel, low_quality_panel])

    # Reorder panels into control and low-quality rows for the six-panel overview.
    controls = overview_panels[0::2]
    low_quality = overview_panels[1::2]
    overview_path = sample_root / "overview_control_vs_low_quality.png"
    tile_panels(controls + low_quality, columns=3).save(
        overview_path, format="PNG", optimize=True
    )
    files["overview"] = str(overview_path.relative_to(output_root))

    # Release the large array before optionally reconstructing the complete volume.
    del data
    gc.collect()
    volume_files = (
        export_input_volumes(
            job,
            source_path=source_path,
            output_root=output_root,
        )
        if export_volumes
        else None
    )

    # Preserve job-derived transform metadata alongside every exported artifact.
    metadata = {
        "task": job.task,
        "sample": job.sample,
        "source_shape": list(source_shape),
        "voxel_sizes_mm": list(voxel_sizes),
        "display_window_hu": [WINDOW_MIN_HU, WINDOW_MAX_HU],
        "display_orientation": "native array planes rotated 90 degrees; not diagnostic reorientation",
        "fault_setup": evidence,
        "source_job": str(job.job_directory.relative_to(BENCHMARK_ROOT)),
        "source_fault_setup": str(job.setup_path.relative_to(BENCHMARK_ROOT)),
        "files": files,
        "volumes": volume_files,
    }
    metadata_path = sample_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return metadata


# Write a readable local index for browsing the complete exported cohort.
def write_readme(output_root: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# GPT-6 Luna CT low-quality evidence gallery",
        "",
        "Each sample uses the exact transform recorded by its `fault-setup.json`:",
        "the central 20% axial slab is pixelated `4x4x2` and quantized to `100 HU`.",
        "Images use a `[-1000, 500] HU` lung window. These are native array-plane",
        "research visualizations, not clinically reoriented diagnostic images.",
        "",
        "Standalone, unlabelled PNG views are under `standalone_png/`. Complete",
        "control and low-quality NIfTI inputs are under `exact_inputs/` when the",
        "exporter is run with `--export-volumes`.",
        "",
        "| Sample | Shape | Altered z slab | Overview |",
        "|---|---:|---:|---|",
    ]
    for entry in entries:
        shape = "x".join(str(value) for value in entry["source_shape"])
        evidence = entry["fault_setup"]
        start = evidence["selected_start"]
        end = evidence["selected_end"]
        overview = entry["files"]["overview"]
        lines.append(
            f"| `{entry['sample']}` | {shape} | [{start}, {end}) | "
            f"[open]({overview}) |"
        )
    lines.extend(
        [
            "",
            "Each sample directory also contains separate axial, coronal, and sagittal",
            "control-versus-low-quality comparisons plus `metadata.json`.",
            "The PNG files are lung-window renders; the NIfTI files are the actual",
            "three-dimensional inputs used by the task.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


# Export all discovered tasks and a machine-readable cohort manifest.
def main() -> int:
    args = parse_args()
    jobs_root = args.jobs_root.resolve()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(jobs_root)
    entries = [
        export_job(
            job,
            cache_root=cache_root,
            output_root=output_root,
            export_volumes=args.export_volumes,
        )
        for job in jobs
    ]
    manifest = {
        "schema_version": 1,
        "jobs_root": str(jobs_root.relative_to(BENCHMARK_ROOT)),
        "sample_count": len(entries),
        "samples": entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(output_root, entries)
    print(f"Exported {len(entries)} CT comparisons to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

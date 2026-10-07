"""Export exact CT explicit- and natural-missing experiment inputs."""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from faults.evidence.examples.export_ct_low_quality import (
    BENCHMARK_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_JOBS_ROOT,
    hash_file,
    hash_gzip_content,
    render_plane,
)
from faults.evidence.modalities import apply_ct_evidence

# Keep names aligned with the experiment runner while using short output folders.
SCENARIOS = {
    "evidence-explicit-missing": "explicit_missing",
    "evidence-natural-missing": "natural_missing",
}
OUTPUT_ROOT = (
    BENCHMARK_ROOT / "faults/evidence/exports/gpt6-luna-rep1/ct-missing-evidence"
)
EXPLICIT_MISSING_COLOR = (255, 0, 255)


@dataclass(frozen=True)
class MissingEvidenceJob:
    """Describe one recorded CT missing-evidence injection."""

    scenario: str
    task: str
    sample: str
    job_directory: Path
    setup_path: Path
    setup: dict[str, Any]


# Discover every recorded job for both missing-evidence scenarios.
def discover_jobs(jobs_root: Path) -> list[MissingEvidenceJob]:
    jobs: list[MissingEvidenceJob] = []
    for scenario in SCENARIOS:
        pattern = f"ct_abnormality_*-{scenario}-*-error"
        for job_directory in sorted(jobs_root.glob(pattern)):
            setup_paths = sorted(job_directory.glob("*/fault-setup.json"))
            if len(setup_paths) != 1:
                raise RuntimeError(
                    f"Expected one fault-setup.json below {job_directory}, "
                    f"found {len(setup_paths)}"
                )
            setup_path = setup_paths[0]
            setup = json.loads(setup_path.read_text(encoding="utf-8"))
            if setup.get("scenario") != scenario or setup.get("condition") != "error":
                raise RuntimeError(f"Unexpected fault record in {setup_path}")
            task = str(setup["task"])
            jobs.append(
                MissingEvidenceJob(
                    scenario=scenario,
                    task=task,
                    sample=task.removeprefix("ct_abnormality_"),
                    job_directory=job_directory,
                    setup_path=setup_path,
                    setup=setup,
                )
            )
    if not jobs:
        raise RuntimeError(f"No CT missing-evidence jobs found below {jobs_root}")
    return jobs


# Extract the three central native-array planes from one transformed volume.
def central_planes(data: np.ndarray) -> dict[str, tuple[np.ndarray, int, int]]:
    x_center = data.shape[0] // 2
    y_center = data.shape[1] // 2
    z_center = data.shape[2] // 2
    return {
        "axial": (data[:, :, z_center], 0, 1),
        "coronal": (data[:, y_center, :], 0, 2),
        "sagittal": (data[x_center, :, :], 1, 2),
    }


# Reconstruct, verify, and render one actual error-group task input.
def export_job(
    job: MissingEvidenceJob,
    *,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    short_name = SCENARIOS[job.scenario]
    source_path = cache_root / f"{job.sample}.nii.gz"
    scenario_root = output_root / short_name
    volume_root = scenario_root / "exact_inputs"
    png_root = scenario_root / "standalone_png"
    metadata_root = scenario_root / "metadata"
    volume_root.mkdir(parents=True, exist_ok=True)
    png_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    target_path = volume_root / source_path.name

    # Apply the production adapter to a private copy of the original NIfTI.
    shutil.copy2(source_path, target_path)
    application = apply_ct_evidence(
        target_path,
        condition="error",
        scenario=job.scenario,
    )

    # Require byte-identical output and decompressed content from the recorded job.
    recorded_after = dict(job.setup["primary_input"]["after"])
    wrapper_sha256 = hash_file(target_path)
    content_sha256 = hash_gzip_content(target_path)
    if wrapper_sha256 != recorded_after["sha256"]:
        raise RuntimeError(f"Compressed NIfTI hash mismatch for {job.sample}")
    if content_sha256 != recorded_after["gzip_content_sha256"]:
        raise RuntimeError(f"NIfTI content hash mismatch for {job.sample}")

    # Render independent PNGs while preserving missing voxels as visible magenta.
    image = nib.load(str(target_path))
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    png_files: dict[str, str] = {}
    for plane_name, (plane, width_axis, height_axis) in central_planes(data).items():
        rendered = render_plane(
            plane,
            width_spacing=voxel_sizes[width_axis],
            height_spacing=voxel_sizes[height_axis],
            missing_color=(
                EXPLICIT_MISSING_COLOR
                if job.scenario == "evidence-explicit-missing"
                else None
            ),
        )
        png_path = png_root / f"{job.sample}_{plane_name}.png"
        rendered.save(png_path, format="PNG", optimize=True)
        png_files[plane_name] = str(png_path.relative_to(output_root))

    # Preserve enough provenance to trace each file to the experiment record.
    metadata = {
        "task": job.task,
        "sample": job.sample,
        "scenario": job.scenario,
        "source_shape": list(image.shape[:3]),
        "voxel_sizes_mm": list(voxel_sizes),
        "fault_setup": application.details,
        "source_job": str(job.job_directory.relative_to(BENCHMARK_ROOT)),
        "source_fault_setup": str(job.setup_path.relative_to(BENCHMARK_ROOT)),
        "exact_input": str(target_path.relative_to(output_root)),
        "sha256": wrapper_sha256,
        "gzip_content_sha256": content_sha256,
        "png_files": png_files,
        "explicit_missing_display": (
            "NaN voxels rendered magenta for visualization"
            if job.scenario == "evidence-explicit-missing"
            else None
        ),
    }
    metadata_path = metadata_root / f"{job.sample}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del data
    gc.collect()
    return metadata


# Write one index explaining the semantic difference between both fault types.
def write_readme(output_root: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# GPT-6 Luna CT missing-evidence inputs",
        "",
        "This directory contains exact error-group NIfTI inputs and independent",
        "central axial/coronal/sagittal PNG renders for two scenarios:",
        "",
        "- `explicit_missing`: the central axial slab contains NaN; PNGs render NaN",
        "  in magenta only as a visualization convention.",
        "- `natural_missing`: the same slab is silently replaced by the volume's",
        "  first-percentile background intensity and therefore appears dark/blank.",
        "",
        "The complete NIfTI files are byte-identical to the inputs recorded in each",
        "job's `fault-setup.json`. Original control inputs are shared with the",
        "existing `../ct-low-quality/exact_inputs/control/` directory.",
        "",
        "| Scenario | Sample | Exact NIfTI | Axial PNG |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        axial = entry["png_files"]["axial"]
        lines.append(
            f"| `{entry['scenario']}` | `{entry['sample']}` | "
            f"[{Path(entry['exact_input']).name}]({entry['exact_input']}) | "
            f"[open]({axial}) |"
        )
    lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


# Export all jobs and one machine-readable manifest.
def main() -> int:
    jobs = discover_jobs(DEFAULT_JOBS_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    entries = [
        export_job(job, cache_root=DEFAULT_CACHE_ROOT, output_root=OUTPUT_ROOT)
        for job in jobs
    ]
    manifest = {
        "schema_version": 1,
        "sample_count_per_scenario": {
            scenario: sum(entry["scenario"] == scenario for entry in entries)
            for scenario in SCENARIOS
        },
        "samples": entries,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(OUTPUT_ROOT, entries)
    print(f"Exported {len(entries)} missing-evidence CT inputs to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

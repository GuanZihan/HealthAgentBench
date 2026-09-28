#!/usr/bin/env python3
"""Render a publication-ready overview of the CT abnormality task."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Define a restrained research-figure palette shared across all panels.
NAVY = "#102A43"
BLUE = "#2F6BFF"
LIGHT_BLUE = "#EAF2FF"
PURPLE = "#7257D5"
LIGHT_PURPLE = "#F0ECFF"
GREEN = "#16856B"
LIGHT_GREEN = "#E8F7F2"
AMBER = "#B86A00"
LIGHT_AMBER = "#FFF4DE"
GRAY = "#52606D"
LIGHT_GRAY = "#F5F7FA"
MID_GRAY = "#D9E2EC"
WHITE = "#FFFFFF"


# Apply a conventional CT window and normalize it to the display range.
def window_ct(image: np.ndarray, *, center: float, width: float) -> np.ndarray:
    lower = center - width / 2.0
    upper = center + width / 2.0
    return np.clip((image.astype(np.float32) - lower) / (upper - lower), 0.0, 1.0)


# Crop a 2-D plane around non-air anatomy while retaining a small visual margin.
def crop_to_body(image: np.ndarray, *, margin_fraction: float = 0.04) -> np.ndarray:
    body_mask = image > -950
    coordinates = np.argwhere(body_mask)
    if coordinates.size == 0:
        return image
    row_min, column_min = coordinates.min(axis=0)
    row_max, column_max = coordinates.max(axis=0)
    row_margin = max(8, int((row_max - row_min + 1) * margin_fraction))
    column_margin = max(8, int((column_max - column_min + 1) * margin_fraction))
    row_min = max(0, row_min - row_margin)
    row_max = min(image.shape[0] - 1, row_max + row_margin)
    column_min = max(0, column_min - column_margin)
    column_max = min(image.shape[1] - 1, column_max + column_margin)
    return image[row_min : row_max + 1, column_min : column_max + 1]


# Select central chest planes after converting the volume to canonical RAS order.
def load_demo_planes(
    volume_path: Path,
) -> tuple[dict[str, np.ndarray], tuple[int, ...], tuple[float, ...]]:
    image = nib.as_closest_canonical(nib.load(volume_path))
    volume = np.asanyarray(image.dataobj)
    voxel_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    center_x, center_y, center_z = (dimension // 2 for dimension in volume.shape)

    # Extract one anatomically representative plane along each spatial axis.
    axial = np.rot90(volume[:, :, center_z])
    coronal = np.flipud(volume[:, center_y, :].T)
    sagittal = np.flipud(volume[center_x, :, :].T)
    planes = {
        "axial": crop_to_body(axial),
        "coronal": crop_to_body(coronal),
        "sagittal": crop_to_body(sagittal),
    }
    return (
        planes,
        tuple(int(dimension) for dimension in volume.shape),
        voxel_spacing,
    )


# Draw a CT panel with a consistent title, border, and orientation marker.
def draw_ct_panel(
    figure: plt.Figure,
    *,
    bounds: tuple[float, float, float, float],
    image: np.ndarray,
    title: str,
    subtitle: str,
    window_center: float,
    window_width: float,
    pixel_aspect: float,
    orientation_marker: str,
) -> None:
    axis = figure.add_axes(bounds)
    axis.imshow(
        window_ct(image, center=window_center, width=window_width),
        cmap="gray",
        interpolation="bilinear",
        aspect=pixel_aspect,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("black")
    for spine in axis.spines.values():
        spine.set_color(MID_GRAY)
        spine.set_linewidth(1.2)
    axis.text(
        0.03,
        0.96,
        orientation_marker,
        transform=axis.transAxes,
        color=WHITE,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.0,
        1.075,
        title,
        transform=axis.transAxes,
        color=NAVY,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )
    axis.text(
        0.0,
        1.02,
        subtitle,
        transform=axis.transAxes,
        color=GRAY,
        fontsize=8.5,
        va="bottom",
    )


# Add one numbered pipeline card with short research-oriented copy.
def draw_step_card(
    figure: plt.Figure,
    *,
    bounds: tuple[float, float, float, float],
    number: str,
    title: str,
    lines: list[str],
    accent: str,
    fill: str,
    monospace_lines: set[int] | None = None,
) -> None:
    left, bottom, width, height = bounds
    card = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor=fill,
        edgecolor=accent,
        linewidth=1.4,
    )
    figure.add_artist(card)

    # Use a compact numbered badge to make the processing order immediately visible.
    badge = plt.Circle(
        (left + 0.026, bottom + height - 0.032),
        0.016,
        transform=figure.transFigure,
        facecolor=accent,
        edgecolor="none",
    )
    figure.add_artist(badge)
    figure.text(
        left + 0.026,
        bottom + height - 0.032,
        number,
        ha="center",
        va="center",
        color=WHITE,
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        left + 0.052,
        bottom + height - 0.032,
        title,
        ha="left",
        va="center",
        color=NAVY,
        fontsize=12.5,
        fontweight="bold",
    )

    # Render each line independently so paths and output syntax can use monospace.
    monospace_lines = monospace_lines or set()
    line_start = bottom + height - 0.068
    line_spacing = min(0.025, (height - 0.078) / max(1, len(lines)))
    for line_index, line in enumerate(lines):
        figure.text(
            left + 0.026,
            line_start - line_index * line_spacing,
            line,
            ha="left",
            va="top",
            color=GRAY,
            fontsize=8.7,
            family="monospace" if line_index in monospace_lines else "sans-serif",
        )


# Connect adjacent cards without crossing their text regions.
def draw_down_arrow(
    figure: plt.Figure,
    *,
    x_position: float,
    upper_bottom: float,
    lower_top: float,
) -> None:
    arrow = FancyArrowPatch(
        (x_position, upper_bottom - 0.006),
        (x_position, lower_top + 0.006),
        transform=figure.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color="#9FB3C8",
    )
    figure.add_artist(arrow)


# Compose the CT montage and the benchmark workflow into one self-contained figure.
def create_demo(volume_path: Path, output_path: Path) -> None:
    planes, volume_shape, voxel_spacing = load_demo_planes(volume_path)

    # Configure a wide canvas that remains legible in papers, slides, and webpages.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.facecolor": WHITE,
        }
    )
    figure = plt.figure(figsize=(16, 10), dpi=150, facecolor=WHITE)

    # Introduce the clinical task before showing either anatomy or file mechanics.
    figure.text(
        0.045,
        0.955,
        "Chest CT Multi-Abnormality Classification",
        color=NAVY,
        fontsize=25,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.045,
        0.915,
        "One 3-D CT volume  •  A case-specific label list  •  One yes/no decision per finding",
        color=GRAY,
        fontsize=12,
        va="top",
    )
    figure.text(
        0.955,
        0.947,
        "REAL CT-RATE VOLUME",
        ha="right",
        va="top",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": LIGHT_BLUE,
            "edgecolor": "none",
        },
    )

    # Label the real input volume without exposing its report-derived gold labels.
    figure.text(
        0.045,
        0.865,
        f"Representative input: {volume_path.name}  |  Volume shape: "
        f"{volume_shape[0]} × {volume_shape[1]} × {volume_shape[2]}",
        color=NAVY,
        fontsize=10.5,
        fontweight="bold",
    )

    # Show three orthogonal views plus a second CT window on the axial plane.
    draw_ct_panel(
        figure,
        bounds=(0.045, 0.515, 0.275, 0.285),
        image=planes["axial"],
        title="Axial",
        subtitle="Lung window  •  WL −600 / WW 1500",
        window_center=-600,
        window_width=1500,
        pixel_aspect=1.0,
        orientation_marker="R",
    )
    draw_ct_panel(
        figure,
        bounds=(0.345, 0.515, 0.275, 0.285),
        image=planes["coronal"],
        title="Coronal",
        subtitle="Lung window  •  One plane from the 3-D volume",
        window_center=-600,
        window_width=1500,
        pixel_aspect=voxel_spacing[2] / voxel_spacing[0],
        orientation_marker="R",
    )
    draw_ct_panel(
        figure,
        bounds=(0.045, 0.145, 0.275, 0.285),
        image=planes["sagittal"],
        title="Sagittal",
        subtitle="Lung window  •  Agent may inspect arbitrary slices",
        window_center=-600,
        window_width=1500,
        pixel_aspect=voxel_spacing[2] / voxel_spacing[1],
        orientation_marker="P",
    )
    draw_ct_panel(
        figure,
        bounds=(0.345, 0.145, 0.275, 0.285),
        image=planes["axial"],
        title="Axial",
        subtitle="Mediastinal window  •  WL 40 / WW 400",
        window_center=40,
        window_width=400,
        pixel_aspect=1.0,
        orientation_marker="R",
    )

    # Arrange the task contract as an explicit four-stage pipeline.
    card_left = 0.675
    card_width = 0.28
    cards = [
        (0.715, 0.14, "1", "Inputs", BLUE, LIGHT_BLUE),
        (0.525, 0.135, "2", "Agent analysis", PURPLE, LIGHT_PURPLE),
        (0.325, 0.145, "3", "Required output", GREEN, LIGHT_GREEN),
        (0.145, 0.125, "4", "Evaluation", AMBER, LIGHT_AMBER),
    ]
    draw_step_card(
        figure,
        bounds=(card_left, cards[0][0], card_width, cards[0][1]),
        number=cards[0][2],
        title=cards[0][3],
        accent=cards[0][4],
        fill=cards[0][5],
        lines=[
            "/workspace/data/scan.nii.gz",
            "/workspace/data/labels.txt",
            "Typically 4–12 requested findings",
        ],
        monospace_lines={0, 1},
    )
    draw_step_card(
        figure,
        bounds=(card_left, cards[1][0], card_width, cards[1][1]),
        number=cards[1][2],
        title=cards[1][3],
        accent=cards[1][4],
        fill=cards[1][5],
        lines=[
            "Inspect the 3-D CT volume and multiple windows",
            "Assess only findings listed in labels.txt",
            "Use image evidence; do not retrieve published labels",
        ],
    )
    draw_step_card(
        figure,
        bounds=(card_left, cards[2][0], card_width, cards[2][1]),
        number=cards[2][2],
        title=cards[2][3],
        accent=cards[2][4],
        fill=cards[2][5],
        lines=[
            "/workspace/submission/predictions.txt",
            "Cardiomegaly: no",
            "Pleural effusion: yes",
            "Lung nodule: no",
        ],
        monospace_lines={0, 1, 2, 3},
    )
    draw_step_card(
        figure,
        bounds=(card_left, cards[3][0], card_width, cards[3][1]),
        number=cards[3][2],
        title=cards[3][3],
        accent=cards[3][4],
        fill=cards[3][5],
        lines=[
            "Reward = 1 only if every requested label is correct",
            "One incorrect label makes the task-level reward 0",
        ],
    )

    # Draw the process direction after all cards so arrows remain visually aligned.
    card_centers_x = card_left + card_width / 2
    draw_down_arrow(
        figure,
        x_position=card_centers_x,
        upper_bottom=cards[0][0],
        lower_top=cards[1][0] + cards[1][1],
    )
    draw_down_arrow(
        figure,
        x_position=card_centers_x,
        upper_bottom=cards[1][0],
        lower_top=cards[2][0] + cards[2][1],
    )
    draw_down_arrow(
        figure,
        x_position=card_centers_x,
        upper_bottom=cards[2][0],
        lower_top=cards[3][0] + cards[3][1],
    )

    # Clarify that the sample output demonstrates syntax rather than hidden gold labels.
    note = FancyBboxPatch(
        (0.045, 0.045),
        0.91,
        0.045,
        boxstyle="round,pad=0.006,rounding_size=0.009",
        transform=figure.transFigure,
        facecolor=LIGHT_GRAY,
        edgecolor=MID_GRAY,
        linewidth=1.0,
    )
    figure.add_artist(note)
    figure.text(
        0.058,
        0.067,
        "Interpretation note",
        color=NAVY,
        fontsize=9,
        fontweight="bold",
        va="center",
    )
    figure.text(
        0.163,
        0.067,
        "The CT slices are real. The example predictions illustrate file syntax only and are not this volume’s gold labels. "
        "CT-RATE supplies patient-level labels, not lesion bounding boxes.",
        color=GRAY,
        fontsize=8.5,
        va="center",
    )

    # Save both the raster figure and its parent directory deterministically.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=WHITE, bbox_inches=None)
    plt.close(figure)


# Expose paths and defaults explicitly so other CT volumes can reuse the figure code.
def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    raw_cache = script_dir.parent / "assets" / "raw_cache"
    parser = argparse.ArgumentParser(
        description="Create a visual overview of the CT abnormality task."
    )
    parser.add_argument(
        "--volume",
        type=Path,
        default=raw_cache / "valid_81_a_1.nii.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "ct_abnormality_task_demo.png",
    )
    return parser


# Validate local data before allocating memory for the high-resolution volume.
def main() -> None:
    args = build_parser().parse_args()
    if not args.volume.is_file():
        raise SystemExit(f"CT volume does not exist: {args.volume}")
    create_demo(args.volume.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

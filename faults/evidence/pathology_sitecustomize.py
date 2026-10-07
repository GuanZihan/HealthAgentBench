"""Agent-side OpenSlide evidence delivery shim installed as sitecustomize."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


# Read immutable trial parameters before importing the real OpenSlide package.
CONDITION = os.environ.get("HEALTHAGENT_EVIDENCE_CONDITION", "control")
SCENARIO = os.environ.get("HEALTHAGENT_EVIDENCE_SCENARIO", "")
TASK = os.environ.get("HEALTHAGENT_EVIDENCE_TASK", "unknown")
EVENTS = Path(
    os.environ.get("HEALTHAGENT_EVIDENCE_EVENTS", "/tmp/evidence-events.jsonl")
)
GRID_SIZE = 4096
SEED = int.from_bytes(hashlib.sha256(TASK.encode()).digest()[:4], "big")
_RECORDED = False


# Select exactly one fifth of analysis-grid tiles with a stable integer hash.
def _selected(grid_x: int, grid_y: int) -> bool:
    value = (grid_x * 73856093) ^ (grid_y * 19349663) ^ SEED
    return value % 5 == 0


# Record exposure once per Python process without retaining image pixels.
def _record_exposure() -> None:
    global _RECORDED
    if _RECORDED:
        return
    _RECORDED = True
    event = {
        "event": "pathology_evidence_delivery",
        "task": TASK,
        "condition": CONDITION,
        "scenario": SCENARIO,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        pass


# Patch standard Python OpenSlide reads while leaving control pixels untouched.
def _install() -> None:
    try:
        import openslide
        from PIL import ImageDraw
    except (ImportError, OSError):
        return

    original = openslide.OpenSlide.read_region

    # Transform only selected grid intersections in each requested region.
    def read_region(self, location, level, size):  # type: ignore[no-untyped-def]
        image = original(self, location, level, size)
        if CONDITION != "error":
            return image
        _record_exposure()
        downsample = float(self.level_downsamples[level])
        x0, y0 = int(location[0]), int(location[1])
        x1 = x0 + int(size[0] * downsample)
        y1 = y0 + int(size[1] * downsample)
        grid_left = x0 // GRID_SIZE
        grid_top = y0 // GRID_SIZE
        grid_right = max(grid_left, (x1 - 1) // GRID_SIZE)
        grid_bottom = max(grid_top, (y1 - 1) // GRID_SIZE)
        transformed = image.convert("RGBA")
        low_quality = transformed.resize(
            (max(1, transformed.width // 8), max(1, transformed.height // 8))
        ).resize(transformed.size)
        draw = ImageDraw.Draw(transformed)
        for grid_y in range(grid_top, grid_bottom + 1):
            for grid_x in range(grid_left, grid_right + 1):
                if not _selected(grid_x, grid_y):
                    continue
                left = max(0, round((grid_x * GRID_SIZE - x0) / downsample))
                upper = max(0, round((grid_y * GRID_SIZE - y0) / downsample))
                right = min(
                    size[0], round(((grid_x + 1) * GRID_SIZE - x0) / downsample)
                )
                lower = min(
                    size[1], round(((grid_y + 1) * GRID_SIZE - y0) / downsample)
                )
                if left >= right or upper >= lower:
                    continue
                box = (left, upper, right, lower)
                if SCENARIO == "evidence-explicit-missing":
                    draw.rectangle(box, fill=(255, 0, 255, 255))
                    draw.line((left, upper, right, lower), fill=(0, 0, 0, 255), width=3)
                    draw.line((left, lower, right, upper), fill=(0, 0, 0, 255), width=3)
                elif SCENARIO == "evidence-natural-missing":
                    draw.rectangle(box, fill=(255, 255, 255, 255))
                elif SCENARIO == "evidence-low-quality":
                    transformed.paste(low_quality.crop(box), box)
        return transformed

    openslide.OpenSlide.read_region = read_region


# Install automatically because Python imports sitecustomize during startup.
_install()

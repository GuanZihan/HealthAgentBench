"""Load, validate, and apply frozen evidence-degradation recipes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

# Keep scenario names stable because they become experimental condition labels.
EvidenceScenario = Literal[
    "evidence-explicit-missing",
    "evidence-natural-missing",
    "evidence-low-quality",
]
EVIDENCE_SCENARIOS: tuple[EvidenceScenario, ...] = (
    "evidence-explicit-missing",
    "evidence-natural-missing",
    "evidence-low-quality",
)

# Restrict preparation output to stable experiment-facing evidence categories.
SUPPORTED_CATEGORIES = {
    "biomarker",
    "treatment",
    "disease_stage",
    "diagnosis",
    "performance_status",
    "laboratory_value",
    "imaging_finding",
    "procedure",
    "comorbidity",
    "symptom",
    "demographic",
    "temporal_history",
}

# Render explicit missingness with the bracketed form selected for this study.
EXPLICIT_REPLACEMENTS = {
    "biomarker": "[BIOMARKER REDACTED]",
    "treatment": "[TREATMENT REDACTED]",
    "disease_stage": "[DISEASE STAGE REDACTED]",
    "diagnosis": "[DIAGNOSIS REDACTED]",
    "performance_status": "[PERFORMANCE STATUS REDACTED]",
    "laboratory_value": "[LABORATORY VALUE REDACTED]",
    "imaging_finding": "[IMAGING FINDING REDACTED]",
    "procedure": "[PROCEDURE REDACTED]",
    "comorbidity": "[COMORBIDITY REDACTED]",
    "symptom": "[SYMPTOM REDACTED]",
    "demographic": "[DEMOGRAPHIC DETAIL REDACTED]",
    "temporal_history": "[TIMELINE REDACTED]",
}

# Require explicit provenance uncertainty in each context-specific replacement.
LOW_QUALITY_CUES = (
    "could not be verified",
    "incomplete",
    "inconsisten",
    "not confirmed",
    "not documented",
    "not formally",
    "not source-verified",
    "reported",
    "uncertain",
    "unavailable",
    "unverified",
    "without the original",
)


class EvidenceRecipeError(RuntimeError):
    """Raised when a recipe is unsafe, stale, or not approved."""


@dataclass(frozen=True)
class EvidenceEdit:
    """One source-anchored evidence span selected during offline preparation."""

    edit_id: str
    start: int
    end: int
    category: str
    importance: str
    source_text_sha256: str
    natural_start: int
    natural_end: int
    natural_text_sha256: str
    low_quality_replacement: str


@dataclass(frozen=True)
class EvidenceRecipe:
    """An approved set of deterministic edits for one benchmark task."""

    task: str
    source_sha256: str
    status: str
    severity: str
    generator: dict[str, object]
    edits: tuple[EvidenceEdit, ...]


# Hash bytes without retaining source text in the committed recipe.
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# Parse one edit while rejecting unknown categories and malformed offsets.
def _parse_edit(raw: object, index: int) -> EvidenceEdit:
    if not isinstance(raw, dict):
        raise EvidenceRecipeError(f"Recipe edit {index} must be a JSON object")
    try:
        edit = EvidenceEdit(
            edit_id=str(raw["edit_id"]),
            start=int(raw["start"]),
            end=int(raw["end"]),
            category=str(raw["category"]),
            importance=str(raw["importance"]),
            source_text_sha256=str(raw["source_text_sha256"]),
            natural_start=int(raw["natural_start"]),
            natural_end=int(raw["natural_end"]),
            natural_text_sha256=str(raw["natural_text_sha256"]),
            low_quality_replacement=str(raw["low_quality_replacement"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceRecipeError(f"Recipe edit {index} is malformed") from error
    if not edit.edit_id:
        raise EvidenceRecipeError(f"Recipe edit {index} has an empty edit_id")
    if edit.start < 0 or edit.end <= edit.start:
        raise EvidenceRecipeError(f"Recipe edit {edit.edit_id} has invalid offsets")
    if edit.category not in SUPPORTED_CATEGORIES:
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has unsupported category {edit.category!r}"
        )
    if edit.importance not in {"supporting", "eligibility_critical"}:
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has invalid importance {edit.importance!r}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", edit.source_text_sha256):
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has an invalid source span digest"
        )
    if edit.natural_start < 0 or edit.natural_end <= edit.natural_start:
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has invalid natural-deletion offsets"
        )
    if edit.natural_start > edit.start or edit.natural_end < edit.end:
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} natural deletion must contain its core span"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", edit.natural_text_sha256):
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has an invalid natural span digest"
        )
    if not edit.low_quality_replacement.strip():
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has an empty low-quality replacement"
        )
    if len(edit.low_quality_replacement) > 160 or "\n" in edit.low_quality_replacement:
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} has an invalid low-quality replacement"
        )
    replacement = edit.low_quality_replacement.casefold()
    if not any(cue in replacement for cue in LOW_QUALITY_CUES):
        raise EvidenceRecipeError(
            f"Recipe edit {edit.edit_id} low-quality replacement lacks an "
            "uncertainty cue"
        )
    return edit


# Load only versioned, human-approved recipes into the runtime path.
def load_evidence_recipe(
    path: Path, *, require_approved: bool = True
) -> EvidenceRecipe:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceRecipeError(f"Could not load evidence recipe: {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise EvidenceRecipeError(
            "Evidence recipe must use schema_version 2; regenerate legacy drafts"
        )
    edits_raw = raw.get("edits")
    if not isinstance(edits_raw, list) or not edits_raw:
        raise EvidenceRecipeError("Evidence recipe must contain at least one edit")
    recipe = EvidenceRecipe(
        task=str(raw.get("task", "")),
        source_sha256=str(raw.get("source_sha256", "")),
        status=str(raw.get("status", "")),
        severity=str(raw.get("severity", "")),
        generator=cast(dict[str, object], raw.get("generator", {})),
        edits=tuple(_parse_edit(edit, index) for index, edit in enumerate(edits_raw)),
    )
    if not recipe.task:
        raise EvidenceRecipeError("Evidence recipe is missing its task name")
    if not re.fullmatch(r"[0-9a-f]{64}", recipe.source_sha256):
        raise EvidenceRecipeError("Evidence recipe has an invalid source digest")
    if recipe.status not in {"draft", "approved"}:
        raise EvidenceRecipeError(
            f"Evidence recipe has invalid status {recipe.status!r}"
        )
    if require_approved and recipe.status != "approved":
        raise EvidenceRecipeError(
            f"Evidence recipe for {recipe.task} is not approved: {recipe.status!r}"
        )
    if recipe.severity not in {"label-preserving", "evidence-insufficient"}:
        raise EvidenceRecipeError(
            f"Evidence recipe has invalid severity {recipe.severity!r}"
        )
    return recipe


# Normalize only whitespace artifacts introduced by span deletion.
def _clean_natural_missing(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    text = re.sub(r":\s*\.", ".", text)
    text = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", text)
    text = re.sub(r"([,;:]){2,}", r"\1", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"^\s*[,.;:]+\s*", "", text)
    return text.strip()


# Validate source anchors before producing a matched control or error input.
def _validate_source(text: str, source: bytes, recipe: EvidenceRecipe) -> None:
    if sha256_bytes(source) != recipe.source_sha256:
        raise EvidenceRecipeError(
            f"Source digest does not match approved recipe for {recipe.task}"
        )
    previous_end = -1
    previous_natural_end = -1
    for edit in sorted(recipe.edits, key=lambda value: value.start):
        if edit.end > len(text):
            raise EvidenceRecipeError(
                f"Recipe edit {edit.edit_id} exceeds source length"
            )
        if edit.start < previous_end:
            raise EvidenceRecipeError("Evidence recipe contains overlapping edits")
        selected = text[edit.start : edit.end].encode("utf-8")
        if sha256_bytes(selected) != edit.source_text_sha256:
            raise EvidenceRecipeError(
                f"Source span digest mismatch for recipe edit {edit.edit_id}"
            )
        selected_text = selected.decode("utf-8").casefold()
        if selected_text in edit.low_quality_replacement.casefold():
            raise EvidenceRecipeError(
                f"Low-quality replacement reveals source span for {edit.edit_id}"
            )
        if edit.natural_end > len(text):
            raise EvidenceRecipeError(
                f"Recipe edit {edit.edit_id} natural deletion exceeds source length"
            )
        if edit.natural_start < previous_natural_end:
            raise EvidenceRecipeError(
                "Evidence recipe contains overlapping natural-deletion spans"
            )
        natural_selected = text[edit.natural_start : edit.natural_end].encode("utf-8")
        if sha256_bytes(natural_selected) != edit.natural_text_sha256:
            raise EvidenceRecipeError(
                f"Natural source span digest mismatch for recipe edit {edit.edit_id}"
            )
        previous_end = edit.end
        previous_natural_end = edit.natural_end


# Apply frozen spans from right to left so source offsets remain stable.
def apply_evidence_recipe(
    source: bytes,
    recipe: EvidenceRecipe,
    *,
    scenario: str,
    condition: str,
) -> tuple[bytes, dict[str, object]]:
    if scenario not in EVIDENCE_SCENARIOS:
        raise EvidenceRecipeError(f"Unsupported evidence scenario: {scenario}")
    if condition not in {"control", "error"}:
        raise EvidenceRecipeError(f"Unsupported evidence condition: {condition}")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceRecipeError("Evidence degradation requires UTF-8 text") from error
    _validate_source(text, source, recipe)

    # Preserve byte-identical control inputs after validating the same recipe.
    if condition == "control":
        return source, {
            "recipe_status": recipe.status,
            "severity": recipe.severity,
            "edit_count": len(recipe.edits),
            "edited_categories": sorted({edit.category for edit in recipe.edits}),
        }

    # Select deterministic category markers or contextual frozen replacements.
    if scenario == "evidence-explicit-missing":
        replacements = EXPLICIT_REPLACEMENTS
    elif scenario == "evidence-natural-missing":
        replacements = {category: "" for category in SUPPORTED_CATEGORIES}
    else:
        replacements = None

    # Apply the approved source-anchored replacements without model calls.
    degraded = text
    if scenario == "evidence-natural-missing":
        positions = [
            (edit.natural_start, edit.natural_end, edit) for edit in recipe.edits
        ]
    else:
        positions = [(edit.start, edit.end, edit) for edit in recipe.edits]
    for start, end, edit in sorted(
        positions,
        key=lambda value: value[0],
        reverse=True,
    ):
        replacement = (
            edit.low_quality_replacement
            if replacements is None
            else replacements[edit.category]
        )
        degraded = degraded[:start] + replacement + degraded[end:]
    if scenario == "evidence-natural-missing":
        degraded = _clean_natural_missing(degraded)
    output = degraded.encode("utf-8")
    if output == source:
        raise EvidenceRecipeError("Evidence error condition did not change the input")
    return output, {
        "recipe_status": recipe.status,
        "severity": recipe.severity,
        "edit_count": len(recipe.edits),
        "edited_categories": sorted({edit.category for edit in recipe.edits}),
        "edit_ids": [edit.edit_id for edit in recipe.edits],
    }

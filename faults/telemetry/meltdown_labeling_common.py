"""Shared utilities for the Agent Meltdown trace-labeling pipeline."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

# Pin the paper version because its appendix prompts are part of the protocol.
PAPER_ID = "2605.19149v1"
PAPER_URL = f"https://arxiv.org/html/{PAPER_ID}"
TAXONOMY_VERSION = f"agent-meltdown-arxiv-{PAPER_ID}"

# Remove bookkeeping fields while preserving decisions, actions, and observations.
BOILERPLATE_KEYS = {
    "cached_tokens",
    "cost_usd",
    "created_at",
    "duration_ms",
    "latency_ms",
    "model_provider",
    "prompt_tokens",
    "response_id",
    "token_usage",
    "total_tokens",
    "updated_at",
    "usage",
}

# Match large base64 data URIs even when they are nested inside serialized strings.
DATA_URI_PATTERN = re.compile(
    r"data:([A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=_-]{128,})"
)


@dataclass(frozen=True)
class TraceRecord:
    """Identify one Harbor trial trace and its stable analysis identifier."""

    trace_id: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class PaperMaterials:
    """Hold the three published prompts and the frozen paper taxonomy."""

    discovery_prompt: str
    aggregation_prompt: str
    annotation_prompt: str
    frozen_taxonomy: str
    paper_url: str


@dataclass(frozen=True)
class ModelResult:
    """Keep model output together with reproducibility metadata."""

    text: str
    metadata: dict[str, Any]


# Compute stable hashes for prompts, cleaned traces, and output provenance.
def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# Write text through a sibling temporary file to avoid partial research artifacts.
def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


# Serialize JSON deterministically so reruns produce reviewable diffs.
def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


# Convert arbitrary task names into filesystem-safe, collision-resistant IDs.
def safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if len(cleaned) <= 180:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:160]}__{digest}"


# Discover one preferred trace per Harbor trial without double-counting codex.txt.
def discover_traces(jobs_root: Path) -> list[TraceRecord]:
    jobs_root = jobs_root.resolve()
    preferred = sorted(jobs_root.rglob("agent/trajectory.json"))
    selected = {path.parent.resolve(): path for path in preferred}
    for codex_path in sorted(jobs_root.rglob("agent/codex.txt")):
        selected.setdefault(codex_path.parent.resolve(), codex_path)

    records: list[TraceRecord] = []
    for path in sorted(selected.values()):
        relative = path.relative_to(jobs_root)
        parts = relative.parts
        trial_name = parts[-3] if len(parts) >= 3 else path.parent.name
        job_name = parts[-4] if len(parts) >= 4 else "job"
        trace_id = safe_identifier(f"{job_name}__{trial_name}")
        records.append(
            TraceRecord(
                trace_id=trace_id,
                path=path,
                relative_path=str(relative),
            )
        )
    return records


# Replace embedded binary payloads while retaining their media type and size.
def _clean_string(value: str) -> str:
    def replace_data_uri(match: re.Match[str]) -> str:
        media_type = match.group(1)
        payload_size = len(match.group(2))
        return (
            f"<embedded {media_type} base64 payload omitted; "
            f"{payload_size} characters>"
        )

    without_payloads = DATA_URI_PATTERN.sub(replace_data_uri, value)
    return without_payloads.replace("\x00", "<NUL>")


# Recursively remove telemetry-only fields from structured Harbor traces.
def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in BOILERPLATE_KEYS:
                continue
            cleaned[str(key)] = _clean_json(item)
        return cleaned
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, str):
        return _clean_string(value)
    return value


# Normalize JSON, JSONL, or plain-text traces while keeping exact action evidence.
def clean_trace(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(_clean_json(parsed), ensure_ascii=False, indent=2)

    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            cleaned_lines.append(_clean_string(line))
        else:
            cleaned_lines.append(
                json.dumps(_clean_json(parsed), ensure_ascii=False, sort_keys=True)
            )
    return "\n".join(cleaned_lines)


# Optionally bound very long traces while preserving both failure onset and ending.
def clip_trace(trace: str, maximum_characters: int) -> str:
    if maximum_characters <= 0 or len(trace) <= maximum_characters:
        return trace
    head_size = int(maximum_characters * 0.60)
    tail_size = maximum_characters - head_size
    omitted = len(trace) - maximum_characters
    marker = f"\n\n<TRACE_MIDDLE_OMITTED characters={omitted}>\n\n"
    return trace[:head_size] + marker + trace[-tail_size:]


# Download or copy the immutable paper HTML used to recover the published prompts.
def _load_paper_html(cache_directory: Path, paper_html: Path | None) -> str:
    cache_directory.mkdir(parents=True, exist_ok=True)
    cached_path = cache_directory / f"{PAPER_ID}.html"
    if paper_html is not None:
        html = paper_html.read_text(encoding="utf-8")
        atomic_write_text(cached_path, html)
        return html
    if cached_path.is_file():
        return cached_path.read_text(encoding="utf-8")

    response = requests.get(
        PAPER_URL,
        headers={"User-Agent": "HealthAgentBench-research-reproduction/1.0"},
        timeout=60,
    )
    response.raise_for_status()
    atomic_write_text(cached_path, response.text)
    return response.text


# Find a paper section heading by stable visible text rather than fragile HTML IDs.
def _find_heading(soup: BeautifulSoup, text: str, level: str) -> Tag:
    normalized_target = " ".join(text.split()).casefold()
    for heading in soup.find_all(level):
        normalized = " ".join(heading.get_text(" ", strip=True).split()).casefold()
        if normalized_target in normalized:
            return heading
    raise RuntimeError(f"Could not find {level} heading containing {text!r}")


# Extract all visible text until the next heading at the same or higher level.
def _extract_section(heading: Tag, stop_levels: set[str]) -> str:
    chunks: list[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name in stop_levels:
            break
        if not isinstance(sibling, Tag):
            continue
        text = sibling.get_text("\n", strip=True)
        if not text or text == "⬇":
            continue
        chunks.append(text)
    if not chunks:
        raise RuntimeError(f"Extracted an empty section after {heading.get_text()!r}")
    return "\n\n".join(chunks).strip()


# Decode the paper's downloadable text listing to preserve its prompt byte-for-byte.
def _extract_prompt_listing(heading: Tag) -> str:
    section = heading.find_parent("section")
    if section is None:
        raise RuntimeError(f"Prompt section not found for {heading.get_text()!r}")
    link = section.select_one(".ltx_listing_data a[href^='data:text/plain;base64,']")
    if not isinstance(link, Tag):
        raise RuntimeError(f"Downloadable prompt not found for {heading.get_text()!r}")
    href = link.get("href")
    if not isinstance(href, str):
        raise RuntimeError(f"Prompt download link is malformed for {heading.get_text()!r}")
    encoded = href.split(",", maxsplit=1)[1]
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8").strip()
    except (binascii.Error, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Prompt download could not be decoded for {heading.get_text()!r}"
        ) from error


# Extract and cache the exact Appendix C prompts and Appendix B frozen taxonomy.
def load_paper_materials(
    cache_directory: Path,
    *,
    paper_html: Path | None = None,
) -> PaperMaterials:
    html = _load_paper_html(cache_directory, paper_html)
    soup = BeautifulSoup(html, "html.parser")
    discovery = _extract_prompt_listing(
        _find_heading(soup, "Behavior discovery prompt", "h3")
    )
    aggregation = _extract_prompt_listing(
        _find_heading(soup, "Category aggregation prompt", "h3")
    )
    annotation = _extract_prompt_listing(
        _find_heading(soup, "Behavior Annotation Prompt", "h3")
    )
    taxonomy = _extract_section(
        _find_heading(soup, "Full Behavior Taxonomy", "h2"), {"h2"}
    )

    # Fail closed if arXiv changes the rendered appendix or placeholders disappear.
    if "<TRACE>" not in discovery:
        raise RuntimeError("Published discovery prompt no longer contains <TRACE>")
    if "<dir>" not in aggregation:
        raise RuntimeError("Published aggregation prompt no longer contains <dir>")
    if "<TRACE>" not in annotation or "<TAXONOMY>" not in annotation:
        raise RuntimeError("Published annotation prompt placeholders were not found")

    materials = {
        "discovery_prompt.txt": discovery,
        "aggregation_prompt.txt": aggregation,
        "annotation_prompt.txt": annotation,
        "paper_frozen_taxonomy.md": taxonomy,
    }
    for filename, content in materials.items():
        atomic_write_text(cache_directory / filename, content + "\n")
    atomic_write_json(
        cache_directory / "materials_manifest.json",
        {
            "paper_id": PAPER_ID,
            "paper_url": PAPER_URL,
            "sha256": {
                filename: sha256_text(content) for filename, content in materials.items()
            },
        },
    )
    return PaperMaterials(
        discovery_prompt=discovery,
        aggregation_prompt=aggregation,
        annotation_prompt=annotation,
        frozen_taxonomy=taxonomy,
        paper_url=PAPER_URL,
    )


# Call the Responses API with the paper's model and reasoning-effort settings.
def call_model(
    *,
    prompt: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    max_retries: int = 3,
) -> ModelResult:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the project dependencies before labeling") from error

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        timeout=600.0,
    )
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=prompt,
                max_output_tokens=max_output_tokens,
                store=False,
            )
            usage = response.usage.model_dump() if response.usage is not None else None
            return ModelResult(
                text=response.output_text,
                metadata={
                    "response_id": response.id,
                    "resolved_model": response.model,
                    "requested_model": model,
                    "reasoning_effort": reasoning_effort,
                    "usage": usage,
                },
            )
        except Exception as error:  # API error types vary across SDK releases.
            last_error = error
            if attempt == max_retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Model request failed after {max_retries} attempts") from last_error


# Parse JSON-only model output while tolerating accidental Markdown fences.
def parse_json_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:-1] if len(lines) >= 3 else lines
        stripped = "\n".join(lines)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Expected one top-level JSON object")
    return parsed


# Repeat the identical prompt only when the model fails to return valid JSON.
def call_json_model(
    *,
    prompt: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    max_parse_attempts: int = 3,
) -> tuple[dict[str, Any], ModelResult]:
    last_error: Exception | None = None
    for _ in range(max_parse_attempts):
        result = call_model(
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        try:
            return parse_json_output(result.text), result
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
    raise RuntimeError("Model output remained invalid JSON") from last_error


# Build a sidecar that makes every annotation auditable and reproducible.
def build_provenance(
    *,
    prompt: str,
    trace: str | None,
    model_result: ModelResult,
    source: dict[str, Any],
) -> dict[str, Any]:
    provenance = {
        "paper_id": PAPER_ID,
        "paper_url": PAPER_URL,
        "prompt_sha256": sha256_text(prompt),
        "source": source,
        "model": model_result.metadata,
    }
    if trace is not None:
        provenance["cleaned_trace_sha256"] = sha256_text(trace)
        provenance["cleaned_trace_characters"] = len(trace)
    return provenance

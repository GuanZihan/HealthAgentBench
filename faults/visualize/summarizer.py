"""GPT-backed structured rollout summarization and cache management."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from faults.visualize.loader import TrialBundle, model_input
from faults.visualize.models import AnalysisDocument, RolloutNarrative


DEFAULT_MODEL = "openai/gpt-5.6-luna"
PROMPT_PATH = Path(__file__).with_name("prompt.md")


def api_model_name(model: str) -> str:
    """Translate Harbor's provider/model notation to an OpenAI model ID."""

    return model.removeprefix("openai/")


def cache_path(bundle: TrialBundle, model: str = DEFAULT_MODEL) -> Path:
    """Return the per-trial path for one model's analysis."""

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-").lower()
    return bundle.path / "visualization" / f"analysis-{slug}.json"


def load_cached_analysis(
    bundle: TrialBundle,
    model: str = DEFAULT_MODEL,
) -> AnalysisDocument | None:
    """Load a cache only when it was generated from the current artifacts."""

    path = cache_path(bundle, model)
    if not path.is_file():
        return None
    try:
        document = AnalysisDocument.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if document.source_fingerprint != bundle.fingerprint:
        return None
    if document.model != model:
        return None
    return document


def save_analysis(document: AnalysisDocument, path: Path) -> None:
    """Atomically persist a generated analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def analyze_with_openai(
    bundle: TrialBundle,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "medium",
    api_key: str | None = None,
    base_url: str | None = None,
    max_input_chars: int = 180_000,
    client: Any | None = None,
) -> AnalysisDocument:
    """Generate a schema-validated narrative with the Responses API."""

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if client is None and not key:
        raise RuntimeError("OPENAI_API_KEY is required to generate an AI analysis")
    if client is None:
        client_kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": 180.0,
            "max_retries": 2,
        }
        selected_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if selected_base_url:
            client_kwargs["base_url"] = selected_base_url
        client = OpenAI(**client_kwargs)

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    response = client.responses.parse(
        model=api_model_name(model),
        reasoning={"effort": reasoning_effort},
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": model_input(bundle, max_chars=max_input_chars)},
        ],
        text_format=RolloutNarrative,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(
            "The model response did not contain a parsed rollout analysis"
        )

    usage: dict[str, int] = {}
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        for source, target in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(raw_usage, source, None)
            if isinstance(value, int):
                usage[target] = value

    document = AnalysisDocument(
        generated_at=datetime.now(timezone.utc),
        generator="openai",
        model=model,
        source_fingerprint=bundle.fingerprint,
        response_id=getattr(response, "id", None),
        usage=usage,
        analysis=parsed,
    )
    _validate_blocks(document, bundle)
    return document


def _validate_blocks(document: AnalysisDocument, bundle: TrialBundle) -> None:
    """Reject structurally valid but trajectory-inconsistent model output."""

    valid_ids = {step.step_id for step in bundle.steps}
    previous_end = -1
    for block in document.analysis.blocks:
        if block.start_step > block.end_step:
            raise ValueError(f"Invalid range in {block.block_id}")
        if block.start_step <= previous_end:
            raise ValueError("Behavior blocks are not chronological")
        if block.start_step not in valid_ids or block.end_step not in valid_ids:
            raise ValueError(f"Unknown step range in {block.block_id}")
        if any(step_id not in valid_ids for step_id in block.evidence_step_ids):
            raise ValueError(f"Unknown evidence step in {block.block_id}")
        previous_end = block.end_step


def generate_and_cache(
    bundle: TrialBundle,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "medium",
    force: bool = False,
    **kwargs: Any,
) -> tuple[AnalysisDocument, Path]:
    """Reuse a valid cache or generate and save a new model analysis."""

    if not force:
        cached = load_cached_analysis(bundle, model)
        if cached is not None:
            return cached, cache_path(bundle, model)
    document = analyze_with_openai(
        bundle,
        model=model,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )
    path = cache_path(bundle, model)
    save_analysis(document, path)
    return document, path


def analysis_as_download(document: AnalysisDocument) -> str:
    """Serialize an analysis consistently for Streamlit downloads."""

    return json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)

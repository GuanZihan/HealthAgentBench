"""Generate auditable clinical-trial evidence recipes with a preparation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .recipes import LOW_QUALITY_CUES, SUPPORTED_CATEGORIES, sha256_bytes

# Resolve all paths relative to the benchmark checkout rather than the caller.
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPICS = (
    BENCHMARK_ROOT / "assets/clinical_trial_matching/assets/raw_cache/topics2021.xml"
)
DEFAULT_RECIPES = Path(__file__).with_name("recipes")
DEFAULT_GENERATED = Path(__file__).with_name("generated")
DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "select_clinical_evidence.md"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.5-2026-04-23"
MAX_NATURAL_SPAN_CHARS = 160


class RecipePreparationError(RuntimeError):
    """Raised when model output cannot become a safe deterministic recipe."""


# Reject pasted shell commands and control characters before building an auth header.
def _api_key_validation_error(value: str) -> str | None:
    if value == "EMPTY":
        return "OPENAI_API_KEY is not set"
    if len(value) < 20:
        return "OPENAI_API_KEY is unexpectedly short"
    if value != value.strip() or any(ord(character) < 33 for character in value):
        return "OPENAI_API_KEY contains whitespace or control characters"
    if value.startswith("export ") or value.startswith("OPENAI_API_KEY="):
        return "OPENAI_API_KEY contains a shell command instead of only the key"
    return None


# Hash the exact prompt and source inputs used by the offline generator.
def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# Resolve one model span conservatively while tolerating letter-case drift only.
def _resolve_unique_span(source: str, proposed: str, *, label: str) -> tuple[int, str]:
    if not proposed:
        raise RecipePreparationError(f"Every {label} span must be nonempty")
    exact_matches = list(re.finditer(re.escape(proposed), source))
    if len(exact_matches) == 1:
        match = exact_matches[0]
        return match.start(), match.group(0)
    if len(exact_matches) > 1:
        raise RecipePreparationError(
            f"The {label} span occurs more than once in its source context"
        )

    # Accept one case-only correction but reject all other fuzzy matching.
    case_matches = list(re.finditer(re.escape(proposed), source, flags=re.IGNORECASE))
    if len(case_matches) == 1:
        match = case_matches[0]
        return match.start(), match.group(0)
    if len(case_matches) > 1:
        raise RecipePreparationError(
            f"The {label} span has more than one case-insensitive source match"
        )
    raise RecipePreparationError(
        f"The {label} span was not found verbatim or by a case-only correction"
    )


# Parse topics with the same whitespace normalization used by task bootstrap.
def load_topics(path: Path) -> dict[int, str]:
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError) as error:
        raise RecipePreparationError(f"Could not parse topics file: {path}") from error
    topics: dict[int, str] = {}
    for element in tree.findall("topic"):
        number = int(element.attrib["number"])
        topics[number] = " ".join((element.text or "").split())
    if not topics:
        raise RecipePreparationError(f"Topics file contains no topics: {path}")
    return topics


# Discover the benchmark task-to-topic mapping from checked-in integer IDs.
def discover_tasks(root: Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for path in sorted(root.glob("tasks/clinical_trial_matching_task_*")):
        topic_path = path / "environment/workspace/topic_id.txt"
        if topic_path.is_file():
            tasks.append((path.name, int(topic_path.read_text().strip())))
    if not tasks:
        raise RecipePreparationError("No clinical-trial matching tasks were found")
    return tasks


# Remove optional Markdown fencing before strict JSON parsing.
def _parse_model_json(value: str) -> dict[str, Any]:
    candidate = value.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise RecipePreparationError(
            "Preparation model returned invalid JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise RecipePreparationError("Preparation-model output must be a JSON object")
    return parsed


# Define one strict API schema instead of asking GPT-5.5 to infer JSON shape.
def _selection_schema(maximum: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "minItems": 1,
                "maxItems": maximum,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "natural_text": {"type": "string"},
                        "low_quality_text": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_CATEGORIES),
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["supporting", "eligibility_critical"],
                        },
                    },
                    "required": [
                        "text",
                        "natural_text",
                        "low_quality_text",
                        "category",
                        "importance",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["selections"],
        "additionalProperties": False,
    }


# Reject broad deletions that would remove unrelated eligibility evidence.
def _validate_natural_scope(
    note: str,
    *,
    natural_start: int,
    natural_text: str,
    selected_text: str,
    category: str,
) -> None:
    maximum = min(
        MAX_NATURAL_SPAN_CHARS,
        max(72, 4 * len(selected_text)),
    )
    if len(natural_text) > maximum:
        raise RecipePreparationError(
            f"Natural-deletion span is too broad ({len(natural_text)} > {maximum} "
            "characters); select only the smallest removable phrase or clause"
        )
    if len(natural_text) > 72 and len(natural_text) / max(len(note), 1) > 0.25:
        raise RecipePreparationError(
            "Natural-deletion span removes more than 25% of the patient note"
        )
    sentence_boundaries = re.findall(r"[.!?](?:\s+|$)", natural_text)
    if len(sentence_boundaries) > 1:
        raise RecipePreparationError(
            "Natural-deletion span crosses more than one sentence boundary"
        )
    demographic = re.search(
        r"\b(?:\d{1,3}[- ]?(?:year|yo)|man|woman|girl|boy|gentleman|male|female)\b",
        natural_text,
        flags=re.IGNORECASE,
    )
    if natural_start == 0 and category != "demographic" and demographic:
        raise RecipePreparationError(
            "Natural-deletion span removes introductory demographic evidence that "
            "is outside the selected category"
        )
    if category == "treatment" and re.fullmatch(
        r"\s*\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?)\s*",
        selected_text,
        flags=re.IGNORECASE,
    ):
        raise RecipePreparationError(
            "A duration-only value must use temporal_history, not treatment"
        )


# Require contextual uncertainty text without leaking the selected medical value.
def _validate_low_quality_text(selected_text: str, replacement: str) -> str:
    value = replacement.strip()
    if not value or len(value) > 160 or "\n" in value:
        raise RecipePreparationError(
            "low_quality_text must be one nonempty line of at most 160 characters"
        )
    lowered = value.casefold()
    if selected_text.casefold() in lowered:
        raise RecipePreparationError(
            "low_quality_text must not reveal the selected source value"
        )
    if not any(cue in lowered for cue in LOW_QUALITY_CUES):
        raise RecipePreparationError(
            "low_quality_text must state that evidence is reported, unavailable, "
            "incomplete, uncertain, or unverified"
        )
    return value


# Reject overlapping core or natural-deletion spans after source resolution.
def _validate_edit_overlaps(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(edits, key=lambda edit: int(edit["start"]))
    for left, right in zip(ordered, ordered[1:], strict=False):
        if int(left["end"]) > int(right["start"]):
            raise RecipePreparationError(
                "Preparation model selected overlapping evidence spans"
            )
        if int(left["natural_end"]) > int(right["natural_start"]):
            raise RecipePreparationError(
                "Preparation model selected overlapping natural-deletion spans"
            )
    return ordered


# Convert verbatim model selections into uniquely source-anchored hashed edits.
def build_edits(
    note: str, response: dict[str, Any], *, maximum: int
) -> list[dict[str, Any]]:
    selections = response.get("selections")
    if not isinstance(selections, list) or not selections:
        raise RecipePreparationError(
            "Preparation model returned no evidence selections"
        )
    edits: list[dict[str, Any]] = []
    for index, raw in enumerate(selections[:maximum], start=1):
        if not isinstance(raw, dict):
            raise RecipePreparationError(
                "Preparation-model selection must be a JSON object"
            )
        selected = str(raw.get("text", ""))
        natural_selected = str(raw.get("natural_text", ""))
        low_quality_selected = str(raw.get("low_quality_text", ""))
        category = str(raw.get("category", ""))
        importance = str(raw.get("importance", ""))
        try:
            natural_start, source_natural_text = _resolve_unique_span(
                note,
                natural_selected,
                label="natural-deletion",
            )
            relative_start, source_selected_text = _resolve_unique_span(
                source_natural_text,
                selected,
                label="core evidence",
            )
        except RecipePreparationError as error:
            raise RecipePreparationError(
                f"Selection {index} is invalid: {error}; "
                f"text={selected!r}; natural_text={natural_selected!r}"
            ) from error
        if category not in SUPPORTED_CATEGORIES:
            raise RecipePreparationError(f"Unsupported evidence category: {category!r}")
        if importance not in {"supporting", "eligibility_critical"}:
            raise RecipePreparationError(f"Unsupported importance: {importance!r}")
        _validate_natural_scope(
            note,
            natural_start=natural_start,
            natural_text=source_natural_text,
            selected_text=source_selected_text,
            category=category,
        )
        low_quality_replacement = _validate_low_quality_text(
            source_selected_text,
            low_quality_selected,
        )
        # Resolve a repeated short value relative to its unique natural-language anchor.
        start = natural_start + relative_start
        edits.append(
            {
                "edit_id": f"e{index:02d}",
                "start": start,
                "end": start + len(source_selected_text),
                "category": category,
                "importance": importance,
                "source_text_sha256": _sha256_text(source_selected_text),
                "natural_start": natural_start,
                "natural_end": natural_start + len(source_natural_text),
                "natural_text_sha256": _sha256_text(source_natural_text),
                "low_quality_replacement": low_quality_replacement,
            }
        )
    return _validate_edit_overlaps(edits)


# Salvage only independently valid non-overlapping selections after retries fail.
def build_partial_edits(
    note: str,
    response: dict[str, Any],
    *,
    maximum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selections = response.get("selections")
    if not isinstance(selections, list):
        return [], [{"selection": None, "validation_error": "Not a selection list"}]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_index, raw in enumerate(selections[:maximum], start=1):
        try:
            candidate = build_edits(
                note,
                {"selections": [raw]},
                maximum=1,
            )[0]
            candidate["edit_id"] = f"e{len(accepted) + 1:02d}"
            _validate_edit_overlaps([*accepted, candidate])
        except RecipePreparationError as error:
            rejected.append(
                {
                    "selection": source_index,
                    "model_selection": raw,
                    "validation_error": str(error),
                }
            )
            continue
        accepted.append(candidate)
    return _validate_edit_overlaps(accepted), rejected


# Ask the local OpenAI-compatible vLLM endpoint for spans, never rewritten prose.
def select_evidence(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    task: str,
    note: str,
    maximum: int,
    api_mode: str,
    reasoning_effort: str,
    validation_feedback: str | None = None,
) -> tuple[str, dict[str, Any]]:
    request = (
        f"{prompt}\n\nTASK: {task}\nMAX_SELECTIONS: {maximum}\n"
        f"PATIENT_NOTE_BEGIN\n{note}\nPATIENT_NOTE_END"
    )
    # Give deterministic corrective feedback only after a rejected model response.
    if validation_feedback:
        request += (
            "\n\nPREVIOUS_OUTPUT_VALIDATION_ERROR:\n"
            f"{validation_feedback}\n"
            "Return corrected JSON rather than repeating the invalid selection. "
            "Every text value must be a literal substring of its own natural_text "
            "with the same word order. Use a longer unique natural_text when a "
            "short value repeats, or omit an invalid selection. Keep natural_text "
            "minimal and correct low_quality_text for the surrounding grammar."
        )

    # Use Responses structured output for GPT-5.5 and retain Chat Completions fallback.
    if api_mode == "responses":
        response = client.responses.create(
            model=model,
            input=request,
            reasoning={"effort": reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "clinical_evidence_recipe",
                    "schema": _selection_schema(maximum),
                    "strict": True,
                },
                "verbosity": "low",
            },
            max_output_tokens=2000,
            prompt_cache_key="healthagentbench-clinical-evidence-v2",
            store=False,
        )
        content = response.output_text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request}],
            temperature=0,
            seed=0,
        )
        content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RecipePreparationError(f"{model} returned an empty response")

    # Preserve request-level provenance without recording credentials or headers.
    usage = getattr(response, "usage", None)
    metadata = {
        "api_mode": api_mode,
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", model),
        "usage": usage.model_dump(mode="json") if usage is not None else None,
    }
    return content, metadata


# Persist only hashes and offsets in recipes; retain raw notes in ignored audit logs.
def prepare_recipe(
    *,
    client: OpenAI,
    model: str,
    model_revision: str,
    api_mode: str,
    reasoning_effort: str,
    prompt: str,
    task: str,
    topic_id: int,
    note: str,
    maximum: int,
    severity: str,
    recipe_directory: Path,
    generated_directory: Path,
    max_retries: int,
) -> Path:
    # Protect approved human-reviewed recipes before spending a model request.
    recipe_path = recipe_directory / f"{task}.json"
    if recipe_path.is_file():
        try:
            existing = json.loads(recipe_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RecipePreparationError(
                f"Existing recipe is invalid JSON: {recipe_path}"
            ) from error
        if isinstance(existing, dict) and existing.get("status") == "approved":
            raise RecipePreparationError(
                f"Refusing to overwrite approved recipe: {recipe_path}"
            )
        # Preserve a manually curated draft even when --overwrite-drafts is used.
        curation = existing.get("curation") if isinstance(existing, dict) else None
        if isinstance(curation, dict) and curation.get("locked") is True:
            raise RecipePreparationError(
                f"Refusing to overwrite locked curated recipe: {recipe_path}"
            )

    # Retry only semantically invalid model output and preserve every attempt locally.
    attempts: list[dict[str, Any]] = []
    validation_feedback: str | None = None
    edits: list[dict[str, Any]] | None = None
    raw_response = ""
    response_metadata: dict[str, Any] = {}
    last_parsed: dict[str, Any] | None = None
    partial_salvage: dict[str, Any] | None = None
    for attempt_number in range(1, max_retries + 2):
        raw_response = ""
        response_metadata = {}
        try:
            raw_response, response_metadata = select_evidence(
                client,
                model=model,
                prompt=prompt,
                task=task,
                note=note,
                maximum=maximum,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                validation_feedback=validation_feedback,
            )
            parsed = _parse_model_json(raw_response)
            last_parsed = parsed
            edits = build_edits(note, parsed, maximum=maximum)
        except RecipePreparationError as error:
            validation_feedback = str(error)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "model_response": raw_response or None,
                    "response_metadata": response_metadata or None,
                    "validation_error": validation_feedback,
                }
            )
            continue
        attempts.append(
            {
                "attempt": attempt_number,
                "model_response": raw_response,
                "response_metadata": response_metadata,
                "validation_error": None,
            }
        )
        break

    # Keep valid selections after retries rather than fuzzily repairing invalid text.
    if edits is None and last_parsed is not None:
        accepted, rejected = build_partial_edits(
            note,
            last_parsed,
            maximum=maximum,
        )
        if accepted:
            edits = accepted
            partial_salvage = {
                "accepted_selection_count": len(accepted),
                "rejected_selection_count": len(rejected),
                "rejected_selections": rejected,
            }
            attempts[-1]["partial_salvage"] = partial_salvage

    # Persist terminal validation failures without creating an unusable recipe.
    if edits is None:
        recorded_at = datetime.now(timezone.utc).isoformat()
        generated_directory.mkdir(parents=True, exist_ok=True)
        failure_path = generated_directory / f"{task}.failed.json"
        failure_path.write_text(
            json.dumps(
                {
                    "task": task,
                    "topic_id": topic_id,
                    "recorded_at": recorded_at,
                    "patient_note": note,
                    "status": "failed",
                    "attempts": attempts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RecipePreparationError(
            f"{model} output remained invalid after {len(attempts)} attempts; "
            f"audit written to {failure_path}: {validation_feedback}"
        )

    # Record the successful deterministic recipe and its exact generator provenance.
    recorded_at = datetime.now(timezone.utc).isoformat()
    recipe = {
        "schema_version": 2,
        "task": task,
        "topic_id": topic_id,
        "status": "draft",
        "severity": severity,
        "source_sha256": sha256_bytes(note.encode("utf-8")),
        "generator": {
            "model": model,
            "model_revision": model_revision,
            "api_mode": api_mode,
            "reasoning_effort": reasoning_effort,
            "store": False if api_mode == "responses" else None,
            "temperature": 0 if api_mode == "chat-completions" else None,
            "seed": 0 if api_mode == "chat-completions" else None,
            "prompt_sha256": _sha256_text(prompt),
            "recorded_at": recorded_at,
            "attempt_count": len(attempts),
            "partial_salvage": partial_salvage is not None,
            "rejected_selection_count": (
                int(partial_salvage["rejected_selection_count"])
                if partial_salvage
                else 0
            ),
        },
        "edits": edits,
    }
    recipe_directory.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Keep the model response and source note local for reproducibility review.
    generated_directory.mkdir(parents=True, exist_ok=True)
    audit_path = generated_directory / f"{task}.json"
    audit_path.write_text(
        json.dumps(
            {
                "task": task,
                "topic_id": topic_id,
                "recorded_at": recorded_at,
                "patient_note": note,
                "model_response": raw_response,
                "attempts": attempts,
                "recipe_path": str(recipe_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return recipe_path


# Expose one reproducible command for generating all nine clinical recipes.
def build_parser() -> argparse.ArgumentParser:
    default_mode = os.getenv("EVIDENCE_API_MODE", "responses")
    default_base_url = os.getenv("EVIDENCE_BASE_URL")
    if not default_base_url and default_mode == "chat-completions":
        default_base_url = os.getenv("EVIDENCE_VLLM_BASE_URL")
    default_api_key = os.getenv("EVIDENCE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not default_api_key and default_mode == "chat-completions":
        default_api_key = os.getenv("EVIDENCE_VLLM_API_KEY", "EMPTY")
    default_revision = os.getenv("EVIDENCE_MODEL_REVISION")
    if not default_revision and default_mode == "chat-completions":
        default_revision = os.getenv("MODEL_REVISION")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=default_base_url or DEFAULT_OPENAI_BASE_URL,
    )
    parser.add_argument("--api-key", default=default_api_key or "EMPTY")
    parser.add_argument(
        "--model", default=os.getenv("EVIDENCE_MODEL", DEFAULT_OPENAI_MODEL)
    )
    parser.add_argument("--model-revision", default=default_revision)
    parser.add_argument(
        "--api-mode",
        choices=("responses", "chat-completions"),
        default=default_mode,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=os.getenv("EVIDENCE_REASONING_EFFORT", "medium"),
    )
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--maximum-selections", type=int, default=2)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of deterministic correction attempts after the first response",
    )
    parser.add_argument(
        "--overwrite-drafts",
        action="store_true",
        help=(
            "Regenerate unlocked draft recipes; approved and manually locked "
            "recipes remain protected"
        ),
    )
    parser.add_argument(
        "--severity",
        choices=("label-preserving", "evidence-insufficient"),
        default="evidence-insufficient",
    )
    return parser


# Generate selected task recipes and report their approval state explicitly.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.maximum_selections < 1:
        raise SystemExit("--maximum-selections must be positive")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be nonnegative")
    if args.api_mode == "responses" and (
        args.base_url.rstrip("/") == DEFAULT_OPENAI_BASE_URL
    ):
        key_error = _api_key_validation_error(args.api_key)
        if key_error:
            raise SystemExit(
                f"{key_error}. Set the variable to the key value only; do not "
                "include export, quotes, or another shell command."
            )
    model_revision = args.model_revision or args.model
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    topics = load_topics(args.topics)
    tasks = discover_tasks(BENCHMARK_ROOT)
    requested = set(args.tasks or [])
    unknown = requested.difference(task for task, _ in tasks)
    if unknown:
        raise SystemExit(f"Unknown clinical tasks: {', '.join(sorted(unknown))}")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    for task, topic_id in tasks:
        if requested and task not in requested:
            continue
        if topic_id not in topics:
            raise RecipePreparationError(f"Topic {topic_id} is missing for {task}")
        # Preserve generated work by default when a valid draft or approval exists.
        existing_path = args.recipes / f"{task}.json"
        if existing_path.is_file() and not args.overwrite_drafts:
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RecipePreparationError(
                    f"Existing recipe is invalid JSON: {existing_path}"
                ) from error
            existing_status = (
                existing.get("status") if isinstance(existing, dict) else None
            )
            if existing_status not in {"draft", "approved"}:
                raise RecipePreparationError(
                    f"Existing recipe has invalid status: {existing_path}"
                )
            if existing.get("schema_version") != 2:
                raise RecipePreparationError(
                    f"Existing recipe uses legacy schema; regenerate it with "
                    f"--overwrite-drafts: {existing_path}"
                )
            print(f"[evidence] kept {existing_path} status={existing_status}")
            continue
        path = prepare_recipe(
            client=client,
            model=args.model,
            model_revision=model_revision,
            api_mode=args.api_mode,
            reasoning_effort=args.reasoning_effort,
            prompt=prompt,
            task=task,
            topic_id=topic_id,
            note=topics[topic_id],
            maximum=args.maximum_selections,
            severity=args.severity,
            recipe_directory=args.recipes,
            generated_directory=args.generated,
            max_retries=args.max_retries,
        )
        print(f"[evidence] wrote {path} status=draft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

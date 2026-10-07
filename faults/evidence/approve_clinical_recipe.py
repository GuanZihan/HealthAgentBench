"""Preview and explicitly approve one frozen clinical evidence recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prepare_clinical_recipes import (
    BENCHMARK_ROOT,
    DEFAULT_GENERATED,
    DEFAULT_RECIPES,
    DEFAULT_TOPICS,
    discover_tasks,
    load_topics,
)
from .recipes import EVIDENCE_SCENARIOS, apply_evidence_recipe, load_evidence_recipe


class RecipeApprovalError(RuntimeError):
    """Raised when a draft cannot pass deterministic approval checks."""


# Hash the ignored generation record so an approval identifies what was reviewed.
def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Load the editable JSON object separately from the strict runtime dataclass.
def _load_raw_recipe(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeApprovalError(f"Could not load recipe JSON: {path}") from error
    if not isinstance(value, dict):
        raise RecipeApprovalError("Recipe JSON must contain one object")
    return value


# Validate the ignored record against both the task and immutable source digest.
def _validate_audit_record(
    path: Path,
    *,
    task: str,
    topic_id: int,
    source: bytes,
) -> str:
    raw = _load_raw_recipe(path)
    if raw.get("task") != task or raw.get("topic_id") != topic_id:
        raise RecipeApprovalError("Generation record does not match the requested task")
    if str(raw.get("patient_note", "")).encode("utf-8") != source:
        raise RecipeApprovalError(
            "Generation record patient note does not match source"
        )
    return _sha256_path(path)


# Print exact selected spans and all three renderings before any status mutation.
def _print_review(task: str, source: bytes, recipe_path: Path) -> None:
    recipe = load_evidence_recipe(recipe_path, require_approved=False)
    text = source.decode("utf-8")
    print(f"[evidence-review] task={task} status={recipe.status}")
    print(
        f"[evidence-review] severity={recipe.severity} "
        f"source_sha256={recipe.source_sha256}"
    )
    for edit in recipe.edits:
        selected = text[edit.start : edit.end]
        print(
            f"[evidence-review] {edit.edit_id} category={edit.category} "
            f"importance={edit.importance} span={selected!r} "
            f"natural_span={text[edit.natural_start : edit.natural_end]!r} "
            f"low_quality_replacement={edit.low_quality_replacement!r}"
        )
    for scenario in EVIDENCE_SCENARIOS:
        output, _ = apply_evidence_recipe(
            source,
            recipe,
            scenario=scenario,
            condition="error",
        )
        print(f"\n--- {scenario} ---\n{output.decode('utf-8')}\n")


# Atomically change only the approval metadata after all checks and human review.
def _approve(
    path: Path,
    *,
    raw: dict[str, Any],
    reviewer: str,
    audit_sha256: str,
) -> None:
    raw["status"] = "approved"
    raw["approval"] = {
        "reviewer": reviewer,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "generation_record_sha256": audit_sha256,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# Require one named task so every approval remains an intentional review event.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--reviewer")
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    return parser


# Preview by default; write approval only with both reviewer and explicit confirmation.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_map = dict(discover_tasks(BENCHMARK_ROOT))
    if args.task not in task_map:
        raise SystemExit(f"Unknown clinical task: {args.task}")
    topic_id = task_map[args.task]
    topics = load_topics(args.topics)
    if topic_id not in topics:
        raise RecipeApprovalError(f"Topic {topic_id} is missing for {args.task}")
    source = topics[topic_id].encode("utf-8")
    recipe_path = args.recipes / f"{args.task}.json"
    audit_path = args.generated / f"{args.task}.json"
    raw = _load_raw_recipe(recipe_path)
    recipe = load_evidence_recipe(recipe_path, require_approved=False)
    if recipe.task != args.task or raw.get("topic_id") != topic_id:
        raise RecipeApprovalError("Recipe task or topic ID does not match benchmark")
    audit_sha256 = _validate_audit_record(
        audit_path,
        task=args.task,
        topic_id=topic_id,
        source=source,
    )
    _print_review(args.task, source, recipe_path)
    if not args.confirm_reviewed:
        print("[evidence-review] preview only; recipe status was not changed")
        return 0
    if recipe.status != "draft":
        raise SystemExit(
            f"Recipe must be draft before approval; observed {recipe.status!r}"
        )
    if not args.reviewer or not args.reviewer.strip():
        raise SystemExit("--reviewer is required with --confirm-reviewed")
    _approve(
        recipe_path,
        raw=raw,
        reviewer=args.reviewer.strip(),
        audit_sha256=audit_sha256,
    )
    print(f"[evidence-review] approved {recipe_path} reviewer={args.reviewer.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

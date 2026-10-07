"""Validate every approved clinical recipe before a long experiment batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from .prepare_clinical_recipes import (
    BENCHMARK_ROOT,
    DEFAULT_RECIPES,
    DEFAULT_TOPICS,
    discover_tasks,
    load_topics,
)
from .recipes import EVIDENCE_SCENARIOS, apply_evidence_recipe, load_evidence_recipe


# Accept alternate paths so tests and archived campaigns can validate snapshots.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument(
        "--allow-drafts",
        action="store_true",
        help="Validate draft structure and renderings without treating them as approved",
    )
    return parser


# Recheck approval, task identity, source hash, and every selected span anchor.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topics = load_topics(args.topics)
    tasks = discover_tasks(BENCHMARK_ROOT)
    failures: list[str] = []
    for task, topic_id in tasks:
        path = args.recipes / f"{task}.json"
        try:
            recipe = load_evidence_recipe(
                path,
                require_approved=not args.allow_drafts,
            )
            if recipe.task != task:
                raise ValueError(f"recipe task is {recipe.task!r}")
            source = topics[topic_id].encode("utf-8")
            for scenario in EVIDENCE_SCENARIOS:
                for condition in ("control", "error"):
                    apply_evidence_recipe(
                        source,
                        recipe,
                        scenario=scenario,
                        condition=condition,
                    )
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(f"{task}: {error}")
    if failures:
        detail = "\n  - ".join(failures)
        raise SystemExit(f"Clinical evidence recipe preflight failed:\n  - {detail}")
    mode = "draft-review" if args.allow_drafts else "approved-preflight"
    print(f"[evidence-preflight] mode={mode} recipes={len(tasks)} status=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

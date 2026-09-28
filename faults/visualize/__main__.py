"""Command-line entry point for precomputing rollout analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from faults.visualize.loader import load_trial
from faults.visualize.summarizer import DEFAULT_MODEL, generate_and_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed agent rollout")
    parser.add_argument(
        "trial", type=Path, help="Trial directory or single-trial job directory"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore an existing valid cache"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_trial(args.trial)
    document, path = generate_and_cache(
        bundle,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        force=args.force,
    )
    print(f"Analyzed {bundle.job_name}/{bundle.trial_name}")
    print(f"Generated {len(document.analysis.blocks)} behavior blocks")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

"""Stage 1: discover candidate harmful behaviors in each cleaned agent trace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from meltdown_labeling_common import (
    PAPER_ID,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    call_json_model,
    clean_trace,
    clip_trace,
    discover_traces,
    load_paper_materials,
    sha256_text,
)

# Match the behavior-discovery configuration reported in Section 4.3.
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "high"


# Expose only experiment-relevant controls and keep paper defaults visible.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Agent Meltdown behavior-discovery prompt over traces."
    )
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-html", type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get("DISCOVERY_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get(
            "DISCOVERY_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ),
    )
    parser.add_argument(
        "--max-trace-characters",
        type=int,
        default=int(os.environ.get("MAX_TRACE_CHARACTERS", "0")),
        help="Zero preserves the full cleaned trace; positive values clip the middle.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# Reject malformed discovery outputs before downstream taxonomy aggregation.
def validate_discovery(value: dict[str, Any], trace_id: str) -> None:
    required = {
        "trace_id",
        "task_summary",
        "rollout_summary",
        "errored_before_agent_logic",
        "has_candidate_behaviors",
        "candidate_behaviors",
        "overall_severity",
        "abstention_reason",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"{trace_id}: discovery output lacks {missing}")
    if not isinstance(value["candidate_behaviors"], list):
        raise ValueError(f"{trace_id}: candidate_behaviors must be a list")


# Run the published discovery prompt once per trace with resumable outputs.
def main() -> int:
    args = parse_args()
    jobs_root = args.jobs_root.resolve()
    output_root = args.output_root.resolve()
    paper_root = output_root / "paper"
    cleaned_root = output_root / "cleaned"
    discovery_root = output_root / "discovery"
    metadata_root = output_root / "metadata/discovery"
    materials = load_paper_materials(paper_root, paper_html=args.paper_html)

    records = discover_traces(jobs_root)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No trajectory.json or codex.txt traces found in {jobs_root}")

    # Rebuild one deterministic manifest instead of appending duplicate rerun rows.
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        cleaned = clip_trace(
            clean_trace(record.path),
            maximum_characters=args.max_trace_characters,
        )
        cleaned_path = cleaned_root / f"{record.trace_id}.txt"
        result_path = discovery_root / f"{record.trace_id}.json"
        metadata_path = metadata_root / f"{record.trace_id}.json"
        atomic_write_text(cleaned_path, cleaned + "\n")
        manifest.append(
            {
                "trace_id": record.trace_id,
                "source_path": record.relative_path,
                "cleaned_path": str(cleaned_path.relative_to(output_root)),
                "cleaned_sha256": sha256_text(cleaned),
                "cleaned_characters": len(cleaned),
                "discovery_path": str(result_path.relative_to(output_root)),
            }
        )

        # Resume expensive API work unless the researcher explicitly requests overwrite.
        if result_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(records)}] reuse {record.trace_id}")
            continue
        if args.dry_run:
            print(f"[{index}/{len(records)}] prepared {record.trace_id}")
            continue

        prompt = materials.discovery_prompt.replace("<TRACE>", cleaned)
        annotation, model_result = call_json_model(
            prompt=prompt,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=32_000,
        )
        validate_discovery(annotation, record.trace_id)

        # Keep source-derived trace identity authoritative across every stage.
        annotation["trace_id"] = record.trace_id
        atomic_write_json(result_path, annotation)
        atomic_write_json(
            metadata_path,
            {
                "provenance": build_provenance(
                    prompt=prompt,
                    trace=cleaned,
                    model_result=model_result,
                    source={
                        "trace_id": record.trace_id,
                        "relative_path": record.relative_path,
                    },
                ),
                "raw_model_output": model_result.text,
            },
        )
        print(f"[{index}/{len(records)}] labeled {record.trace_id}")

    # Record stage settings without persisting API credentials or base URLs.
    atomic_write_json(output_root / "trace_manifest.json", manifest)
    atomic_write_json(
        output_root / "stage_01_config.json",
        {
            "paper_id": PAPER_ID,
            "jobs_root": str(jobs_root),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_trace_characters": args.max_trace_characters,
            "trace_count": len(records),
            "dry_run": args.dry_run,
        },
    )
    print(f"Stage 1 complete: {len(records)} traces under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

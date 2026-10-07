"""Stage 2: aggregate discovered behaviors into a candidate taxonomy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from meltdown_labeling_common import (
    PAPER_ID,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    call_model,
    load_paper_materials,
    sha256_text,
)

# Match the category-aggregation configuration reported in Section 4.3.
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "medium"


# Keep aggregation inputs explicit because this stage can approach context limits.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Agent Meltdown behavior-discovery JSON files."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-html", type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get("AGGREGATION_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get(
            "AGGREGATION_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ),
    )
    parser.add_argument(
        "--max-input-characters",
        type=int,
        default=int(os.environ.get("MAX_AGGREGATION_CHARACTERS", "0")),
        help="Zero allows the full discovery corpus; a positive value fails if exceeded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# Serialize the directory in stable filename order for one model request.
def build_directory_snapshot(discovery_root: Path) -> tuple[str, list[Path]]:
    paths = sorted(discovery_root.glob("*.json"))
    if not paths:
        raise RuntimeError(f"No discovery JSON files found in {discovery_root}")
    chunks: list[str] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        chunks.append(
            f"===== {path.name} =====\n"
            + json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        )
    return "\n\n".join(chunks), paths


# Run the published aggregation prompt and preserve the paper taxonomy separately.
def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    paper_root = output_root / "paper"
    discovery_root = output_root / "discovery"
    taxonomy_root = output_root / "taxonomy"
    candidate_path = taxonomy_root / "candidate_taxonomy.md"
    metadata_path = output_root / "metadata/aggregation.json"
    materials = load_paper_materials(paper_root, paper_html=args.paper_html)
    snapshot, discovery_paths = build_directory_snapshot(discovery_root)
    if args.max_input_characters > 0 and len(snapshot) > args.max_input_characters:
        raise RuntimeError(
            f"Discovery snapshot has {len(snapshot)} characters, exceeding "
            f"--max-input-characters={args.max_input_characters}"
        )

    # The API has no local-directory tool, so instantiate <dir> with an inline snapshot.
    directory_payload = (
        f"{discovery_root}\n\n"
        "The directory contents are provided inline below, one JSON file per block.\n\n"
        f"{snapshot}"
    )
    prompt = materials.aggregation_prompt.replace("<dir>", directory_payload)
    atomic_write_text(taxonomy_root / "aggregation_input.txt", prompt + "\n")
    atomic_write_text(
        taxonomy_root / "paper_frozen_taxonomy.md",
        materials.frozen_taxonomy + "\n",
    )

    # Candidate taxonomy generation is resumable because it can be an expensive call.
    if candidate_path.is_file() and not args.overwrite:
        print(f"Stage 2 reuse: {candidate_path}")
    elif args.dry_run:
        print(f"Stage 2 prepared {len(discovery_paths)} discovery summaries")
    else:
        model_result = call_model(
            prompt=prompt,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=32_000,
        )
        atomic_write_text(candidate_path, model_result.text.strip() + "\n")
        atomic_write_json(
            metadata_path,
            {
                "provenance": build_provenance(
                    prompt=prompt,
                    trace=None,
                    model_result=model_result,
                    source={
                        "discovery_directory": str(discovery_root),
                        "discovery_count": len(discovery_paths),
                        "snapshot_sha256": sha256_text(snapshot),
                    },
                ),
                "raw_model_output": model_result.text,
            },
        )
        print(f"Stage 2 wrote candidate taxonomy: {candidate_path}")

    # Make the manual-review boundary explicit rather than silently treating output as final.
    atomic_write_json(
        output_root / "stage_02_config.json",
        {
            "paper_id": PAPER_ID,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "discovery_count": len(discovery_paths),
            "snapshot_characters": len(snapshot),
            "candidate_requires_human_review": True,
            "final_annotation_default": "paper_frozen_taxonomy",
            "dry_run": args.dry_run,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

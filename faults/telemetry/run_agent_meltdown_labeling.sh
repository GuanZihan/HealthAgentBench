#!/bin/bash
set -euo pipefail

# Resolve repository paths independently of the caller's current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOBS_ROOT="${1:-${JOBS_ROOT:-$BENCHMARK_ROOT/jobs/gpt6-luna-rep1}}"
RUN_NAME="${RUN_NAME:-$(basename "$JOBS_ROOT")}"
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-$BENCHMARK_ROOT/results/telemetry/meltdown_labels/$RUN_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-$BENCHMARK_ROOT/.venv/bin/python}"

# Use the three model configurations reported in Agent Meltdowns Section 4.3.
export DISCOVERY_MODEL="${DISCOVERY_MODEL:-gpt-5.4-mini}"
export DISCOVERY_REASONING_EFFORT="${DISCOVERY_REASONING_EFFORT:-high}"
export AGGREGATION_MODEL="${AGGREGATION_MODEL:-gpt-5.5}"
export AGGREGATION_REASONING_EFFORT="${AGGREGATION_REASONING_EFFORT:-medium}"
export ANNOTATION_MODEL="${ANNOTATION_MODEL:-gpt-5.4-mini}"
export ANNOTATION_REASONING_EFFORT="${ANNOTATION_REASONING_EFFORT:-high}"

# Keep labeling credentials separate from any local-vLLM experiment configuration.
if [[ -n "${LABELING_OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$LABELING_OPENAI_API_KEY"
fi
if [[ -n "${LABELING_OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="$LABELING_OPENAI_BASE_URL"
else
  unset OPENAI_BASE_URL || true
fi

# Build optional arguments without shell-string interpolation.
COMMON_ARGS=()
if [[ -n "${PAPER_HTML:-}" ]]; then
  COMMON_ARGS+=(--paper-html "$PAPER_HTML")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi
LIMIT_ARGS=()
if [[ "${LIMIT:-0}" != "0" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

# A dry run prepares and counts traces without making paid API calls.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/label_meltdowns_01_discover.py" \
    --jobs-root "$JOBS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    --dry-run
  echo "Dry run complete; no model requests were sent."
  exit 0
fi

# Fail before incurring partial costs when cloud credentials are absent.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY or LABELING_OPENAI_API_KEY is required." >&2
  exit 2
fi

# # Stage 1 discovers candidate behaviors independently in every cleaned trace.
# "$PYTHON_BIN" "$SCRIPT_DIR/label_meltdowns_01_discover.py" \
#   --jobs-root "$JOBS_ROOT" \
#   --output-root "$OUTPUT_ROOT" \
#   "${COMMON_ARGS[@]}" \
#   "${LIMIT_ARGS[@]}"

# Stage 2 aggregates discovery outputs into a candidate taxonomy for expert review.
"$PYTHON_BIN" "$SCRIPT_DIR/label_meltdowns_02_aggregate.py" \
  --output-root "$OUTPUT_ROOT" \
  "${COMMON_ARGS[@]}"

# Support the paper's manual-review boundary before a new taxonomy is adopted.
if [[ "${STOP_AFTER_AGGREGATION:-0}" == "1" ]]; then
  echo "Stopped after Stage 2 for expert taxonomy review."
  echo "Candidate: $OUTPUT_ROOT/taxonomy/candidate_taxonomy.md"
  exit 0
fi

# Stage 3 defaults to the paper's frozen Appendix B taxonomy, not the unreviewed candidate.
TAXONOMY_ARGS=()
if [[ "${ANNOTATION_TAXONOMY:-paper}" == "candidate" ]]; then
  if [[ "${CANDIDATE_TAXONOMY_REVIEWED:-0}" != "1" ]]; then
    echo "Set CANDIDATE_TAXONOMY_REVIEWED=1 after expert review." >&2
    exit 2
  fi
  TAXONOMY_ARGS+=(--taxonomy "$OUTPUT_ROOT/taxonomy/candidate_taxonomy.md")
elif [[ "${ANNOTATION_TAXONOMY:-paper}" != "paper" ]]; then
  TAXONOMY_ARGS+=(--taxonomy "$ANNOTATION_TAXONOMY")
fi
"$PYTHON_BIN" "$SCRIPT_DIR/label_meltdowns_03_annotate.py" \
  --output-root "$OUTPUT_ROOT" \
  "${COMMON_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${TAXONOMY_ARGS[@]}"

# Print only non-sensitive artifact paths at handoff.
echo "Agent Meltdown labeling complete."
echo "Annotations: $OUTPUT_ROOT/annotations"
echo "Summary:     $OUTPUT_ROOT/annotation_summary.csv"
echo "Candidate:   $OUTPUT_ROOT/taxonomy/candidate_taxonomy.md"
echo "Frozen:      $OUTPUT_ROOT/taxonomy/paper_frozen_taxonomy.md"

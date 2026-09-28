#!/bin/bash
set -euo pipefail

# Resolve the repository independently of the caller's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

# Load the Rivanna Apptainer module used by the benchmark runtime.
module load apptainer/1.5.0

# Keep all transient Apptainer state outside shared project storage.
run_tmp="$(mktemp -d "/tmp/${USER:-hab}-ehr-meds.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$run_tmp" 2>/dev/null || true
  rm -rf -- "$run_tmp"
}
trap cleanup EXIT

# Configure repository-local caches that are excluded from Git.
export APPTAINER_IGNORE_PROOT=1
export APPTAINER_TMPDIR="$run_tmp"
export APPTAINER_CACHEDIR="$PWD/.apptainer-cache"

mkdir -p "$APPTAINER_CACHEDIR" "$PWD/.cache/harbor-sif" "$PWD/jobs"

# Run one ordinary EHR-to-MEDS smoke trial with overridable model settings.
uv run harbor run \
  --path tasks/ehr_to_meds_etl \
  --agent "${AGENT:-codex}" \
  --model "${MODEL:-gpt-5.6-sol}" \
  --agent-kwarg "reasoning_effort=${REASONING_EFFORT:-xhigh}" \
  --agent-setup-timeout-multiplier 4 \
  --env singularity \
  --environment-kwarg "singularity_image_cache_dir=$PWD/.cache/harbor-sif" \
  --mounts '[{"type":"bind","source":"/etc/resolv.conf","target":"/etc/resolv.conf","read_only":true}]' \
  --n-attempts "${N_ATTEMPTS:-1}" \
  --n-concurrent "${N_CONCURRENT:-1}" \
  --job-name "${JOB_NAME:-ehr-meds-codex-smoke}" \
  --jobs-dir "$PWD/jobs" \
  --yes \
  "$@"

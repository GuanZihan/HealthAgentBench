#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  printf 'Usage: %s TASK_NAME SCENARIO {control|error} [harbor arguments...]\n' "$0" >&2
  exit 2
fi

task_name="$1"
scenario="$2"
condition="$3"
shift 3

case "$scenario" in
  file-not-found | corrupt-file | truncated-file | missing-dependency) ;;
  *)
    printf 'Unsupported scenario: %s\n' "$scenario" >&2
    exit 2
    ;;
esac
if [[ "$condition" != "control" && "$condition" != "error" ]]; then
  printf 'Condition must be control or error: %s\n' "$condition" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$(cd -- "$script_dir/../.." && pwd -P)"
task_dir="$benchmark_root/tasks/$task_name"

if [[ ! -f "$task_dir/task.toml" ]]; then
  printf 'Unknown task: %s\n' "$task_name" >&2
  exit 2
fi
if [[ "$task_name" == "ehr_to_meds_etl" ]]; then
  printf 'Use the dedicated run_ehr_to_meds_* scripts for ehr_to_meds_etl.\n' >&2
  exit 2
fi

module load apptainer/1.5.0

run_tmp="$(mktemp -d "/tmp/${USER:-hab}-family-fault.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$run_tmp" 2>/dev/null || true
  rm -rf -- "$run_tmp"
}
trap cleanup EXIT

export APPTAINER_IGNORE_PROOT=1
export APPTAINER_TMPDIR="$run_tmp"
export APPTAINER_CACHEDIR="$benchmark_root/.apptainer-cache"
export PYTHONPATH="$benchmark_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
  "$APPTAINER_CACHEDIR" \
  "$benchmark_root/.cache/harbor-sif" \
  "$benchmark_root/jobs"

cd "$benchmark_root"

# The environment reads dataset credentials from .env for bootstrap only.
# Exported copies must not leak into the long-lived Agent process.
unset HF_TOKEN REDIVIS_API_TOKEN PN_USER PN_PASS \
  PHYSIONET_USER PHYSIONET_PASSWORD

uv run python -m faults.run_fault \
  --task "tasks/$task_name" \
  --scenario "$scenario" \
  --condition "$condition" \
  -- \
  --agent "${AGENT:-codex}" \
  --model "${MODEL:-gpt-5.6-luna}" \
  --agent-kwarg "reasoning_effort=${REASONING_EFFORT:-medium}" \
  --agent-setup-timeout-multiplier 4 \
  --n-attempts "${N_ATTEMPTS:-1}" \
  --n-concurrent "${N_CONCURRENT:-1}" \
  --job-name "${JOB_NAME:-${task_name}-${scenario}-${condition}}" \
  --jobs-dir "$benchmark_root/jobs" \
  --yes \
  "$@"

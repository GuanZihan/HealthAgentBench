#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s TASK_NAME [additional harbor run arguments...]\n' "$0" >&2
  exit 2
fi

task_name="$1"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$(cd -- "$script_dir/.." && pwd -P)"
task_dir="$benchmark_root/tasks/$task_name"

if [[ ! -f "$task_dir/task.toml" ]]; then
  printf 'Unknown task: %s\n' "$task_name" >&2
  exit 2
fi
if [[ "$task_name" == "ehr_to_meds_etl" ]]; then
  printf 'Use faults/scripts/run_ehr_to_meds_condition.sh for ehr_to_meds_etl.\n' >&2
  exit 2
fi

module load apptainer/1.5.0

run_tmp="$(mktemp -d "/tmp/${USER:-hab}-${task_name}.XXXXXX")"
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

# FamilySingularityEnvironment reads these from .env only for bootstrap.
# Remove exported copies so Apptainer cannot inherit them into the Agent.
unset HF_TOKEN REDIVIS_API_TOKEN PN_USER PN_PASS \
  PHYSIONET_USER PHYSIONET_PASSWORD

uv run harbor run \
  --path "tasks/$task_name" \
  --agent "${AGENT:-codex}" \
  --model "${MODEL:-gpt-5.6-luna}" \
  --agent-kwarg "reasoning_effort=${REASONING_EFFORT:-medium}" \
  --agent-setup-timeout-multiplier 4 \
  --environment-import-path \
    faults.family_singularity:FamilySingularityEnvironment \
  --environment-kwarg \
    "singularity_image_cache_dir=$benchmark_root/.cache/harbor-sif" \
  --mounts \
    '[{"type":"bind","source":"/etc/resolv.conf","target":"/etc/resolv.conf","read_only":true}]' \
  --n-attempts "${N_ATTEMPTS:-1}" \
  --n-concurrent "${N_CONCURRENT:-1}" \
  --job-name "${JOB_NAME:-apptainer-$task_name}" \
  --jobs-dir "$benchmark_root/jobs" \
  --yes \
  "$@"

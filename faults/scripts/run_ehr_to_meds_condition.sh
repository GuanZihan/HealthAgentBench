#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "control" && "$1" != "error" ) ]]; then
  printf 'Usage: %s {control|error} [harbor arguments...]\n' "$0" >&2
  exit 2
fi

condition="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$(cd -- "$script_dir/../.." && pwd -P)"
task_image="$benchmark_root/containers/healthagentbench-ehr-to-meds.sif"
fault_scenario="${FAULT_SCENARIO:-file-not-found}"

case "$fault_scenario" in
  file-not-found | checksum-mismatch | recovery-http-403) ;;
  *)
    printf 'Unsupported FAULT_SCENARIO: %s\n' "$fault_scenario" >&2
    printf 'Supported values: file-not-found, checksum-mismatch, recovery-http-403\n' >&2
    exit 2
    ;;
esac

experiment_name="${EXPERIMENT_NAME:-ehr-meds-${fault_scenario}}"

if [[ ! "$experiment_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'EXPERIMENT_NAME contains unsupported characters: %s\n' \
    "$experiment_name" >&2
  exit 2
fi

if [[ ! -f "$task_image" ]]; then
  printf 'Missing EHR-to-MEDS SIF: %s\n' "$task_image" >&2
  exit 1
fi

module load apptainer/1.5.0

run_tmp="$(mktemp -d "/tmp/${USER:-hab}-ehr-meds-${fault_scenario}-${condition}.XXXXXX")"
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

uv run harbor run \
  --path tasks/ehr_to_meds_etl \
  --agent codex \
  --model "${MODEL:-gpt-5.6-luna}" \
  --agent-kwarg "reasoning_effort=${REASONING_EFFORT:-medium}" \
  --agent-setup-timeout-multiplier 4 \
  --environment-import-path faults.singularity_fault:EhrToMedsFaultEnvironment \
  --environment-kwarg "fault_condition=$condition" \
  --environment-kwarg "fault_scenario=$fault_scenario" \
  --environment-kwarg \
    "singularity_image_cache_dir=$benchmark_root/.cache/harbor-sif" \
  --mounts \
    '[{"type":"bind","source":"/etc/resolv.conf","target":"/etc/resolv.conf","read_only":true}]' \
  --n-attempts "${N_ATTEMPTS:-1}" \
  --n-concurrent "${N_CONCURRENT:-1}" \
  --job-name "${experiment_name}-${condition}" \
  --jobs-dir "$benchmark_root/jobs" \
  --yes \
  "$@"

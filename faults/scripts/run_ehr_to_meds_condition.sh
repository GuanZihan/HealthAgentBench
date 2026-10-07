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
  file-not-found | checksum-mismatch | recovery-http-403 | \
    permission-denied-recoverable | protected-file-sudo | \
    evidence-explicit-missing | evidence-natural-missing | evidence-low-quality) ;;
  *)
    printf 'Unsupported FAULT_SCENARIO: %s\n' "$fault_scenario" >&2
    printf '%s\n' \
      'Supported values: file-not-found, checksum-mismatch, recovery-http-403,' \
      'permission-denied-recoverable, protected-file-sudo, evidence-explicit-missing,' \
      'evidence-natural-missing, evidence-low-quality' >&2
    exit 2
    ;;
esac

experiment_name="${EXPERIMENT_NAME:-ehr-meds-${fault_scenario}}"
agent_name="${AGENT:-codex}"
model_name="${MODEL:-gpt-5.6-luna}"

# Route the text-only Qwen alias through the adapter that removes view_image.
agent_selector=(--agent "$agent_name")
if [[ "$agent_name" == "codex" && "$model_name" == "qwen-coder" ]]; then
  agent_selector=(--agent-import-path faults.codex_agent:TextOnlyCodex)
fi

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

# Keep each experiment campaign in an explicitly selectable Harbor jobs root.
jobs_dir="${JOBS_DIR:-$benchmark_root/jobs}"

mkdir -p \
  "$APPTAINER_CACHEDIR" \
  "$benchmark_root/.cache/harbor-sif" \
  "$jobs_dir"
jobs_dir="$(cd -- "$jobs_dir" && pwd -P)"

cd "$benchmark_root"

uv run harbor run \
  --path tasks/ehr_to_meds_etl \
  "${agent_selector[@]}" \
  --model "$model_name" \
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
  --jobs-dir "$jobs_dir" \
  --yes \
  "$@"

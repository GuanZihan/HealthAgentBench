#!/bin/bash
set -euo pipefail

# Run every task in one shared-SIF family under matched control/error conditions.
usage() {
  cat <<'EOF'
Usage:
  bash run_family_experiments.sh FAMILY [additional Harbor arguments...]

Families:
  clinical-trial-matching
  ct-abnormality
  ehr-data-quality
  ehr-event-modelling
  pathology
  xray-report

Environment controls:
  MODEL                Agent model name (default: gpt-5.6-luna).
  REASONING_EFFORT     Agent reasoning effort (default: medium).
  REPLICATE            Replicate label used in job names (default: rep1).
  JOBS_DIR             Harbor output root (default: HealthAgentBench/jobs).
  SCENARIOS            Space-separated scenario subset (default: nine faults).
  CAPTURE_TELEMETRY    Set to 1 to capture each trial's vLLM calls.
  VLLM_UPSTREAM_URL    Upstream vLLM origin used in telemetry mode.
  PLAN_ONLY            Set to 1 to print the matrix without running Harbor.
EOF
}

# Reject incomplete invocations before discovering task directories.
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

# Resolve the requested family to its stable HealthAgentBench task-name prefix.
family="$1"
shift
case "$family" in
  clinical-trial-matching) task_prefix="clinical_trial_matching_" ;;
  ct-abnormality) task_prefix="ct_abnormality_" ;;
  ehr-data-quality) task_prefix="ehr_data_quality_" ;;
  ehr-event-modelling) task_prefix="ehr_event_modelling_" ;;
  pathology) task_prefix="tumor_area_selection_pathology_" ;;
  xray-report) task_prefix="xray_report_correction_" ;;
  *)
    printf 'Unsupported family: %s\n' "$family" >&2
    usage >&2
    exit 2
    ;;
esac

# Anchor task discovery and child launchers to this repository checkout.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$script_dir"
cd "$benchmark_root"

# Discover tasks dynamically so upstream task additions need no script edit.
shopt -s nullglob
candidate_dirs=("$benchmark_root"/tasks/"$task_prefix"*)
tasks=()
for task_dir in "${candidate_dirs[@]}"; do
  if [[ -f "$task_dir/task.toml" ]]; then
    tasks+=("${task_dir##*/}")
  fi
done
if [[ ${#tasks[@]} -eq 0 ]]; then
  printf 'No tasks found for family %s with prefix %s\n' \
    "$family" "$task_prefix" >&2
  exit 1
fi

# Resolve reproducibility controls once and reuse them for every matched pair.
model="${MODEL:-gpt-5.6-luna}"
reasoning_effort="${REASONING_EFFORT:-medium}"
replicate="${REPLICATE:-rep1}"
capture_telemetry="${CAPTURE_TELEMETRY:-0}"
plan_only="${PLAN_ONLY:-0}"
if [[ ! "$replicate" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'REPLICATE contains unsupported characters: %s\n' "$replicate" >&2
  exit 2
fi
for binary_value in "$capture_telemetry" "$plan_only"; do
  if [[ "$binary_value" != "0" && "$binary_value" != "1" ]]; then
    printf 'CAPTURE_TELEMETRY and PLAN_ONLY must be 0 or 1\n' >&2
    exit 2
  fi
done

# Use the same generic and evidence scenario names across all shared-SIF families.
read -r -a scenarios <<< \
  "${SCENARIOS:-file-not-found corrupt-file truncated-file missing-dependency permission-denied-recoverable protected-file-sudo evidence-explicit-missing evidence-natural-missing evidence-low-quality}"
if [[ ${#scenarios[@]} -eq 0 ]]; then
  printf 'SCENARIOS must contain at least one scenario\n' >&2
  exit 2
fi
for scenario in "${scenarios[@]}"; do
  case "$scenario" in
    file-not-found | corrupt-file | truncated-file | missing-dependency | \
      permission-denied-recoverable | protected-file-sudo) ;;
    evidence-explicit-missing | evidence-natural-missing | evidence-low-quality) ;;
    *)
      printf 'Unsupported shared-family scenario: %s\n' "$scenario" >&2
      exit 2
      ;;
  esac
done

# Fail before the first costly trial if a clinical evidence recipe is unavailable.
if [[ "$family" == "clinical-trial-matching" && "$plan_only" != "1" ]]; then
  for scenario in "${scenarios[@]}"; do
    if [[ "$scenario" == evidence-* ]]; then
      uv run python -m faults.evidence.validate_clinical_recipes
      break
    fi
  done
fi

# Preserve optional endpoint and agent overrides for every Harbor invocation.
harbor_arguments=("$@")

# Print the resolved matrix before the first potentially long-running trial.
printf '[family-batch] family=%s tasks=%d scenarios=%d conditions=2 trials=%d replicate=%s telemetry=%s\n' \
  "$family" "${#tasks[@]}" "${#scenarios[@]}" \
  "$((${#tasks[@]} * ${#scenarios[@]} * 2))" "$replicate" "$capture_telemetry"

# Run each control immediately before its matched error trial.
for task in "${tasks[@]}"; do
  for scenario in "${scenarios[@]}"; do
    for condition in control error; do
      job_name="${task}-${scenario}-${replicate}-${condition}"
      printf '[family-batch] task=%s scenario=%s condition=%s job=%s\n' \
        "$task" "$scenario" "$condition" "$job_name"
      if [[ "$plan_only" == "1" ]]; then
        continue
      fi

      # Route official local-model runs through a per-trial proxy when requested.
      if [[ "$capture_telemetry" == "1" ]]; then
        MODEL="$model" \
        REASONING_EFFORT="$reasoning_effort" \
        JOB_NAME="$job_name" \
        TELEMETRY_RUN_ID="$job_name" \
          bash faults/telemetry/run_fault_with_capture.sh \
          "$task" "$scenario" "$condition" "${harbor_arguments[@]}"
      else
        MODEL="$model" \
        REASONING_EFFORT="$reasoning_effort" \
        JOB_NAME="$job_name" \
          bash faults/scripts/run_family_condition.sh \
          "$task" "$scenario" "$condition" "${harbor_arguments[@]}"
      fi
    done
  done
done

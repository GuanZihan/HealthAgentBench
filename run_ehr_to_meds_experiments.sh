#!/bin/bash
set -euo pipefail

# Resolve this repository before launching the dedicated EHR-to-MEDS environment.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

# Resolve reproducibility and optional telemetry controls once for the matrix.
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

# Use all dedicated EHR-to-MEDS scenarios unless a subset is requested.
read -r -a scenarios <<< \
  "${SCENARIOS:-file-not-found checksum-mismatch recovery-http-403 permission-denied-recoverable protected-file-sudo evidence-explicit-missing evidence-natural-missing evidence-low-quality}"
if [[ ${#scenarios[@]} -eq 0 ]]; then
  printf 'SCENARIOS must contain at least one scenario\n' >&2
  exit 2
fi
for scenario in "${scenarios[@]}"; do
  case "$scenario" in
    file-not-found | checksum-mismatch | recovery-http-403 | \
      permission-denied-recoverable | protected-file-sudo | \
      evidence-explicit-missing | evidence-natural-missing | evidence-low-quality) ;;
    *)
      printf 'Unsupported EHR-to-MEDS scenario: %s\n' "$scenario" >&2
      exit 2
      ;;
  esac
done

# Preserve optional endpoint and agent overrides for every Harbor invocation.
harbor_arguments=("$@")

# Print the resolved experiment matrix before starting the first trial.
printf '[ehr-meds-batch] scenarios=%d conditions=2 trials=%d replicate=%s telemetry=%s\n' \
  "${#scenarios[@]}" "$((${#scenarios[@]} * 2))" "$replicate" "$capture_telemetry"

# Run each control immediately before its matched error trial.
for scenario in "${scenarios[@]}"; do
  for condition in control error; do
    experiment_name="ehr-meds-${scenario}-${replicate}"
    job_name="${experiment_name}-${condition}"
    printf '[ehr-meds-batch] scenario=%s condition=%s job=%s\n' \
      "$scenario" "$condition" "$job_name"
    if [[ "$plan_only" == "1" ]]; then
      continue
    fi

    # Route official local-model runs through a per-trial proxy when requested.
    if [[ "$capture_telemetry" == "1" ]]; then
      MODEL="$model" \
      REASONING_EFFORT="$reasoning_effort" \
      EXPERIMENT_NAME="$experiment_name" \
      JOB_NAME="$job_name" \
      TELEMETRY_RUN_ID="$job_name" \
        bash faults/telemetry/run_fault_with_capture.sh \
        ehr_to_meds_etl "$scenario" "$condition" "${harbor_arguments[@]}"
    else
      MODEL="$model" \
      REASONING_EFFORT="$reasoning_effort" \
      FAULT_SCENARIO="$scenario" \
      EXPERIMENT_NAME="$experiment_name" \
        bash faults/scripts/run_ehr_to_meds_condition.sh \
        "$condition" "${harbor_arguments[@]}"
    fi
  done
done

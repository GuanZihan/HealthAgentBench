#!/bin/bash
set -euo pipefail

# Run the complete benchmark matrix with GPT-5.5-mini and no telemetry proxy.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

# Remove local-Qwen endpoints inherited from an earlier interactive shell.
unset OPENAI_BASE_URL VLLM_UPSTREAM_URL

# Keep the GPT campaign independently configurable without changing the Qwen matrix.
model="${MODEL:-gpt-6-luna}"
reasoning_effort="${REASONING_EFFORT:-medium}"
replicate_prefix="${REPLICATE_PREFIX:-gpt6-luna}"
jobs_root="${JOBS_ROOT:-$script_dir/jobs}"
read -r -a replicate_numbers <<< "${REPLICATES:-1 2 3}"
# read -r -a families <<< \
#   "${FAMILIES:-ct xray pathology clinical ehr-data-quality ehr-event-modelling ehr-to-meds}"
read -r -a families <<< \
  "${FAMILIES:-ehr-data-quality ehr-event-modelling ehr-to-meds}"


# Map the compact family names to the benchmark's public launchers.
family_launcher() {
  case "$1" in
    ct) printf '%s\n' run_ct_experiments.sh ;;
    xray) printf '%s\n' run_xray_experiments.sh ;;
    pathology) printf '%s\n' run_pathology_experiments.sh ;;
    clinical) printf '%s\n' run_clinical_trial_experiments.sh ;;
    ehr-data-quality) printf '%s\n' run_ehr_data_quality_experiments.sh ;;
    ehr-event-modelling) printf '%s\n' run_ehr_event_modelling_experiments.sh ;;
    ehr-to-meds) printf '%s\n' run_ehr_to_meds_experiments.sh ;;
    *)
      printf 'Unsupported family in FAMILIES: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

# Validate the requested matrix before starting any paid model calls.
if [[ ${#replicate_numbers[@]} -eq 0 || ${#families[@]} -eq 0 ]]; then
  printf 'REPLICATES and FAMILIES must each contain at least one value\n' >&2
  exit 2
fi
for family in "${families[@]}"; do
  family_launcher "$family" >/dev/null
done

# Run every replicate in its own Harbor jobs directory without request capture.
for rep_number in "${replicate_numbers[@]}"; do
  if [[ ! "$rep_number" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'Unsupported replicate identifier: %s\n' "$rep_number" >&2
    exit 2
  fi
  replicate="${replicate_prefix}-rep${rep_number}"
  replicate_jobs="$jobs_root/$replicate"
  printf '[gpt-matrix] model=%s replicate=%s telemetry=0 jobs=%s\n' \
    "$model" "$replicate" "$replicate_jobs"

  # Run families sequentially so dataset setup and result attribution stay clear.
  for family in "${families[@]}"; do
    launcher="$(family_launcher "$family")"
    MODEL="$model" \
    REASONING_EFFORT="$reasoning_effort" \
    REPLICATE="$replicate" \
    JOBS_DIR="$replicate_jobs" \
    CAPTURE_TELEMETRY=0 \
      bash "$launcher"
  done
done

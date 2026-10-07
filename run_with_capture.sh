#!/bin/bash
set -euo pipefail

# Run every task family sequentially with one telemetry proxy per trial.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

# Require an explicit local endpoint so a publishable run cannot hit OpenAI by accident.
export MODEL="${MODEL:-qwen-coder}"
export REASONING_EFFORT="${REASONING_EFFORT:-medium}"
export REPLICATE="${REPLICATE:-qwen-telemetry-rep1}"
export CAPTURE_TELEMETRY=1
export JOBS_DIR="${JOBS_DIR:-$script_dir/jobs/all-${REPLICATE}}"

# Stop before any real trial if the declared vLLM Responses server is unavailable.
if [[ "${PLAN_ONLY:-0}" != "1" ]]; then
  : "${VLLM_UPSTREAM_URL:?Set VLLM_UPSTREAM_URL, for example http://127.0.0.1:8010}"
  models_url="${VLLM_UPSTREAM_URL%/}/v1/models"
  if ! curl --fail --silent --show-error --max-time 10 "$models_url" >/dev/null; then
    printf 'vLLM is not ready at %s\n' "$models_url" >&2
    exit 1
  fi
fi

# Allow resumable family subsets while keeping the complete benchmark as default.
read -r -a families <<< \
  "${FAMILIES:-ct xray pathology clinical ehr-data-quality ehr-event-modelling ehr-to-meds}"

# Validate all clinical recipes before spending GPU time on earlier families.
if [[ "${PLAN_ONLY:-0}" != "1" ]]; then
  for family in "${families[@]}"; do
    if [[ "$family" == "clinical" ]]; then
      .venv/bin/python -m faults.evidence.validate_clinical_recipes
      break
    fi
  done
fi

# Execute one family at a time so the fixed telemetry port never collides.
for family in "${families[@]}"; do
  case "$family" in
    ct) launcher="run_ct_experiments.sh" ;;
    xray) launcher="run_xray_experiments.sh" ;;
    pathology) launcher="run_pathology_experiments.sh" ;;
    clinical) launcher="run_clinical_trial_experiments.sh" ;;
    ehr-data-quality) launcher="run_ehr_data_quality_experiments.sh" ;;
    ehr-event-modelling) launcher="run_ehr_event_modelling_experiments.sh" ;;
    ehr-to-meds) launcher="run_ehr_to_meds_experiments.sh" ;;
    *)
      printf 'Unsupported family in FAMILIES: %s\n' "$family" >&2
      exit 2
      ;;
  esac
  printf '[all-families] starting family=%s launcher=%s\n' "$family" "$launcher"
  bash "$launcher" "$@"
done

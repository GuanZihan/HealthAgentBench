#!/bin/bash
#SBATCH --job-name=healthagent-gpt6-sol
#SBATCH --partition=bii
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Resolve the repository from the submission directory because Slurm copies this file.
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
submit_dir="${SLURM_SUBMIT_DIR:-$source_dir}"
if [[ -f "$submit_dir/run_family_experiments.sh" ]]; then
  script_dir="$submit_dir"
elif [[ -f "$submit_dir/HealthAgentBench/run_family_experiments.sh" ]]; then
  script_dir="$submit_dir/HealthAgentBench"
else
  printf 'Cannot locate HealthAgentBench from submission directory: %s\n' \
    "$submit_dir" >&2
  exit 1
fi
script_dir="$(cd -- "$script_dir" && pwd -P)"
cd "$script_dir"

# Remove local-Qwen endpoints inherited from an earlier interactive shell.
unset OPENAI_BASE_URL VLLM_UPSTREAM_URL

# Keep the GPT campaign independently configurable without changing the Qwen matrix.
model="${MODEL:-gpt-6-sol}"
reasoning_effort="${REASONING_EFFORT:-medium}"
replicate_prefix="${REPLICATE_PREFIX:-gpt6-sol}"
jobs_root="${JOBS_ROOT:-$script_dir/jobs}"
read -r -a replicate_numbers <<< "${REPLICATES:-1 2 3}"
read -r -a families <<< \
  "${FAMILIES:-ct xray pathology clinical ehr-data-quality ehr-event-modelling ehr-to-meds}"

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
      bash "$script_dir/$launcher"
  done
done

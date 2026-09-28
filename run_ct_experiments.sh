#!/bin/bash
set -euo pipefail

# Resolve the repository independently of the caller's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

# Define the ten CT-RATE volumes used by the benchmark task family.
tasks=(
  ct_abnormality_valid_16_a_1
  ct_abnormality_valid_46_a_1
  ct_abnormality_valid_81_a_1
  ct_abnormality_valid_137_a_1
  ct_abnormality_valid_144_a_1
  ct_abnormality_valid_265_a_2
  ct_abnormality_valid_304_a_1
  ct_abnormality_valid_481_a_2
  ct_abnormality_valid_636_a_1
  ct_abnormality_valid_670_a_1
)

# Define the matched environmental conditions evaluated for each volume.
scenarios=(
  file-not-found
  corrupt-file
  truncated-file
  missing-dependency
)

# Allow research runs to override the model, effort, or replicate identifier.
model="${MODEL:-gpt-5.6-luna}"
reasoning_effort="${REASONING_EFFORT:-medium}"
replicate="${REPLICATE:-rep1}"

# Run one control/error pair for every task-scenario combination.
for task in "${tasks[@]}"; do
  for scenario in "${scenarios[@]}"; do
    MODEL="$model" \
    REASONING_EFFORT="$reasoning_effort" \
    JOB_NAME="${task}-${scenario}-${replicate}-control" \
      bash faults/scripts/run_family_control.sh "$task" "$scenario"

    MODEL="$model" \
    REASONING_EFFORT="$reasoning_effort" \
    JOB_NAME="${task}-${scenario}-${replicate}-error" \
      bash faults/scripts/run_family_error.sh "$task" "$scenario"
  done
done

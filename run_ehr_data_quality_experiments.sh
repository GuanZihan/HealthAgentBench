#!/bin/bash
set -euo pipefail

# Run every MIMIC-IV Demo data-quality task through the shared batch driver.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_family_experiments.sh" ehr-data-quality "$@"

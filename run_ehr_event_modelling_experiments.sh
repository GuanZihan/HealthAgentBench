#!/bin/bash
set -euo pipefail

# Run every EHRSHOT event-modelling task through the shared batch driver.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_family_experiments.sh" ehr-event-modelling "$@"

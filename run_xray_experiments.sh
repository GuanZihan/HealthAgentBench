#!/bin/bash
set -euo pipefail

# Run every MIMIC-CXR report-correction task through the shared batch driver.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_family_experiments.sh" xray-report "$@"

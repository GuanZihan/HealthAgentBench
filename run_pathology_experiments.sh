#!/bin/bash
set -euo pipefail

# Run every CAMELYON16 pathology task through the shared fault batch driver.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_family_experiments.sh" pathology "$@"

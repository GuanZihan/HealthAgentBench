#!/bin/bash
set -euo pipefail

# Keep the historical CT entry point as a thin wrapper around the shared batch driver.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_family_experiments.sh" ct-abnormality "$@"

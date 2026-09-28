#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'Usage: %s TASK_NAME SCENARIO [harbor arguments...]\n' "$0" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_family_condition.sh" "$1" "$2" control "${@:3}"

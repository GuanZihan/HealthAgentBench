#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$(cd -- "$script_dir/.." && pwd -P)"
definition_dir="$script_dir/definitions"

declare -A definitions=(
  [clinical-trial-matching]="clinical-trial-matching.def"
  [ct-abnormality]="ct-abnormality.def"
  [ehr-data-quality]="ehr-data-quality.def"
  [ehr-event-modelling]="ehr-event-modelling.def"
  [pathology]="pathology.def"
  [xray-report]="xray-report.def"
)

usage() {
  printf 'Usage: %s [--force] [all|FAMILY ...]\n' "$0"
  printf 'Families: %s\n' "${!definitions[*]}"
}

force=0
if [[ "${1:-}" == "--force" ]]; then
  force=1
  shift
fi

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
  families=(
    clinical-trial-matching
    ct-abnormality
    ehr-data-quality
    ehr-event-modelling
    pathology
    xray-report
  )
else
  families=("$@")
fi

if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer/1.5.0
fi
if ! command -v apptainer >/dev/null 2>&1; then
  printf 'apptainer is not on PATH. On Rivanna run: module load apptainer/1.5.0\n' >&2
  exit 1
fi

build_tmp="$(mktemp -d "/tmp/${USER:-hab}-hab-sif-build.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$build_tmp" 2>/dev/null || true
  rm -rf -- "$build_tmp"
}
trap cleanup EXIT

export APPTAINER_IGNORE_PROOT=1
export APPTAINER_TMPDIR="$build_tmp"
export APPTAINER_CACHEDIR="$benchmark_root/.apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"

for family in "${families[@]}"; do
  definition="${definitions[$family]:-}"
  if [[ -z "$definition" ]]; then
    printf 'Unknown family: %s\n' "$family" >&2
    usage >&2
    exit 2
  fi

  source_path="$definition_dir/$definition"
  output_path="$script_dir/healthagentbench-${family}.sif"
  if [[ -e "$output_path" && "$force" -ne 1 ]]; then
    printf '[skip] %s already exists (use --force to rebuild)\n' "$output_path"
    continue
  fi

  printf '[build] %s -> %s\n' "$source_path" "$output_path"
  build_args=(build --fakeroot)
  if [[ "$force" -eq 1 ]]; then
    build_args+=(--force)
  fi
  apptainer "${build_args[@]}" "$output_path" "$source_path"
done

printf '[done] family SIF build completed\n'

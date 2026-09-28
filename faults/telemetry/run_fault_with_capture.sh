#!/bin/bash
set -euo pipefail

# Run one fault condition through a local capture proxy before forwarding to vLLM.

# Print the stable command contract used by batch scripts and interactive runs.
usage() {
  cat <<'EOF'
Usage:
  bash faults/telemetry/run_fault_with_capture.sh \
    TASK_NAME SCENARIO {control|error} [harbor arguments...]

Required precondition:
  A vLLM OpenAI-compatible server is already running at VLLM_UPSTREAM_URL.

Common environment variables:
  VLLM_UPSTREAM_URL   Upstream vLLM origin (default: http://127.0.0.1:8000).
  TELEMETRY_PORT      Local capture proxy port (default: 8100).
  TELEMETRY_DIR       Capture artifact directory.
  MODEL               Served model alias (default: qwen-coder).
  JOB_NAME            Harbor job name for non-EHR task families.
EOF
}

# Reject incomplete invocations before creating background processes.
if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

# Resolve experiment identifiers before shifting optional Harbor arguments.
task_name="$1"
scenario="$2"
condition="$3"
shift 3
if [[ "$condition" != "control" && "$condition" != "error" ]]; then
  printf 'Condition must be control or error: %s\n' "$condition" >&2
  exit 2
fi

# Anchor all paths to the nested HealthAgentBench checkout.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$(cd -- "$script_dir/../.." && pwd -P)"
upstream_url="${VLLM_UPSTREAM_URL:-http://127.0.0.1:8000}"
telemetry_port="${TELEMETRY_PORT:-8100}"
model="${MODEL:-qwen-coder}"
run_id="${TELEMETRY_RUN_ID:-${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)}"
telemetry_dir="${TELEMETRY_DIR:-$benchmark_root/results/telemetry/${task_name}-${scenario}-${condition}-${run_id}}"
ready_file="$telemetry_dir/READY.json"
proxy_log="$telemetry_dir/proxy.log"

# Validate shell-controlled values before including them in file paths and URLs.
if [[ ! "$telemetry_port" =~ ^[1-9][0-9]*$ ]] || ((telemetry_port > 65535)); then
  printf 'TELEMETRY_PORT must be an integer between 1 and 65535\n' >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'TELEMETRY_RUN_ID contains unsupported characters: %s\n' "$run_id" >&2
  exit 2
fi

# Create the artifact directory before redirecting the proxy process log.
mkdir -p "$telemetry_dir"
cd "$benchmark_root"
export PYTHONPATH="$benchmark_root${PYTHONPATH:+:$PYTHONPATH}"

# Stop only the proxy process created by this wrapper on every exit path.
proxy_pid=''
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
    kill "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

# Start the standard-library proxy independently of the GPU-serving container.
python -m faults.telemetry.proxy \
  --upstream "$upstream_url" \
  --listen-port "$telemetry_port" \
  --output-dir "$telemetry_dir" \
  --ready-file "$ready_file" \
  --metadata "task=$task_name" \
  --metadata "scenario=$scenario" \
  --metadata "condition=$condition" \
  --metadata "model=$model" \
  --metadata "run_id=$run_id" \
  >"$proxy_log" 2>&1 &
proxy_pid=$!

# Wait briefly for the proxy socket and fail with its log if startup is unsuccessful.
deadline=$((SECONDS + 30))
while ((SECONDS < deadline)); do
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    wait "$proxy_pid" || true
    printf 'Telemetry proxy exited during startup; inspect %s\n' "$proxy_log" >&2
    exit 1
  fi
  if [[ -f "$ready_file" ]] && curl --fail --silent --max-time 2 \
    "http://127.0.0.1:${telemetry_port}/_telemetry/health" >/dev/null; then
    break
  fi
  sleep 0.2
done
if [[ ! -f "$ready_file" ]]; then
  printf 'Telemetry proxy was not ready after 30 seconds; inspect %s\n' "$proxy_log" >&2
  exit 1
fi

# Route only the Agent endpoint through telemetry while retaining the original task runtime.
export MODEL="$model"
agent_endpoint="http://127.0.0.1:${telemetry_port}/v1"
printf '[telemetry] capture directory: %s\n' "$telemetry_dir"
printf '[telemetry] agent endpoint: %s\n' "$agent_endpoint"
if [[ "$task_name" == "ehr_to_meds_etl" ]]; then
  export FAULT_SCENARIO="$scenario"
  "$benchmark_root/faults/scripts/run_ehr_to_meds_condition.sh" \
    "$condition" \
    --agent-env "OPENAI_BASE_URL=$agent_endpoint" \
    --agent-env OPENAI_API_KEY=dummy \
    "$@"
else
  "$benchmark_root/faults/scripts/run_family_condition.sh" \
    "$task_name" "$scenario" "$condition" \
    --agent-env "OPENAI_BASE_URL=$agent_endpoint" \
    --agent-env OPENAI_API_KEY=dummy \
    "$@"
fi

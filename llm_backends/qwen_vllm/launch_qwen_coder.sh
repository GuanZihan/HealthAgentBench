#!/bin/bash
set -euo pipefail

# Launch a Qwen Coder model behind vLLM's OpenAI Responses API.
#
# Design goals:
#   - run on a Rivanna GPU node through Apptainer (no Docker daemon);
#   - make the experimental configuration explicit and reproducible;
#   - preserve server logs, the resolved command, and a SIF checksum;
#   - expose a Codex-compatible endpoint only after it passes readiness checks.

usage() {
  cat <<'EOF'
Usage:
  MODEL_ID=Qwen/model-name \
  VLLM_SIF=/path/to/vllm.sif \
    bash launch_qwen_coder.sh [--dry-run] [-- VLLM_EXTRA_ARGS...]

Required environment variables:
  MODEL_ID                 Exact Hugging Face model ID or absolute local path.
  VLLM_SIF                 vLLM Apptainer image. Defaults to
                           HealthAgentBench/containers/vllm-openai-v0.29.0-cu129.sif.

Common experiment controls:
  MODEL_REVISION           Hugging Face commit/tag. Strongly recommended.
  SERVED_MODEL_NAME        API model alias (default: qwen-coder).
  TENSOR_PARALLEL_SIZE     Number of GPUs used by vLLM (default: 1).
  MAX_MODEL_LEN            Context length exposed by vLLM (default: 32768).
  DTYPE                    vLLM dtype (default: auto).
  GPU_MEMORY_UTILIZATION   Fraction of GPU memory available to vLLM (default: 0.90).
  SEED                     Server seed (default: 0).
  TOOL_CALL_PARSER         Tool parser (default: qwen3_coder).

Service and output controls:
  SERVER_HOST              Bind address (default: 127.0.0.1).
  CLIENT_HOST              Address written to client configs (default: 127.0.0.1).
  PORT                     API port (default: 8000).
  STARTUP_TIMEOUT_SECONDS  Readiness timeout (default: 1800).
  MODEL_CACHE_DIR          Persistent Hugging Face cache directory.
  OUTPUT_DIR               Directory for logs and configuration snapshots.
  RECORD_SIF_SHA256        Set to 0 to skip hashing a very large SIF (default: 1).

Any arguments after -- are appended verbatim to `vllm serve`. For example:
  -- --enable-expert-parallel --quantization fp8

The script stays in the foreground after the service becomes ready. Stop it
with Ctrl-C or cancel the enclosing Slurm job.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '[qwen-vllm] %s\n' "$*"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
benchmark_root="$(cd -- "$script_dir/../.." && pwd -P)"

dry_run=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
extra_vllm_args=("$@")

model_id="${MODEL_ID:-}"
vllm_sif="${VLLM_SIF:-$benchmark_root/containers/vllm-openai-v0.29.0-cu129.sif}"
model_revision="${MODEL_REVISION:-}"
served_model_name="${SERVED_MODEL_NAME:-qwen-coder}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
max_model_len="${MAX_MODEL_LEN:-32768}"
dtype="${DTYPE:-auto}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
seed="${SEED:-0}"
tool_call_parser="${TOOL_CALL_PARSER:-qwen3_coder}"
server_host="${SERVER_HOST:-127.0.0.1}"
client_host="${CLIENT_HOST:-127.0.0.1}"
port="${PORT:-8000}"
startup_timeout_seconds="${STARTUP_TIMEOUT_SECONDS:-1800}"
model_cache_dir="${MODEL_CACHE_DIR:-$benchmark_root/.cache/qwen-vllm}"
record_sif_sha256="${RECORD_SIF_SHA256:-1}"

[[ -n "$model_id" ]] || die "MODEL_ID is required; see $script_dir/README.md"
[[ "$model_id" != *$'\n'* && "$model_id" != *'"'* ]] || \
  die "MODEL_ID must not contain a newline or double quote"
[[ "$served_model_name" =~ ^[A-Za-z0-9._:-]+$ ]] || \
  die "SERVED_MODEL_NAME may contain only letters, numbers, dot, underscore, colon, and dash"
[[ "$tool_call_parser" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "TOOL_CALL_PARSER contains unsupported characters"
is_positive_integer "$tensor_parallel_size" || \
  die "TENSOR_PARALLEL_SIZE must be a positive integer"
is_positive_integer "$max_model_len" || die "MAX_MODEL_LEN must be a positive integer"
is_positive_integer "$port" || die "PORT must be a positive integer"
((port <= 65535)) || die "PORT must be at most 65535"
is_positive_integer "$startup_timeout_seconds" || \
  die "STARTUP_TIMEOUT_SECONDS must be a positive integer"
is_nonnegative_integer "$seed" || die "SEED must be a non-negative integer"
[[ "$record_sif_sha256" == "0" || "$record_sif_sha256" == "1" ]] || \
  die "RECORD_SIF_SHA256 must be 0 or 1"

if [[ "$dry_run" == "0" ]]; then
  if ! command -v apptainer >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
      module load apptainer/1.5.0
    fi
  fi
  command -v apptainer >/dev/null 2>&1 || \
    die "apptainer is unavailable; on Rivanna run: module load apptainer/1.5.0"
  command -v curl >/dev/null 2>&1 || die "curl is required for readiness checks"
  [[ -f "$vllm_sif" ]] || die "VLLM_SIF does not exist: $vllm_sif"
  if [[ "$model_id" == /* ]]; then
    [[ -d "$model_id" ]] || die "local MODEL_ID directory does not exist: $model_id"
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    note "warning: SLURM_JOB_ID is unset; make sure this is a GPU compute node"
  fi
fi

run_id="${RUN_ID:-${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-$benchmark_root/results/vllm/$run_id}"
server_log="$output_dir/server.log"
resolved_command="$output_dir/command.sh"
run_config="$output_dir/run-config.env"
codex_config="$output_dir/codex-config.toml"
endpoint_env="$output_dir/endpoint.env"
models_response="$output_dir/models.json"
package_versions="$output_dir/python-packages.txt"
container_preflight_log="$output_dir/container-preflight.log"
ready_marker="$output_dir/READY"
base_url="http://$client_host:$port/v1"

mkdir -p "$output_dir" "$model_cache_dir" "$benchmark_root/.apptainer-cache"

apptainer_args=(
  exec
  --cleanenv
  --nv
  --bind "$model_cache_dir:/model-cache"
)
if [[ "$model_id" == /* ]]; then
  apptainer_args+=(--bind "$model_id:$model_id:ro")
fi

vllm_args=(
  vllm serve "$model_id"
  --served-model-name "$served_model_name"
  --host "$server_host"
  --port "$port"
  --tensor-parallel-size "$tensor_parallel_size"
  --max-model-len "$max_model_len"
  --dtype "$dtype"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --seed "$seed"
  --enable-auto-tool-choice
  --tool-call-parser "$tool_call_parser"
)
if [[ -n "$model_revision" && "$model_id" != /* ]]; then
  vllm_args+=(--revision "$model_revision")
elif [[ -n "$model_revision" ]]; then
  note "local model: MODEL_REVISION is recorded as a provenance label only"
else
  note "warning: MODEL_REVISION is unset; pin a commit or tag for publishable runs"
fi
vllm_args+=("${extra_vllm_args[@]}")

command=(apptainer "${apptainer_args[@]}" "$vllm_sif" "${vllm_args[@]}")

{
  printf '#!/bin/bash\n'
  printf 'set -euo pipefail\n'
  printf 'export APPTAINERENV_HF_HOME=%q\n' '/model-cache/huggingface'
  printf 'export APPTAINERENV_TOKENIZERS_PARALLELISM=%q\n' 'false'
  printf 'if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"; fi\n'
  printf 'if [[ -n ${HF_TOKEN:-} ]]; then export APPTAINERENV_HF_TOKEN="$HF_TOKEN"; fi\n'
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$resolved_command"
chmod u+x "$resolved_command"

sif_sha256="not-recorded"
vllm_version="not-recorded"
apptainer_version="not-recorded"
if [[ "$record_sif_sha256" == "1" && "$dry_run" == "0" ]]; then
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
  note "computing the SIF checksum for the run record"
  sif_sha256="$(sha256sum "$vllm_sif" | awk '{print $1}')"
fi
if [[ "$dry_run" == "0" ]]; then
  apptainer_version="$(apptainer --version 2>/dev/null || printf 'unknown')"
  if apptainer exec --cleanenv "$vllm_sif" python3 -c \
    'from importlib.metadata import PackageNotFoundError, version
for package in ("vllm", "torch", "torchvision", "torchaudio"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=not-installed")' >"$package_versions" 2>"$container_preflight_log"; then
    vllm_version="$(awk -F= '$1 == "vllm" {print $2}' "$package_versions")"
  else
    printf 'package metadata inspection failed\n' >>"$container_preflight_log"
    printf '%s\n' 'vllm=unknown' >"$package_versions"
    vllm_version='unknown'
  fi
fi

{
  printf 'CREATED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'HOSTNAME=%q\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'SLURM_JOB_ID=%q\n' "${SLURM_JOB_ID:-}"
  printf 'MODEL_ID=%q\n' "$model_id"
  printf 'MODEL_REVISION=%q\n' "$model_revision"
  printf 'SERVED_MODEL_NAME=%q\n' "$served_model_name"
  printf 'VLLM_SIF=%q\n' "$vllm_sif"
  printf 'VLLM_SIF_SHA256=%q\n' "$sif_sha256"
  printf 'VLLM_VERSION=%q\n' "$vllm_version"
  printf 'APPTAINER_VERSION=%q\n' "$apptainer_version"
  printf 'TENSOR_PARALLEL_SIZE=%q\n' "$tensor_parallel_size"
  printf 'MAX_MODEL_LEN=%q\n' "$max_model_len"
  printf 'DTYPE=%q\n' "$dtype"
  printf 'GPU_MEMORY_UTILIZATION=%q\n' "$gpu_memory_utilization"
  printf 'SEED=%q\n' "$seed"
  printf 'TOOL_CALL_PARSER=%q\n' "$tool_call_parser"
  printf 'BASE_URL=%q\n' "$base_url"
  printf 'OUTPUT_DIR=%q\n' "$output_dir"
} >"$run_config"

cat >"$codex_config" <<EOF
model = "$served_model_name"
model_provider = "vllm"

[model_providers.vllm]
name = "Local vLLM"
base_url = "$base_url"
env_key = "VLLM_API_KEY"
wire_api = "responses"
supports_websockets = false
EOF

{
  printf 'OPENAI_BASE_URL=%q\n' "$base_url"
  printf 'OPENAI_API_KEY=%q\n' 'dummy'
  printf 'MODEL=%q\n' "$served_model_name"
  printf 'VLLM_API_KEY=%q\n' 'dummy'
} >"$endpoint_env"

note "resolved command: $resolved_command"
note "run configuration: $run_config"
if [[ "$dry_run" == "1" ]]; then
  note "dry run; the service was not started"
  printf '\n'
  sed -n '1,200p' "$resolved_command"
  exit 0
fi

export APPTAINER_CACHEDIR="$benchmark_root/.apptainer-cache"
export APPTAINERENV_HF_HOME='/model-cache/huggingface'
export APPTAINERENV_TOKENIZERS_PARALLELISM='false'
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi

command -v nvidia-smi >/dev/null 2>&1 || \
  die "nvidia-smi is unavailable; run this launcher inside a GPU allocation"
if ! nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader >"$output_dir/gpus.csv"; then
  die "no NVIDIA GPU is visible; request a GPU allocation before launching vLLM"
fi
visible_gpu_count="$(wc -l <"$output_dir/gpus.csv")"
visible_gpu_count="${visible_gpu_count//[[:space:]]/}"
is_positive_integer "$visible_gpu_count" || \
  die "no NVIDIA GPU is visible; request a GPU allocation before launching vLLM"
((visible_gpu_count >= tensor_parallel_size)) || \
  die "TENSOR_PARALLEL_SIZE=$tensor_parallel_size, but only $visible_gpu_count GPU(s) are visible"

note "checking the SIF's PyTorch and torchvision compatibility"
if ! apptainer "${apptainer_args[@]}" "$vllm_sif" python3 -c \
  'import torch
import torchvision
if not hasattr(torch.ops, "torchvision") or not hasattr(torch.ops.torchvision, "nms"):
    raise RuntimeError("torchvision::nms is unavailable")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")' \
  >>"$container_preflight_log" 2>&1; then
  die "the SIF has an incompatible torch/torchvision stack; inspect $package_versions and $container_preflight_log, then build the pinned image from $script_dir/README.md"
fi

server_pid=''
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    note "stopping vLLM process $server_pid"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

rm -f "$ready_marker"
note "starting vLLM; initialization can take several minutes"
"${command[@]}" > >(tee -a "$server_log") 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$output_dir/server.pid"

deadline=$((SECONDS + startup_timeout_seconds))
while ((SECONDS < deadline)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    die "vLLM exited before becoming ready; inspect $server_log"
  fi
  if curl --fail --silent --show-error --max-time 5 \
    "$base_url/models" >"$models_response.tmp" 2>/dev/null; then
    mv "$models_response.tmp" "$models_response"
    touch "$ready_marker"
    note "service is ready: $base_url"
    note "client environment: source $endpoint_env"
    note "Codex config snapshot: $codex_config"
    note "server log: $server_log"
    break
  fi
  sleep 5
done

[[ -f "$ready_marker" ]] || \
  die "vLLM was not ready after ${startup_timeout_seconds}s; inspect $server_log"

# Keep the service attached to the allocation. The EXIT/TERM trap forwards
# cancellation to vLLM and leaves all run records in OUTPUT_DIR.
wait "$server_pid"

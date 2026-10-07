set -euo pipefail

for rep_number in 1 2 3; do
  replicate="qwen-next-fp8-rep${rep_number}"

  MODEL=qwen-coder \
  VLLM_UPSTREAM_URL=http://127.0.0.1:8010 \
  REASONING_EFFORT=medium \
  N_ATTEMPTS=1 \
  N_CONCURRENT=1 \
  REPLICATE="$replicate" \
  JOBS_DIR="$PWD/jobs/$replicate" \
    bash run_with_capture.sh
done

agent_args=(
  --agent-env OPENAI_BASE_URL=http://127.0.0.1:8010/v1
  --agent-env OPENAI_API_KEY=dummy
)

# JOBS_DIR="$PWD/jobs/xray-qwen30b-rep1" \
# MODEL=qwen-coder \
# REASONING_EFFORT=medium \
# REPLICATE=qwen30b-rep1 \
#   bash run_xray_experiments.sh "${agent_args[@]}"


# JOBS_DIR="$PWD/jobs/pathology-qwen30b-rep1" \
# MODEL=qwen-coder \
# REASONING_EFFORT=medium \
# REPLICATE=qwen30b-rep1 \
#   bash run_pathology_experiments.sh "${agent_args[@]}"



# JOBS_DIR="$PWD/jobs/clinical-trial-qwen30b-rep1" \
# MODEL=qwen-coder \
# REASONING_EFFORT=medium \
# REPLICATE=qwen30b-rep1 \
#   bash run_clinical_trial_experiments.sh "${agent_args[@]}"



JOBS_DIR="$PWD/jobs/ehr-dq-qwen30b-rep1" \
MODEL=qwen-coder \
REASONING_EFFORT=medium \
REPLICATE=qwen30b-rep1 \
  bash run_ehr_data_quality_experiments.sh "${agent_args[@]}"



JOBS_DIR="$PWD/jobs/ehr-meds-qwen30b-rep1" \
MODEL=qwen-coder \
REASONING_EFFORT=medium \
REPLICATE=qwen30b-rep1 \
  bash run_ehr_to_meds_experiments.sh "${agent_args[@]}"
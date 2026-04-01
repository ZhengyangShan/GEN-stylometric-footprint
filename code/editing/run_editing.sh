#!/bin/bash

# ============================================================
# EditLens Experiment
# ============================================================
#
# Applies 303 editing prompts from Thai et al. (2025)
# to sampled documents with random revision ratios (1-100%).
#
# Usage:
#   sbatch run_sample_experiment.sh                  # all 303 prompts
#   sbatch run_sample_experiment.sh --max-prompts 10 # quick test
#
# Extra args are forwarded to python, e.g.:
#   sbatch run_sample_experiment.sh --prompt-split train
#   sbatch run_sample_experiment.sh --prompt-category tone_and_style
# ============================================================
# bash editing/run_editing.sh --prompt-split test --samples-per-domain 200 --min-revision-pct 50 --max-revision-pct 50
set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" #"meta-llama/Llama-3.3-70B-Instruct", "openai/gpt-oss-120b"
CSV_PATH="editing/sampled_subset_1000.csv"

# vLLM settings
export TENSOR_PARALLEL_SIZE=4
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=4096

# Create logs directory
mkdir -p logs

echo "============================================================"
echo "EditLens Experiment"
echo "============================================================"
echo "Model      : ${MODEL_NAME}"
echo "Input      : ${CSV_PATH}"
echo "Args       : $@"
echo "============================================================"

python -m editing.run_editing \
    --csv-path "${CSV_PATH}" \
    --model-name "${MODEL_NAME}" \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-new-tokens 1024 \
    --seed 42 \
    "$@"

echo "============================================================"
echo "EditLens experiment complete!"
echo "============================================================"

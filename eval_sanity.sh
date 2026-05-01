#!/bin/bash
set -euo pipefail

cd /home/an34232/Repos/entrgi/main_expts

BASE_MODEL="Dream-org/Dream-v0-Instruct-7B"
HELDOUT_JSON="/home/an34232/Repos/entrgi/main_expts/data/reward_bench2_diag_split/prompt_only/test.jsonl"
RESULTS_DIR="/hdd1/an34232/entrgi_sft_results_rb2_diag"
MODEL_DIR="/hdd1/an34232/entrgi_sft_models/dream-reward-bench2-diag-split-trunc128-sft-lora-r32-alllinear"
MODEL_NAME="dream-reward-bench2-diag-split-trunc128-sft-lora-r32-alllinear"
SUBSET_SIZE="${SUBSET_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "${RESULTS_DIR}"

for SEED in 1 2 3; do
  echo "Running base, seed ${SEED}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((29510 + SEED)) bon_infer.py \
    --dream_model "${BASE_MODEL}" \
    --K 1 \
    --T 128 \
    --max_new_tokens 128 \
    --dataset_path "${HELDOUT_JSON}" \
    --split heldout \
    --prompt_field prompt \
    --subset_size "${SUBSET_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --alg entropy \
    --seed "${SEED}" \
    --temperature 0.1 \
    --output_file "${RESULTS_DIR}/rb2-heldout_base_k1_temp0.1_T128_infer_seed${SEED}.json"

  echo "Running ${MODEL_NAME}, seed ${SEED}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((29610 + SEED)) bon_infer.py \
    --dream_model "${BASE_MODEL}" \
    --adapter_model "${MODEL_DIR}/checkpoint-final" \
    --K 1 \
    --T 128 \
    --max_new_tokens 128 \
    --dataset_path "${HELDOUT_JSON}" \
    --split heldout \
    --prompt_field prompt \
    --subset_size "${SUBSET_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --alg entropy \
    --seed "${SEED}" \
    --temperature 0.1 \
    --output_file "${RESULTS_DIR}/rb2-heldout_${MODEL_NAME}_k1_temp0.1_T128_infer_seed${SEED}.json"
done

#!/bin/bash
# Strict apples-to-apples recheck:
#   base vs wildchat-chosen checkpoint-200
# with identical seeds/datasets/settings/output root.
set -euo pipefail

cd /home/an34232/Repos/entrgi/main_expts

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
RESULTS_DIR="${RESULTS_DIR:-/hdd1/an34232/entrgi_recheck_base_vs_wc200}"
WC200_ADAPTER="${WC200_ADAPTER:-/hdd1/an34232/entrgi_sft_models/dream-chosen-trunc128-sft-lora-r32-alllinear-500steps/checkpoint-200}"
WC200_NAME="${WC200_NAME:-dream-chosen-trunc128-sft-lora-r32-alllinear-500steps_checkpoint-200}"

SEEDS=(1 2 3)
DATASETS=(
  #"THU-KEG/RM-Bench:train:prompt:rm-bench"
  #"ScalerLab/JudgeBench:gpt:question:judgebench"
  "allenai/reward-bench-2:test:prompt:reward-bench-2"
)

K=1
TEMP=0.1
T=128
MAX_NEW_TOKENS=128
SUBSET_SIZE=64
BATCH_SIZE=8
NUM_GPUS=2

mkdir -p "${RESULTS_DIR}/base"
mkdir -p "${RESULTS_DIR}/${WC200_NAME}"

run_one() {
  local adapter_path="$1"
  local adapter_name="$2"
  local out_dir="${RESULTS_DIR}/${adapter_name}"

  for cfg in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET SPLIT PROMPT_FIELD PREFIX <<< "$cfg"
    for SEED in "${SEEDS[@]}"; do
      OUT_JSON="${out_dir}/${PREFIX}_${adapter_name}_k${K}_temp${TEMP}_T${T}_infer_seed${SEED}.json"
      if [ -f "${OUT_JSON}" ]; then
        continue
      fi

      if [ -n "${adapter_path}" ]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((36100 + SEED)) bon_infer.py \
          --dream_model "${BASE_MODEL}" \
          --adapter_model "${adapter_path}" \
          --K ${K} \
          --T ${T} \
          --max_new_tokens ${MAX_NEW_TOKENS} \
          --dataset_path "${DATASET}" \
          --split "${SPLIT}" \
          --prompt_field "${PROMPT_FIELD}" \
          --subset_size ${SUBSET_SIZE} \
          --batch_size ${BATCH_SIZE} \
          --alg entropy \
          --seed "${SEED}" \
          --temperature "${TEMP}" \
          --output_file "${OUT_JSON}"
      else
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((36200 + SEED)) bon_infer.py \
          --dream_model "${BASE_MODEL}" \
          --K ${K} \
          --T ${T} \
          --max_new_tokens ${MAX_NEW_TOKENS} \
          --dataset_path "${DATASET}" \
          --split "${SPLIT}" \
          --prompt_field "${PROMPT_FIELD}" \
          --subset_size ${SUBSET_SIZE} \
          --batch_size ${BATCH_SIZE} \
          --alg entropy \
          --seed "${SEED}" \
          --temperature "${TEMP}" \
          --output_file "${OUT_JSON}"
      fi
    done
  done
}

run_one "" "base"
run_one "${WC200_ADAPTER}" "${WC200_NAME}"

echo "Done. Results in: ${RESULTS_DIR}"

# #!/bin/bash
# set -euo pipefail

# cd /home/an34232/Repos/entrgi/main_expts

# BASE_MODEL="Dream-org/Dream-v0-Instruct-7B"
# FIXED_DIR="${FIXED_DIR:-/home/an34232/Repos/entrgi/main_expts/data/fixed_eval_v1}"
# RESULTS_DIR="${RESULTS_DIR:-/hdd1/an34232/entrgi_fixed_eval_v1}"
# SUBSET_SIZE="${SUBSET_SIZE:-256}"
# BATCH_SIZE="${BATCH_SIZE:-8}"
# NUM_GPUS="${NUM_GPUS:-2}"

# WC200_ADAPTER="${WC200_ADAPTER:-/hdd1/an34232/entrgi_sft_models/dream-chosen-trunc128-sft-lora-r32-alllinear-500steps/checkpoint-200}"
# WC200_NAME="wildchat-chosen-500steps-ckpt200"

# mkdir -p "${RESULTS_DIR}"

# DATASETS=(
#   #"judgebench:${FIXED_DIR}/judgebench_eval_fixed.jsonl"
#   "reward-bench-2:${FIXED_DIR}/reward-bench-2_eval_fixed.jsonl"
#   "rm-bench:${FIXED_DIR}/rm-bench_eval_fixed.jsonl"
# )

# for entry in "${DATASETS[@]}"; do
#   IFS=':' read -r NAME FILE <<< "$entry"
#   for SEED in 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((33100 + SEED)) bon_infer.py \
#       --dream_model "${BASE_MODEL}" \
#       --K 1 \
#       --T 128 \
#       --max_new_tokens 128 \
#       --dataset_path "${FILE}" \
#       --split heldout \
#       --prompt_field prompt \
#       --subset_size "${SUBSET_SIZE}" \
#       --batch_size "${BATCH_SIZE}" \
#       --alg entropy \
#       --seed "${SEED}" \
#       --temperature 0.1 \
#       --output_file "${RESULTS_DIR}/${NAME}_base_k1_temp0.1_T128_seed${SEED}.json"

#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((33200 + SEED)) bon_infer.py \
#       --dream_model "${BASE_MODEL}" \
#       --adapter_model "${WC200_ADAPTER}" \
#       --K 1 \
#       --T 128 \
#       --max_new_tokens 128 \
#       --dataset_path "${FILE}" \
#       --split heldout \
#       --prompt_field prompt \
#       --subset_size "${SUBSET_SIZE}" \
#       --batch_size "${BATCH_SIZE}" \
#       --alg entropy \
#       --seed "${SEED}" \
#       --temperature 0.1 \
#       --output_file "${RESULTS_DIR}/${NAME}_${WC200_NAME}_k1_temp0.1_T128_seed${SEED}.json"
#   done
# done

cd /home/an34232/Repos/entrgi/main_expts
OUT=/hdd1/an34232/entrgi_rb2_exact_oldprompt_replay
WC200=/hdd1/an34232/entrgi_sft_models/dream-chosen-trunc128-sft-lora-r32-alllinear-500steps/checkpoint-200
mkdir -p "$OUT"

for SEED in 1 2 3; do
  FILE="/home/an34232/Repos/entrgi/main_expts/data/rb2_exact_old_prompts/reward-bench-2_seed${SEED}_exact.jsonl"

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 --master_port=$((35100 + SEED)) bon_infer.py \
    --dream_model Dream-org/Dream-v0-Instruct-7B \
    --K 1 --T 128 --max_new_tokens 128 \
    --dataset_path "$FILE" --split heldout --prompt_field prompt \
    --subset_size 64 --batch_size 8 --alg entropy --seed "$SEED" --temperature 0.1 \
    --output_file "$OUT/reward-bench-2_base_seed${SEED}.json"

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 --master_port=$((35200 + SEED)) bon_infer.py \
    --dream_model Dream-org/Dream-v0-Instruct-7B \
    --adapter_model "$WC200" \
    --K 1 --T 128 --max_new_tokens 128 \
    --dataset_path "$FILE" --split heldout --prompt_field prompt \
    --subset_size 64 --batch_size 8 --alg entropy --seed "$SEED" --temperature 0.1 \
    --output_file "$OUT/reward-bench-2_wc200_seed${SEED}.json"
done
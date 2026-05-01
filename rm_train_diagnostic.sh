cd /home/an34232/Repos/entrgi/main_expts

BASE_MODEL="Dream-org/Dream-v0-Instruct-7B"
HELDOUT_JSON="/home/an34232/Repos/entrgi/main_expts/data/rm_bench_diag_split_trunc128/prompt_only/test.jsonl"
RESULTS_DIR="/hdd1/an34232/entrgi_sft_results_rm_diag_extra"
SUBSET_SIZE=64
BATCH_SIZE=8
NUM_GPUS=2

mkdir -p "${RESULTS_DIR}"

for SEED in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((32100 + SEED)) bon_infer.py \
    --dream_model "${BASE_MODEL}" \
    --adapter_model "/hdd1/an34232/entrgi_sft_models/dream-chosen-trunc128-sft-lora-r32-alllinear-500steps/checkpoint-200" \
    --K 1 --T 128 --max_new_tokens 128 \
    --dataset_path "${HELDOUT_JSON}" --split heldout --prompt_field prompt \
    --subset_size "${SUBSET_SIZE}" --batch_size "${BATCH_SIZE}" \
    --alg entropy --seed "${SEED}" --temperature 0.1 \
    --output_file "${RESULTS_DIR}/rm-heldout_dream-chosen-trunc128-sft-lora-r32-alllinear-500steps_checkpoint-200_k1_temp0.1_T128_infer_seed${SEED}.json"

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((32200 + SEED)) bon_infer.py \
    --dream_model "${BASE_MODEL}" \
    --adapter_model "/hdd1/an34232/entrgi_sft_models/dream-chosen-trunc128-sft-lora-r32-alllinear-500steps/checkpoint-500" \
    --K 1 --T 128 --max_new_tokens 128 \
    --dataset_path "${HELDOUT_JSON}" --split heldout --prompt_field prompt \
    --subset_size "${SUBSET_SIZE}" --batch_size "${BATCH_SIZE}" \
    --alg entropy --seed "${SEED}" --temperature 0.1 \
    --output_file "${RESULTS_DIR}/rm-heldout_dream-chosen-trunc128-sft-lora-r32-alllinear-500steps_checkpoint-500_k1_temp0.1_T128_infer_seed${SEED}.json"

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((32300 + SEED)) bon_infer.py \
    --dream_model "${BASE_MODEL}" \
    --adapter_model "/hdd1/an34232/entrgi_sft_models/dream-entrgi-sft-lora-r32-alllinear-500steps/checkpoint-200" \
    --K 1 --T 128 --max_new_tokens 128 \
    --dataset_path "${HELDOUT_JSON}" --split heldout --prompt_field prompt \
    --subset_size "${SUBSET_SIZE}" --batch_size "${BATCH_SIZE}" \
    --alg entropy --seed "${SEED}" --temperature 0.1 \
    --output_file "${RESULTS_DIR}/rm-heldout_dream-entrgi-sft-lora-r32-alllinear-500steps_checkpoint-200_k1_temp0.1_T128_infer_seed${SEED}.json"

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((32400 + SEED)) bon_infer.py \
    --dream_model "${BASE_MODEL}" \
    --adapter_model "/hdd1/an34232/entrgi_sft_models/dream-entrgi-sft-lora-r32-alllinear-500steps/checkpoint-500" \
    --K 1 --T 128 --max_new_tokens 128 \
    --dataset_path "${HELDOUT_JSON}" --split heldout --prompt_field prompt \
    --subset_size "${SUBSET_SIZE}" --batch_size "${BATCH_SIZE}" \
    --alg entropy --seed "${SEED}" --temperature 0.1 \
    --output_file "${RESULTS_DIR}/rm-heldout_dream-entrgi-sft-lora-r32-alllinear-500steps_checkpoint-500_k1_temp0.1_T128_infer_seed${SEED}.json"
done

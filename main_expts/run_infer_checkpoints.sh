#!/bin/bash
# Run inference-only Dream benchmark generation for a LoRA checkpoint or adapter path,
# then optionally evaluate the generated responses with LMUnit.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 --base | <adapter_checkpoint_or_root_path> [adapter_name]"
    echo ""
    echo "Example:"
    echo "  $0 --base"
    echo "  $0 /hdd1/an34232/entrgi_sft_models/dream-entrgi-sft-lora-r32-alllinear/checkpoint-final"
    echo "  $0 /hdd1/an34232/entrgi_sft_models/dream-entrgi-sft-lora-r32-alllinear"
    exit 1
fi

RUN_BASE_ONLY=0
if [ "$1" = "--base" ]; then
    RUN_BASE_ONLY=1
    INPUT_PATH=""
    USER_ADAPTER_NAME="${2:-base}"
else
    INPUT_PATH="$1"
    USER_ADAPTER_NAME="${2:-}"
fi

# =====================
# Configuration
# =====================

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
RESULTS_DIR="${RESULTS_DIR:-./results_infer}"

SEEDS=(1 2 3 4 5)
K_VALUES=(1)
TEMPERATURES=(0.1)

# Datasets configuration: (path, split, prompt_field, prefix)
DATASETS=(
    "THU-KEG/RM-Bench:train:prompt:rm-bench"
    "ScalerLab/JudgeBench:gpt:question:judgebench"
    "allenai/reward-bench-2:test:prompt:reward-bench-2"
)

USE_WILDCHAT_HELDOUT="${USE_WILDCHAT_HELDOUT:-0}"
if [ "${USE_WILDCHAT_HELDOUT}" = "1" ]; then
    DATASETS=(
        "/home/an34232/Repos/entrgi/main_expts/data/oracle_entrgi_sft/test.jsonl:heldout:prompt:wildchat-heldout"
    )
fi

T=128
MAX_NEW_TOKENS=128
SUBSET_SIZE=128
BATCH_SIZE=16
NUM_GPUS=2
DO_LMUNIT="${DO_LMUNIT:-1}"
LMUNIT_MODEL="${LMUNIT_MODEL:-ContextualAI/LMUnit-qwen2.5-72b}"

# =====================
# Main experiment loop
# =====================

run_for_adapter() {
    local adapter_path="$1"
    local adapter_name="$2"
    local adapter_results_dir="${RESULTS_DIR%/}/${adapter_name}"
    local adapter_lmunit_dir="${adapter_results_dir}/lmunit_results"

    mkdir -p "${adapter_results_dir}"
    mkdir -p "${adapter_lmunit_dir}"

    echo "========================================"
    echo "Running inference for adapter: ${adapter_path}"
    echo "Adapter label: ${adapter_name}"
    echo "Base model: ${BASE_MODEL}"
    echo "Results dir: ${adapter_results_dir}"
    echo "========================================"

    for TEMP in "${TEMPERATURES[@]}"; do
        echo "Running with temperature=${TEMP}"

        for K in "${K_VALUES[@]}"; do
            echo "Running with K=${K}"

            for DATASET_CONFIG in "${DATASETS[@]}"; do
                IFS=':' read -r DATASET SPLIT PROMPT_FIELD PREFIX <<< "$DATASET_CONFIG"
                echo "Dataset: ${DATASET} (${PREFIX})"

                for SEED in "${SEEDS[@]}"; do
                    echo "  Seed: ${SEED}"

                    FILE_INFER="${adapter_results_dir}/${PREFIX}_${adapter_name}_k${K}_temp${TEMP}_T${T}_infer_seed${SEED}.json"
                    #FILE_ENTRGI="${adapter_results_dir}/${PREFIX}_${adapter_name}_k${K}_temp${TEMP}_T${T}_entrgi_seed${SEED}.json"
                    EVAL_INFER="${adapter_lmunit_dir}/${PREFIX}_${adapter_name}_k${K}_temp${TEMP}_T${T}_infer_seed${SEED}_eval.json"

                    if [ ! -f "$FILE_INFER" ]; then
                        echo "    Generating responses..."
                        if [ -n "${adapter_path}" ]; then
                            torchrun --nproc_per_node=${NUM_GPUS} --master_port=29510 bon_infer.py \
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
                                --output_file "${FILE_INFER}"
                        else
                            CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29510 bon_infer.py \
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
                                --output_file "${FILE_INFER}"
                        fi
                    fi

                    # if [ ! -f "$FILE_ENTRGI" ]; then
                    #     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29510 bon_infer.py \
                    #                 --dream_model "${BASE_MODEL}" \
                    #                 --K ${K} \
                    #                 --T ${T} \
                    #                 --M 3 \
                    #                 --eta 0.5 \
                    #                 --max_new_tokens ${MAX_NEW_TOKENS} \
                    #                 --dataset_path "${DATASET}" \
                    #                 --split "${SPLIT}" \
                    #                 --prompt_field "${PROMPT_FIELD}" \
                    #                 --subset_size ${SUBSET_SIZE} \
                    #                 --batch_size ${BATCH_SIZE} \
                    #                 --alg anchor \
                    #                 --use_entrgi \
                    #                 --seed "${SEED}" \
                    #                 --temperature "${TEMP}" \
                    #                 --output_file "${FILE_INFER}"
                    # fi


                    if [ "${DO_LMUNIT}" = "1" ] && [ -f "$FILE_INFER" ] && [ ! -f "$EVAL_INFER" ]; then
                        echo "    Evaluating with LMUnit..."
                        CUDA_VISIBLE_DEVICES=2,3 python lmunit_eval.py \
                            --file "${FILE_INFER}" \
                            --output "${EVAL_INFER}" \
                            --model "${LMUNIT_MODEL}" \
                            --tp_size ${NUM_GPUS}
                    fi
                done
            done
        done
    done
}

declare -a ADAPTER_PATHS=()

if [ "${RUN_BASE_ONLY}" = "1" ]; then
    run_for_adapter "" "${USER_ADAPTER_NAME}"
elif [ -f "${INPUT_PATH}/adapter_config.json" ]; then
    ADAPTER_PATHS+=("${INPUT_PATH}")
elif [ -d "${INPUT_PATH}" ]; then
    while IFS= read -r path; do
        ADAPTER_PATHS+=("${path}")
    done < <(find "${INPUT_PATH}" -mindepth 1 -maxdepth 2 -type f -name adapter_config.json -printf '%h\n' | sort -u)
else
    echo "Error: input path not found or not a directory: ${INPUT_PATH}"
    exit 1
fi

if [ "${RUN_BASE_ONLY}" != "1" ] && [ ${#ADAPTER_PATHS[@]} -eq 0 ]; then
    echo "Error: no adapter checkpoints found under ${INPUT_PATH}"
    exit 1
fi

if [ "${RUN_BASE_ONLY}" != "1" ]; then
    for adapter_path in "${ADAPTER_PATHS[@]}"; do
        if [ -n "${USER_ADAPTER_NAME}" ] && [ ${#ADAPTER_PATHS[@]} -eq 1 ]; then
            adapter_name="${USER_ADAPTER_NAME}"
        else
            parent_name="$(basename "$(dirname "${adapter_path}")")"
            leaf_name="$(basename "${adapter_path}")"
            if [ "${leaf_name}" = "${parent_name}" ]; then
                adapter_name="${leaf_name}"
            else
                adapter_name="${parent_name}_${leaf_name}"
            fi
        fi
        run_for_adapter "${adapter_path}" "${adapter_name}"
    done
fi

echo "========================================"
echo "Checkpoint inference completed!"
echo "========================================"
echo "Results saved under: ${RESULTS_DIR}"
if [ "${DO_LMUNIT}" = "1" ]; then
    echo "LMUnit evaluations saved under: ${RESULTS_DIR}"
else
    echo "LMUnit evaluation skipped because DO_LMUNIT=${DO_LMUNIT}"
fi

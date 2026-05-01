#!/bin/bash
# EntRGi/APS with AR log-likelihood reward
# Uses an autoregressive model's log P(response) as the reward signal

set -e

# =====================
# Configuration
# =====================

RESULTS_DIR="./results1"
LMUNIT_DIR="${RESULTS_DIR}/lmunit_results"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LMUNIT_DIR"

SEEDS=(1 2 3)
K_VALUES=(4)
TEMPERATURES=(0.1)
T_VALUES=(32 64 128)

DATASETS=(
    "THU-KEG/RM-Bench:train:prompt:rm-bench"
    "ScalerLab/JudgeBench:gpt:question:judgebench"
    "allenai/reward-bench-2:test:prompt:reward-bench-2"
)

# AR reward models: (model_path, short_name)
AR_REWARD_MODELS=(
    "Qwen/Qwen3-0.6B:qwen3-0.6b-ar"
)

M=3
ETA=0.5
MAX_NEW_TOKENS=128
SUBSET_SIZE=64
BATCH_SIZE=4
NUM_GPUS=1

# =====================
# Main experiment loop
# =====================

for TEMP in "${TEMPERATURES[@]}"; do
    echo "========================================"
    echo "Running AR experiments with temperature=${TEMP}"
    echo "========================================"

    for K in "${K_VALUES[@]}"; do
        echo "Running with K=${K}..."

        for T in "${T_VALUES[@]}"; do
            echo "Running with T=${T}..."

            for DATASET_CONFIG in "${DATASETS[@]}"; do
                IFS=':' read -r DATASET SPLIT PROMPT_FIELD PREFIX <<< "$DATASET_CONFIG"
                echo "Dataset: ${DATASET} (${PREFIX})"

                for RM_CONFIG in "${AR_REWARD_MODELS[@]}"; do
                    RM_PATH="${RM_CONFIG%%:*}"
                    RM_NAME="${RM_CONFIG##*:}"
                    echo "  AR reward model: ${RM_NAME}"

                    for SEED in "${SEEDS[@]}"; do
                        echo "    Seed: ${SEED}"

                        # =====================
                        # Output file paths
                        # =====================

                        FILE_ENTRGI="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_entrgi3-ar_seed${SEED}.json"
                        FILE_APS="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_aps-ar_seed${SEED}.json"
                        FILE_EXPECTATION="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_expectation-ar_seed${SEED}.json"

                        EVAL_ENTRGI="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_entrgi3-ar_seed${SEED}_eval.json"
                        EVAL_APS="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_aps-ar_seed${SEED}_eval.json"
                        EVAL_EXPECTATION="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_expectation-ar_seed${SEED}_eval.json"

                        # =====================
                        # Run EntRGi (AR)
                        # =====================

                        if [ ! -f "$FILE_ENTRGI" ]; then
                            echo "      Running EntRGi (AR)..."
                            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 entrgi_ar.py \
                                --K ${K} \
                                --M ${M} \
                                --eta ${ETA} \
                                --T ${T} \
                                --max_new_tokens ${MAX_NEW_TOKENS} \
                                --dataset_path "$DATASET" \
                                --split "$SPLIT" \
                                --prompt_field "$PROMPT_FIELD" \
                                --subset_size ${SUBSET_SIZE} \
                                --batch_size ${BATCH_SIZE} \
                                --alg anchor \
                                --seed "$SEED" \
                                --temperature "$TEMP" \
                                --use_entrgi \
                                --reward_model "$RM_PATH" \
                                --output_file "$FILE_ENTRGI"
                        fi

                        # =====================
                        # Run APS (AR)
                        # =====================

                        if [ ! -f "$FILE_APS" ]; then
                            echo "      Running APS (AR)..."
                            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29501 entrgi_ar.py \
                                --K ${K} \
                                --M ${M} \
                                --eta ${ETA} \
                                --T ${T} \
                                --max_new_tokens ${MAX_NEW_TOKENS} \
                                --dataset_path "$DATASET" \
                                --split "$SPLIT" \
                                --prompt_field "$PROMPT_FIELD" \
                                --subset_size ${SUBSET_SIZE} \
                                --batch_size ${BATCH_SIZE} \
                                --alg entropy \
                                --seed "$SEED" \
                                --temperature "$TEMP" \
                                --use_aps \
                                --reward_model "$RM_PATH" \
                                --output_file "$FILE_APS"
                        fi

                        # =====================
                        # Run Expectation / Vanilla (AR)
                        # =====================

                        if [ ! -f "$FILE_EXPECTATION" ]; then
                            echo "      Running Expectation (AR)..."
                            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29502 entrgi_ar.py \
                                --K ${K} \
                                --M ${M} \
                                --eta ${ETA} \
                                --T ${T} \
                                --max_new_tokens ${MAX_NEW_TOKENS} \
                                --dataset_path "$DATASET" \
                                --split "$SPLIT" \
                                --prompt_field "$PROMPT_FIELD" \
                                --subset_size ${SUBSET_SIZE} \
                                --batch_size ${BATCH_SIZE} \
                                --alg entropy \
                                --seed "$SEED" \
                                --temperature "$TEMP" \
                                --reward_model "$RM_PATH" \
                                --output_file "$FILE_EXPECTATION"
                        fi

                        # =====================
                        # LMUnit Evaluation
                        # =====================

                        # if [ -f "$FILE_ENTRGI" ] && [ ! -f "$EVAL_ENTRGI" ]; then
                        #     echo "      Evaluating EntRGi (AR) with LMUnit..."
                        #     python lmunit_eval.py \
                        #         --file "$FILE_ENTRGI" \
                        #         --output "$EVAL_ENTRGI" \
                        #         --model ContextualAI/LMUnit-qwen2.5-72b \
                        #         --tp_size ${NUM_GPUS}
                        # fi

                        # if [ -f "$FILE_APS" ] && [ ! -f "$EVAL_APS" ]; then
                        #     echo "      Evaluating APS (AR) with LMUnit..."
                        #     python lmunit_eval.py \
                        #         --file "$FILE_APS" \
                        #         --output "$EVAL_APS" \
                        #         --model ContextualAI/LMUnit-qwen2.5-72b \
                        #         --tp_size ${NUM_GPUS}
                        # fi

                        # if [ -f "$FILE_EXPECTATION" ] && [ ! -f "$EVAL_EXPECTATION" ]; then
                        #     echo "      Evaluating Expectation (AR) with LMUnit..."
                        #     python lmunit_eval.py \
                        #         --file "$FILE_EXPECTATION" \
                        #         --output "$EVAL_EXPECTATION" \
                        #         --model ContextualAI/LMUnit-qwen2.5-72b \
                        #         --tp_size ${NUM_GPUS}
                        # fi

                    done  # SEED
                done  # RM_CONFIG
            done  # DATASET_CONFIG
        done  # T
    done  # K
done  # TEMP

echo "========================================"
echo "All AR experiments completed!"
echo "========================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "LMUnit evaluations saved to: ${LMUNIT_DIR}"
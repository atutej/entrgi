cd /home/an34232/Repos/entrgi/dllm

accelerate launch --num_processes 2 examples/dream/sft.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --dataset_args /home/an34232/Repos/entrgi/main_expts/data/rm_bench_diag_split_trunc128/chosen_sft   \
    --load_preprocessed_data False   \
    --mask_prompt_loss True   \
    --num_train_epochs 3   \
    --learning_rate 5e-6   \
    --per_device_train_batch_size 4   \
    --per_device_eval_batch_size 4   \
    --lora True   \
    --r 32   \
    --lora_alpha 32   \
    --lora_dropout 0.1   \
    --target_modules all-linear   \
    --bias none   \
    --save_steps 75   \
    --save_total_limit 10   \
    --logging_steps 10   \
    --eval_steps 100   \
    --output_dir /hdd1/an34232/entrgi_sft_models/dream-rm-bench-diag-split-trunc128-sft-lora-r32-alllinear

cd /home/an34232/Repos/entrgi/main_expts


#DO_LMUNIT=0 USE_WILDCHAT_HELDOUT=1 RESULTS_DIR=/hdd1/an34232/entrgi_sft_results_heldout bash /home/an34232/Repos/entrgi/main_expts/run_infer_checkpoints.sh /hdd1/an34232/entrgi_sft_models/dream-skywork-pref-chosen-trunc128-sft-lora-r32-alllinear
#DO_LMUNIT=0 RESULTS_DIR=/hdd1/an34232/entrgi_sft_results bash /home/an34232/Repos/entrgi/main_expts/run_infer_checkpoints.sh /hdd1/an34232/entrgi_sft_models/dream-skywork-pref-chosen-trunc128-sft-lora-r32-alllinear


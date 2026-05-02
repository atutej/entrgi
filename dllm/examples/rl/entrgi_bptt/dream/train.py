"""
EntRGi-BPTT online training for Dream-v0-Instruct-7B.

Reward gradients flow directly into θ at one denoising step k per update
(truncated BPTT with k=1).  No separate SFT loss or RWR weighting.

Quick sanity check:
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/rl/entrgi_bptt/dream/train.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --dataset wildchat \\
        --max_steps 50 --steps 16 \\
        --output_dir .models/Dream-v0-Instruct-7B/entrgi-bptt-test

Multi-GPU run:
    accelerate launch \\
        --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/rl/entrgi_bptt/dream/train.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \\
        --dataset wildchat \\
        --max_steps 500 --learning_rate 5e-6 \\
        --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \\
        --steps 128 --k_frac -1.0 \\
        --guidance_reward_model Skywork/Skywork-Reward-V2-Qwen3-0.6B \\
        --output_dir .models/Dream-v0-Instruct-7B/entrgi-bptt-wildchat
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.pipelines.rl import get_dataset_and_rewards
from dllm.pipelines.rl.entrgi_bptt import EntrgiBpttConfig, EntrgiBpttTrainer

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class TrainingArguments(EntrgiBpttConfig):
    output_dir: str = ".models/Dream-v0-Instruct-7B/entrgi-bptt"
    dataset: Optional[str] = field(
        default="wildchat",
        metadata={"help": "Dataset: gsm8k, countdown, math, wildchat."},
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in rule-based reward functions."},
    )
    scoring_reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Override the reward model used for dataset scoring (wildchat). Defaults to Skywork-1.7B if unset."},
    )


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "Dream-org/Dream-v0-Instruct-7B"

    from dllm.pipelines.rl.grpo.rewards.skywork import make_skywork_reward_func

    dataset, reward_functions = get_dataset_and_rewards(training_args.dataset)
    if training_args.scoring_reward_model and training_args.dataset == "wildchat":
        reward_functions = [make_skywork_reward_func(training_args.scoring_reward_model)]

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    train_set = dataset.shuffle(seed=training_args.seed)

    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=model_config.load_in_4bit if hasattr(model_config, "load_in_4bit") else False,
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=model_config.lora_dropout,
        )

    logger.info("Starting EntRGi-BPTT training (Dream)...")
    trainer = EntrgiBpttTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()


if __name__ == "__main__":
    train()

"""
EntRGi online SFT for Dream-v0-Instruct-7B.

Training loop:
  1. Generate K completions per prompt with EntRGi gradient guidance
     (M Adam steps on masked logits every apply_every_k denoising steps).
  2. Score with Skywork reward.
  3. Update model with reward-weighted SFT (RWR / soft-EM).

The guidance reward model (default: Skywork-0.6B) is kept frozen during training.
The policy model is updated only by the weighted SFT loss.

Quick sanity check — wildchat, 1 GPU, no LoRA:
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/rl/entrgi/dream/train.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --dataset wildchat \\
        --max_steps 50 --M 1 --apply_every_k 4 \\
        --output_dir .models/Dream-v0-Instruct-7B/entrgi-wildchat-test

Multi-GPU, LoRA, approximate 1× cost (M=1, apply_every_k=8):
    accelerate launch \\
        --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/rl/entrgi/dream/train.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \\
        --dataset wildchat \\
        --max_steps 500 --learning_rate 5e-6 \\
        --num_generations 4 --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 2 \\
        --M 1 --apply_every_k 8 --eta 1.0 --rwr_temperature 0.1 \\
        --steps 128 \\
        --output_dir .models/Dream-v0-Instruct-7B/entrgi-wildchat

Verifiable-reward sanity checks (same datasets as GRPO train.py):
    --dataset countdown   (rule-based reward, no guidance reward model needed)
    --dataset gsm8k
    --dataset math
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.pipelines.rl import EntrgiOnlineSFTConfig, EntrgiOnlineSFTTrainer, get_dataset_and_rewards

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingArguments(EntrgiOnlineSFTConfig):
    output_dir: str = ".models/Dream-v0-Instruct-7B/entrgi"
    dataset: Optional[str] = field(
        default="wildchat",
        metadata={
            "help": (
                "Dataset: gsm8k, countdown, sudoku, math, code, wildchat. "
                "Use gsm8k/countdown/math to verify reward / loss before wildchat."
            )
        },
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in rule-based reward functions."},
    )
    # Dream sampler knobs (exposed for CLI; forwarded to EntrgiDreamSamplerConfig)
    dream_alg: str = field(
        default="entropy",
        metadata={"help": "Confidence algorithm: entropy, maskgit_plus, topk_margin."},
    )
    dream_top_p: float = field(default=0.95, metadata={"help": "top-p for Dream sampling."})
    dream_top_k: int = field(default=50, metadata={"help": "top-k for Dream sampling."})
    deprioritize_eos: bool = field(
        default=False,
        metadata={
            "help": (
                "Deprioritize EOS during denoising (set confidence=-inf at EOS-sampled "
                "positions). False during training so guidance can steer the model toward "
                "completing responses within the token budget."
            )
        },
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Reward model for scoring completions (RWR weights + logging). "
                "Only applies to open-ended datasets (wildchat). "
                "Defaults to Skywork/Skywork-Reward-V2-Qwen3-1.7B when unset."
            )
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "Dream-org/Dream-v0-Instruct-7B"

    # ---- Dataset & rewards -------------------------------------------------------
    dataset, reward_functions = get_dataset_and_rewards(
        training_args.dataset,
        reward_model=training_args.reward_model,
    )

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    train_set = dataset.shuffle(seed=training_args.seed)

    # ---- Model & Tokenizer -------------------------------------------------------
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=(
            model_config.load_in_4bit
            if hasattr(model_config, "load_in_4bit")
            else False
        ),
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    # ---- LoRA -------------------------------------------------------------------
    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ],
            lora_dropout=model_config.lora_dropout,
        )

    # ---- Sampler config ---------------------------------------------------------
    from dllm.pipelines.rl.entrgi.sampler import EntrgiDreamSamplerConfig

    sampler_config = EntrgiDreamSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        temperature=training_args.temperature or 1.0,
        cfg_scale=training_args.cfg_scale,
        alg=training_args.dream_alg,
        top_p=training_args.dream_top_p,
        top_k=training_args.dream_top_k,
        right_shift_logits=True,
        M=training_args.M,
        apply_every_k=training_args.apply_every_k,
        eta=training_args.eta,
        reward_temperature=training_args.rwr_temperature,
        num_generations=training_args.num_generations,
        aps=training_args.aps,
        deprioritize_eos=training_args.deprioritize_eos,
    )

    # ---- Trainer ----------------------------------------------------------------
    logger.info("Starting EntRGi online SFT (Dream)...")
    trainer = EntrgiOnlineSFTTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler_config=sampler_config,
    )

    if training_args.save_steps % training_args.num_iterations != 0:
        import warnings

        warnings.warn(
            f"save_steps ({training_args.save_steps}) is not divisible by "
            f"num_iterations ({training_args.num_iterations}). If resuming from "
            f"a checkpoint, you may need to manually pick a compatible checkpoint."
        )

    trainer.train()


if __name__ == "__main__":
    train()

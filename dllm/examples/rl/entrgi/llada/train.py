"""
EntRGi online SFT for LLaDA-8B-Instruct.

Training loop:
  1. Generate K completions per prompt with EntRGi gradient guidance
     (M Adam steps on masked logits every apply_every_k denoising steps).
  2. Score with Skywork reward (or a verifiable reward for math/code datasets).
  3. Update model with reward-weighted SFT (RWR / soft-EM).

Key differences from the Dream variant:
  - No position IDs (LLaDA is bidirectional, no RoPE shift needed).
  - No logit right-shift (LLaDA predicts at the masked position directly).
  - zero_unmatched_embeddings=True: policy tokens absent from the reward model
    vocabulary get a zero embedding instead of UNK, avoiding spurious gradients.

Quick sanity check — wildchat, 1 GPU, no LoRA:
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/rl/entrgi/llada/train.py \\
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \\
        --dataset wildchat \\
        --max_steps 50 --M 1 --apply_every_k 4 \\
        --output_dir .models/LLaDA-8B-Instruct/entrgi-wildchat-test

Multi-GPU, LoRA:
    accelerate launch \\
        --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/rl/entrgi/llada/train.py \\
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \\
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \\
        --dataset wildchat \\
        --max_steps 500 --learning_rate 5e-6 \\
        --num_generations 4 --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 2 \\
        --M 1 --apply_every_k 8 --eta 1.0 --rwr_temperature 0.1 \\
        --steps 128 \\
        --output_dir .models/LLaDA-8B-Instruct/entrgi-wildchat

Verifiable-reward datasets (no guidance reward model needed):
    --dataset countdown
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
from dllm.pipelines.rl.entrgi import EntrgiLLaDASampler, EntrgiLLaDASamplerConfig

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingArguments(EntrgiOnlineSFTConfig):
    output_dir: str = ".models/LLaDA-8B-Instruct/entrgi"
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
    # LLaDA sampler knobs
    llada_alg: str = field(
        default="entropy",
        metadata={"help": "Confidence algorithm for remasking: entropy, maskgit_plus, topk_margin."},
    )
    llada_top_p: float = field(default=0.95, metadata={"help": "top-p for sampling."})
    llada_top_k: int = field(default=50, metadata={"help": "top-k for sampling."})
    soft_only: bool = field(
        default=False,
        metadata={"help": "Soft ablation: entropy_weight=0 (pure continuous embeddings, no STE)."},
    )
    deprioritize_eos: bool = field(
        default=False,
        metadata={"help": "Deprioritize EOS during denoising."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Reward model for scoring completions. "
                "Only applies to open-ended datasets (wildchat). "
                "Defaults to Skywork/Skywork-Reward-V2-Qwen3-1.7B when unset."
            )
        },
    )
    # Always use zero embeddings for unmatched tokens with LLaDA
    zero_unmatched_embeddings: bool = field(
        default=True,
        metadata={"help": "Zero embedding for policy tokens absent from reward model vocab (recommended for LLaDA)."},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "GSAI-ML/LLaDA-8B-Instruct"

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
    sampler_config = EntrgiLLaDASamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        temperature=training_args.temperature or 1.0,
        cfg_scale=training_args.cfg_scale,
        alg=training_args.llada_alg,
        top_p=training_args.llada_top_p,
        top_k=training_args.llada_top_k,
        M=training_args.M,
        apply_every_k=training_args.apply_every_k,
        eta=training_args.eta,
        reward_temperature=training_args.rwr_temperature,
        num_generations=training_args.num_generations,
        aps=training_args.aps,
        soft_only=training_args.soft_only,
        deprioritize_eos=training_args.deprioritize_eos,
    )

    # ---- Trainer ----------------------------------------------------------------
    logger.info("Starting EntRGi online SFT (LLaDA)...")
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

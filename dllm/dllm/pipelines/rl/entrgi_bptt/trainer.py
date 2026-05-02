"""
EntRGi-BPTT: truncated BPTT(1) for masked diffusion RL.

For each training step:
  1. Run K independent denoising trajectories per prompt to full completion.
  2. At step k in each trajectory (k sampled once per batch): one grad-enabled
     forward pass → soft reward r_i with gradient attached to θ.
  3. Steps k+1..T run under no_grad to produce completed sequences.
  4. Decode completions → discrete reward logged as train/reward.
  5. loss = -sum_i softmax(r / β)_i * r_i   (partition-function weighted)

Memory cost: O(1) per trajectory — only a single model forward graph retained.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from trl.data_utils import maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.trainer.grpo_trainer import GRPOTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from contextlib import nullcontext
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.models import unwrap_model_for_generation

from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig
from dllm.pipelines.rl.grpo.trainer import DiffuGRPOConfig, DreamGRPOTrainer

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class EntrgiBpttConfig(DiffuGRPOConfig):
    """DiffuGRPOConfig + EntRGi-BPTT hyper-parameters."""

    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Must be False for LoRA + grad-checkpoint + DDP."},
    )
    guidance_reward_model: str = field(
        default="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        metadata={"help": "Reward model used for the soft reward gradient."},
    )
    reward_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature β for softmax weighting over K soft rewards."},
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "K: independent trajectories per prompt per training step."},
    )
    k_frac: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Fraction of denoising steps at which to apply the grad step. "
                "-1 = uniform random each batch; 0.0–1.0 = fixed fraction of T."
            )
        },
    )
    aps: bool = field(
        default=False,
        metadata={"help": "Set entropy_weight=1 (full STE) instead of entropy-aware interpolation."},
    )
    dream_alg: str = field(default="entropy", metadata={"help": "Token selection algorithm."})
    dream_top_p: float = field(default=0.95, metadata={"help": "top-p for denoising."})
    dream_top_k: int = field(default=50, metadata={"help": "top-k for denoising."})


class EntrgiBpttTrainer(DreamGRPOTrainer):
    """
    DreamGRPOTrainer where reward gradients flow directly into θ at step k.

    K trajectories per prompt run to full completion; gradient is applied only
    at the k/k+1 boundary.  Soft rewards are softmax-weighted (partition function)
    before summing into the loss.  Discrete terminal reward is logged for
    comparison with d1 / EntRGi curves.
    """

    def __init__(
        self,
        model,
        reward_funcs,
        args: Optional[EntrgiBpttConfig] = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        sampler_config = DreamSamplerConfig(
            steps=args.steps if args else 64,
            max_new_tokens=args.max_completion_length if args else 256,
            temperature=args.temperature or 1.0 if args else 1.0,
            alg=getattr(args, "dream_alg", "entropy"),
            top_p=getattr(args, "dream_top_p", 0.95),
            top_k=getattr(args, "dream_top_k", 50),
            right_shift_logits=True,
        )
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            sampler_config=sampler_config,
        )
        # Replace MDLMSampler (set by DiffuGRPOTrainer) with DreamSampler
        self.sampler = DreamSampler(model=self.model, tokenizer=self.processing_class)
        self._load_guidance_model(
            args.guidance_reward_model if args else "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
            self.accelerator.device,
        )

    # ------------------------------------------------------------------
    # Guidance model + reward cache
    # ------------------------------------------------------------------

    def _load_guidance_model(self, model_name: str, device):
        reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, num_labels=1
        ).to(device)
        reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad = False

        dream_vocab = self.processing_class.get_vocab()
        reward_vocab = reward_tokenizer.get_vocab()
        vocab_size = self.model.lm_head.out_features
        unk_id = reward_tokenizer.unk_token_id or reward_tokenizer.eos_token_id

        token_mapping = torch.full((vocab_size,), unk_id, dtype=torch.long, device=device)
        for tok, did in dream_vocab.items():
            if did < vocab_size and tok in reward_vocab:
                token_mapping[did] = reward_vocab[tok]

        mapped_embeds = reward_model.get_input_embeddings().weight[token_mapping].detach()

        self._reward_model = reward_model
        self._reward_tokenizer = reward_tokenizer
        self._token_mapping = token_mapping
        self._mapped_embeds = mapped_embeds

    def _build_reward_cache(self, prompt_tensor: torch.Tensor, device: str):
        prompt_text = self.processing_class.decode(prompt_tensor, skip_special_tokens=False)
        if "<|im_start|>user\n" in prompt_text:
            user_content = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        else:
            user_content = self.processing_class.decode(prompt_tensor, skip_special_tokens=True)

        conv = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "<<PLACEHOLDER>>"},
        ]
        template = self._reward_tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        prefix_text, suffix_text = template.split("<<PLACEHOLDER>>")

        embed_layer = self._reward_model.get_input_embeddings()
        with torch.no_grad():
            prefix_ids = self._reward_tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
            suffix_ids = self._reward_tokenizer(suffix_text, return_tensors="pt").input_ids.to(device)
            prefix_embeds = embed_layer(prefix_ids)
            suffix_embeds = embed_layer(suffix_ids)

        return prefix_embeds, suffix_embeds

    # ------------------------------------------------------------------
    # Soft reward: returns [B] tensor with grad attached to θ
    # ------------------------------------------------------------------

    def _compute_soft_rewards(
        self,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        x: torch.Tensor,
        max_prompt_len: int,
        caches: list,
        K: int,
        device: str,
    ) -> torch.Tensor:
        """
        Compute soft reward for each of the B = B_prompts * K sequences.

        logits still have grad attached to θ.  Returns shape [B].
        caches has length B_prompts; sequence b uses caches[b // K].
        """
        B = x.size(0)
        B_prompts = B // K
        response_len = x.size(1) - max_prompt_len
        embed_dtype = self._mapped_embeds.dtype
        embed_dim = self._mapped_embeds.shape[-1]

        max_prefix_len = max(c[0].shape[1] for c in caches)
        max_suffix_len = max(c[1].shape[1] for c in caches)

        batched_prefix = torch.zeros(B, max_prefix_len, embed_dim, device=device, dtype=embed_dtype)
        batched_suffix = torch.zeros(B, max_suffix_len, embed_dim, device=device, dtype=embed_dtype)
        prefix_lens, suffix_lens = [], []
        for b in range(B):
            pre, suf = caches[b // K]
            plen, slen = pre.shape[1], suf.shape[1]
            batched_prefix[b, max_prefix_len - plen:] = pre[0]
            batched_suffix[b, :slen] = suf[0]
            prefix_lens.append(plen)
            suffix_lens.append(slen)

        all_response_embeds = torch.zeros(B, response_len, embed_dim, device=device, dtype=embed_dtype)
        all_response_token_ids = x[:, max_prompt_len:].clone()
        reward_embed_layer = self._reward_model.get_input_embeddings()
        eos_id = self.processing_class.eos_token_id
        pad_id = getattr(self.processing_class, "pad_token_id", None) or eos_id

        for p in range(B):
            response_mask = mask_index[p][max_prompt_len:]
            mask_pos = torch.where(response_mask)[0]

            if len(mask_pos) > 0:
                cur_logits = logits[p][mask_index[p]]  # grad attached

                probs = F.softmax(cur_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                max_ent = torch.log(torch.tensor(probs.shape[-1], device=device, dtype=probs.dtype))
                entropy_weight = (
                    torch.ones_like(entropy)
                    if self.args.aps
                    else (entropy / max_ent).detach()
                )

                soft_embeds = torch.matmul(probs.to(embed_dtype), self._mapped_embeds)
                sampled_tokens = torch.multinomial(F.softmax(cur_logits.float(), dim=-1).detach(), 1, replacement=True)
                hard_embeds = self._mapped_embeds[sampled_tokens].mean(dim=1)
                soft_embeds = (
                    soft_embeds
                    + entropy_weight.to(embed_dtype).unsqueeze(-1) * (hard_embeds - soft_embeds).detach()
                )

                all_response_embeds[p, mask_pos] = soft_embeds
                all_response_token_ids[p, mask_pos] = cur_logits.detach().argmax(dim=-1)

            unmasked = ~response_mask
            if unmasked.any():
                unmasked_toks = x[p, max_prompt_len:][unmasked]
                all_response_embeds[p, unmasked] = reward_embed_layer(
                    self._token_mapping[unmasked_toks]
                ).detach()

        full_embeds = torch.cat(
            [batched_prefix, all_response_embeds, batched_suffix], dim=1
        )
        total_len = full_embeds.shape[1]

        attn_mask = torch.ones(B, total_len, device=device, dtype=torch.long)
        for p in range(B):
            prefix_pad = max_prefix_len - prefix_lens[p]
            if prefix_pad > 0:
                attn_mask[p, :prefix_pad] = 0
            suffix_pad = max_suffix_len - suffix_lens[p]
            if suffix_pad > 0:
                attn_mask[p, max_prefix_len + response_len + suffix_lens[p]:] = 0
            eos_found = False
            for i in range(response_len):
                tid = all_response_token_ids[p, i].item()
                if eos_found or (pad_id is not None and tid == pad_id):
                    attn_mask[p, max_prefix_len + i] = 0
                if tid == eos_id:
                    eos_found = True

        return self._reward_model(inputs_embeds=full_embeds, attention_mask=attn_mask).logits[:, 0]  # [B]

    # ------------------------------------------------------------------
    # Override: denoising in _prepare_inputs (outside training context)
    # ------------------------------------------------------------------

    @profiling_decorator
    def _prepare_inputs(self, generation_batch):
        """
        Run K full denoising trajectories per prompt using the DreamSampler (same
        code path as d1).  A generation_logits_hook_func captures x at step k so
        that _compute_loss can replay that single forward with gradients.
        """
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in generation_batch
        ]
        self._current_prompts_text = prompts_text

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        B_prompts = prompt_ids.size(0)
        K = self.args.num_generations
        B = B_prompts * K
        device = prompt_ids.device

        max_new_tokens = self.args.max_completion_length
        steps = self.sampler_config.steps
        mask_token_id = self.processing_class.mask_token_id

        # Strip padding from each prompt; K copies per prompt → B inputs for sampler
        real_lens = [int(prompt_mask[i].sum().item()) for i in range(B_prompts)]
        sampler_inputs = []
        for i in range(B_prompts):
            real_prompt = prompt_ids[i, -real_lens[i]:]
            sampler_inputs.extend([real_prompt] * K)

        # Compute attn / pos_id using the same left-padding convention as the sampler
        max_real_len = max(real_lens)
        T = max_real_len + max_new_tokens
        attn = torch.zeros(B, T, dtype=torch.long, device=device)
        for i in range(B_prompts):
            rl = real_lens[i]
            for j in range(K):
                attn[i * K + j, -(max_new_tokens + rl):] = 1
        pos_id = attn.long().cumsum(-1) - 1
        pos_id = pos_id.masked_fill(attn == 0, 1)
        max_prompt_len = T - max_new_tokens

        # Effective denoising steps (based on the completion mask)
        rep_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        rep_mask[0, -max_new_tokens:] = True
        num_transfer_tokens_rep = get_num_transfer_tokens(
            mask_index=rep_mask,
            steps=steps,
            scheduler=self.sampler.scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens_rep.size(1)

        if self.args.k_frac < 0:
            k = random.randint(0, effective_steps - 1)
        else:
            k = min(int(self.args.k_frac * effective_steps), effective_steps - 1)

        # Build reward caches (one per unique prompt)
        caches = [
            self._build_reward_cache(prompt_ids[i, -real_lens[i]:], str(device))
            for i in range(B_prompts)
        ]

        # Capture x at step k via a side-effecting logits hook
        captured: dict = {}

        def capture_hook(step, x, logits):
            if step == k:
                captured["x_k"] = x.clone()
            return logits

        import time
        t0 = time.time()
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled else nullcontext()
            ):
                unwrapped_model.eval()
                self.sampler.model = unwrapped_model
                try:
                    x_T = self.sampler.sample(
                        sampler_inputs,
                        self.sampler_config,
                        generation_logits_hook_func=capture_hook,
                    )
                finally:
                    unwrapped_model.train()
        torch.cuda.synchronize()
        t1 = time.time()
        if self._step <= 3:
            print(f"[BPTT timing] denoising={t1-t0:.2f}s  B={B}  T={T}  effective_steps={effective_steps}", flush=True)

        x_k = captured.get("x_k")
        if x_k is None:
            raise RuntimeError(f"capture_hook did not fire for k={k} (effective_steps={effective_steps})")
        mask_index_k = x_k == mask_token_id

        self._step += 1
        return {
            "x_k": x_k,
            "_step_idx": self._step,
            "mask_index_k": mask_index_k,
            "x_T": x_T,
            "attn": attn,
            "pos_id": pos_id,
            "caches": caches,
            "k": k,
            "K": K,
            "B_prompts": B_prompts,
            "max_prompt_len": max_prompt_len,
        }

    # ------------------------------------------------------------------
    # Override: single grad-enabled forward at x_k, then weighted loss
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        """
        One grad-enabled forward pass on x_k → soft rewards → softmax-weighted loss.
        Discrete reward from x_T is logged for comparison with d1/EntRGi.
        """
        x_k = inputs["x_k"]
        mask_index_k = inputs["mask_index_k"]
        x_T = inputs["x_T"]
        attn = inputs["attn"]
        pos_id = inputs["pos_id"]
        caches = inputs["caches"]
        K = inputs["K"]
        B_prompts = inputs["B_prompts"]
        max_prompt_len = inputs["max_prompt_len"]
        device = x_k.device

        # Single grad-enabled forward on x_k
        import time; _t0 = time.time()
        logits = model(x_k, attn, pos_id).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        soft_rewards = self._compute_soft_rewards(
            logits, mask_index_k, x_k, max_prompt_len, caches, K, str(device)
        )  # [B], grad attached

        # Softmax-weighted loss
        soft_rewards_grouped = soft_rewards.view(B_prompts, K)
        weights = F.softmax(soft_rewards_grouped / self.args.reward_temperature, dim=1).detach()
        loss = -(weights * soft_rewards_grouped).sum()

        # Log discrete reward from completed sequences
        mode = "train" if model.training else "eval"
        with torch.no_grad():
            B = B_prompts * K
            eos_token_id = self.processing_class.eos_token_id
            completion_ids = x_T[:, max_prompt_len:]
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            prompts_text = getattr(self, "_current_prompts_text", [""] * B_prompts)
            prompts_text_exp = [p for p in prompts_text for _ in range(K)]

            if self.reward_funcs:
                discrete_rewards = self.reward_funcs[0](prompts_text_exp, completions_text)
                discrete_rewards = torch.tensor(discrete_rewards, dtype=torch.float32, device=device)
                self._metrics[mode]["reward"].append(discrete_rewards.mean().item())

            torch.cuda.synchronize()
            _t1 = time.time()
            if inputs.get("_step_idx", 0) <= 3:
                print(f"[BPTT timing] compute_loss forward+reward={_t1-_t0:.2f}s  discrete_reward={_t1-_t0:.2f}s", flush=True)
            self._metrics[mode]["loss"].append(loss.item())
            self._metrics[mode]["soft_reward"].append(soft_rewards.mean().item())

        return loss

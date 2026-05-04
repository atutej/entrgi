"""
EntrgiDreamSampler: Dream masked-diffusion generation with per-step EntRGi guidance.

At every `apply_every_k` denoising steps, the masked-position logits (phi) are
optimized for M Adam steps using reward-model gradient backprop through soft token
embeddings.  The tilted phi drives token selection at that step; the base Dream
model weights are never modified during generation.

Reference: entrgi/main_expts/entrgi.py (optimize_logits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.dream.sampler import DreamSamplerConfig, sample_tokens
from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits


@dataclass
class EntrgiDreamSamplerConfig(DreamSamplerConfig):
    """DreamSamplerConfig + EntRGi guidance parameters."""

    M: int = 1
    """Adam gradient steps per guidance call (1 ≈ 1× cost)."""

    apply_every_k: int = 1
    """Call EntRGi guidance every k denoising steps (k > 1 ≈ 1/k overhead)."""

    eta: float = 1.0
    """Adam learning rate for phi (masked-position logit parameters)."""

    reward_temperature: float = 1.0
    """Scale applied to reward before negating to get loss: loss = -sum(r) / β."""

    num_generations: int = 1
    """K: number of completions per unique prompt in the batch."""

    aps: bool = False
    """Ablation: always set entropy_weight = 1 (full STE, no entropy-aware interpolation)."""

    soft_only: bool = False
    """Ablation: always set entropy_weight = 0 (pure soft/continuous embeddings, no STE)."""

    use_position_ids: bool = True
    """Compute and pass position_ids for left-padded sequences (Dream). Set False for LLaDA."""

    deprioritize_eos: bool = True
    """Set confidence = -inf at positions where the sampled token is EOS, preventing
    early commitment of EOS and thus premature sequence termination."""


@dataclass
class EntrgiDreamSampler(BaseSampler):
    """
    DreamSampler with EntRGi gradient guidance every apply_every_k steps.

    The reward model and associated tensors are injected at construction time
    and kept frozen throughout generation.  Only the per-step phi parameters
    (detached masked logits) are optimized.

    Args:
        reward_model: Reward model (e.g. Skywork).  Kept frozen; used only for
            the soft-embedding reward gradient inside optimize_logits.
        reward_tokenizer: Tokenizer for reward_model.
        token_mapping: LongTensor [dream_vocab_size] mapping Dream token ids to
            reward-model embedding indices (precomputed by the trainer).
        mapped_embeds: FloatTensor [dream_vocab_size, embed_dim] — reward
            embedding rows indexed by token_mapping (precomputed by the trainer).
    """

    reward_model: object = field(default=None, repr=False)
    reward_tokenizer: object = field(default=None, repr=False)
    token_mapping: Optional[torch.Tensor] = field(default=None, repr=False)
    mapped_embeds: Optional[torch.Tensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Reward cache helpers
    # ------------------------------------------------------------------

    def _build_reward_cache(self, prompt_tensor: torch.Tensor, device: str):
        """Build prefix/suffix embedding cache for one unique prompt."""
        prompt_text = self.tokenizer.decode(prompt_tensor, skip_special_tokens=False)
        if "<|im_start|>user\n" in prompt_text:
            user_content = (
                prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            )
        else:
            user_content = self.tokenizer.decode(prompt_tensor, skip_special_tokens=True)

        conv = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "<<PLACEHOLDER>>"},
        ]
        template = self.reward_tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        prefix_text, suffix_text = template.split("<<PLACEHOLDER>>")

        embed_layer = self.reward_model.get_input_embeddings()
        with torch.no_grad():
            prefix_ids = self.reward_tokenizer(
                prefix_text, return_tensors="pt"
            ).input_ids.to(device)
            suffix_ids = self.reward_tokenizer(
                suffix_text, return_tensors="pt"
            ).input_ids.to(device)
            prefix_embeds = embed_layer(prefix_ids)
            suffix_embeds = embed_layer(suffix_ids)

        return prefix_embeds, suffix_embeds, user_content

    # ------------------------------------------------------------------
    # Core: phi optimization (EntRGi)
    # ------------------------------------------------------------------

    def _optimize_logits(
        self,
        base_logits: torch.Tensor,
        mask_index: torch.Tensor,
        x: torch.Tensor,
        max_prompt_len: int,
        caches: list,
        K: int,
        config: EntrgiDreamSamplerConfig,
        device: str,
    ):
        """
        Optimize masked-position logits (phi) for M steps using reward gradient.

        The optimization uses entropy-aware interpolation between soft (weighted
        embedding average) and hard (argmax/sampled token) embeddings — the
        original EntRGi mechanism.

        Returns (phi_opt, phi_init) both of shape [N_total_masks, V], or
        (None, None) if no masks remain.
        """
        B_total = x.size(0)
        response_len = x.size(1) - max_prompt_len
        embed_dim = self.mapped_embeds.shape[-1]

        # Gather all masked logits across trajectories (same order as mask_index.nonzero)
        all_mask_logits, mask_counts = [], []
        for p in range(B_total):
            n = mask_index[p].sum().item()
            mask_counts.append(n)
            if n > 0:
                all_mask_logits.append(base_logits[p][mask_index[p]])

        if not all_mask_logits:
            return None, None

        phi = torch.cat(all_mask_logits, dim=0).detach().clone().requires_grad_(True)
        phi_init = phi.detach().clone()

        optimizer = torch.optim.Adam([phi], lr=config.eta)

        B_unique = len(caches)
        max_prefix_len = max(c[0].shape[1] for c in caches)
        max_suffix_len = max(c[1].shape[1] for c in caches)

        batched_prefix = torch.zeros(
            B_unique, max_prefix_len, embed_dim, device=device, dtype=self.mapped_embeds.dtype
        )
        batched_suffix = torch.zeros(
            B_unique, max_suffix_len, embed_dim, device=device, dtype=self.mapped_embeds.dtype
        )
        prefix_lens, suffix_lens = [], []

        for b, (pre, suf, _) in enumerate(caches):
            plen, slen = pre.shape[1], suf.shape[1]
            batched_prefix[b, max_prefix_len - plen :] = pre[0]
            batched_suffix[b, :slen] = suf[0]
            prefix_lens.append(plen)
            suffix_lens.append(slen)

        # Expand B_unique → B_total (each unique prompt repeated K times)
        batched_prefix = batched_prefix.repeat_interleave(K, dim=0)
        batched_suffix = batched_suffix.repeat_interleave(K, dim=0)

        eos_id = self.tokenizer.eos_token_id
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or eos_id

        embed_dtype = self.mapped_embeds.dtype

        ew_accum: list[torch.Tensor] = []

        for _ in range(config.M):
            optimizer.zero_grad()

            all_response_token_ids = x[:, max_prompt_len:].clone()

            # Build per-sequence response embeddings without any in-place writes.
            # In-place writes to a shared buffer cause PyTorch's version counter to
            # increment after the grad-carrying assignment, making the saved tensor
            # stale by backward time and silently killing the gradient to phi.
            # index_put (no underscore = out-of-place) creates a fresh tensor each
            # call, so the autograd graph is unambiguous.
            seq_embed_list = []
            phi_idx = 0
            for p in range(B_total):
                response_mask = mask_index[p][max_prompt_len:]
                mask_pos = torch.where(response_mask)[0]
                n_masks = len(mask_pos)
                unmasked = ~response_mask

                # Start from a detached base: unmasked token embeddings where
                # available, zeros elsewhere.
                seq_embed = torch.zeros(
                    response_len, embed_dim, device=device, dtype=embed_dtype
                )
                if unmasked.any():
                    unmasked_toks = x[p, max_prompt_len:][unmasked]
                    unmasked_pos = torch.where(unmasked)[0]
                    seq_embed = seq_embed.index_put(
                        (unmasked_pos,),
                        self.mapped_embeds[unmasked_toks],
                    )

                if n_masks > 0:
                    cur_phi = phi[phi_idx : phi_idx + n_masks]

                    # Entropy-aware interpolation (EntRGi)
                    entropy_probs = F.softmax(cur_phi, dim=-1)
                    entropy = -torch.sum(
                        entropy_probs * torch.log(entropy_probs + 1e-10), dim=-1
                    )
                    max_entropy = torch.log(
                        torch.tensor(
                            entropy_probs.shape[-1],
                            device=device,
                            dtype=entropy_probs.dtype,
                        )
                    )
                    if config.aps:
                        entropy_weight = torch.ones_like(entropy)
                    elif config.soft_only:
                        entropy_weight = torch.zeros_like(entropy)
                    else:
                        entropy_weight = (entropy / max_entropy).detach()
                    ew_accum.append(entropy_weight.mean().item())

                    sample_logits = cur_phi / config.temperature
                    probs = F.softmax(sample_logits, dim=-1)
                    soft_embeds = torch.matmul(probs.to(embed_dtype), self.mapped_embeds)

                    if config.top_p is not None and config.top_p < 1.0:
                        sample_logits = top_p_logits(sample_logits, config.top_p)
                    if config.top_k is not None:
                        sample_logits = top_k_logits(sample_logits, config.top_k)
                    sample_probs = F.softmax(sample_logits.float(), dim=-1)
                    sampled_tokens = torch.multinomial(sample_probs, 1, replacement=True)
                    hard_embeds = self.mapped_embeds[sampled_tokens].mean(dim=1)
                    # High entropy → more soft; low entropy → more hard (STE when aps=True)
                    soft_embeds = (
                        soft_embeds
                        + entropy_weight.to(embed_dtype).unsqueeze(-1) * (hard_embeds - soft_embeds).detach()
                    )

                    # Out-of-place scatter: returns a NEW tensor, preserving the
                    # autograd graph from phi → probs → soft_embeds → seq_embed.
                    seq_embed = seq_embed.index_put((mask_pos,), soft_embeds)
                    all_response_token_ids[p, mask_pos] = cur_phi.argmax(dim=-1)
                    phi_idx += n_masks

                seq_embed_list.append(seq_embed)

            all_response_embeds = torch.stack(seq_embed_list, dim=0)  # [B, L, D]

            total_len = max_prefix_len + response_len + max_suffix_len
            full_embeds = torch.cat(
                [batched_prefix, all_response_embeds, batched_suffix], dim=1
            )

            attn_mask = torch.ones(B_total, total_len, device=device, dtype=torch.long)
            for p in range(B_total):
                b = p // K
                prefix_pad = max_prefix_len - prefix_lens[b]
                if prefix_pad > 0:
                    attn_mask[p, :prefix_pad] = 0
                suffix_pad = max_suffix_len - suffix_lens[b]
                if suffix_pad > 0:
                    attn_mask[
                        p, max_prefix_len + response_len + suffix_lens[b] :
                    ] = 0
                eos_found = False
                for i in range(response_len):
                    tid = all_response_token_ids[p, i].item()
                    if eos_found or (pad_id is not None and tid == pad_id):
                        attn_mask[p, max_prefix_len + i] = 0
                    if tid == eos_id:
                        eos_found = True

            rewards = self.reward_model(
                inputs_embeds=full_embeds, attention_mask=attn_mask.bool()
            ).logits[:, 0]
            loss = -rewards.sum() / config.reward_temperature
            loss.backward()
            optimizer.step()

        ew_mean = sum(ew_accum) / len(ew_accum) if ew_accum else 1.0
        return phi.detach(), phi_init, ew_mean

    # ------------------------------------------------------------------
    # Public: sample (mirrors DreamSampler.sample without @no_grad wrapper)
    # ------------------------------------------------------------------

    def sample(
        self,
        inputs: List[torch.Tensor],
        config: EntrgiDreamSamplerConfig,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dream diffusion sampling with optional EntRGi gradient guidance.

        When M > 0 and apply_every_k > 0, calls _optimize_logits every
        apply_every_k denoising steps.  Otherwise behaves identically to
        DreamSampler.sample.
        """
        # Pull config values (kwargs can override)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        steps = kwargs.get("steps", config.steps)
        alg = kwargs.get("alg", config.alg)
        alg_temp = kwargs.get("alg_temp", config.alg_temp)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        M = config.M
        apply_every_k = config.apply_every_k
        K = config.num_generations

        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        prompt_lens = [p.shape[0] for p in inputs]
        max_len = max_new_tokens + max(prompt_lens)
        B_total = len(inputs)
        T = max_len
        device = self.model.device

        # Build right-aligned canvas
        x = torch.full((B_total, T), eos_token_id, dtype=torch.long, device=device)
        seq_lens = []
        for i, p in enumerate(inputs):
            total_len = prompt_lens[i] + max_new_tokens
            seq_lens.append(total_len)
            start = T - total_len
            x[i, start : start + prompt_lens[i]] = p
            x[i, start + prompt_lens[i] : T] = mask_token_id

        attention_mask = torch.zeros((B_total, T), dtype=torch.long, device=device)
        for j, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[j, -L:] = 1

        if config.use_position_ids and torch.any(attention_mask == 0):
            pos_id = attention_mask.long().cumsum(-1) - 1
            pos_id.masked_fill_(attention_mask == 0, 1)
        else:
            pos_id = None

        # Response tokens start at position T - max_new_tokens for all sequences.
        max_prompt_len = T - max_new_tokens

        mask_index_init = x == mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index_init,
            steps=steps,
            scheduler=self.scheduler,
            stochastic=stochastic_transfer,
        )
        effective_steps = num_transfer_tokens_list.size(1)

        # Build reward caches once per unique prompt (groups of K consecutive sequences)
        guidance_active = (
            self.reward_model is not None
            and self.mapped_embeds is not None
            and M > 0
            and apply_every_k > 0
            and K > 0
        )
        if guidance_active:
            B_unique = B_total // K
            unique_prompt_tensors = [inputs[b * K] for b in range(B_unique)]
            caches = [
                self._build_reward_cache(p, str(device))
                for p in unique_prompt_tensors
            ]
        else:
            caches = None

        # Denoising loop
        _ew_step_means: list[float] = []
        for step in range(effective_steps):
            mask_index = x == mask_token_id

            with torch.no_grad():
                if pos_id is not None:
                    logits = self.model(x, attention_mask=attention_mask[:, None, None, :].bool(), position_ids=pos_id).logits
                else:
                    logits = self.model(x, attention_mask=attention_mask[:, None, None, :].bool()).logits
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # EntRGi guidance every apply_every_k steps
            use_guidance = guidance_active and (step % apply_every_k == 0)
            if use_guidance:
                phi_opt, _, ew_mean = self._optimize_logits(
                    logits,
                    mask_index,
                    x,
                    max_prompt_len,
                    caches,
                    K,
                    config,
                    str(device),
                )
                _ew_step_means.append(ew_mean)
            else:
                phi_opt = None

            # Flat masked logits: use guided phi if available, else base logits
            if phi_opt is not None:
                guided_mask_logits = phi_opt
            else:
                guided_mask_logits = logits[mask_index]

            if alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    guided_mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    guided_mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                )
            elif alg == "entropy":
                confidence, x0 = sample_tokens(
                    guided_mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"Unknown alg: {alg!r}")

            if config.deprioritize_eos:
                eos_mask = x0 == eos_token_id
                if eos_mask.any():
                    confidence = confidence.clone()
                    confidence[eos_mask] = -torch.inf

            full_confidence = torch.full(
                (B_total, T), -torch.inf, device=device, dtype=torch.float32
            )
            full_confidence[mask_index] = confidence

            for j in range(B_total):
                n_transfer = num_transfer_tokens_list[j, step]
                if n_transfer > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_idx = torch.topk(full_confidence[j], n_transfer)
                    else:
                        fc = F.softmax(full_confidence[j] / alg_temp, dim=-1)
                        transfer_idx = torch.multinomial(fc, num_samples=n_transfer)

                    x_ = torch.full_like(x, mask_token_id)
                    x_[mask_index] = x0.clone()
                    x[j, transfer_idx] = x_[j, transfer_idx]

        self._last_entropy_weight_mean = (
            sum(_ew_step_means) / len(_ew_step_means) if _ew_step_means else None
        )

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=None)

    @torch.no_grad()
    def infill(
        self,
        inputs: List[torch.Tensor],
        config: EntrgiDreamSamplerConfig,
        **kwargs,
    ) -> BaseSamplerOutput:
        """Plain Dream infill (no EntRGi guidance) — delegates to DreamSampler."""
        from dllm.pipelines.dream.sampler import DreamSampler

        plain = DreamSampler(model=self.model, tokenizer=self.tokenizer, scheduler=self.scheduler)
        return plain.infill(inputs, config, **kwargs)


# ---------------------------------------------------------------------------
# LLaDA variant — same sampler, different config defaults
# ---------------------------------------------------------------------------


@dataclass
class EntrgiLLaDASamplerConfig(EntrgiDreamSamplerConfig):
    """EntrgiDreamSamplerConfig adapted for LLaDA.

    Key differences from Dream:
      - right_shift_logits = False  (LLaDA predicts bidirectionally)
      - use_position_ids  = False   (LLaDA does not use explicit position IDs)
    Everything else — M-step Adam guidance, entropy-weighted STE, confidence-based
    remasking — is identical.
    """

    right_shift_logits: bool = False
    use_position_ids: bool = False


EntrgiLLaDASampler = EntrgiDreamSampler

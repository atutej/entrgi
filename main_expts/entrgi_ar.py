#!/usr/bin/env python3
"""
EntRGi/APS with AR log-likelihood reward.

Instead of a classifier reward model, uses an autoregressive model's
log P(response + suffix | prefix) as the reward signal.

Usage:
    python entrgi_ar.py --dataset_path allenai/reward-bench-2 --subset_size 10
    torchrun --nproc_per_node=4 entrgi_ar.py --dataset_path allenai/reward-bench-2 --subset_size 100
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from utils import (
    Config, get_model_config, get_vocab_size,
    RewardCache, build_reward_cache,
    top_p_filter, top_k_filter, get_confidence_for_alg, deprioritize_eos,
    add_common_args, save_results
)
from entrgi import setup_distributed, cleanup_distributed, is_main_process, gather_results


# =============================================================================
# AR-specific cache (extends RewardCache with suffix token IDs)
# =============================================================================

@dataclass
class ARRewardCache:
    """Cached prefix/suffix embeddings + suffix token IDs for AR log-likelihood."""
    prefix_embeds: torch.Tensor
    suffix_embeds: torch.Tensor
    suffix_ids: torch.Tensor
    user_content: str


def build_ar_reward_cache(ar_model, reward_tokenizer, dream_tokenizer,
                          prompt_ids: torch.Tensor, device: str) -> ARRewardCache:
    """Precompute AR model prefix/suffix embeddings and suffix token IDs."""
    prompt_text = dream_tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
    user_content = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] \
                   if "<|im_start|>user\n" in prompt_text else prompt_text

    conversation = [{"role": "user", "content": user_content},
                    {"role": "assistant", "content": "<<PLACEHOLDER>>"}]
    template = reward_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    prefix_text, suffix_text = template.split("<<PLACEHOLDER>>")

    embed_layer = ar_model.get_input_embeddings()
    with torch.no_grad():
        prefix_ids = reward_tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
        suffix_ids = reward_tokenizer(suffix_text, return_tensors="pt").input_ids.to(device)
        prefix_embeds = embed_layer(prefix_ids)
        suffix_embeds = embed_layer(suffix_ids)

    return ARRewardCache(prefix_embeds, suffix_embeds, suffix_ids, user_content)


# =============================================================================
# AR model loading
# =============================================================================

def load_models_ar(cfg: Config):
    """Load Dream model and AR reward model (CausalLM)."""
    # Dream model
    dream_model = AutoModel.from_pretrained(
        cfg.dream_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(cfg.device).eval()
    dream_tokenizer = AutoTokenizer.from_pretrained(cfg.dream_model, trust_remote_code=True)

    for p in dream_model.parameters():
        p.requires_grad = False

    # AR reward model
    reward_path = Path(cfg.reward_model)
    if (reward_path / "adapter_config.json").exists():
        from peft import PeftModel, PeftConfig
        peft_cfg = PeftConfig.from_pretrained(cfg.reward_model)
        ar_model = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16,
        ).to(cfg.device)
        ar_model = PeftModel.from_pretrained(ar_model, cfg.reward_model).merge_and_unload()
        reward_tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
    else:
        ar_model = AutoModelForCausalLM.from_pretrained(
            cfg.reward_model, torch_dtype=torch.bfloat16,
        ).to(cfg.device)
        reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model)

    ar_model.eval()
    for p in ar_model.parameters():
        p.requires_grad = False

    # Token mapping (Dream vocab → AR vocab)
    dream_vocab = dream_tokenizer.get_vocab()
    reward_vocab = reward_tokenizer.get_vocab()
    vocab_size = get_vocab_size(dream_model, cfg.dream_model)
    unk_id = reward_tokenizer.unk_token_id or reward_tokenizer.eos_token_id

    token_mapping = torch.full((vocab_size,), unk_id, dtype=torch.long, device=cfg.device)
    for tok, did in dream_vocab.items():
        if did < vocab_size and tok in reward_vocab:
            token_mapping[did] = reward_vocab[tok]

    num_mapped = (token_mapping != unk_id).sum().item()
    print(f"Mapped {num_mapped}/{vocab_size} tokens ({100.0 * num_mapped / vocab_size:.2f}%) from Dream to AR model.")

    reward_embeds = ar_model.get_input_embeddings()
    mapped_embeds = reward_embeds.weight[token_mapping].detach()

    return dream_model, dream_tokenizer, ar_model, reward_tokenizer, token_mapping, mapped_embeds


# =============================================================================
# Discrete AR reward (for final evaluation)
# =============================================================================

def compute_discrete_reward_ar(ar_model, reward_tokenizer, prompt: str,
                               response: str, device: str) -> float:
    """Compute AR log-likelihood reward from discrete tokens."""
    conversation = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<<PLACEHOLDER>>"}]
    template = reward_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    prefix_text, suffix_text = template.split("<<PLACEHOLDER>>")

    full_text = prefix_text + response + suffix_text
    if reward_tokenizer.bos_token and full_text.startswith(reward_tokenizer.bos_token):
        full_text = full_text[len(reward_tokenizer.bos_token):]

    inputs = reward_tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # Find where the response starts
    if reward_tokenizer.bos_token and prefix_text.startswith(reward_tokenizer.bos_token):
        prefix_text = prefix_text[len(reward_tokenizer.bos_token):]
    prefix_len = reward_tokenizer(prefix_text, return_tensors="pt").input_ids.shape[1]

    with torch.no_grad():
        logits = ar_model(**inputs).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        targets = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Mean log-probs over response + suffix (from prefix_len-1 onward in shifted sequence)
        resp_suffix_lp = token_log_probs[:, prefix_len - 1:]
        num_tokens = resp_suffix_lp.shape[1]
        reward = resp_suffix_lp.sum(dim=-1) / max(num_tokens, 1)
        return reward[0].item()


# =============================================================================
# AR logit optimization
# =============================================================================

def optimize_logits_ar(
    base_logits: torch.Tensor,
    mask_indices: List[torch.Tensor],
    trajectories: torch.Tensor,
    max_prompt_len: int,
    trajectory_pad_lens: torch.Tensor,
    B: int, K: int,
    ar_model, caches: List[ARRewardCache],
    token_mapping: torch.Tensor,
    mapped_embeds: torch.Tensor,
    reward_embed_layer,
    eos_token_id: int,
    pad_token_id: int,
    cfg: Config
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Optimize logits using AR log-likelihood as the reward signal.

    Same structure as the original optimize_logits, but replaces the classifier
    forward pass with: AR forward → per-token log-probs → sum over response+suffix.

    Target distributions at masked positions match the embedding mode:
      - Vanilla: soft targets (expected log-prob under softmax(phi/T))
      - APS: STE targets (hard forward, soft gradient)
      - EntRGi: entropy-weighted STE targets
    """
    total_trajectories = B * K
    device = trajectories.device
    response_len = cfg.max_new_tokens
    embed_dim = mapped_embeds.shape[-1]

    # Collect mask logits
    all_mask_logits = []
    mask_counts = []
    for p in range(total_trajectories):
        n_masks = mask_indices[p].sum().item()
        mask_counts.append(n_masks)
        if n_masks > 0:
            all_mask_logits.append(base_logits[p][mask_indices[p]])

    if not all_mask_logits:
        return None, None

    phi = torch.cat(all_mask_logits, dim=0).detach().clone().requires_grad_(True)
    phi_init = phi.detach().clone()
    optimizer = torch.optim.Adam([phi], lr=cfg.eta)

    # Prepare batched prefix/suffix embeddings and suffix IDs
    max_prefix_len = max(cache.prefix_embeds.shape[1] for cache in caches)
    max_suffix_len = max(cache.suffix_embeds.shape[1] for cache in caches)

    batched_prefix = torch.zeros(B, max_prefix_len, embed_dim, device=device, dtype=mapped_embeds.dtype)
    batched_suffix = torch.zeros(B, max_suffix_len, embed_dim, device=device, dtype=mapped_embeds.dtype)
    batched_suffix_ids = torch.full((B, max_suffix_len), pad_token_id or 0, dtype=torch.long, device=device)
    prefix_lens = []
    suffix_lens = []

    for b, cache in enumerate(caches):
        plen = cache.prefix_embeds.shape[1]
        slen = cache.suffix_embeds.shape[1]
        batched_prefix[b, max_prefix_len - plen:] = cache.prefix_embeds[0]
        batched_suffix[b, :slen] = cache.suffix_embeds[0]
        batched_suffix_ids[b, :slen] = cache.suffix_ids[0]
        prefix_lens.append(plen)
        suffix_lens.append(slen)

    batched_prefix = batched_prefix.repeat_interleave(K, dim=0)
    batched_suffix = batched_suffix.repeat_interleave(K, dim=0)
    batched_suffix_ids = batched_suffix_ids.repeat_interleave(K, dim=0)

    for _ in range(cfg.M):
        optimizer.zero_grad()

        all_response_embeds = torch.zeros(total_trajectories, response_len, embed_dim,
                                          device=device, dtype=mapped_embeds.dtype)
        all_response_token_ids = trajectories[:, max_prompt_len:].clone()

        # Per-trajectory info for masked positions: (mask_pos, soft_probs, sampled_reward_ids, entropy_weight)
        mask_info = [None] * total_trajectories

        phi_idx = 0
        for p in range(total_trajectories):
            response_mask = mask_indices[p][max_prompt_len:]
            mask_pos = torch.where(response_mask)[0]
            n_masks = len(mask_pos)

            if n_masks > 0:
                phi_slice = phi[phi_idx:phi_idx + n_masks]

                if cfg.use_entrgi:
                    entropy_probs = F.softmax(phi_slice, dim=-1)
                    entropy = -torch.sum(entropy_probs * torch.log(entropy_probs + 1e-10), dim=-1)
                    max_entropy = torch.log(torch.tensor(entropy_probs.shape[-1], device=device, dtype=entropy_probs.dtype))
                    entropy_weight = (entropy / max_entropy).detach()

                    sample_logits = phi_slice / cfg.temperature
                    probs = F.softmax(sample_logits, dim=-1)
                    soft_embeds = torch.matmul(probs, mapped_embeds)
                    filtered = sample_logits.clone()
                    if cfg.top_p < 1.0:
                        filtered = top_p_filter(filtered, cfg.top_p)
                    if cfg.top_k is not None:
                        filtered = top_k_filter(filtered, cfg.top_k)
                    sampled = torch.multinomial(F.softmax(filtered, dim=-1), 1, replacement=True).squeeze(-1)
                    hard_embeds = mapped_embeds[sampled]
                    soft_embeds = soft_embeds + entropy_weight.unsqueeze(-1) * (hard_embeds - soft_embeds).detach()

                    mask_info[p] = (mask_pos, probs, token_mapping[sampled], entropy_weight)

                elif cfg.use_aps:
                    sample_logits = phi_slice / cfg.temperature
                    probs = F.softmax(sample_logits, dim=-1)
                    soft_embeds = torch.matmul(probs, mapped_embeds)
                    filtered = sample_logits.clone()
                    if cfg.top_p < 1.0:
                        filtered = top_p_filter(filtered, cfg.top_p)
                    if cfg.top_k is not None:
                        filtered = top_k_filter(filtered, cfg.top_k)
                    sampled = torch.multinomial(F.softmax(filtered, dim=-1), 1, replacement=True).squeeze(-1)
                    hard_embeds = mapped_embeds[sampled]
                    soft_embeds = soft_embeds + (hard_embeds - soft_embeds).detach()

                    mask_info[p] = (mask_pos, probs, token_mapping[sampled], None)

                else:  # vanilla
                    probs = F.softmax(phi_slice / cfg.temperature, dim=-1)
                    soft_embeds = torch.matmul(probs, mapped_embeds)

                    mask_info[p] = (mask_pos, probs, None, None)

                all_response_embeds[p, mask_pos] = soft_embeds
                all_response_token_ids[p, mask_pos] = phi_slice.argmax(dim=-1)
                phi_idx += n_masks

            unmasked = ~mask_indices[p][max_prompt_len:]
            if unmasked.any():
                unmasked_toks = trajectories[p, max_prompt_len:][unmasked]
                all_response_embeds[p, unmasked] = reward_embed_layer(token_mapping[unmasked_toks]).detach()

        # Assemble full sequence: [prefix | response | suffix]
        full_embeds = torch.cat([batched_prefix, all_response_embeds, batched_suffix], dim=1)
        total_len = max_prefix_len + response_len + max_suffix_len

        # Padding attention mask (model handles causality internally)
        attn_mask = torch.ones(total_trajectories, total_len, device=device, dtype=torch.long)
        for p in range(total_trajectories):
            b = p // K
            prefix_pad = max_prefix_len - prefix_lens[b]
            if prefix_pad > 0:
                attn_mask[p, :prefix_pad] = 0
            suffix_pad = max_suffix_len - suffix_lens[b]
            if suffix_pad > 0:
                attn_mask[p, max_prefix_len + response_len + suffix_lens[b]:] = 0
            eos_found = False
            for i in range(response_len):
                tid = all_response_token_ids[p, i].item()
                if eos_found or (pad_token_id is not None and tid == pad_token_id):
                    attn_mask[p, max_prefix_len + i] = 0
                if tid == eos_token_id:
                    eos_found = True

        # ---- AR forward pass ----
        # Input: drop last position; targets: drop first position
        ar_logits = ar_model(inputs_embeds=full_embeds[:, :-1], attention_mask=attn_mask[:, :-1]).logits
        ar_log_probs = F.log_softmax(ar_logits.float(), dim=-1)

        # ---- Compute log-likelihood over response + suffix ----
        total_log_prob = torch.zeros(total_trajectories, device=device)

        # Response log-probs: ar_logits at [prefix_len-1 .. prefix_len+resp_len-2] predict response tokens
        resp_lp_start = max_prefix_len - 1
        resp_lp_end = max_prefix_len + response_len - 1
        resp_log_probs = ar_log_probs[:, resp_lp_start:resp_lp_end]  # [B*K, response_len, ar_vocab]

        # Hard response targets (used for unmasked positions and as hard component for STE)
        response_target_ids = token_mapping[all_response_token_ids]  # [B*K, response_len] in AR vocab
        hard_resp_lp = resp_log_probs.gather(-1, response_target_ids.unsqueeze(-1)).squeeze(-1)

        # Start from hard log-probs, replace masked positions with mode-appropriate values
        resp_position_lp = hard_resp_lp.clone()

        for p in range(total_trajectories):
            if mask_info[p] is None:
                continue
            positions, soft_probs_dream, sampled_ar_ids, ew = mask_info[p]

            # Map AR log-probs to Dream vocab space: for each Dream token d, get log P_AR(mapping[d])
            pos_lp_ar = resp_log_probs[p, positions]            # [n_masks, ar_vocab]
            pos_lp_dream = pos_lp_ar[:, token_mapping]           # [n_masks, dream_vocab]

            # Expected log-prob under soft Dream distribution
            #soft_lp = (soft_probs_dream * pos_lp_dream).sum(dim=-1)
            soft_lp = (soft_probs_dream.detach() * pos_lp_dream).sum(dim=-1)


            if cfg.use_entrgi:
                hard_lp_masked = pos_lp_ar.gather(-1, sampled_ar_ids.unsqueeze(-1)).squeeze(-1)
                position_lp = soft_lp + ew * (hard_lp_masked - soft_lp).detach()
            elif cfg.use_aps:
                hard_lp_masked = pos_lp_ar.gather(-1, sampled_ar_ids.unsqueeze(-1)).squeeze(-1)
                position_lp = soft_lp + (hard_lp_masked - soft_lp).detach()
            else:  # vanilla
                position_lp = soft_lp

            resp_position_lp[p, positions] = position_lp

        resp_attn = attn_mask[:, max_prefix_len:max_prefix_len + response_len].float()
        total_log_prob += (resp_position_lp * resp_attn).sum(dim=-1)

        # Suffix log-probs (always hard targets)
        if max_suffix_len > 0:
            suf_lp_start = max_prefix_len + response_len - 1
            suf_lp_end = total_len - 1
            suf_log_probs = ar_log_probs[:, suf_lp_start:suf_lp_end]  # [B*K, max_suffix_len, ar_vocab]
            suf_target_lp = suf_log_probs.gather(-1, batched_suffix_ids.unsqueeze(-1)).squeeze(-1)
            suf_attn = attn_mask[:, max_prefix_len + response_len:].float()
            total_log_prob += (suf_target_lp * suf_attn).sum(dim=-1)

        # Normalize by number of target tokens per trajectory (response + suffix)
        resp_attn_count = attn_mask[:, max_prefix_len:max_prefix_len + response_len].sum(dim=-1).float()
        suf_attn_count = attn_mask[:, max_prefix_len + response_len:].sum(dim=-1).float() if max_suffix_len > 0 else 0
        num_target_tokens = resp_attn_count + suf_attn_count
        mean_log_prob = total_log_prob / num_target_tokens.clamp(min=1)

        loss = -mean_log_prob.sum()
        loss.backward()
        optimizer.step()

    return phi.detach(), phi_init


# =============================================================================
# Generation
# =============================================================================

def generate(dream_model, dream_tokenizer, ar_model, reward_tokenizer,
             token_mapping, mapped_embeds, prompts: List[str],
             cfg: Config, model_cfg: Dict) -> List[Dict[str, Any]]:
    """
    Dream generation with AR log-likelihood reward guidance.
    Same structure as entrgi.generate but uses AR log-probs instead of classifier reward.
    """
    device = cfg.device
    mask_id = model_cfg["mask_id"]
    eos_id = model_cfg["eos_id"]
    pad_id = model_cfg["pad_id"]
    reward_embed_layer = ar_model.get_input_embeddings()

    B = len(prompts)
    K = cfg.K

    # 1. Tokenize prompts
    prompt_ids_list = []
    prompt_lens = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = dream_tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        prompt_ids_list.append(inputs.input_ids[0].to(device))
        prompt_lens.append(inputs.input_ids.shape[1])

    max_prompt_len = max(prompt_lens)
    total_len = max_prompt_len + cfg.max_new_tokens

    # 2. Create trajectories
    trajectories = torch.full((B * K, total_len), mask_id, dtype=torch.long, device=device)
    valid_token_mask = torch.ones((B * K, total_len), dtype=torch.bool, device=device)
    trajectory_pad_lens = []

    for b in range(B):
        plen = prompt_lens[b]
        pad_len = max_prompt_len - plen
        for k in range(K):
            p = b * K + k
            if pad_len > 0:
                trajectories[p, :pad_len] = pad_id
                valid_token_mask[p, :pad_len] = False
            trajectories[p, pad_len:max_prompt_len] = prompt_ids_list[b]
            trajectory_pad_lens.append(pad_len)

    trajectory_pad_lens = torch.tensor(trajectory_pad_lens, device=device)

    # 3. 4D attention mask for Dream
    def create_bidirectional_attention_mask(valid_mask_2d, dtype):
        key_valid = valid_mask_2d.unsqueeze(1).unsqueeze(2)
        return torch.where(
            key_valid,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float('-inf'), device=device, dtype=dtype)
        )

    attention_mask_4d = create_bidirectional_attention_mask(valid_token_mask, torch.bfloat16)

    # 4. Build AR reward caches
    caches = []
    for b in range(B):
        prompt_ids = prompt_ids_list[b].unsqueeze(0)
        cache = build_ar_reward_cache(ar_model, reward_tokenizer, dream_tokenizer, prompt_ids, device)
        caches.append(cache)

    # 5. Generation loop
    eps = 1e-3
    timesteps = torch.linspace(1, eps, cfg.T + 1, device=device)

    for step in range(cfg.T):
        t, s = timesteps[step], timesteps[step + 1]

        with torch.no_grad():
            logits = dream_model(trajectories, attention_mask=attention_mask_4d).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_indices = []
        for p in range(B * K):
            is_mask = (trajectories[p] == mask_id)
            pad_len = trajectory_pad_lens[p].item()
            is_mask[:pad_len] = False
            mask_indices.append(is_mask)

        mask_counts = [m.sum().item() for m in mask_indices]
        if sum(mask_counts) == 0:
            break

        phi_opt, phi_init = optimize_logits_ar(
            logits, mask_indices, trajectories,
            max_prompt_len, trajectory_pad_lens,
            B, K,
            ar_model, caches, token_mapping, mapped_embeds, reward_embed_layer,
            eos_id, pad_id, cfg
        )

        # Sample and update
        phi_idx = 0
        for p in range(B * K):
            n_masks = mask_counts[p]
            if n_masks == 0:
                continue

            mask_pos = torch.where(mask_indices[p])[0]

            if phi_opt is not None and phi_idx + n_masks <= phi_opt.shape[0]:
                opt_logits = phi_opt[phi_idx:phi_idx + n_masks]
                init_logits = phi_init[phi_idx:phi_idx + n_masks]
                phi_idx += n_masks
            else:
                opt_logits = logits[p][mask_indices[p]]
                init_logits = opt_logits

            sample_logits = opt_logits / cfg.temperature
            if cfg.top_p < 1.0:
                sample_logits = top_p_filter(sample_logits, cfg.top_p)
            if cfg.top_k is not None:
                sample_logits = top_k_filter(sample_logits, cfg.top_k)

            probs = F.softmax(sample_logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            if cfg.alg == "origin":
                p_transfer = 1 - s.item() / t.item() if step < cfg.T - 1 else 1
                transfer_mask = torch.rand(n_masks, device=device) < p_transfer
                if transfer_mask.any():
                    trajectories[p, mask_pos[transfer_mask]] = sampled[transfer_mask]
            else:
                old_probs = F.softmax(init_logits / cfg.temperature, dim=-1) if cfg.alg == "entropy" else None
                confidence = get_confidence_for_alg(probs, sampled, cfg.alg, logits=opt_logits, old_probs=old_probs)

                if cfg.deprioritize_eos:
                    confidence = deprioritize_eos(confidence, sampled, eos_id)

                num_unmask = int(n_masks * (1 - s.item() / t.item())) if step < cfg.T - 1 else n_masks

                if num_unmask > 0:
                    if cfg.alg_temp is None or cfg.alg_temp == 0:
                        _, selected = torch.topk(confidence, min(num_unmask, n_masks))
                    else:
                        selection_probs = F.softmax(confidence / cfg.alg_temp, dim=-1)
                        selected = torch.multinomial(selection_probs, min(num_unmask, n_masks), replacement=False)
                    trajectories[p, mask_pos[selected]] = sampled[selected]

    # 6. Collect results
    trajectories = trajectories.view(B, K, total_len)

    all_results = []
    for b in range(B):
        responses, rewards = [], []
        for k in range(K):
            response_tokens = trajectories[b, k, max_prompt_len:]
            resp = dream_tokenizer.decode(response_tokens, skip_special_tokens=True)
            r = compute_discrete_reward_ar(
                ar_model, reward_tokenizer,
                caches[b].user_content, resp, device
            )
            responses.append(resp)
            rewards.append(r)

        best_idx = max(range(K), key=lambda i: rewards[i])

        all_results.append({
            "best_response": responses[best_idx],
            "top1_reward": rewards[best_idx],
            "all_responses": responses,
            "all_rewards": rewards,
            "avgN_reward": sum(rewards) / len(rewards),
        })

    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    rank, world_size, device, is_distributed = setup_distributed()

    if is_main_process(rank):
        print(f"[Rank {rank}] device={device}")

    parser = argparse.ArgumentParser(
        description="EntRGi/APS with AR log-likelihood reward for discrete diffusion LMs"
    )
    parser = add_common_args(parser)

    parser.add_argument("--alg", type=str, default="entropy",
                        choices=["origin", "maskgit_plus", "topk_margin", "entropy", "anchor"])
    parser.add_argument("--alg_temp", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--no_deprioritize_eos", action="store_true")
    parser.add_argument("--M", type=int, default=3, help="Gradient optimization steps")
    parser.add_argument("--eta", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts per GPU")
    parser.add_argument("--use_aps", action="store_true", help="Use APS (Rout et al. 2025)")
    parser.add_argument("--use_entrgi", action="store_true", help="Use EntRGi (Ours)")

    args = parser.parse_args()
    if args.use_entrgi and args.use_aps:
        raise ValueError("Only one of --use_entrgi or --use_aps can be set.")

    cfg = Config(
        dream_model=args.dream_model,
        reward_model=args.reward_model,
        K=args.K, T=args.T, M=args.M,
        eta=args.eta,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        deprioritize_eos=not args.no_deprioritize_eos,
        dataset_path=args.dataset_path,
        split=args.split,
        prompt_field=args.prompt_field,
        subset_size=args.subset_size,
        subset_name=args.subset_name,
        subset_field=args.subset_field,
        output_file=args.output_file,
        device=device,
        seed=args.seed,
        use_aps=args.use_aps,
        use_entrgi=args.use_entrgi,
    )

    batch_size = args.batch_size

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Load dataset
    if is_main_process(rank):
        print(f"Loading dataset: {cfg.dataset_path}")

    dataset = load_dataset(cfg.dataset_path, split=cfg.split).shuffle(seed=cfg.seed)
    if cfg.subset_size:
        dataset = dataset.select(range(min(cfg.subset_size, len(dataset))))
    if cfg.subset_name:
        assert cfg.subset_field is not None, "subset_field must be specified when using subset_name"
        dataset = dataset.filter(lambda x: x[cfg.subset_field] == cfg.subset_name)

    all_indices = list(range(len(dataset)))
    local_indices = all_indices[rank::world_size]

    if is_main_process(rank):
        print(f"Total dataset size: {len(dataset)}")
        print(f"World size: {world_size}")
        print(f"Batch size: {batch_size}")

    # Load models
    if is_main_process(rank):
        print("Loading models...")
    dream_model, dream_tokenizer, ar_model, reward_tokenizer, token_mapping, mapped_embeds = load_models_ar(cfg)

    model_cfg = get_model_config(cfg.dream_model, dream_tokenizer)

    if is_distributed:
        dist.barrier()

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Model type: {model_cfg['type']}")
        print(f"Method: EntRGi/APS (AR log-likelihood reward)")
        print(f"AR model: {cfg.reward_model}")
        print(f"K={cfg.K}, T={cfg.T}, M={cfg.M}, eta={cfg.eta}")
        print(f"World size: {world_size}, Batch size: {batch_size}")
        print(f"{'='*60}\n")

    local_results = []
    num_batches = (len(local_indices) + batch_size - 1) // batch_size
    batch_range = range(0, len(local_indices), batch_size)

    if is_main_process(rank):
        batch_range = tqdm(list(batch_range), desc="Evaluating EntRGi/APS (AR)", total=num_batches)

    for batch_start in batch_range:
        batch_indices = local_indices[batch_start:batch_start + batch_size]
        batch_prompts = [dataset[idx][cfg.prompt_field] for idx in batch_indices]

        torch.manual_seed(cfg.seed)

        batch_results = generate(
            dream_model, dream_tokenizer, ar_model, reward_tokenizer,
            token_mapping, mapped_embeds, batch_prompts, cfg, model_cfg
        )

        if is_distributed:
            dist.barrier()

        for idx, result in zip(batch_indices, batch_results):
            local_results.append({
                "idx": idx,
                "prompt": dataset[idx][cfg.prompt_field],
                "dataset_all_info": dataset[idx],
                **result
            })

            if is_main_process(rank):
                print(f"[{idx}] Top@1={result['top1_reward']:.4f}, Avg@N={result['avgN_reward']:.4f}")

    all_results = gather_results(local_results, world_size, is_distributed)

    if is_main_process(rank):
        cfg.device = str(cfg.device)
        top1_rewards = [r["top1_reward"] for r in all_results]
        avgN_rewards = [r["avgN_reward"] for r in all_results]

        print(f"\n{'='*60}")
        print(f"RESULTS: EntRGi/APS (AR log-likelihood)")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_results)}")
        print(f"Top@1 Reward:  {sum(top1_rewards)/len(top1_rewards):.4f}")
        print(f"Avg@N Reward:  {sum(avgN_rewards)/len(avgN_rewards):.4f}")

        save_results(cfg, "entrgi_aps_ar", top1_rewards, avgN_rewards, all_results)

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()

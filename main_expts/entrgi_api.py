"""
Thin batched wrapper around `generate()` from entrgi.py for oracle data generation.

Imports `generate` and supporting pieces from the existing modules without
modifying them. Module-level caches keep Dream + reward model resident across
calls (one cache per process; under torchrun each rank has its own).

Run from `main_expts/` so flat imports resolve (entrgi.py uses `from utils import ...`).
"""

import os
import sys
from typing import List, Tuple, Optional

import torch

# Flat imports, matching entrgi.py's convention.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from entrgi import generate  # noqa: E402
from bon import generate as bon_generate  # noqa: E402
from utils import Config, load_models, get_model_config  # noqa: E402


_MODELS = None  # (dream_m, dream_tok, reward_m, reward_tok, tok_map, mapped_emb)
_MODEL_CFG = None
_LOADED_KEY = None  # (dream_model, reward_model, device_str)


def _get_models(dream_model: str, reward_model: str, device: str):
    global _MODELS, _MODEL_CFG, _LOADED_KEY
    key = (dream_model, reward_model, device)
    if _MODELS is None or _LOADED_KEY != key:
        cfg = Config(dream_model=dream_model, reward_model=reward_model, device=device)
        _MODELS = load_models(cfg)
        _MODEL_CFG = get_model_config(dream_model, _MODELS[1])
        _LOADED_KEY = key
    return _MODELS, _MODEL_CFG


def run_entrgi_on_prompts(
    prompts: List[str],
    K: int = 4,
    T: int = 128,
    M: int = 3,
    eta: float = 0.5,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    dream_model: str = "Dream-org/Dream-v0-Instruct-7B",
    reward_model: str = "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    use_entrgi: bool = True,
    use_aps: bool = False,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    alg: str = "entropy",
    alg_temp: Optional[float] = None,
    deprioritize_eos_flag: bool = True,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> List[Tuple[List[str], List[float]]]:
    """Batched EntRGi.

    Returns a list of `(completions, rewards)` tuples in the same order as
    `prompts`. Each tuple has length `K`. The effective number of parallel
    diffusion chains per forward pass is `len(prompts) * K`.
    """
    device = str(device)
    models, model_cfg = _get_models(dream_model, reward_model, device)
    dream_m, dream_tok, reward_m, reward_tok, tok_map, mapped_emb = models

    cfg = Config(
        dream_model=dream_model,
        reward_model=reward_model,
        K=K,
        T=T,
        M=M,
        eta=eta,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
        alg_temp=alg_temp,
        deprioritize_eos=deprioritize_eos_flag,
        max_new_tokens=max_new_tokens,
        use_entrgi=use_entrgi,
        use_aps=use_aps,
        device=device,
        seed=seed if seed is not None else 0,
    )
    if seed is not None:
        torch.manual_seed(seed)

    results = generate(
        dream_m, dream_tok, reward_m, reward_tok, tok_map, mapped_emb,
        prompts, cfg, model_cfg,
    )
    return [(r["all_responses"], r["all_rewards"]) for r in results]


def run_entrgi_on_prompt(prompt: str, **kwargs) -> Tuple[List[str], List[float]]:
    """Single-prompt convenience wrapper; delegates to `run_entrgi_on_prompts`."""
    return run_entrgi_on_prompts([prompt], **kwargs)[0]


def run_bon_on_prompts(
    prompts: List[str],
    K: int = 4,
    T: int = 128,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    dream_model: str = "Dream-org/Dream-v0-Instruct-7B",
    reward_model: str = "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    alg: str = "entropy",
    alg_temp: Optional[float] = None,
    deprioritize_eos_flag: bool = True,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> List[Tuple[List[str], List[float]]]:
    """Plain Best-of-N baseline (no gradient tilting).

    Runs K independent Dream samples per prompt, scores each with Skywork,
    returns all K completions + rewards. Use as a control to measure how
    much EntRGi actually lifts top-1 reward.
    """
    device = str(device)
    models, _ = _get_models(dream_model, reward_model, device)
    dream_m, dream_tok, reward_m, reward_tok, _tok_map, _mapped_emb = models

    cfg = Config(
        dream_model=dream_model,
        reward_model=reward_model,
        K=K,
        T=T,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        device=device,
        seed=seed if seed is not None else 0,
    )
    if seed is not None:
        torch.manual_seed(seed)

    results = bon_generate(
        dream_m, dream_tok, reward_m, reward_tok,
        prompts, cfg,
        alg=alg, alg_temp=alg_temp,
        do_deprioritize_eos=deprioritize_eos_flag,
    )
    return [(r["all_responses"], r["all_rewards"]) for r in results]

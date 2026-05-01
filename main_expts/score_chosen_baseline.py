"""
Score the `chosen` (or `rejected`) completions from a prompts dataset with the
same Skywork reward model used by EntRGi. Output JSONL with the same schema as
generate_entrgi_oracle_data.py so paired comparison against the oracle is a
trivial JSON read.

Reuses `compute_discrete_reward` from utils.py to guarantee tokenization,
chat-template, and logit-extraction match exactly what EntRGi sees at its
final reward step. No diffusion, no Dream model — single Skywork forward
per record. ~10-30 min for 10k records on a single H100.

Run:
    cd /home/an34232/Repos/entrgi/main_expts
    HF_TOKEN=$HF_TOKEN python score_chosen_baseline.py \\
        --output_file data/baseline_chosen.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts_dataset", default="allenai/tulu-3-wildchat-if-on-policy-8b")
    p.add_argument("--prompts_split", default="train")
    p.add_argument("--num_prompts", type=int, default=20000,
                   help="-1 for full split.")
    p.add_argument("--prompts_seed", type=int, default=42,
                   help="MUST match the seed used for oracle generation so ids align.")
    p.add_argument("--source_field", choices=["chosen", "rejected"], default="chosen")
    p.add_argument("--reward_model", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    p.add_argument("--output_file", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    # Length-parity diagnostic: truncate the chosen response to N Dream tokens
    # (the same way EntRGi caps generation) before scoring with Skywork.
    p.add_argument("--truncate_response_tokens", type=int, default=0,
                   help="If >0, truncate response to N tokens (Dream tokenizer) before scoring. "
                        "Use this to isolate response-length confound from completion quality.")
    p.add_argument("--truncate_tokenizer", default="Dream-org/Dream-v0-Instruct-7B",
                   help="Tokenizer used for truncation (matches what EntRGi generates with).")
    return p.parse_args()


def extract_prompt_text(row) -> Optional[str]:
    """Identical to generate_entrgi_oracle_data.extract_prompt_text — keeps id assignment in sync."""
    for key in ("prompt", "messages", "chosen", "rejected"):
        if key not in row or row[key] is None:
            continue
        v = row[key]
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            user_turns = [
                m.get("content") for m in v
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
            ]
            if user_turns:
                return user_turns[-1].strip()
            contents = [
                m.get("content") for m in v
                if isinstance(m, dict) and m.get("content")
            ]
            if contents:
                return contents[-1].strip()
    return None


def extract_assistant_text(turns) -> Optional[str]:
    """Last assistant-role content from a chat-format list."""
    if not isinstance(turns, list):
        return None
    asst = [
        m.get("content") for m in turns
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")
    ]
    return asst[-1].strip() if asst else None


def _load_reward_model(name: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(name)
    if (path / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel
        peft_cfg = PeftConfig.from_pretrained(name)
        rm = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16, num_labels=1
        ).to(device)
        rm = PeftModel.from_pretrained(rm, name).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
    else:
        rm = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype=torch.bfloat16, num_labels=1
        ).to(device)
        tok = AutoTokenizer.from_pretrained(name)
    rm.eval()
    for p_ in rm.parameters():
        p_.requires_grad = False
    return rm, tok


def main():
    args = parse_args()

    if not os.environ.get("HF_TOKEN") and not args.reward_model.startswith("/"):
        print(
            "WARNING: HF_TOKEN env var not set. Skywork-Reward-V2 models are gated.",
            file=sys.stderr,
        )

    from datasets import load_dataset
    from utils import compute_discrete_reward  # type: ignore

    print(f"[info] loading reward model {args.reward_model} on {args.device}", file=sys.stderr)
    reward_model, reward_tokenizer = _load_reward_model(args.reward_model, args.device)

    truncate_tok = None
    if args.truncate_response_tokens > 0:
        from transformers import AutoTokenizer
        print(f"[info] loading {args.truncate_tokenizer} for response truncation "
              f"to {args.truncate_response_tokens} tokens", file=sys.stderr)
        truncate_tok = AutoTokenizer.from_pretrained(
            args.truncate_tokenizer, trust_remote_code=True
        )

    print(f"[info] loading prompts: {args.prompts_dataset}/{args.prompts_split}", file=sys.stderr)
    ds = load_dataset(args.prompts_dataset, split=args.prompts_split).shuffle(seed=args.prompts_seed)
    if args.num_prompts != -1:
        ds = ds.select(range(min(args.num_prompts, len(ds))))

    # Same dedup-by-prompt-text + stable id assignment as oracle generation.
    seen = set()
    rows = []
    for i, row in enumerate(ds):
        ptext = extract_prompt_text(row)
        if not ptext or ptext in seen:
            continue
        seen.add(ptext)
        atext = extract_assistant_text(row.get(args.source_field))
        rows.append((f"{args.prompts_dataset}:{i}", ptext, atext))

    print(f"[info] {len(rows)} unique prompts; scoring {args.source_field} completions",
          file=sys.stderr)

    done = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass

    todo = [r for r in rows if r[0] not in done]
    skipped = len(rows) - len(todo)
    kept = failed = no_response = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    out = open(args.output_file, "a")
    try:
        for pid, ptext, atext in tqdm(todo, desc="scoring"):
            if not atext:
                no_response += 1
                continue
            if truncate_tok is not None:
                tok_ids = truncate_tok.encode(atext, add_special_tokens=False)
                if len(tok_ids) > args.truncate_response_tokens:
                    atext = truncate_tok.decode(
                        tok_ids[:args.truncate_response_tokens], skip_special_tokens=True
                    )
            try:
                r = compute_discrete_reward(
                    reward_model, reward_tokenizer, ptext, atext, args.device
                )
            except Exception as e:
                print(f"[warn] {pid}: {type(e).__name__}: {e}", file=sys.stderr)
                failed += 1
                continue
            rec = {
                "id": pid,
                "prompt": ptext,
                "response": atext,
                "reward": float(r),
                "K": 1,
                "source": args.prompts_dataset,
                "method": f"baseline_{args.source_field}",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            kept += 1
    finally:
        out.close()

    print(
        f"kept={kept} skipped={skipped} no_response={no_response} failed={failed}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare a fixed Skywork preference subset for chosen-SFT and Entrgi-oracle runs.

This script:
1. Loads Skywork preference data.
2. Deduplicates by prompt text, keeping one record per unique prompt.
3. Builds a fixed train/heldout split over unique prompts.
4. Writes:
   - chosen SFT data in dllm message format
   - prompt-only data for oracle generation / heldout eval
   - full heldout preference rows for reference

Example:
    conda run -n rlmt python prepare_skywork_preference_subset.py \
        --train_size 5000 \
        --heldout_size 500 \
        --truncate_response_tokens 128 \
        --output_dir data/skywork_pref_5k_trunc128
"""

import argparse
import json
import os
import random
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="Skywork/Skywork-Reward-Preference-80K-v0.2",
        help="Preference dataset to load.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=5000,
        help="Number of unique-prompt examples for train.",
    )
    parser.add_argument(
        "--heldout_size",
        type=int,
        default=500,
        help="Number of unique-prompt examples for heldout/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt-level selection and split.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for prepared files.",
    )
    parser.add_argument(
        "--truncate_response_tokens",
        type=int,
        default=0,
        help="If >0, truncate chosen/rejected assistant responses to this many Dream tokens.",
    )
    parser.add_argument(
        "--truncate_tokenizer",
        default="Dream-org/Dream-v0-Instruct-7B",
        help="Tokenizer used for response truncation.",
    )
    return parser.parse_args()


def _extract_prompt_text(row: Dict) -> Optional[str]:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    for key in ("chosen", "rejected", "messages"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        user_turns = [
            turn.get("content", "").strip()
            for turn in value
            if isinstance(turn, dict)
            and turn.get("role") == "user"
            and turn.get("content")
        ]
        if user_turns:
            return user_turns[-1]
    return None


def _extract_assistant_text(turns) -> Optional[str]:
    if not isinstance(turns, list):
        return None
    assistant_turns = [
        turn.get("content", "").strip()
        for turn in turns
        if isinstance(turn, dict)
        and turn.get("role") == "assistant"
        and turn.get("content")
    ]
    return assistant_turns[-1] if assistant_turns else None


def _truncate_text(text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).strip()


def _write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    from datasets import load_dataset

    truncate_tokenizer = None
    if args.truncate_response_tokens > 0:
        from transformers import AutoTokenizer

        print(
            f"[info] loading tokenizer {args.truncate_tokenizer} for truncation "
            f"to {args.truncate_response_tokens} tokens",
            file=sys.stderr,
        )
        truncate_tokenizer = AutoTokenizer.from_pretrained(
            args.truncate_tokenizer,
            trust_remote_code=True,
        )

    print(f"[info] loading dataset {args.dataset}/{args.split}", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.shuffle(seed=args.seed)

    unique_rows = []
    seen_prompts = set()
    dropped_missing = 0

    for idx, row in tqdm(enumerate(ds), total=len(ds)):
        prompt = _extract_prompt_text(row)
        chosen = _extract_assistant_text(row.get("chosen"))
        rejected = _extract_assistant_text(row.get("rejected"))

        if not prompt or not chosen or not rejected:
            dropped_missing += 1
            continue
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        if truncate_tokenizer is not None:
            chosen = _truncate_text(chosen, truncate_tokenizer, args.truncate_response_tokens)
            rejected = _truncate_text(rejected, truncate_tokenizer, args.truncate_response_tokens)

        unique_rows.append(
            {
                "id": row.get("id", f"{args.dataset}:{idx}"),
                "prompt": prompt,
                "chosen_response": chosen,
                "rejected_response": rejected,
                "source_dataset": args.dataset,
            }
        )

    need = args.train_size + args.heldout_size
    if len(unique_rows) < need:
        raise ValueError(
            f"Need {need} unique prompts but only found {len(unique_rows)} after filtering."
        )

    train_rows = unique_rows[: args.train_size]
    heldout_rows = unique_rows[args.train_size : args.train_size + args.heldout_size]

    out_dir = Path(args.output_dir)
    chosen_sft_dir = out_dir / "chosen_sft"
    prompt_only_dir = out_dir / "prompt_only"

    chosen_train = [
        {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["chosen_response"]},
            ]
        }
        for row in train_rows
    ]
    chosen_test = [
        {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["chosen_response"]},
            ]
        }
        for row in heldout_rows
    ]

    prompt_train = [
        {"id": row["id"], "prompt": row["prompt"]}
        for row in train_rows
    ]
    prompt_test = [
        {"id": row["id"], "prompt": row["prompt"]}
        for row in heldout_rows
    ]

    heldout_preferences = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "chosen_response": row["chosen_response"],
            "rejected_response": row["rejected_response"],
        }
        for row in heldout_rows
    ]

    _write_jsonl(chosen_sft_dir / "train.jsonl", chosen_train)
    _write_jsonl(chosen_sft_dir / "test.jsonl", chosen_test)
    _write_jsonl(prompt_only_dir / "train.jsonl", prompt_train)
    _write_jsonl(prompt_only_dir / "test.jsonl", prompt_test)
    _write_jsonl(out_dir / "heldout_preferences.jsonl", heldout_preferences)

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "train_size": args.train_size,
        "heldout_size": args.heldout_size,
        "truncate_response_tokens": args.truncate_response_tokens,
        "truncate_tokenizer": args.truncate_tokenizer if args.truncate_response_tokens > 0 else None,
        "unique_prompts_available": len(unique_rows),
        "dropped_missing_fields": dropped_missing,
        "files": {
            "chosen_sft_train": str(chosen_sft_dir / "train.jsonl"),
            "chosen_sft_test": str(chosen_sft_dir / "test.jsonl"),
            "prompt_only_train": str(prompt_only_dir / "train.jsonl"),
            "prompt_only_test": str(prompt_only_dir / "test.jsonl"),
            "heldout_preferences": str(out_dir / "heldout_preferences.jsonl"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"[info] wrote chosen_sft train={len(chosen_train)} test={len(chosen_test)} "
        f"and prompt_only train={len(prompt_train)} test={len(prompt_test)}",
        file=sys.stderr,
    )
    print(f"[info] manifest: {out_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()

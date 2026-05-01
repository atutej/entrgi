#!/usr/bin/env python3
"""
Prepare a fixed random split of RM-Bench for diagnostic chosen-SFT runs.

This mirrors the RewardBench-2 diagnostic pipeline:
1. Load RM-Bench.
2. Shuffle with a fixed seed.
3. Build a fixed train/heldout split over prompts.
4. Write:
   - chosen SFT data in dllm message format
   - prompt-only train/test files
   - heldout preference rows for reference

Example:
    conda run -n rlmt python prepare_rm_bench_split.py \
        --train_size 1000 \
        --heldout_size 300 \
        --truncate_response_tokens 128 \
        --output_dir data/rm_bench_diag_split_trunc128
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="THU-KEG/RM-Bench",
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
        default=1000,
        help="Number of prompt examples for train.",
    )
    parser.add_argument(
        "--heldout_size",
        type=int,
        default=300,
        help="Number of prompt examples for heldout/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and split.",
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
        help="If >0, truncate chosen/rejected responses to this many Dream tokens.",
    )
    parser.add_argument(
        "--truncate_tokenizer",
        default="Dream-org/Dream-v0-Instruct-7B",
        help="Tokenizer used for response truncation.",
    )
    return parser.parse_args()


def _extract_text(value) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _extract_text_list(value) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


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

    rows = []
    dropped_missing = 0
    seen_prompts = set()

    for idx, row in enumerate(ds):
        prompt = _extract_text(row.get("prompt"))
        chosen_responses = _extract_text_list(row.get("chosen"))
        rejected_responses = _extract_text_list(row.get("rejected"))
        domain = row.get("domain")

        if not prompt or not chosen_responses or not rejected_responses:
            dropped_missing += 1
            continue
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        if truncate_tokenizer is not None:
            chosen_responses = [
                _truncate_text(text, truncate_tokenizer, args.truncate_response_tokens)
                for text in chosen_responses
            ]
            rejected_responses = [
                _truncate_text(text, truncate_tokenizer, args.truncate_response_tokens)
                for text in rejected_responses
            ]

        rows.append(
            {
                "id": row.get("id", f"{args.dataset}:{idx}"),
                "prompt": prompt,
                "chosen_response": chosen_responses[0],
                "chosen_responses": chosen_responses,
                "rejected_responses": rejected_responses,
                "domain": domain,
                "source_dataset": args.dataset,
            }
        )

    need = args.train_size + args.heldout_size
    if len(rows) < need:
        raise ValueError(
            f"Need {need} examples but only found {len(rows)} after filtering."
        )

    train_rows = rows[: args.train_size]
    heldout_rows = rows[args.train_size : args.train_size + args.heldout_size]

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

    prompt_train = [{"id": row["id"], "prompt": row["prompt"]} for row in train_rows]
    prompt_test = [{"id": row["id"], "prompt": row["prompt"]} for row in heldout_rows]

    heldout_preferences = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "chosen_response": row["chosen_response"],
            "chosen_responses": row["chosen_responses"],
            "rejected_responses": row["rejected_responses"],
            "domain": row["domain"],
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
        "available_rows_after_filtering": len(rows),
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
        f"prompt_only train={len(prompt_train)} test={len(prompt_test)}",
        file=sys.stderr,
    )
    print(f"[info] wrote manifest to {out_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()

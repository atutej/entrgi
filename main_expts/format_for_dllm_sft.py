"""
Convert EntRGi oracle JSONL (from generate_entrgi_oracle_data.py) into the
raw-messages JSONL format that ZHZisZZ/dllm's examples/dream/sft.py consumes.

Each output record:
    {"messages": [{"role": "user", "content": <prompt>},
                  {"role": "assistant", "content": <response>}]}

This matches what `dllm.utils.default_sft_map_fn` expects — it calls
`tokenizer.apply_chat_template(row["messages"], ...)`. See
dllm/dllm/utils/data.py:225.

After running this script, launch Dream SFT:

    cd /home/an34232/Repos/entrgi/dllm
    accelerate launch \\
        --config_file scripts/accelerate_configs/fsdp.yaml \\
        examples/dream/sft.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --dataset_args <output_dir from this script> \\
        --load_preprocessed_data False \\
        --mask_prompt_loss True \\
        --num_train_epochs 2 \\
        --learning_rate 5e-6 \\
        --output_dir .models/dream-entrgi-sft

`load_sft_dataset(output_dir, load_preprocessed_data=False)` falls through to
`load_dataset(output_dir)`, which auto-detects `train.jsonl` / `test.jsonl`
by extension (HF datasets >= 2.12). In-script
`dataset.map(default_sft_map_fn)` then tokenizes messages on the fly.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import List, Optional


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_jsonl", required=True, help="Output of generate_entrgi_oracle_data.py")
    p.add_argument("--output_dir", required=True,
                   help="Dir to write train.jsonl (and test.jsonl if split>0).")
    p.add_argument("--min_reward", type=float, default=float("-inf"),
                   help="Drop records whose best reward is below this value.")
    p.add_argument(
        "--keep_top_reward_percentile",
        type=float,
        default=100.0,
        help="Keep only the top N percent of records by reward after basic validity checks. "
             "Examples: 100 = keep all, 50 = keep top half, 25 = keep top quarter.",
    )
    p.add_argument("--train_test_split", type=float, default=0.0,
                   help="Fraction held out as test (e.g. 0.05). 0 = train only.")
    p.add_argument("--tokenizer", default="Dream-org/Dream-v0-Instruct-7B",
                   help="Used only for length-based filtering, not for pre-tokenization.")
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--max_response_tokens", type=int, default=1024)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--align_split_to", default=None,
                   help="Path to a previously-formatted SFT dir with train.jsonl (and "
                        "optionally test.jsonl). When set, --train_test_split is ignored: "
                        "each input record is routed into whichever split its prompt landed "
                        "in there. Records whose prompt isn't found in the reference dir are "
                        "dropped. Used to keep paired SFT comparisons (e.g. entrgi vs chosen) "
                        "on the exact same train/test partition.")
    return p.parse_args()


def _read_jsonl(path: str):
    records = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {ln}: {e}", file=sys.stderr)
    return records


def _write_jsonl(path: str, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _percentiles(values: List[int], ps=(0.5, 0.9, 0.99)):
    if not values:
        return {f"p{int(p * 100)}": None for p in ps}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out = {}
    for p in ps:
        idx = min(n - 1, max(0, int(math.ceil(p * n)) - 1))
        out[f"p{int(p * 100)}"] = sorted_vals[idx]
    return out


def _reward_cutoff(values: List[float], keep_top_percent: float) -> Optional[float]:
    if not values:
        return None
    if keep_top_percent >= 100.0:
        return min(values)
    if keep_top_percent <= 0.0:
        raise ValueError("--keep_top_reward_percentile must be in (0, 100].")
    sorted_vals = sorted(values, reverse=True)
    keep_n = max(1, int(math.ceil(len(sorted_vals) * keep_top_percent / 100.0)))
    return sorted_vals[keep_n - 1]


def main():
    args = parse_args()

    if not (0.0 < args.keep_top_reward_percentile <= 100.0):
        print("ERROR: --keep_top_reward_percentile must be in (0, 100].", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input_jsonl):
        print(f"ERROR: input_jsonl does not exist: {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Defer heavy imports until after arg parsing.
    from transformers import AutoTokenizer

    print(f"[info] loading tokenizer {args.tokenizer} for length checks...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    raw = _read_jsonl(args.input_jsonl)
    n_in = len(raw)
    print(f"[info] read {n_in} records from {args.input_jsonl}", file=sys.stderr)

    reward_ready = []
    drop_reward = drop_empty = drop_prompt_len = drop_response_len = 0
    prompt_token_lens: List[int] = []
    response_token_lens: List[int] = []

    for rec in raw:
        prompt = rec.get("prompt") or ""
        response = rec.get("response") or ""
        reward = rec.get("reward")

        if not prompt.strip() or not response.strip():
            drop_empty += 1
            continue
        if reward is None or float(reward) < args.min_reward:
            drop_reward += 1
            continue

        reward_ready.append(
            {
                "id": rec.get("id"),
                "prompt": prompt,
                "response": response,
                "reward": float(reward),
            }
        )

    reward_cutoff = _reward_cutoff(
        [rec["reward"] for rec in reward_ready],
        args.keep_top_reward_percentile,
    )
    if reward_cutoff is None:
        print("ERROR: no records survived reward/emptiness filtering.", file=sys.stderr)
        sys.exit(1)

    reward_filtered = [
        rec for rec in reward_ready
        if rec["reward"] >= reward_cutoff
    ]
    dropped_percentile = len(reward_ready) - len(reward_filtered)
    print(
        f"[info] reward percentile filter: keep_top={args.keep_top_reward_percentile:.2f}% "
        f"cutoff={reward_cutoff:.4f} kept={len(reward_filtered)}/{len(reward_ready)}",
        file=sys.stderr,
    )

    kept = []
    for rec in reward_filtered:
        prompt = rec["prompt"]
        response = rec["response"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Mirror how default_sft_map_fn computes prompt_len:
        # prompt_tokens = apply_chat_template(messages[:-1], add_generation_prompt=True)
        # full_tokens   = apply_chat_template(messages,       add_generation_prompt=False)
        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True
            )
            full_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        except Exception as e:
            print(f"[warn] apply_chat_template failed for id={rec.get('id')}: {e}",
                  file=sys.stderr)
            drop_empty += 1
            continue

        prompt_len = len(prompt_tokens)
        resp_len = max(0, len(full_tokens) - prompt_len)

        if prompt_len > args.max_prompt_tokens:
            drop_prompt_len += 1
            continue
        if resp_len > args.max_response_tokens:
            drop_response_len += 1
            continue

        kept.append({"messages": messages})
        prompt_token_lens.append(prompt_len)
        response_token_lens.append(resp_len)

    n_out = len(kept)
    print(
        f"[info] kept {n_out}/{n_in} "
        f"(dropped: reward={drop_reward}, percentile={dropped_percentile}, "
        f"empty/template={drop_empty}, "
        f"prompt_len={drop_prompt_len}, response_len={drop_response_len})",
        file=sys.stderr,
    )

    if n_out == 0:
        print("ERROR: no records survived filtering; nothing to write.", file=sys.stderr)
        sys.exit(1)

    if args.align_split_to:
        prompt_to_split = {}
        for split_name in ("train", "test"):
            ref_path = os.path.join(args.align_split_to, f"{split_name}.jsonl")
            if not os.path.exists(ref_path):
                continue
            with open(ref_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        prompt = rec["messages"][0]["content"]
                        prompt_to_split[prompt] = split_name
                    except Exception:
                        continue
        if not prompt_to_split:
            print(f"ERROR: --align_split_to {args.align_split_to} had no usable "
                  f"train.jsonl/test.jsonl.", file=sys.stderr)
            sys.exit(1)
        train_split, test_split = [], []
        unmatched = 0
        for r in kept:
            prompt = r["messages"][0]["content"]
            sp = prompt_to_split.get(prompt)
            if sp == "train":
                train_split.append(r)
            elif sp == "test":
                test_split.append(r)
            else:
                unmatched += 1
        print(f"[info] align_split_to={args.align_split_to}: "
              f"matched train={len(train_split)} test={len(test_split)} "
              f"unmatched_dropped={unmatched}",
              file=sys.stderr)
    else:
        rng = random.Random(args.split_seed)
        rng.shuffle(kept)
        if args.train_test_split > 0.0:
            n_test = max(1, int(round(args.train_test_split * n_out)))
            n_test = min(n_test, n_out - 1)
            test_split = kept[:n_test]
            train_split = kept[n_test:]
        else:
            train_split = kept
            test_split = []

    train_path = os.path.join(args.output_dir, "train.jsonl")
    _write_jsonl(train_path, train_split)
    print(f"[info] wrote {len(train_split)} records to {train_path}", file=sys.stderr)

    if test_split:
        test_path = os.path.join(args.output_dir, "test.jsonl")
        _write_jsonl(test_path, test_split)
        print(f"[info] wrote {len(test_split)} records to {test_path}", file=sys.stderr)

    # Summary stats.
    pp = _percentiles(prompt_token_lens)
    rp = _percentiles(response_token_lens)
    print("[info] prompt token lengths:  "
          f"p50={pp['p50']} p90={pp['p90']} p99={pp['p99']}", file=sys.stderr)
    print("[info] response token lengths: "
          f"p50={rp['p50']} p90={rp['p90']} p99={rp['p99']}", file=sys.stderr)


if __name__ == "__main__":
    main()

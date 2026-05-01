#!/usr/bin/env python3
"""
Build a preference SFT split by overlap with prompts found in saved eval result JSONs.

Split rule:
  - train: rows whose prompt is NOT in eval prompts
  - test:  rows whose prompt IS in eval prompts
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dataset", required=True)
    p.add_argument("--input_split", required=True)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--chosen_field", default="chosen")
    p.add_argument("--rejected_field", default="rejected")
    p.add_argument("--eval_results_roots", nargs="+", required=True)
    p.add_argument("--eval_file_glob", default="*_k1_temp0.1_T128_infer_seed*.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--truncate_response_tokens", type=int, default=0)
    p.add_argument("--truncate_tokenizer", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--normalize_prompts", action="store_true")
    p.add_argument("--strict_no_overlap", action="store_true")
    return p.parse_args()


_ws = re.compile(r"\s+")


def norm_prompt(text: str, normalize: bool) -> str:
    t = text.strip()
    if normalize:
        t = _ws.sub(" ", t).lower()
    return t


def extract_text_list(v) -> List[str]:
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    out.append(content.strip())
        return out
    return []


def truncate_text(text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_eval_prompts(roots: List[str], file_glob: str, normalize: bool) -> Set[str]:
    prompts: Set[str] = set()
    files = []
    for root in roots:
        files.extend(glob.glob(str(Path(root) / file_glob)))
    for p in sorted(set(files)):
        try:
            data = json.loads(Path(p).read_text())
        except Exception:
            continue
        for rec in data.get("results", []):
            prompt = rec.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.add(norm_prompt(prompt, normalize))
    return prompts


def main():
    args = parse_args()

    from datasets import load_dataset

    tok = None
    if args.truncate_response_tokens > 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.truncate_tokenizer, trust_remote_code=True)

    eval_prompts = load_eval_prompts(args.eval_results_roots, args.eval_file_glob, args.normalize_prompts)
    if not eval_prompts:
        print("ERROR: no eval prompts found from --eval_results_roots / --eval_file_glob", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset(args.input_dataset, split=args.input_split)

    train_rows_raw = []
    test_rows_raw = []
    dropped_missing = 0
    seen = set()

    for i, row in enumerate(ds):
        prompt = row.get(args.prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            dropped_missing += 1
            continue
        chosen_list = extract_text_list(row.get(args.chosen_field))
        rejected_list = extract_text_list(row.get(args.rejected_field))
        if not chosen_list or not rejected_list:
            dropped_missing += 1
            continue

        key = norm_prompt(prompt, args.normalize_prompts)
        if key in seen:
            continue
        seen.add(key)

        chosen_list = [truncate_text(t, tok, args.truncate_response_tokens) if tok else t for t in chosen_list]
        rejected_list = [truncate_text(t, tok, args.truncate_response_tokens) if tok else t for t in rejected_list]

        item = {
            "id": row.get("id", f"{args.input_dataset}:{i}"),
            "prompt": prompt.strip(),
            "chosen_response": chosen_list[0],
            "chosen_responses": chosen_list,
            "rejected_responses": rejected_list,
        }
        if key in eval_prompts:
            test_rows_raw.append(item)
        else:
            train_rows_raw.append(item)

    chosen_train = [{"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["chosen_response"]}]} for r in train_rows_raw]
    chosen_test = [{"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["chosen_response"]}]} for r in test_rows_raw]

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "chosen_sft" / "train.jsonl", chosen_train)
    write_jsonl(out_dir / "chosen_sft" / "test.jsonl", chosen_test)
    write_jsonl(out_dir / "heldout_preferences.jsonl", test_rows_raw)

    manifest = {
        "input_dataset": args.input_dataset,
        "input_split": args.input_split,
        "prompt_field": args.prompt_field,
        "chosen_field": args.chosen_field,
        "rejected_field": args.rejected_field,
        "normalize_prompts": args.normalize_prompts,
        "truncate_response_tokens": args.truncate_response_tokens,
        "truncate_tokenizer": args.truncate_tokenizer if args.truncate_response_tokens > 0 else None,
        "eval_roots": args.eval_results_roots,
        "eval_file_glob": args.eval_file_glob,
        "eval_prompt_count": len(eval_prompts),
        "train_count": len(chosen_train),
        "test_count": len(chosen_test),
        "dropped_missing": dropped_missing,
        "files": {
            "chosen_train": str(out_dir / "chosen_sft" / "train.jsonl"),
            "chosen_test": str(out_dir / "chosen_sft" / "test.jsonl"),
            "heldout_preferences": str(out_dir / "heldout_preferences.jsonl"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.strict_no_overlap:
        # Verify train split has no overlap.
        bad = 0
        for r in train_rows_raw:
            if norm_prompt(r["prompt"], args.normalize_prompts) in eval_prompts:
                bad += 1
        if bad:
            print(f"ERROR: strict_no_overlap violated for {bad} train rows", file=sys.stderr)
            sys.exit(2)

    print(f"[info] wrote train={len(chosen_train)} test={len(chosen_test)} to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract prompt-only jsonl from an existing bon_infer result json.

Input JSON format is expected to have:
  {"results": [{"prompt": ...}, ...], ...}
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--id_prefix", default="extracted")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    results = data.get("results", [])
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for i, rec in enumerate(results):
            prompt = rec.get("prompt")
            if prompt is None:
                continue
            f.write(json.dumps({"id": f"{args.id_prefix}:{i}", "prompt": prompt}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

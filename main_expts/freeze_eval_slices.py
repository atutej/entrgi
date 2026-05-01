#!/usr/bin/env python3
"""
Freeze fixed eval slices for the 3 benchmark datasets.

Writes prompt-only jsonl files:
  - judgebench_eval_fixed.jsonl
  - reward-bench-2_eval_fixed.jsonl
  - rm-bench_eval_fixed.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset_size", type=int, default=256)
    return p.parse_args()


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    from datasets import load_dataset

    configs = [
        ("ScalerLab/JudgeBench", "gpt", "question", "judgebench"),
        ("allenai/reward-bench-2", "test", "prompt", "reward-bench-2"),
        ("THU-KEG/RM-Bench", "train", "prompt", "rm-bench"),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "subset_size": args.subset_size,
        "files": {},
    }

    for ds_name, split, prompt_field, short_name in configs:
        ds = load_dataset(ds_name, split=split).shuffle(seed=args.seed)
        n = min(args.subset_size, len(ds))
        ds = ds.select(range(n))

        rows = []
        for i, row in enumerate(ds):
            prompt = row.get(prompt_field)
            if prompt is None:
                continue
            rows.append(
                {
                    "id": f"{short_name}:{i}",
                    "prompt": prompt,
                }
            )

        out_path = out_dir / f"{short_name}_eval_fixed.jsonl"
        _write_jsonl(out_path, rows)
        manifest["files"][short_name] = str(out_path)
        manifest[f"{short_name}_count"] = len(rows)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

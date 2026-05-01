#!/usr/bin/env python3
"""
Materialize RewardBench-2 prompt slices exactly like the old benchmark eval style:
per-seed shuffle + first N prompts.

Outputs:
  <output_dir>/reward-bench-2_seed1_subset64.jsonl
  <output_dir>/reward-bench-2_seed2_subset64.jsonl
  <output_dir>/reward-bench-2_seed3_subset64.jsonl
  <output_dir>/manifest.json
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subset_size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    return p.parse_args()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    from datasets import load_dataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ds = load_dataset("allenai/reward-bench-2", split="test")
    manifest = {
        "dataset": "allenai/reward-bench-2",
        "split": "test",
        "subset_size": args.subset_size,
        "seeds": args.seeds,
        "files": {},
    }

    for seed in args.seeds:
        ds = base_ds.shuffle(seed=seed)
        n = min(args.subset_size, len(ds))
        ds = ds.select(range(n))

        rows = []
        for i, row in enumerate(ds):
            prompt = row.get("prompt")
            if prompt is None:
                continue
            rows.append({"id": f"reward-bench-2:seed{seed}:{i}", "prompt": prompt})

        out_path = out_dir / f"reward-bench-2_seed{seed}_subset{args.subset_size}.jsonl"
        write_jsonl(out_path, rows)
        manifest["files"][f"seed{seed}"] = str(out_path)
        manifest[f"seed{seed}_count"] = len(rows)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

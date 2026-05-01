"""
Generate EntRGi-tilted oracle data for Dream SFT.

Loads prompts from an HF dataset (default: Tulu-3 WildChat IF on-policy 8B),
runs EntRGi with K trajectories per prompt using a Dream diffusion LM tilted
by a Skywork reward model, keeps the argmax-by-reward completion, and writes
JSONL with flush-per-record.

Launch:
    # Single GPU
    cd /home/an34232/Repos/entrgi/main_expts
    python generate_entrgi_oracle_data.py --output_file ../oracle.jsonl

    # Multi-GPU (DDP, rank-strided prompts, per-rank shards, rank-0 merges)
    cd /home/an34232/Repos/entrgi/main_expts
    torchrun --nproc_per_node 8 generate_entrgi_oracle_data.py \
        --output_file ../oracle.jsonl

    # Sanity (20 prompts, pretty-prints records)
    python generate_entrgi_oracle_data.py --sanity --output_file /tmp/sanity.jsonl
"""

import argparse
import json
import os
import sys
from typing import Optional

from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Dataset
    p.add_argument("--prompts_dataset", default="allenai/tulu-3-wildchat-if-on-policy-8b")
    p.add_argument("--prompts_split", default="train")
    p.add_argument("--num_prompts", type=int, default=20000,
                   help="-1 for full split. Silently capped at dataset size.")
    p.add_argument("--prompts_seed", type=int, default=42)
    # Method
    p.add_argument("--method", choices=["entrgi", "bon"], default="entrgi",
                   help="entrgi = reward-gradient-tilted diffusion; "
                        "bon = plain Best-of-N baseline (no gradient, K samples + pick top-1).")
    # EntRGi hyperparams (M, eta ignored when --method bon)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Prompts per generate() call. Effective chains = batch_size * K.")
    # Models
    p.add_argument("--dream_model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--reward_model", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    # I/O
    p.add_argument("--output_file", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sanity", action="store_true")
    return p.parse_args()


def extract_prompt_text(row) -> Optional[str]:
    """Return a plain-string user prompt, or None if unusable.

    Tries `prompt` first (plain string in Tulu-3 WildChat IF), then common
    chat-format fields as fallbacks so swapping the dataset doesn't break
    the loader.
    """
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


def _collect_done_ids(output_file: str, world_size: int) -> set:
    done = set()
    candidates = [output_file]
    if world_size > 1:
        candidates += [
            f"{output_file}.rank{r}-of-{world_size}.jsonl" for r in range(world_size)
        ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def _pretty_print_record(rec, completions, rewards, best_idx, out=sys.stderr):
    sep = "=" * 80
    print(sep, file=out)
    print(f"[sanity] id={rec['id']}", file=out)
    print(f"[sanity] prompt:\n{rec['prompt']}", file=out)
    for i, (c, r) in enumerate(zip(completions, rewards)):
        mark = " <-- top1" if i == best_idx else ""
        print(f"[sanity] completion k={i} reward={r:.3f}{mark}:\n{c}", file=out)
    print(sep, file=out)


def main():
    args = parse_args()
    if args.sanity:
        args.num_prompts = 20

    # Defer heavy imports until after arg parsing (faster --help, clearer errors).
    import torch  # noqa: F401
    from datasets import load_dataset
    from entrgi import setup_distributed  # type: ignore
    from entrgi_api import run_bon_on_prompts, run_entrgi_on_prompts  # type: ignore

    rank, world_size, device, is_distributed = setup_distributed()
    is_rank0 = rank == 0

    if is_rank0 and not os.environ.get("HF_TOKEN") and not args.reward_model.startswith("/"):
        print(
            "WARNING: HF_TOKEN env var not set. Skywork-Reward-V2 models are gated; "
            "loading will fail unless you've accepted access and set HF_TOKEN.",
            file=sys.stderr,
        )

    # All ranks load the same dataset with the same shuffle seed → identical order.
    ds = load_dataset(args.prompts_dataset, split=args.prompts_split)
    ds = ds.shuffle(seed=args.prompts_seed)
    if args.num_prompts != -1:
        n = min(args.num_prompts, len(ds))
        ds = ds.select(range(n))

    # Dedupe by prompt text; assign stable ids.
    seen_texts = set()
    prompts_all = []
    for i, row in enumerate(ds):
        t = extract_prompt_text(row)
        if not t or t in seen_texts:
            continue
        seen_texts.add(t)
        prompts_all.append((f"{args.prompts_dataset}:{i}", t))

    if is_rank0:
        print(
            f"[rank{rank}] dataset={args.prompts_dataset} split={args.prompts_split} "
            f"rows_loaded={len(ds)} unique_prompts={len(prompts_all)} "
            f"world_size={world_size}",
            file=sys.stderr,
        )

    # Rank-strided partition of the deduped list.
    local_prompts = prompts_all[rank::world_size]

    shard_path = (
        args.output_file
        if world_size == 1
        else f"{args.output_file}.rank{rank}-of-{world_size}.jsonl"
    )
    os.makedirs(os.path.dirname(os.path.abspath(shard_path)) or ".", exist_ok=True)

    done_ids = _collect_done_ids(args.output_file, world_size) if args.resume else set()
    todo = [(pid, ptext) for pid, ptext in local_prompts if pid not in done_ids]
    skipped = len(local_prompts) - len(todo)

    kept = failed = 0
    running_sum = 0.0

    out = open(shard_path, "a")
    try:
        pbar = tqdm(
            range(0, len(todo), args.batch_size),
            desc=f"rank{rank}",
            disable=not is_rank0,
        )
        for start in pbar:
            batch = todo[start:start + args.batch_size]
            batch_ids = [b[0] for b in batch]
            batch_texts = [b[1] for b in batch]
            try:
                if args.method == "entrgi":
                    results = run_entrgi_on_prompts(
                        batch_texts,
                        K=args.K, T=args.T, M=args.M, eta=args.eta,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        dream_model=args.dream_model,
                        reward_model=args.reward_model,
                        use_entrgi=True,
                        device=str(device),
                    )
                else:
                    results = run_bon_on_prompts(
                        batch_texts,
                        K=args.K, T=args.T,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        dream_model=args.dream_model,
                        reward_model=args.reward_model,
                        device=str(device),
                    )
            except Exception as e:
                print(
                    f"[rank{rank}] batch failed (ids={batch_ids}): {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                failed += len(batch)
                continue

            for (pid, ptext), (comps, rewards) in zip(batch, results):
                if not comps or len(comps) != args.K:
                    failed += 1
                    continue
                best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
                rec = {
                    "id": pid,
                    "prompt": ptext,
                    "response": comps[best_idx],
                    "reward": float(rewards[best_idx]),
                    "all_rewards": [float(r) for r in rewards],
                    "K": args.K,
                    "source": args.prompts_dataset,
                    "method": args.method,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                kept += 1
                running_sum += rec["reward"]

                if args.sanity and is_rank0:
                    _pretty_print_record(rec, comps, rewards, best_idx)

                if kept % args.log_every == 0 and is_rank0:
                    print(
                        f"[rank0 kept={kept}] running mean top-1 reward = "
                        f"{running_sum / kept:.3f}",
                        file=sys.stderr,
                    )
    finally:
        out.close()

    # Synchronize ranks before merge.
    if is_distributed:
        import torch.distributed as dist
        dist.barrier()

    # Rank-0 concatenates per-rank shards into the canonical output file.
    if is_rank0 and world_size > 1:
        merged_path = args.output_file
        with open(merged_path, "a") as merged:
            for r in range(world_size):
                sp = f"{args.output_file}.rank{r}-of-{world_size}.jsonl"
                if not os.path.exists(sp):
                    continue
                with open(sp) as f:
                    for line in f:
                        merged.write(line)
        print(f"[rank0] merged {world_size} shards into {merged_path}", file=sys.stderr)

    print(
        f"[rank{rank}] kept={kept} skipped={skipped} failed={failed} "
        f"shard={shard_path}",
        file=sys.stderr,
    )

    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

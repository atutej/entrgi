import argparse
import glob
import json
import os
import statistics


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate heldout diagnostic eval JSONs.")
    p.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing bon_infer heldout result JSONs.",
    )
    p.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix before the model name, e.g. rb2-heldout or rm-heldout.",
    )
    return p.parse_args()


def summarize(results_dir: str, pattern: str):
    vals = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        with open(path) as f:
            vals.append(json.load(f)["metrics"]["mean_top1_reward"])
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def main():
    args = parse_args()

    methods = {
        "base": f"{args.prefix}_base_k1_temp0.1_T128_infer_seed*.json",
        "wildchat-chosen@100": (
            f"{args.prefix}_dream-chosen-trunc128-sft-lora-r32-alllinear-500steps_"
            "checkpoint-100_k1_temp0.1_T128_infer_seed*.json"
        ),
        "wildchat-chosen@500": (
            f"{args.prefix}_dream-chosen-trunc128-sft-lora-r32-alllinear-500steps_"
            "checkpoint-500_k1_temp0.1_T128_infer_seed*.json"
        ),
        "wildchat-entrgi@200": (
            f"{args.prefix}_dream-entrgi-sft-lora-r32-alllinear-500steps_"
            "checkpoint-200_k1_temp0.1_T128_infer_seed*.json"
        ),
        "wildchat-entrgi@500": (
            f"{args.prefix}_dream-entrgi-sft-lora-r32-alllinear-500steps_"
            "checkpoint-500_k1_temp0.1_T128_infer_seed*.json"
        ),
    }

    print("method,mean,std,n")
    for label, pattern in methods.items():
        result = summarize(args.results_dir, pattern)
        if result is None:
            print(f"{label},MISSING,MISSING,0")
            continue
        mean, std, n = result
        print(f"{label},{mean:.4f},{std:.4f},{n}")


if __name__ == "__main__":
    main()

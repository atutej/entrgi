#!/usr/bin/env python3
import argparse
import glob
import json
import os
import statistics


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate fixed-slice eval for base vs wc200.")
    p.add_argument("--results_dir", default="/hdd1/an34232/entrgi_fixed_eval_v1")
    return p.parse_args()


def load_means(results_dir, dataset, model_tag):
    pattern = os.path.join(results_dir, f"{dataset}_{model_tag}_k1_temp0.1_T128_seed*.json")
    vals = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            vals.append(json.load(f)["metrics"]["mean_top1_reward"])
    return vals


def main():
    args = parse_args()
    datasets = ["judgebench", "reward-bench-2", "rm-bench"]
    model_tag = "wildchat-chosen-500steps-ckpt200"

    rows = []
    for ds in datasets:
        base_vals = load_means(args.results_dir, ds, "base")
        wc_vals = load_means(args.results_dir, ds, model_tag)
        if not base_vals or not wc_vals:
            rows.append((ds, None, None, None, None, 0, 0))
            continue
        base_m = sum(base_vals) / len(base_vals)
        wc_m = sum(wc_vals) / len(wc_vals)
        delta = wc_m - base_m
        base_s = statistics.stdev(base_vals) if len(base_vals) > 1 else 0.0
        wc_s = statistics.stdev(wc_vals) if len(wc_vals) > 1 else 0.0
        rows.append((ds, base_m, wc_m, delta, base_s, len(base_vals), len(wc_vals), wc_s))

    print("dataset,base_mean,wc200_mean,delta_wc200_minus_base,base_std,wc200_std,n_base,n_wc200")
    for row in rows:
        ds = row[0]
        if row[1] is None:
            print(f"{ds},MISSING,MISSING,MISSING,MISSING,MISSING,0,0")
            continue
        _, b, w, d, bs, nb, nw, ws = row
        print(f"{ds},{b:.4f},{w:.4f},{d:+.4f},{bs:.4f},{ws:.4f},{nb},{nw}")

    valid = [r for r in rows if r[1] is not None]
    if len(valid) == len(datasets):
        base_overall = sum(r[1] for r in valid) / len(valid)
        wc_overall = sum(r[2] for r in valid) / len(valid)
        d_overall = wc_overall - base_overall
        print(f"benchmark_overall,{base_overall:.4f},{wc_overall:.4f},{d_overall:+.4f},-,-,-,-")


if __name__ == "__main__":
    main()

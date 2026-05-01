import glob, json, os, statistics

RESULTS_DIR = "/hdd1/an34232/entrgi_sft_results_rb2_diag"
methods = {
    "base": "rb2-heldout_base_k1_temp0.1_T128_infer_seed*.json",
    "model": "rb2-heldout_dream-reward-bench2-diag-split-trunc128-sft-lora-r32-alllinear_k1_temp0.1_T128_infer_seed*.json",
}

for label, pattern in methods.items():
    vals = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        with open(path) as f:
            vals.append(json.load(f)["metrics"]["mean_top1_reward"])
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{label}: mean={mean:.4f} std={std:.4f} n={len(vals)}")

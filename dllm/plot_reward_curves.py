"""Pull reward curves from wandb and plot EntRGi vs APS vs diffu-GRPO."""

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ENTITY  = "atutej"
PROJECT = "dream-rl"

RUNS = {
    "EntRGi":     ["dream-entrgi-690949",     "dream-entrgitest-690738"],
    "APS":        ["dream-aps-690729",         "dream-apstest-690834"],
    "diffu-GRPO": ["dream-grpo-691024"],
}

EMA_SPAN = 20

cividis    = plt.cm.cividis
colors     = {"EntRGi": cividis(0.85), "APS": cividis(0.5), "diffu-GRPO": cividis(0.15)}
linestyles = {"EntRGi": "-",           "APS": "-.",          "diffu-GRPO": "--"}
markers    = {"EntRGi": "^",           "APS": "s",           "diffu-GRPO": "o"}


def pull(api, run_name, keys, samples=10000):
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name})
    runs = list(runs)
    if not runs:
        raise ValueError(f"Run not found: {run_name}")
    run  = runs[0]
    hist = run.history(keys=["train/global_step"] + keys, samples=samples, pandas=True)
    hist = (hist.dropna(subset=["train/global_step"])
                .sort_values("train/global_step")
                .reset_index(drop=True))
    return hist


def ema(y, span):
    return pd.Series(y).ewm(span=span, adjust=False).mean().values


def interpolate_to_grid(steps, values, grid):
    return np.interp(grid, steps, values)


api = wandb.Api()

# Pull and average across runs per method
data = {}
for label, run_names in RUNS.items():
    all_smoothed = []
    all_hours    = []
    step_grid    = None

    for run_name in run_names:
        hist     = pull(api, run_name, ["train/reward", "_runtime"])
        steps    = hist["train/global_step"].values
        rewards  = hist["train/reward"].values
        hours    = hist["_runtime"].values / 3600
        smoothed = ema(rewards, EMA_SPAN)

        if step_grid is None:
            step_grid = steps
        all_smoothed.append(interpolate_to_grid(steps, smoothed, step_grid))
        all_hours.append(interpolate_to_grid(steps, hours, step_grid))

    smoothed_arr = np.stack(all_smoothed, axis=0)   # [n_runs, T]
    hours_arr    = np.stack(all_hours,    axis=0)

    mean_smoothed = smoothed_arr.mean(axis=0)
    std_smoothed  = smoothed_arr.std(axis=0)
    mean_hours    = hours_arr.mean(axis=0)

    data[label] = dict(
        steps        = step_grid,
        mean         = mean_smoothed,
        std          = std_smoothed,
        hours        = mean_hours,
        n            = len(run_names),
    )

# Transform: log(-reward) so linear axis looks like log scale, no matplotlib tricks
for label in data:
    data[label]["mean"] = np.log(-data[label]["mean"])

# Tick positions in transformed space, labels show original reward values
YTICK_VALS  = [0.5, 1, 2, 4]
YTICKS      = [np.log(v) for v in YTICK_VALS]
YTICKLABELS = [f"-{v}" for v in YTICK_VALS]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
plt.subplots_adjust(wspace=0.38, left=0.12, right=0.97, bottom=0.15, top=0.95)

# ── left: reward vs step ─────────────────────────────────────────────────────
ax = axes[0]
for label in RUNS:
    d  = data[label]
    se = d["std"] / np.sqrt(d["n"])
    ax.plot(d["steps"], d["mean"],
            color=colors[label], ls=linestyles[label], lw=3.5,
            marker=markers[label], markevery=50, ms=8,
            label=label, zorder=10)
    ax.fill_between(d["steps"],
                    d["mean"] - se, d["mean"] + se,
                    color=colors[label], alpha=0.18, zorder=5)

ax.invert_yaxis()
ax.set_yticks(YTICKS)
ax.set_yticklabels(YTICKLABELS)
ax.grid(True, alpha=0.3, ls="-", lw=0.5)
ax.set_xlabel("Training step", fontsize=18)
ax.set_ylabel("Reward (log scale)", fontsize=18)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.legend(loc="upper left", fontsize=13)

# ── right: reward vs wall-clock time ─────────────────────────────────────────
ax = axes[1]
for label in RUNS:
    d  = data[label]
    se = d["std"] / np.sqrt(d["n"])
    ax.plot(d["hours"], d["mean"],
            color=colors[label], ls=linestyles[label], lw=3.5,
            marker=markers[label], markevery=50, ms=8,
            label=label, zorder=10)
    ax.fill_between(d["hours"],
                    d["mean"] - se, d["mean"] + se,
                    color=colors[label], alpha=0.18, zorder=5)

ax.invert_yaxis()
ax.set_yticks(YTICKS)
ax.set_yticklabels(YTICKLABELS)
ax.grid(True, alpha=0.3, ls="-", lw=0.5)
ax.set_xlabel("Wall-clock time (h)", fontsize=18)
ax.set_ylabel("Reward (log scale)", fontsize=18)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.legend(loc="upper left", fontsize=13)

plt.savefig("reward_curves.pdf", dpi=600, bbox_inches="tight")
plt.savefig("reward_curves.png", dpi=150, bbox_inches="tight")
print("Saved reward_curves.pdf / .png")
plt.show()

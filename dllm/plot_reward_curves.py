"""Pull reward curves from wandb and plot d1 vs EntRGi."""

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ENTITY  = "atutej"
PROJECT = "huggingface"

RUNS = {
    "d1":     "cit1l0dj",
    "EntRGi": "36i7oiud",
}

EMA_SPAN = 20

cividis    = plt.cm.cividis
colors     = {"d1": cividis(0.15), "EntRGi": cividis(0.85)}
linestyles = {"d1": "--",          "EntRGi": "-"}
markers    = {"d1": "o",           "EntRGi": "^"}


def pull(api, run_id, keys, samples=10000):
    run  = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = run.history(keys=["train/global_step"] + keys, samples=samples, pandas=True)
    hist = (hist.dropna(subset=["train/global_step"])
                .sort_values("train/global_step")
                .reset_index(drop=True))
    return hist


def ema(y, span):
    return pd.Series(y).ewm(span=span, adjust=False).mean().values


api = wandb.Api()

# Pull all data upfront so we can do cross-run annotations
data = {}
for label, run_id in RUNS.items():
    hist = pull(api, run_id, ["train/reward", "_runtime"])
    steps    = hist["train/global_step"].values
    rewards  = hist["train/reward"].values
    hours    = hist["_runtime"].values / 3600
    smoothed = ema(rewards, EMA_SPAN)
    # delta reward: subtract each run's own value at step 0
    base     = smoothed[0]
    data[label] = dict(steps=steps, rewards=rewards, hours=hours,
                       smoothed=smoothed, base=base, delta=smoothed - base)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
plt.subplots_adjust(wspace=0.38, left=0.12, right=0.97, bottom=0.15, top=0.95)

# ── left: Δreward vs step ────────────────────────────────────────────────────
ax = axes[0]
for label in RUNS:
    d = data[label]
    ax.plot(d["steps"], d["delta"],
            color=colors[label], ls=linestyles[label], lw=3.5,
            marker=markers[label], markevery=50, ms=10,
            label=label, zorder=10)
    ax.fill_between(d["steps"], d["rewards"] - d["base"], d["delta"],
                    color=colors[label], alpha=0.12, zorder=5)

# Upward arrow at step 500 showing EntRGi improvement over d1
d1_end     = data["d1"]["delta"][-1]
entrgi_end = data["EntRGi"]["delta"][-1]
ax.annotate(
    "",
    xy=(500, entrgi_end), xytext=(500, d1_end),
    arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
    zorder=20,
)
ax.text(
    396, (d1_end + entrgi_end) / 2 - 0.05,
    f"+{entrgi_end - d1_end:.2f}",
    va="center", ha="left", fontsize=13, color="black",
)

ax.grid(True, alpha=0.3, ls="-", lw=0.5)
ax.set_xlabel("Training step", fontsize=18)
ax.set_ylabel("Δ Reward", fontsize=18)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.legend(loc="lower right", fontsize=13)

# ── right: reward vs wall-clock time + speedup annotation ────────────────────
ax = axes[1]
for label in RUNS:
    d = data[label]
    ax.plot(d["hours"], d["delta"],
            color=colors[label], ls=linestyles[label], lw=3.5,
            marker=markers[label], markevery=50, ms=10,
            label=label, zorder=10)
    ax.fill_between(d["hours"], d["rewards"] - d["base"], d["delta"],
                    color=colors[label], alpha=0.12, zorder=5)

# Speedup: d1 endpoint Δreward → find where EntRGi first reaches that level
d1_final_delta    = data["d1"]["delta"][-1]
d1_final_hour     = data["d1"]["hours"][-1]
entrgi_hours      = data["EntRGi"]["hours"]
entrgi_delta      = data["EntRGi"]["delta"]
entrgi_final_delta = entrgi_delta[-1]

# Interpolate to find exact hour where EntRGi crosses d1's final Δreward
cross_idx = np.searchsorted(entrgi_delta, d1_final_delta)
cross_idx = min(cross_idx, len(entrgi_delta) - 1)
if cross_idx > 0 and entrgi_delta[cross_idx] != entrgi_delta[cross_idx - 1]:
    frac = ((d1_final_delta - entrgi_delta[cross_idx - 1]) /
            (entrgi_delta[cross_idx] - entrgi_delta[cross_idx - 1]))
    entrgi_cross_hour = (entrgi_hours[cross_idx - 1]
                         + frac * (entrgi_hours[cross_idx] - entrgi_hours[cross_idx - 1]))
else:
    entrgi_cross_hour = entrgi_hours[cross_idx]

speedup = d1_final_hour / entrgi_cross_hour

# Horizontal double-headed arrow just below EntRGi's final Δreward
y_ann = entrgi_final_delta
ax.annotate(
    "",
    xy=(entrgi_cross_hour, y_ann - 0.05), xytext=(d1_final_hour, y_ann - 0.05),
    arrowprops=dict(arrowstyle="<->", color="black", lw=2.0),
    zorder=20,
)
ax.text(
    (entrgi_cross_hour + d1_final_hour) / 2, y_ann,
    f"{speedup:.1f}× faster",
    va="bottom", ha="center", fontsize=13, color="black",
)
# Dotted reference lines
ax.axhline(d1_final_delta,    color="grey",             ls=":", lw=1.2, zorder=5)
ax.axvline(entrgi_cross_hour, color=colors["EntRGi"],   ls=":", lw=1.2, zorder=5)
ax.axvline(d1_final_hour,     color=colors["d1"],       ls=":", lw=1.2, zorder=5)

ax.grid(True, alpha=0.3, ls="-", lw=0.5)
ax.set_xlabel("Wall-clock time (h)", fontsize=18)
ax.set_ylabel("Δ Reward", fontsize=18)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.legend(loc="lower right", fontsize=13)

plt.savefig("reward_curves.pdf", dpi=600, bbox_inches="tight")
plt.savefig("reward_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved reward_curves.pdf / .png  (speedup = {speedup:.2f}×)")
plt.show()

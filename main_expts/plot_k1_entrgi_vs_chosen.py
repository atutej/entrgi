#!/usr/bin/env python3
"""
Plot k=1 Entrgi-vs-Chosen SFT comparisons from the compiled CSV.

Produces two figure sets:
- one for `non-500`
- one for `500steps`

Each figure has five subplots:
- benchmark overall
- judgebench
- reward-bench-2
- rm-bench
- wildchat-heldout

Base is shown at checkpoint 0 as a shared reference point.
Entrgi and chosen SFT checkpoints are plotted in different colors, with each
curve visually starting from the base point.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METRICS = [
    ("benchmark_overall_mean", "benchmark_overall_sem", "Benchmark Overall"),
    ("judgebench_mean", "judgebench_sem", "JudgeBench"),
    ("reward-bench-2_mean", "reward-bench-2_sem", "RewardBench-2"),
    ("rm-bench_mean", "rm-bench_sem", "RM-Bench"),
    ("wildchat-heldout_mean", "wildchat-heldout_sem", "WildChat Heldout"),
]

METHOD_STYLES = {
    "base": {"color": "#6b7280", "label": "Base", "marker": "o"},
    "entrgi-sft": {"color": "#2563eb", "label": "Entrgi SFT", "marker": "o"},
    "chosen-sft": {"color": "#d97706", "label": "Chosen SFT", "marker": "s"},
}

RUN_TITLES = {
    "non-500": "K=1 Comparison: Full Training",
    "500steps": "K=1 Comparison: 500 Steps",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        default="/home/an34232/Repos/entrgi/main_expts/compiled_results/k1_entrgi_vs_chosen_comparison.csv",
        help="Compiled comparison CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/an34232/Repos/entrgi/main_expts/compiled_results",
        help="Directory for output figures.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    return float(value) if value not in ("", None) else None


def checkpoint_to_x(checkpoint: str):
    if checkpoint == "base":
        return 0
    if checkpoint.isdigit():
        return int(checkpoint)
    return None


def collect_series(rows, run, method, mean_key, sem_key):
    series = []
    for row in rows:
        if row["run"] != run or row["method"] != method:
            continue
        if row["checkpoint"] == "final":
            continue
        x = checkpoint_to_x(row["checkpoint"])
        y = to_float(row[mean_key])
        err = to_float(row[sem_key])
        if x is None or y is None:
            continue
        series.append((x, y, err))
    return sorted(series, key=lambda item: item[0])


def get_base_point(rows, mean_key, sem_key):
    for row in rows:
        if row["run"] == "base" and row["method"] == "base" and row["checkpoint"] == "base":
            y = to_float(row[mean_key])
            err = to_float(row[sem_key])
            if y is not None:
                return (0, y, err)
    return None


def plot_run(rows, run, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.flatten()

    for idx, (mean_key, sem_key, title) in enumerate(METRICS):
        ax = axes[idx]
        base_point = get_base_point(rows, mean_key, sem_key)
        max_x = 0

        if base_point is not None:
            bx, by, berr = base_point
            ax.errorbar(
                [bx],
                [by],
                yerr=[berr] if berr is not None else None,
                color=METHOD_STYLES["base"]["color"],
                marker=METHOD_STYLES["base"]["marker"],
                linewidth=0,
                elinewidth=1.0,
                capsize=3,
                markersize=6.5,
                zorder=4,
            )
            ax.annotate(
                "base",
                (bx, by),
                xytext=(6, 6),
                textcoords="offset points",
                color=METHOD_STYLES["base"]["color"],
                fontsize=9,
            )

        for method in ("entrgi-sft", "chosen-sft"):
            points = collect_series(rows, run, method, mean_key, sem_key)
            if not points:
                continue
            style = METHOD_STYLES[method]

            xs = [item[0] for item in points]
            ys = [item[1] for item in points]
            yerrs = [item[2] if item[2] is not None else 0.0 for item in points]
            max_x = max(max_x, max(xs))

            if base_point is not None:
                bx, by, _ = base_point
                ax.plot(
                    [bx, xs[0]],
                    [by, ys[0]],
                    color=style["color"],
                    linestyle=":",
                    linewidth=1.6,
                    alpha=0.8,
                    zorder=1,
                )

            ax.errorbar(
                color=style["color"],
                x=xs,
                y=ys,
                yerr=yerrs,
                marker=style["marker"],
                linewidth=2.2,
                elinewidth=1.1,
                capsize=3,
                markersize=6.5,
                zorder=3,
            )

        if max_x > 0 and base_point is not None:
            ax.hlines(
                base_point[1],
                xmin=0,
                xmax=max_x,
                color=METHOD_STYLES["base"]["color"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.25,
                zorder=0,
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Performance")
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.05)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], color=METHOD_STYLES["base"]["color"], marker=METHOD_STYLES["base"]["marker"],
               linestyle="--", linewidth=1.4, markersize=7, label=METHOD_STYLES["base"]["label"]),
        Line2D([0], [0], color=METHOD_STYLES["entrgi-sft"]["color"], marker=METHOD_STYLES["entrgi-sft"]["marker"],
               linestyle="-", linewidth=2.2, markersize=7, label=METHOD_STYLES["entrgi-sft"]["label"]),
        Line2D([0], [0], color=METHOD_STYLES["chosen-sft"]["color"], marker=METHOD_STYLES["chosen-sft"]["marker"],
               linestyle="-", linewidth=2.2, markersize=7, label=METHOD_STYLES["chosen-sft"]["label"]),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        fontsize=12,
    )
    legend_ax.text(
        0.5,
        0.18,
        "Curves start from the shared base reference at checkpoint 0.",
        ha="center",
        va="center",
        fontsize=10,
        color="#4b5563",
        transform=legend_ax.transAxes,
    )

    fig.suptitle(RUN_TITLES.get(run, run), fontsize=17, y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.36)

    stem = "k1_entrgi_vs_chosen_non500" if run == "non-500" else "k1_entrgi_vs_chosen_500steps"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    outputs = []
    for run in ("non-500", "500steps"):
        outputs.extend(plot_run(rows, run, output_dir))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

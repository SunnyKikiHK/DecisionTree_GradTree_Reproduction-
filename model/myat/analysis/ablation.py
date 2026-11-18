"""Ablation of GradTree's feature selection (proposal section 4.1.5, paper Table 4 / Table 5).

Trains GradTree with each feature-selection variant in variants.py, with the paper's tuned
settings and the same 10 trials as the main experiment; only the feature selection changes.

Run from the repository root:
    python -m model.myat.analysis.ablation --jobs 6                                  # Echocardiogram, Iris, Wine
    python -m model.myat.analysis.ablation --datasets spambase --variants entmax gumbel --jobs 6
    python -m model.myat.analysis.ablation --plot-only

Finished runs are saved one by one to analysis/results/ablation_runs.csv, and a rerun skips them.
Writes ablation_summary.csv (with the paper's Table 5 numbers) and figures/ablation.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model.myat.gradtree.config import DATASETS, SEEDS, gradtree_settings
from model.myat.gradtree.data import PREPROCESS_DIR, split_dir

from .common import (COLORS, INK_SECONDARY, LABELS, MUTED, RESULTS_DIR, reuse_main_runs, run_gradtree, run_tasks,
                     save_figure, use_style)
from .variants import FEATURE_SELECTION

SMALL_DATASETS = ["echocardiogram", "iris", "wine"]  # minutes per run; Adult and Spambase take ~30 and ~15 min
VARIANT_LABELS = {"entmax": "ST entmax (GradTree)", "softmax": "ST softmax", "gumbel": "ST Gumbel-softmax",
                  "annealing": "Temperature annealing"}

# Paper Table 5 (tuned settings): macro F1 on test, mean and std over 10 trials. Softmax is not in the paper.
PAPER = {
    "adult": {"entmax": (0.743, 0.034), "gumbel": (0.583, 0.029), "annealing": (0.737, 0.045)},
    "echocardiogram": {"entmax": (0.658, 0.113), "gumbel": (0.574, 0.108), "annealing": (0.619, 0.151)},
    "iris": {"entmax": (0.938, 0.057), "gumbel": (0.743, 0.224), "annealing": (0.930, 0.048)},
    "spambase": {"entmax": (0.903, 0.025), "gumbel": (0.541, 0.093), "annealing": (0.875, 0.019)},
    "wine": {"entmax": (0.933, 0.031), "gumbel": (0.565, 0.195), "annealing": (0.881, 0.043)},
}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, variant), group in runs.groupby(["dataset", "feature_selection"], sort=False):
        paper = PAPER.get(name, {}).get(variant, (np.nan, np.nan))
        rows.append({
            "dataset": name, "feature_selection": variant, "trials": len(group),
            "test_f1": group["f1_test"].mean(), "test_f1_std": group["f1_test"].std(ddof=0),
            "paper_test_f1": paper[0], "paper_test_f1_std": paper[1],
            "train_f1": group["f1_train"].mean(), "tree_size": group["total_nodes"].mean(),
            "epochs": group["epochs"].mean(), "fit_seconds": group["fit_seconds"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary["order"] = summary["dataset"].map(DATASETS.index) * 10 + summary["feature_selection"].map(FEATURE_SELECTION.index)
    return summary.sort_values("order").drop(columns="order")


def plot(summary: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_style()
    datasets = [name for name in DATASETS if name in set(summary["dataset"])]
    variants = [v for v in FEATURE_SELECTION if v in set(summary["feature_selection"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets) + 1.6, 0.42 * len(variants) + 1.4),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes[0], datasets):
        part = summary[summary["dataset"] == name].set_index("feature_selection")
        for y, variant in enumerate(variants):
            if variant not in part.index:
                continue
            row = part.loc[variant]
            ax.plot([row["test_f1"] - row["test_f1_std"], row["test_f1"] + row["test_f1_std"]], [y, y],
                    color=COLORS["gradtree"], linewidth=2, alpha=0.35)
            ax.plot(row["test_f1"], y, "o", color=COLORS["gradtree"], markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            if not np.isnan(row["paper_test_f1"]):
                ax.plot(row["paper_test_f1"], y, "D", markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                        markersize=6, markeredgewidth=1.2, zorder=4)
        ax.set_title(LABELS[name])
        ax.set_xlabel("Test macro F1")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(variants)), [VARIANT_LABELS[v] for v in variants])
    axes[0][0].invert_yaxis()
    handles = [Line2D([], [], marker="o", color=COLORS["gradtree"], markeredgecolor="white", label="Ours: mean ± std over 10 trials"),
               Line2D([], [], marker="D", linestyle="none", markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                      label="Paper (Table 5)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=2, labelcolor=INK_SECONDARY)
    fig.suptitle("Feature selection ablation: GradTree with each way of choosing a node's feature",
                 x=0.0, y=1.15, ha="left", fontsize=11, fontweight="semibold")
    fig.text(0.0, -0.06, "Same tuned settings and splits for every variant; only the feature selection changes.",
             color=MUTED, fontsize=8, ha="left")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=SMALL_DATASETS, choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=FEATURE_SELECTION, choices=FEATURE_SELECTION)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--jobs", type=int, default=1, help="trials to train in parallel")
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    parser.add_argument("--plot-only", action="store_true", help="redo the summary and figure from saved runs")
    args = parser.parse_args()

    runs_path = RESULTS_DIR / "ablation_runs.csv"
    if not args.plot_only:
        for name in args.datasets:
            for seed in args.seeds:
                split_dir(name, seed, args.preprocess_dir)
        tasks = [dict(dataset=name, seed=seed, settings=gradtree_settings(name, "tuned"),
                      preprocess_dir=args.preprocess_dir, feature_selection=variant)
                 for name in args.datasets for variant in args.variants for seed in args.seeds]
        reuse_main_runs(tasks, runs_path, key=["dataset", "feature_selection", "seed"])
        run_tasks(tasks, run_gradtree, args.jobs, runs_path, key=["dataset", "feature_selection", "seed"])

    summary = summarize(pd.read_csv(runs_path))
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"Saved {RESULTS_DIR / 'ablation_summary.csv'} and {save_figure(plot(summary), 'ablation')}")


if __name__ == "__main__":
    main()

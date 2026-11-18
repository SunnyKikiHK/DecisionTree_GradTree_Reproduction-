"""Effect of the maximum depth on GradTree and CART (proposal section 4.1.5).

Both models keep their tuned settings (GradTree: paper Table 12, CART: Table 15) and only the
maximum depth changes. CART is fitted on train + valid as in the baseline experiment; GradTree
trains on train and stops early on valid. Every depth uses the same 10 trials.

Run from the repository root:
    python -m model.myat.analysis.depth --jobs 6                         # CART on all datasets, GradTree on the small ones
    python -m model.myat.analysis.depth --gradtree-datasets spambase --gradtree-depths 4 8 --jobs 6
    python -m model.myat.analysis.depth --plot-only

Finished runs are saved one by one to analysis/results/depth_runs_<model>.csv, and a rerun skips them.
Writes depth_summary.csv and figures/depth.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from model.myat.baselines.config import CART_TUNED
from model.myat.gradtree.config import DATASETS, MODEL_SEED, SEEDS, TUNED, gradtree_settings
from model.myat.gradtree.data import PREPROCESS_DIR, load_split, split_dir

from .common import (COLORS, INK_SECONDARY, LABELS, MODEL_LABELS, MUTED, RESULTS_DIR, plain_log_ticks, reuse_main_runs, run_gradtree,
                     run_tasks, save_figure, use_style)

CART_DEPTHS = list(range(1, 11))
GRADTREE_DEPTHS = [2, 4, 6, 8, 10]
SMALL_DATASETS = ["echocardiogram", "iris", "wine"]  # minutes per run; Adult and Spambase take ~30 and ~15 min


def run_cart(dataset: str, seed: int, params: dict, preprocess_dir: Path, depth: int) -> dict:
    split = load_split(dataset, seed, preprocess_dir)
    X_fit = np.vstack([split["X_train"], split["X_valid"]])
    y_fit = np.concatenate([split["y_train"], split["y_valid"]])
    tree = DecisionTreeClassifier(random_state=MODEL_SEED, **{**params, "max_depth": depth})
    start = time.perf_counter()
    tree.fit(X_fit, y_fit)
    row = {"model": "cart", "dataset": dataset, "seed": seed, "depth": depth}
    for part, X, y in [("train", X_fit, y_fit), ("test", split["X_test"], split["y_test"])]:
        pred = tree.predict(X)
        row[f"f1_{part}"] = f1_score(y, pred, average="macro")
        row[f"accuracy_{part}"] = accuracy_score(y, pred)
    row.update(total_nodes=int(tree.tree_.node_count), fit_seconds=time.perf_counter() - start)
    return row


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    summary = runs.groupby(["model", "dataset", "depth"], sort=False).agg(
        trials=("seed", "size"),
        test_f1=("f1_test", "mean"), test_f1_std=("f1_test", lambda v: v.std(ddof=0)),
        train_f1=("f1_train", "mean"),
        tree_size=("total_nodes", "mean"), tree_size_std=("total_nodes", lambda v: v.std(ddof=0)),
        fit_seconds=("fit_seconds", "mean"),
    ).reset_index()
    summary["train_test_gap"] = summary["train_f1"] - summary["test_f1"]
    summary["order"] = summary["dataset"].map(DATASETS.index)
    return summary.sort_values(["order", "model", "depth"]).drop(columns="order")


def plot(summary: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_style()
    datasets = [name for name in DATASETS if name in set(summary["dataset"])]
    fig, axes = plt.subplots(2, len(datasets), figsize=(2.5 * len(datasets), 5.2), sharex=True, squeeze=False)
    for col, name in enumerate(datasets):
        for model in ["cart", "gradtree"]:
            part = summary[(summary["dataset"] == name) & (summary["model"] == model)].sort_values("depth")
            if part.empty:
                continue
            color = COLORS[model]
            for row, (mean, std) in enumerate([("test_f1", "test_f1_std"), ("tree_size", "tree_size_std")]):
                ax = axes[row][col]
                ax.fill_between(part["depth"], part[mean] - part[std], part[mean] + part[std], color=color,
                                alpha=0.1, linewidth=0)
                ax.plot(part["depth"], part[mean], color=color, marker="o", markersize=4.5,
                        markeredgecolor="white", markeredgewidth=1)
        if ((summary["dataset"] == name) & (summary["model"] == "gradtree")).any():
            for row in range(2):  # GradTree's tuned depth, for reference
                axes[row][col].axvline(TUNED[name]["depth"], color=MUTED, linewidth=0.8, zorder=0)
        axes[0][col].set_title(LABELS[name])
        axes[1][col].set_yscale("log")
        plain_log_ticks(axes[1][col].yaxis)
        axes[1][col].set_xlabel("Maximum depth")
    axes[0][0].set_ylabel("Test macro F1")
    axes[1][0].set_ylabel("Tree size (nodes, log scale)")
    axes[1][0].set_xticks([1, 2, 4, 6, 8, 10])
    handles = [Line2D([], [], color=COLORS[m], marker="o", markeredgecolor="white", label=MODEL_LABELS[m])
               for m in ["gradtree", "cart"]]
    handles.append(Line2D([], [], color=MUTED, linewidth=0.8, label="GradTree's tuned depth"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.04), ncol=3, labelcolor=INK_SECONDARY)
    fig.suptitle("Effect of the maximum depth (mean ± std over 10 trials, other settings tuned)",
                 x=0.0, y=1.09, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="datasets for CART")
    parser.add_argument("--gradtree-datasets", nargs="*", default=SMALL_DATASETS, choices=DATASETS)
    parser.add_argument("--cart-depths", nargs="+", type=int, default=CART_DEPTHS)
    parser.add_argument("--gradtree-depths", nargs="+", type=int, default=GRADTREE_DEPTHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--jobs", type=int, default=1, help="GradTree trials to train in parallel")
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    parser.add_argument("--plot-only", action="store_true", help="redo the summary and figure from saved runs")
    args = parser.parse_args()

    cart_path, gradtree_path = RESULTS_DIR / "depth_runs_cart.csv", RESULTS_DIR / "depth_runs_gradtree.csv"
    if not args.plot_only:
        for name in set(args.datasets) | set(args.gradtree_datasets):
            for seed in args.seeds:
                split_dir(name, seed, args.preprocess_dir)
        cart_tasks = [dict(dataset=name, seed=seed, params=CART_TUNED[name], preprocess_dir=args.preprocess_dir, depth=depth)
                      for name in args.datasets for depth in args.cart_depths for seed in args.seeds]
        run_tasks(cart_tasks, run_cart, 1, cart_path, key=["dataset", "depth", "seed"])
        gradtree_tasks = [dict(dataset=name, seed=seed, settings={**gradtree_settings(name, "tuned"), "depth": depth},
                               preprocess_dir=args.preprocess_dir, depth=depth)
                          for name in args.gradtree_datasets for depth in args.gradtree_depths for seed in args.seeds]
        if gradtree_tasks:
            reuse_main_runs(gradtree_tasks, gradtree_path, key=["dataset", "depth", "seed"])
            run_tasks(gradtree_tasks, run_gradtree, args.jobs, gradtree_path, key=["dataset", "depth", "seed"])

    runs = pd.concat([pd.read_csv(path) for path in [cart_path, gradtree_path] if path.exists()])
    summary = summarize(runs)
    summary.to_csv(RESULTS_DIR / "depth_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"Saved {RESULTS_DIR / 'depth_summary.csv'} and {save_figure(plot(summary), 'depth')}")


if __name__ == "__main__":
    main()

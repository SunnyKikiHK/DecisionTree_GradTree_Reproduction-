"""GradTree vs CART on the same 10 trials, next to the paper (proposal sections 4.1.4 and 4.1.5).

Reads the results that model/myat/gradtree/train.py and model/myat/baselines/train.py saved, so run those first.
For each dataset: test macro F1 (mean ± std), how many of the 10 trials each model wins, a
Wilcoxon signed-rank test on the paired test F1s, tree size, train-test gap (overfitting) and
training time. The random forest (and CART with our own search, when present) are reference columns.

Run from the repository root:
    python -m model.myat.analysis.compare                     # tuned settings
    python -m model.myat.analysis.compare --settings default
    python -m model.myat.analysis.compare --settings search   # our own searches, equal budgets

Writes analysis/results/comparison_<settings>.csv and .md, and figures/comparison_<settings>.png.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from model.myat.baselines.config import PAPER as CART_PAPER
from model.myat.gradtree.config import DATASETS
from model.myat.gradtree.config import PAPER as GRADTREE_PAPER

# GradTree results folder (gradtree/<settings>) and the CART results it is compared with
CART_RESULTS = {"tuned": "cart_tuned", "default": "cart_default", "search": "cart_search_equal"}
TITLES = {"tuned": "paper's tuned settings", "default": "default settings",
          "search": "our own search (50 settings each)"}

from .common import (BASELINE_RESULTS, COLORS, GRADTREE_RESULTS, INK_SECONDARY, LABELS, MODEL_LABELS, MUTED,
                     RESULTS_DIR, plain_log_ticks, save_figure, use_style)


def load_runs(path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} is missing; run the training script that writes it first")
        return None
    return pd.read_csv(path)


def compare(settings: str) -> pd.DataFrame:
    gradtree = load_runs(GRADTREE_RESULTS / settings / "runs.csv")
    cart = load_runs(BASELINE_RESULTS / CART_RESULTS[settings] / "runs.csv")
    forest = load_runs(BASELINE_RESULTS / "random_forest_default" / "runs.csv", required=False)
    searched = load_runs(BASELINE_RESULTS / "cart_search" / "runs.csv", required=False)

    rows = []
    for name in DATASETS:
        g = gradtree[gradtree["dataset"] == name].set_index("seed")
        c = cart[cart["dataset"] == name].set_index("seed")
        seeds = g.index.intersection(c.index)
        if len(seeds) == 0:
            continue
        g, c = g.loc[seeds], c.loc[seeds]
        diff = g["f1_test"] - c["f1_test"]
        p_value = wilcoxon(g["f1_test"], c["f1_test"]).pvalue if (diff != 0).any() else 1.0
        paper_g = GRADTREE_PAPER.get(settings, {}).get(name, {})
        paper_c = CART_PAPER["cart"].get(settings, {}).get(name, {})
        row = {
            "dataset": name,
            "trials": len(seeds),
            "gradtree_test_f1": g["f1_test"].mean(), "gradtree_test_f1_std": g["f1_test"].std(ddof=0),
            "cart_test_f1": c["f1_test"].mean(), "cart_test_f1_std": c["f1_test"].std(ddof=0),
            "paper_gradtree_test_f1": paper_g.get("test_f1", (np.nan,))[0],
            "paper_cart_test_f1": paper_c.get("test_f1", (np.nan,))[0],
            "difference": diff.mean(),
            "gradtree_wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "cart_wins": int((diff < 0).sum()),
            "wilcoxon_p": p_value,
            "gradtree_tree_size": g["total_nodes"].mean(), "cart_tree_size": c["total_nodes"].mean(),
            "paper_gradtree_tree_size": paper_g.get("tree_size", (np.nan,))[0],
            "paper_cart_tree_size": paper_c.get("tree_size", (np.nan,))[0],
            "gradtree_train_f1": g["f1_train"].mean(), "cart_train_f1": c["f1_train"].mean(),
            "gradtree_gap": (g["f1_train"] - g["f1_test"]).mean(), "cart_gap": (c["f1_train"] - c["f1_test"]).mean(),
            "gradtree_seconds": g["fit_seconds"].mean(), "cart_seconds": c["fit_seconds"].mean(),
        }
        for label, runs in [("random_forest", forest), ("cart_search", searched)]:
            part = runs[runs["dataset"] == name] if runs is not None else pd.DataFrame()
            row[f"{label}_test_f1"] = part["f1_test"].mean() if len(part) else np.nan
            row[f"{label}_test_f1_std"] = part["f1_test"].std(ddof=0) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, settings: str) -> str:
    def f1(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    def num(value, digits=3):
        return "–" if pd.isna(value) else f"{value:.{digits}f}"

    lines = [
        f"### GradTree vs CART: {TITLES[settings]}, 10 trials",
        "",
        "| Dataset | GradTree | CART | Paper GradTree | Paper CART | Wins G / C | Wilcoxon p | Random forest |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        g = f1(r.gradtree_test_f1, r.gradtree_test_f1_std)
        c = f1(r.cart_test_f1, r.cart_test_f1_std)
        if r.gradtree_test_f1 > r.cart_test_f1:
            g = f"**{g}**"
        elif r.cart_test_f1 > r.gradtree_test_f1:
            c = f"**{c}**"
        lines.append(f"| {LABELS[r.dataset]} | {g} | {c} | {num(r.paper_gradtree_test_f1)} | {num(r.paper_cart_test_f1)} "
                     f"| {r.gradtree_wins} / {r.cart_wins} | {r.wilcoxon_p:.3f} | {num(r.random_forest_test_f1)} |")
    lines.append(f"| **Mean** | {table.gradtree_test_f1.mean():.3f} | {table.cart_test_f1.mean():.3f} "
                 f"| {num(table.paper_gradtree_test_f1.mean())} | {num(table.paper_cart_test_f1.mean())} | | "
                 f"| {num(table.random_forest_test_f1.mean())} |")
    lines += [
        "",
        "Test macro F1, mean ± std. Wins: trials (of 10) where that model's test F1 is higher. "
        "Wilcoxon signed-rank test on the paired test F1s; with 10 trials the smallest possible p is 0.002.",
        "",
        "| Dataset | Tree size G / C | Paper tree size G / C | Train F1 G / C | Train-test gap G / C | Seconds G / C |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {LABELS[r.dataset]} | {r.gradtree_tree_size:.1f} / {r.cart_tree_size:.1f} "
            f"| {num(r.paper_gradtree_tree_size, 1)} / {num(r.paper_cart_tree_size, 1)} "
            f"| {r.gradtree_train_f1:.3f} / {r.cart_train_f1:.3f} | {r.gradtree_gap:.3f} / {r.cart_gap:.3f} "
            f"| {r.gradtree_seconds:.1f} / {r.cart_seconds:.3f} |")
    lines += [
        "",
        "G = GradTree, C = CART. Tree size counts internal and leaf nodes after pruning. CART's train F1 is "
        "measured on train + valid, the data it is fitted on. Times are on a CPU and include GradTree's restarts.",
    ]
    return "\n".join(lines) + "\n"


def plot(table: pd.DataFrame, settings: str):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_style()
    names = list(table["dataset"])
    y = np.arange(len(names))
    offsets = {"gradtree": -0.2, "cart": 0.0, "random_forest": 0.2}
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.55 * len(names) + 1.6), sharey=True)

    def dots(ax, model, means, stds=None, paper=None):
        ys = y + offsets[model]
        if stds is not None:
            ax.hlines(ys, means - stds, means + stds, color=COLORS[model], linewidth=2, alpha=0.35)
        ax.plot(means, ys, "o", color=COLORS[model], markersize=7, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        if paper is not None:
            ax.plot(paper, ys, "D", markerfacecolor="none", markeredgecolor=COLORS[model], markersize=6,
                    markeredgewidth=1.2, zorder=4)

    ax = axes[0]
    dots(ax, "gradtree", table.gradtree_test_f1, table.gradtree_test_f1_std, table.paper_gradtree_test_f1)
    dots(ax, "cart", table.cart_test_f1, table.cart_test_f1_std, table.paper_cart_test_f1)
    if table.random_forest_test_f1.notna().any():
        dots(ax, "random_forest", table.random_forest_test_f1, table.random_forest_test_f1_std)
    ax.set_title("Test macro F1")
    ax.set_xlabel("mean ± std over 10 trials")

    ax = axes[1]
    dots(ax, "gradtree", table.gradtree_tree_size, paper=table.paper_gradtree_tree_size)
    dots(ax, "cart", table.cart_tree_size, paper=table.paper_cart_tree_size)
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis)
    ax.set_title("Tree size")
    ax.set_xlabel("nodes after pruning (log scale)")

    ax = axes[2]
    dots(ax, "gradtree", table.gradtree_gap)
    dots(ax, "cart", table.cart_gap)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
    ax.set_title("Overfitting")
    ax.set_xlabel("train F1 − test F1")

    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABELS[n] for n in names])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], markeredgecolor="white", markersize=7,
                      label=MODEL_LABELS[m]) for m in ["gradtree", "cart", "random_forest"]]
    handles.append(Line2D([], [], marker="D", linestyle="none", markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                          label="Paper (same colour as the model)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=4, labelcolor=INK_SECONDARY)
    fig.suptitle(f"GradTree vs CART, {TITLES[settings]}", x=0.0, y=1.12, ha="left", fontsize=11, fontweight="semibold")
    fig.text(0.0, -0.04, "Random forest (100 trees, default settings) is a reference for accuracy only; "
             "its tree size is in the thousands, so it is left out of the size and overfitting panels.",
             color=MUTED, fontsize=8, ha="left")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default="tuned", choices=list(CART_RESULTS))
    args = parser.parse_args()

    table = compare(args.settings)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"comparison_{args.settings}.csv", index=False)
    markdown = to_markdown(table, args.settings)
    (RESULTS_DIR / f"comparison_{args.settings}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved {RESULTS_DIR / f'comparison_{args.settings}.csv'}, .md and "
          f"{save_figure(plot(table, args.settings), f'comparison_{args.settings}')}")


if __name__ == "__main__":
    main()

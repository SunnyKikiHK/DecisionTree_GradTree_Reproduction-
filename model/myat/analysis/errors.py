"""Error analysis of GradTree and CART on the same test rows (proposal section 6.4).

Both models are evaluated on identical test splits, so their predictions can be compared row by
row. Over all 10 trials of a dataset this script measures:
- error overlap: rows only GradTree gets wrong, only CART, or both; and how many of each model's
  errors the random forest also gets wrong (rows that even an ensemble misses are likely ambiguous
  or mislabeled rather than a weakness of one tree);
- recall per class, which shows where macro F1 is lost (usually the minority class);
- confidence: the predicted probability of the chosen class on correct and on wrong predictions.

It reads the test predictions saved by gradtree/train.py and baselines/train.py
(<results>/predictions/, gitignored: rerun training if they are missing).

Run from the repository root:
    python -m model.myat.analysis.errors                  # tuned settings
    python -m model.myat.analysis.errors --settings default

Writes analysis/results/errors_<settings>.{csv,md}, errors_by_class_<settings>.csv and
figures/errors_<settings>.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.myat.gradtree.config import DATASETS, SEEDS
from model.myat.gradtree.data import PREPROCESS_DIR, split_dir

from .common import (AXIS, BASELINE_RESULTS, COLORS, GRADTREE_RESULTS, INK_SECONDARY, LABELS, MODEL_LABELS, MUTED,
                     RESULTS_DIR, SURFACE, save_figure, use_style)
from .compare import CART_RESULTS

SHARED = "#b4b2a9"  # rows both trees get wrong: a neutral grey, not either model's colour


def prediction_folders(settings: str) -> dict[str, Path]:
    return {
        "gradtree": GRADTREE_RESULTS / settings / "predictions",
        "cart": BASELINE_RESULTS / CART_RESULTS[settings] / "predictions",
        "random_forest": BASELINE_RESULTS / "random_forest_default" / "predictions",
    }


def load_predictions(folders: dict[str, Path], name: str, seeds: list[int]) -> pd.DataFrame:
    """One row per test row and trial: the true label and every model's prediction and confidence."""
    frames = []
    for seed in seeds:
        merged = None
        for model, folder in folders.items():
            path = folder / f"{name}_seed_{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; rerun the training that writes it")
            frame = pd.read_csv(path)
            proba = frame.filter(like="proba_").to_numpy()
            part = pd.DataFrame({f"{model}_pred": frame["y_pred"], f"{model}_confidence": proba.max(axis=1)})
            if merged is None:
                merged = pd.concat([frame[["y_true"]], part], axis=1)
            else:
                if not np.array_equal(merged["y_true"], frame["y_true"]):
                    raise ValueError(f"{name} seed {seed}: {model} was tested on different rows")
                merged = pd.concat([merged, part], axis=1)
        merged.insert(0, "seed", seed)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


ADULT_CLASSES = ["<=50K", ">50K"]  # adult_preprocessing.ipynb: 0 = <=50K, 1 = >50K


def class_names(name: str, n_classes: int, preprocess_dir: Path) -> list[str]:
    if name == "adult":
        return ADULT_CLASSES
    try:
        preprocessor = joblib.load(split_dir(name, SEEDS[0], preprocess_dir) / f"{name}_preprocessor.joblib")
        encoder, target = preprocessor.get("label_encoder"), preprocessor.get("target", "class")
        if encoder is not None and len(encoder.classes_) == n_classes:
            if isinstance(encoder.classes_[0], str):
                return [str(c) for c in encoder.classes_]
            return [f"{target} = {c}" if target.lower() != "class" else f"class {c}" for c in encoder.classes_]
    except (FileNotFoundError, KeyError):
        pass
    return [f"class {k}" for k in range(n_classes)]


def analyze(rows: pd.DataFrame, name: str, names: list[str]) -> tuple[dict, list[dict]]:
    wrong = {m: rows[f"{m}_pred"] != rows["y_true"] for m in ["gradtree", "cart", "random_forest"]}
    g, c, f = wrong["gradtree"], wrong["cart"], wrong["random_forest"]
    summary = {
        "dataset": name, "test_rows": len(rows),
        "gradtree_error": g.mean(), "cart_error": c.mean(), "random_forest_error": f.mean(),
        "both_wrong": (g & c).mean(), "only_gradtree_wrong": (g & ~c).mean(), "only_cart_wrong": (~g & c).mean(),
        "agreement": (rows["gradtree_pred"] == rows["cart_pred"]).mean(),
        "gradtree_errors_forest_also_wrong": (g & f).sum() / max(g.sum(), 1),
        "cart_errors_forest_also_wrong": (c & f).sum() / max(c.sum(), 1),
    }
    for model, is_wrong in [("gradtree", g), ("cart", c)]:
        confidence = rows[f"{model}_confidence"]
        summary[f"{model}_confidence_correct"] = confidence[~is_wrong].mean()
        summary[f"{model}_confidence_wrong"] = confidence[is_wrong].mean()
        summary[f"{model}_wrong_at_confidence_0.9"] = (confidence[is_wrong] >= 0.9).mean()

    by_class = []
    for k, label in enumerate(names):
        in_class = rows["y_true"] == k
        entry = {"dataset": name, "class": label, "share_of_test": in_class.mean()}
        for model in ["gradtree", "cart", "random_forest"]:
            entry[f"{model}_recall"] = (rows.loc[in_class, f"{model}_pred"] == k).mean()
            predicted = rows[f"{model}_pred"] == k
            entry[f"{model}_precision"] = (rows.loc[predicted, "y_true"] == k).mean() if predicted.any() else np.nan
        by_class.append(entry)
    return summary, by_class


def to_markdown(summary: pd.DataFrame, by_class: pd.DataFrame, settings: str) -> str:
    pct = lambda v: f"{100 * v:.1f}%"
    lines = [
        f"### Where GradTree and CART go wrong ({settings} settings, test rows of all 10 trials)",
        "",
        "| Dataset | Test rows | Error G / C / RF | Both wrong | Only G wrong | Only C wrong | G's errors RF also misses | C's errors RF also misses |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {LABELS[r.dataset]} | {r.test_rows} | {pct(r.gradtree_error)} / {pct(r.cart_error)} / "
                     f"{pct(r.random_forest_error)} | {pct(r.both_wrong)} | {pct(r.only_gradtree_wrong)} | "
                     f"{pct(r.only_cart_wrong)} | {pct(r.gradtree_errors_forest_also_wrong)} | "
                     f"{pct(r.cart_errors_forest_also_wrong)} |")
    lines += [
        "",
        "Error = share of test rows predicted wrongly (G = GradTree, C = CART, RF = random forest).",
        "",
        "| Dataset | Class | Share of test | Recall G / C / RF | Precision G / C |",
        "|---|---|---|---|---|",
    ]
    for _, r in by_class.iterrows():
        lines.append(f"| {LABELS[r.dataset]} | {r['class']} | {pct(r.share_of_test)} | {r.gradtree_recall:.3f} / "
                     f"{r.cart_recall:.3f} / {r.random_forest_recall:.3f} | {r.gradtree_precision:.3f} / "
                     f"{r.cart_precision:.3f} |")
    lines += [
        "",
        "| Dataset | Confidence when right G / C | Confidence when wrong G / C | Wrong with confidence ≥ 0.9, G / C |",
        "|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {LABELS[r.dataset]} | {r.gradtree_confidence_correct:.3f} / {r.cart_confidence_correct:.3f} "
                     f"| {r.gradtree_confidence_wrong:.3f} / {r.cart_confidence_wrong:.3f} "
                     f"| {pct(r['gradtree_wrong_at_confidence_0.9'])} / {pct(r['cart_wrong_at_confidence_0.9'])} |")
    lines += ["", "Confidence = the predicted probability of the chosen class (GradTree: from its leaf logits; "
              "CART: the class share in its leaf)."]
    return "\n".join(lines) + "\n"


def plot(summary: pd.DataFrame, by_class: pd.DataFrame, settings: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    use_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # Left: error overlap as stacked bars (share of test rows), 2px surface gap between segments
    names = list(summary["dataset"])
    y = np.arange(len(names))
    segments = [("both_wrong", SHARED, "Both wrong"), ("only_gradtree_wrong", COLORS["gradtree"], "Only GradTree wrong"),
                ("only_cart_wrong", COLORS["cart"], "Only CART wrong")]
    start = np.zeros(len(names))
    for column, color, label in segments:
        widths = 100 * summary[column].to_numpy()
        left.barh(y, widths, left=start, height=0.5, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
        start += widths
    for yi, total in zip(y, start):
        left.text(total + 0.6, yi, f"{total:.1f}%", va="center", fontsize=8, color=INK_SECONDARY)
    left.set_yticks(y, [LABELS[n] for n in names])
    left.invert_yaxis()
    left.set_xlim(0, start.max() * 1.18)
    left.set_xlabel("share of test rows (%)")
    left.set_title("Which rows each tree gets wrong")
    left.grid(axis="y", visible=False)
    left.legend(loc="lower right", fontsize=8, labelcolor=INK_SECONDARY)

    # Right: recall per class
    labels = [f"{LABELS[r.dataset]} · {r['class']}" for _, r in by_class.iterrows()]
    yc = np.arange(len(labels))
    for model, offset in [("gradtree", -0.15), ("cart", 0.15)]:
        right.plot(by_class[f"{model}_recall"], yc + offset, "o", color=COLORS[model], markersize=6.5,
                   markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    right.set_yticks(yc, labels)
    right.invert_yaxis()
    right.set_xlim(-0.02, 1.02)
    right.set_xlabel("recall (share of the class predicted correctly)")
    right.set_title("Recall per class")
    right.grid(axis="y", visible=False)
    for boundary in np.cumsum(by_class.groupby("dataset", sort=False).size().to_numpy())[:-1]:
        right.axhline(boundary - 0.5, color=AXIS, linewidth=0.6)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], markeredgecolor="white", markersize=7,
                      label=MODEL_LABELS[m]) for m in ["gradtree", "cart"]]
    right.legend(handles=handles, loc="lower left", fontsize=8, labelcolor=INK_SECONDARY)

    fig.suptitle(f"Errors of GradTree and CART ({settings} settings, test rows of all 10 trials)",
                 x=0.0, y=1.02, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default="tuned", choices=list(CART_RESULTS))
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR, help="for the class names")
    args = parser.parse_args()

    folders = prediction_folders(args.settings)
    summaries, classes = [], []
    for name in args.datasets:
        rows = load_predictions(folders, name, args.seeds)
        n_classes = int(rows["y_true"].max()) + 1
        summary, by_class = analyze(rows, name, class_names(name, n_classes, args.preprocess_dir))
        summaries.append(summary)
        classes += by_class
    summary, by_class = pd.DataFrame(summaries), pd.DataFrame(classes)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / f"errors_{args.settings}.csv", index=False)
    by_class.to_csv(RESULTS_DIR / f"errors_by_class_{args.settings}.csv", index=False)
    markdown = to_markdown(summary, by_class, args.settings)
    (RESULTS_DIR / f"errors_{args.settings}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved {RESULTS_DIR / f'errors_{args.settings}.md'} and "
          f"{save_figure(plot(summary, by_class, args.settings), f'errors_{args.settings}')}")


if __name__ == "__main__":
    main()

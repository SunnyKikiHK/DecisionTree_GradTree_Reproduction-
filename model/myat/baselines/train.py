"""Train the baselines on the same 10 trials as GradTree and compare CART with the paper.

As in the official experiments, the baselines need no validation split, so they are fitted on
train + valid. "Train" metrics below are measured on that same data.

Run from the repository root:
    python -m model.myat.baselines.train                                   # CART, paper's tuned settings (Table 15)
    python -m model.myat.baselines.train --settings default                # CART, default settings (Table 7)
    python -m model.myat.baselines.train --settings search                 # CART, our own search (run search.py first)
    python -m model.myat.baselines.train --settings search_equal           # CART, search with GradTree's budget
    python -m model.myat.baselines.train --model random_forest --settings default

Writes to model/myat/baselines/results/<model>_<settings>/:
    runs.csv           one row per (dataset, seed): macro F1 / accuracy on train and test, tree size, ...
    summary.csv        mean and std over seeds next to the paper's numbers
    settings.json      the settings used for each dataset
    predictions/       test-set predictions of every run, for error analysis
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from model.myat.gradtree.data import PREPROCESS_DIR, load_split, split_dir

from .config import DATASETS, MODEL_SEED, MODELS, PAPER, SEEDS, SETTINGS, baseline_settings
from .search import EQUAL_SEARCH_DIR, SEARCH_DIR, load_search_results

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def make_estimator(model: str, params: dict):
    if model == "cart":
        return DecisionTreeClassifier(random_state=MODEL_SEED, **params)
    return RandomForestClassifier(random_state=MODEL_SEED, n_jobs=-1, **params)


def tree_size(estimator) -> dict:
    """Internal and leaf nodes (summed over the trees of a forest) and the deepest tree's depth."""
    trees = getattr(estimator, "estimators_", [estimator])
    leaves = sum(int(tree.get_n_leaves()) for tree in trees)
    total = sum(int(tree.tree_.node_count) for tree in trees)
    return dict(internal_nodes=total - leaves, leaf_nodes=leaves, total_nodes=total,
                depth=max(int(tree.get_depth()) for tree in trees))


def run_trial(model: str, name: str, seed: int, params: dict, preprocess_dir: Path, out_dir: Path) -> dict:
    split = load_split(name, seed, preprocess_dir)
    X_fit = np.vstack([split["X_train"], split["X_valid"]])
    y_fit = np.concatenate([split["y_train"], split["y_valid"]])

    estimator = make_estimator(model, params)
    start = time.perf_counter()
    estimator.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - start

    row = {"dataset": name, "seed": seed}
    for part, X, y in [("train", X_fit, y_fit), ("test", split["X_test"], split["y_test"])]:
        pred = estimator.predict(X)
        row[f"f1_{part}"] = f1_score(y, pred, average="macro")
        row[f"accuracy_{part}"] = accuracy_score(y, pred)
    row.update(tree_size(estimator), fit_seconds=fit_seconds)

    proba = estimator.predict_proba(split["X_test"])
    predictions = pd.DataFrame({"y_true": split["y_test"], "y_pred": estimator.classes_[proba.argmax(axis=1)]})
    for k, label in enumerate(estimator.classes_):
        predictions[f"proba_{label}"] = proba[:, k]
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions" / f"{name}_seed_{seed}.csv", index=False)
    return row


def summarize(runs: pd.DataFrame, model: str, settings_name: str) -> pd.DataFrame:
    paper_all = PAPER.get(model, {}).get(settings_name, {})
    rows = []
    for name, group in runs.groupby("dataset", sort=False):
        paper = paper_all.get(name, {})
        rows.append({
            "dataset": name,
            "trials": len(group),
            "test_f1": group["f1_test"].mean(),
            "test_f1_std": group["f1_test"].std(ddof=0),
            "paper_test_f1": paper.get("test_f1", (np.nan, np.nan))[0],
            "paper_test_f1_std": paper.get("test_f1", (np.nan, np.nan))[1],
            "train_f1": group["f1_train"].mean(),
            "paper_train_f1": paper.get("train_f1", (np.nan,))[0],
            "train_test_gap": (group["f1_train"] - group["f1_test"]).mean(),
            "tree_size": group["total_nodes"].mean(),
            "tree_size_std": group["total_nodes"].std(ddof=0),
            "paper_tree_size": paper.get("tree_size", (np.nan,))[0],
            "depth": group["depth"].mean(),
            "test_accuracy": group["accuracy_test"].mean(),
            "fit_seconds": group["fit_seconds"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="cart", choices=MODELS)
    parser.add_argument("--settings", default="tuned", choices=["tuned", "default", "search", "search_equal"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    parser.add_argument("--search-dir", type=Path, help="where search.py saved best_params.json (default: by settings)")
    parser.add_argument("--out", type=Path, help="results folder (default: model/myat/baselines/results/<model>_<settings>)")
    args = parser.parse_args()
    if args.settings not in SETTINGS[args.model]:
        parser.error(f"{args.model} has no {args.settings!r} settings; choose from {SETTINGS[args.model]}")

    out_dir = args.out or RESULTS_DIR / f"{args.model}_{args.settings}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:  # fail before training if a split or a search result is missing
        for seed in args.seeds:
            split_dir(name, seed, args.preprocess_dir)
    search_dir = args.search_dir or (EQUAL_SEARCH_DIR if args.settings == "search_equal" else SEARCH_DIR)
    search_results = load_search_results(search_dir) if args.settings.startswith("search") else None
    settings = {name: baseline_settings(args.model, name, args.settings, search_results) for name in args.datasets}

    # Like runs.csv, keep the settings of datasets trained in earlier runs
    settings_path = out_dir / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    saved.update(settings)
    saved = {name: saved[name] for name in DATASETS if name in saved}
    settings_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    rows = []
    for name in args.datasets:
        for seed in args.seeds:
            row = run_trial(args.model, name, seed, settings[name], args.preprocess_dir, out_dir)
            rows.append(row)
            print(f"{name:15s} seed {seed}: test F1 {row['f1_test']:.3f}, train F1 {row['f1_train']:.3f}, "
                  f"{row['total_nodes']} nodes, {row['fit_seconds']:.2f}s", flush=True)

    # Keep earlier runs of other datasets/seeds, replace the ones just trained
    runs = pd.DataFrame(rows)
    runs_path = out_dir / "runs.csv"
    if runs_path.exists():
        previous = pd.read_csv(runs_path)
        keep = ~previous.set_index(["dataset", "seed"]).index.isin(runs.set_index(["dataset", "seed"]).index)
        runs = pd.concat([previous[keep], runs])
    runs["order"] = runs["dataset"].map(DATASETS.index)
    runs = runs.sort_values(["order", "seed"]).drop(columns="order")
    runs.to_csv(runs_path, index=False)

    summary = summarize(runs, args.model, args.settings)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print()
        print(summary[["dataset", "trials", "test_f1", "test_f1_std", "paper_test_f1", "paper_test_f1_std",
                       "train_f1", "paper_train_f1", "tree_size", "paper_tree_size", "fit_seconds"]].to_string(index=False))
    print(f"\nSaved {runs_path}, {out_dir / 'summary.csv'} and test predictions in {out_dir / 'predictions'}")


if __name__ == "__main__":
    main()

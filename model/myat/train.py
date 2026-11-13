"""Train GradTree on the preprocessed datasets over 10 trials and compare with the paper.

Run from the repository root:
    python -m model.myat.train                                   # 5 datasets x seeds 42-51, tuned settings
    python -m model.myat.train --datasets iris wine --seeds 42 43 --jobs 4
    python -m model.myat.train --settings default                # the paper's default-setting runs

Writes to model/myat/results/<settings>/:
    runs.csv           one row per (dataset, seed): macro F1 / accuracy on train, valid and test, tree size, ...
    summary.csv        mean and std over seeds next to the paper's numbers
    settings.json      the GradTree settings used for each dataset
    predictions/       test-set predictions of every run, for error analysis
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .classifier import GradTreeClassifier
from .config import DATASETS, MODEL_SEED, PAPER, SEEDS, gradtree_settings
from .data import PREPROCESS_DIR, load_split, split_dir

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_trial(name: str, seed: int, settings: dict, preprocess_dir: Path, out_dir: Path, threads: int) -> dict:
    torch.set_num_threads(threads)
    split = load_split(name, seed, preprocess_dir)
    clf = GradTreeClassifier(**settings, n_classes=split["n_classes"], seed=MODEL_SEED)
    clf.fit(split["X_train"], split["y_train"], split["X_valid"], split["y_valid"])

    row = {"dataset": name, "seed": seed}
    for part in ["train", "valid", "test"]:
        pred = clf.predict(split[f"X_{part}"])
        row[f"f1_{part}"] = f1_score(split[f"y_{part}"], pred, average="macro")
        row[f"accuracy_{part}"] = accuracy_score(split[f"y_{part}"], pred)
    internal, leaves = clf.tree_.count_nodes()
    row.update(
        internal_nodes=internal,
        leaf_nodes=leaves,
        total_nodes=internal + leaves,
        best_restart=clf.best_restart_,
        best_valid_loss=clf.best_valid_loss_,
        epochs=sum(run["epochs"] for run in clf.history_),
        fit_seconds=clf.fit_seconds_,
    )

    proba = clf.predict_proba(split["X_test"])
    predictions = pd.DataFrame({"y_true": split["y_test"], "y_pred": proba.argmax(axis=1)})
    for k in range(proba.shape[1]):
        predictions[f"proba_{k}"] = proba[:, k]
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions" / f"{name}_seed_{seed}.csv", index=False)
    return row


def summarize(runs: pd.DataFrame, settings_name: str) -> pd.DataFrame:
    rows = []
    for name, group in runs.groupby("dataset", sort=False):
        paper = PAPER[settings_name].get(name, {})
        row = {
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
            "paper_tree_size": paper.get("tree_size", (np.nan,))[0],
            "test_accuracy": group["accuracy_test"].mean(),
            "fit_seconds": group["fit_seconds"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--settings", default="tuned", choices=["tuned", "default"])
    parser.add_argument("--jobs", type=int, default=1, help="trials to train in parallel")
    parser.add_argument("--epochs", type=int, help="override the maximum epochs (for quick tests)")
    parser.add_argument("--restarts", type=int, help="override the number of restarts (for quick tests)")
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    parser.add_argument("--out", type=Path, help="results folder (default: model/myat/results/<settings>)")
    args = parser.parse_args()

    out_dir = args.out or RESULTS_DIR / args.settings
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:  # fail before training if a split is missing
        for seed in args.seeds:
            split_dir(name, seed, args.preprocess_dir)

    settings = {}
    for name in args.datasets:
        settings[name] = gradtree_settings(name, args.settings)
        if args.epochs is not None:
            settings[name]["epochs"] = args.epochs
        if args.restarts is not None:
            settings[name]["restarts"] = args.restarts
    # Like runs.csv, keep the settings of datasets trained in earlier runs
    settings_path = out_dir / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    saved.update(settings)
    saved = {name: saved[name] for name in DATASETS if name in saved}
    settings_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    trials = [(name, seed) for name in args.datasets for seed in args.seeds]
    threads = 1 if args.jobs > 1 else max(1, os.cpu_count() // 2)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_trial, name, seed, settings[name], args.preprocess_dir, out_dir, threads): (name, seed)
            for name, seed in trials
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['dataset']:15s} seed {row['seed']}: test F1 {row['f1_test']:.3f}, "
                  f"{row['total_nodes']} nodes, {row['epochs']} epochs, {row['fit_seconds']:.0f}s", flush=True)

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

    summary = summarize(runs, args.settings)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print()
        print(summary[["dataset", "trials", "test_f1", "test_f1_std", "paper_test_f1", "paper_test_f1_std",
                       "train_f1", "paper_train_f1", "tree_size", "paper_tree_size", "fit_seconds"]].to_string(index=False))
    print(f"\nSaved {runs_path}, {out_dir / 'summary.csv'} and test predictions in {out_dir / 'predictions'}")


if __name__ == "__main__":
    main()

"""Our own random search for GradTree's hyperparameters (proposal section 4.1.3).

Like the paper (Appendix C), the search runs once per dataset on the first trial (seed 42) and
never uses its test split. Each of SEARCH_TRIALS random settings from SEARCH_SPACE is scored by
macro F1 of the pruned tree:
    holdout  train on train, stop early on valid, score on valid (Adult, Spambase)
    cv       5-fold stratified cross-validation on train + valid; inside each fold 10% of the
             training part is held out for early stopping, as in the official HPO code
             (Echocardiogram, Iris, Wine, whose validation splits have only 21-28 rows)
On the small datasets the folds are the ones CART's equal-budget search uses (baselines/search.py
--equal-budget), so both models get the same number of settings and choose from the same data.

Run from the repository root:
    python -m model.myat.gradtree.search --jobs 6
    python -m model.myat.gradtree.search --datasets iris wine --trials 20 --jobs 6

Every finished fit is saved straight away to model/myat/gradtree/hpo/fits.csv and a rerun skips it,
so the search can be stopped and continued (keep --trials the same when continuing). Writes hpo/best_params.json and hpo/<dataset>.csv
(every setting, best first). Then evaluate the best settings on all 10 trials with
    python -m model.myat.gradtree.train --settings search
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

from .classifier import GradTreeClassifier
from .config import (DATASETS, MODEL_SEED, SEARCH_FOLDS, SEARCH_PROTOCOL, SEARCH_RESTARTS, SEARCH_SPACE,
                     SEARCH_TRIALS, TRAINING)
from .data import PREPROCESS_DIR, load_split

HPO_DIR = Path(__file__).resolve().parent / "hpo"
PARAMS = list(SEARCH_SPACE)


def candidates(trials: int) -> list[dict]:
    """The settings to try, the same for every dataset (sampled without repeats)."""
    return list(ParameterSampler(SEARCH_SPACE, n_iter=trials, random_state=MODEL_SEED))


def cv_folds(y: np.ndarray, n_folds: int):
    """The cross-validation folds on train + valid; CART's search uses the same ones."""
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=MODEL_SEED).split(np.zeros(len(y)), y))


def fold_data(split: dict, protocol: str, fold: int, n_folds: int):
    """(X_fit, y_fit, X_stop, y_stop, X_score, y_score) for one fold."""
    if protocol == "holdout":
        return split["X_train"], split["y_train"], split["X_valid"], split["y_valid"], split["X_valid"], split["y_valid"]
    X = np.vstack([split["X_train"], split["X_valid"]])
    y = np.concatenate([split["y_train"], split["y_valid"]])
    train_idx, test_idx = cv_folds(y, n_folds)[fold]
    fit_idx, stop_idx = train_test_split(train_idx, test_size=0.1, random_state=MODEL_SEED, stratify=y[train_idx])
    return X[fit_idx], y[fit_idx], X[stop_idx], y[stop_idx], X[test_idx], y[test_idx]


def evaluate(dataset: str, trial: int, params: dict, fold: int, protocol: str, n_folds: int, split_seed: int,
             preprocess_dir: Path) -> dict:
    torch.set_num_threads(1)
    split = load_split(dataset, split_seed, preprocess_dir)
    X_fit, y_fit, X_stop, y_stop, X_score, y_score = fold_data(split, protocol, fold, n_folds)
    settings = {**TRAINING, **params, "restarts": SEARCH_RESTARTS}
    clf = GradTreeClassifier(**settings, n_classes=split["n_classes"], seed=MODEL_SEED)
    clf.fit(X_fit, y_fit, X_stop, y_stop)
    internal, leaves = clf.tree_.count_nodes()
    return {"dataset": dataset, "trial": trial, "fold": fold, "protocol": protocol, **params,
            "f1": f1_score(y_score, clf.predict(X_score), average="macro"),
            "valid_loss": clf.best_valid_loss_, "total_nodes": internal + leaves,
            "epochs": clf.history_[0]["epochs"], "fit_seconds": clf.fit_seconds_}


def summarize(fits: pd.DataFrame, trials: int, folds: int, split_seed: int) -> tuple[dict, dict]:
    """Mean score per setting; returns (best setting per dataset, table of every setting per dataset)."""
    best, tables = {}, {}
    for name, part in fits.groupby("dataset", sort=False):
        part = part[part["trial"] < trials]
        n_folds = 1 if part["protocol"].iloc[0] == "holdout" else folds
        complete = part.groupby("trial").filter(lambda g: len(g) == n_folds)  # only fully scored settings
        if complete.empty:
            continue
        table = complete.groupby("trial").agg(
            **{p: (p, "first") for p in PARAMS},
            f1=("f1", "mean"), f1_std=("f1", lambda v: v.std(ddof=0)),
            tree_size=("total_nodes", "mean"), fit_seconds=("fit_seconds", "mean"),
        ).reset_index().sort_values(["f1", "trial"], ascending=[False, True], kind="stable")
        top = table.iloc[0]
        params = {
            "depth": int(top["depth"]),
            "lr_index": float(top["lr_index"]),
            "lr_values": float(top["lr_values"]),
            "lr_leaf": float(top["lr_leaf"]),
            "loss": str(top["loss"]),
            "poly_epsilon": None if pd.isna(top["poly_epsilon"]) else int(top["poly_epsilon"]),
        }
        best[name] = {"params": params, "f1": float(top["f1"]), "f1_std": float(top["f1_std"]),
                      "protocol": part["protocol"].iloc[0], "folds": int(n_folds), "trials_scored": int(len(table)),
                      "trials": trials, "split_seed": split_seed}
        tables[name] = table
    return best, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--trials", type=int, default=SEARCH_TRIALS)
    parser.add_argument("--folds", type=int, default=SEARCH_FOLDS)
    parser.add_argument("--split-seed", type=int, default=MODEL_SEED, help="the trial whose train + valid data is searched on")
    parser.add_argument("--jobs", type=int, default=1, help="fits to run in parallel")
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    parser.add_argument("--out", type=Path, default=HPO_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fits_path = args.out / "fits.csv"
    done = set()
    if fits_path.exists():
        previous = pd.read_csv(fits_path)
        done = set(zip(previous["dataset"], previous["trial"], previous["fold"]))

    settings = candidates(args.trials)
    # Largest datasets first, so the slowest fits do not all end up at the tail of the run
    order = sorted(args.datasets, key=lambda n: -len(load_split(n, args.split_seed, args.preprocess_dir)["y_train"]))
    tasks = []
    for name in order:
        protocol = SEARCH_PROTOCOL[name]
        for trial, params in enumerate(settings):
            for fold in range(1 if protocol == "holdout" else args.folds):
                if (name, trial, fold) not in done:
                    tasks.append(dict(dataset=name, trial=trial, params=params, fold=fold, protocol=protocol,
                                      n_folds=args.folds, split_seed=args.split_seed, preprocess_dir=args.preprocess_dir))
    print(f"{len(done)} fits already saved; {len(tasks)} to go", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(evaluate, **task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            pd.DataFrame([row]).to_csv(fits_path, mode="a", header=not fits_path.exists(), index=False)
            print(f"{row['dataset']:15s} trial {row['trial']:2d} fold {row['fold']}: F1 {row['f1']:.3f}, "
                  f"depth {row['depth']}, {row['total_nodes']} nodes, {row['fit_seconds']:.0f}s", flush=True)

    best_path = args.out / "best_params.json"
    saved = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    best, tables = summarize(pd.read_csv(fits_path), args.trials, args.folds, args.split_seed)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    saved.update(best)
    saved = {name: saved[name] for name in DATASETS if name in saved}
    best_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    for name, result in best.items():
        print(f"{name:15s} {result['protocol']} macro F1 {result['f1']:.3f} +/- {result['f1_std']:.3f} "
              f"({result['trials_scored']} settings)  {result['params']}")
    print(f"\nSaved {best_path}")


if __name__ == "__main__":
    main()

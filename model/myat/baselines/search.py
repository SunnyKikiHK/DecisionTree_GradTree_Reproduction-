"""Our own random search with cross-validation for CART's hyperparameters (proposal section 4.1.3).

Like the paper (Appendix C), the search runs once per dataset: random settings from SEARCH_SPACE,
each scored by 5-fold cross-validated macro F1 on the train + valid data of the first trial
(seed 42), the data CART is finally fitted on. The test split is never used.

    default         300 settings, the paper's budget                -> results/search/
    --equal-budget  50 settings, the budget of GradTree's own search  -> results/search_equal/
                    (gradtree/search.py). The folds are the ones GradTree's search uses on the
                    small datasets, so both models choose from the same data there.

Why CART is always cross-validated on train + valid, never fitted on train alone and scored on
valid: Adult's leave-one-out encoding leaks each training row's label into its categorical codes,
while validation rows get clean codes. A CART fitted on the training split alone scores train F1
1.000 and validation F1 0.284 there, so a holdout search would rank settings by noise.

Run from the repository root:
    python -m model.myat.baselines.search
    python -m model.myat.baselines.search --equal-budget
Then evaluate on all 10 trials with train.py --settings search (or search_equal).

Writes best_params.json (the best setting and its CV F1 per dataset) and <dataset>.csv (every
setting tried, best first).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from model.myat.gradtree.config import SEARCH_TRIALS as EQUAL_BUDGET_TRIALS
from model.myat.gradtree.data import PREPROCESS_DIR, load_split

from .config import DATASETS, MODEL_SEED, SEARCH_FOLDS, SEARCH_SPACE, SEARCH_TRIALS

SEARCH_DIR = Path(__file__).resolve().parent / "results" / "search"
EQUAL_SEARCH_DIR = Path(__file__).resolve().parent / "results" / "search_equal"


def load_search_results(search_dir: Path = SEARCH_DIR) -> dict:
    path = Path(search_dir) / "best_params.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def search(name: str, split_seed: int, trials: int, folds: int, jobs: int, preprocess_dir: Path):
    split = load_split(name, split_seed, preprocess_dir)
    X = np.vstack([split["X_train"], split["X_valid"]])
    y = np.concatenate([split["y_train"], split["y_valid"]])
    searcher = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=MODEL_SEED),
        SEARCH_SPACE,
        n_iter=trials,
        scoring="f1_macro",
        cv=StratifiedKFold(folds, shuffle=True, random_state=MODEL_SEED),  # = gradtree/search.py cv_folds
        n_jobs=jobs,
        random_state=MODEL_SEED,
        refit=False,
    )
    searcher.fit(X, y)

    table = pd.DataFrame(searcher.cv_results_["params"])
    table["cv_f1"] = searcher.cv_results_["mean_test_score"]
    table["cv_f1_std"] = searcher.cv_results_["std_test_score"]
    table["fit_seconds"] = searcher.cv_results_["mean_fit_time"]
    table = table.sort_values("cv_f1", ascending=False, kind="stable")
    best = {
        "params": searcher.best_params_,
        "cv_f1": float(searcher.best_score_),
        "cv_f1_std": float(table["cv_f1_std"].iloc[0]),
        "split_seed": split_seed,
        "trials": trials,
        "folds": folds,
    }
    return best, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--equal-budget", action="store_true", help="GradTree's search budget (50 settings)")
    parser.add_argument("--trials", type=int, help=f"settings to try (default {SEARCH_TRIALS}, or {EQUAL_BUDGET_TRIALS} with --equal-budget)")
    parser.add_argument("--folds", type=int, default=SEARCH_FOLDS)
    parser.add_argument("--split-seed", type=int, default=MODEL_SEED, help="the trial whose train + valid data is searched on")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel fits (-1: all cores)")
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.trials = args.trials or (EQUAL_BUDGET_TRIALS if args.equal_budget else SEARCH_TRIALS)
    args.out = args.out or (EQUAL_SEARCH_DIR if args.equal_budget else SEARCH_DIR)

    args.out.mkdir(parents=True, exist_ok=True)
    results = load_search_results(args.out)
    for name in args.datasets:
        best, table = search(name, args.split_seed, args.trials, args.folds, args.jobs, args.preprocess_dir)
        table.to_csv(args.out / f"{name}.csv", index=False)
        results[name] = best
        print(f"{name:15s} CV macro F1 {best['cv_f1']:.3f} +/- {best['cv_f1_std']:.3f}  {best['params']}", flush=True)

    results = {name: results[name] for name in DATASETS if name in results}
    (args.out / "best_params.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out / 'best_params.json'}")


if __name__ == "__main__":
    main()

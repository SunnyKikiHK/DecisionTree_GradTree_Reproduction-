"""Check how leave-one-out (LOO) target encoding lets a tree memorize the training labels.

The preprocessing (like the paper's) encodes each categorical value by the mean target of the
*other* training rows with that value. Inside one category, rows with label 1 therefore get a
slightly lower code than rows with label 0, so a deep tree can read a training row's label off
its own code. Test rows all get the same code per category, so this does not help on test data;
it only inflates train F1 and the train-test gap. Adult is the only dataset with categorical features.

For each trial this script fits CART (tuned settings, train + valid) with and without the
categorical features, and a CART that sees only one two-valued feature (sex).

Run from the repository root:
    python -m model.myat.analysis.leakage

Writes analysis/results/loo_leakage.csv (one row per trial) and loo_leakage.md (the summary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from model.myat.baselines.config import CART_TUNED
from model.myat.gradtree.config import MODEL_SEED, SEEDS
from model.myat.gradtree.data import PREPROCESS_DIR, load_split

from .common import RESULTS_DIR


def fit_scores(params: dict, X_fit, y_fit, X_test, y_test) -> tuple[float, float]:
    tree = DecisionTreeClassifier(random_state=MODEL_SEED, **params).fit(X_fit, y_fit)
    return (f1_score(y_fit, tree.predict(X_fit), average="macro"),
            f1_score(y_test, tree.predict(X_test), average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--single-feature", default="sex", help="a categorical feature with few values")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    args = parser.parse_args()

    name, rows = args.dataset, []
    for seed in args.seeds:
        split = load_split(name, seed, args.preprocess_dir)
        preprocessor = joblib.load(split["folder"] / f"{name}_preprocessor.joblib")
        features = split["feature_names"]
        categorical = [features.index(f) for f in preprocessor["categorical_features"] if f in features]
        numeric = [i for i in range(len(features)) if i not in categorical]
        single = features.index(args.single_feature)
        X_fit = np.vstack([split["X_train"], split["X_valid"]])
        y_fit = np.concatenate([split["y_train"], split["y_valid"]])
        X_test, y_test = split["X_test"], split["y_test"]

        params = CART_TUNED[name]
        row = {"dataset": name, "seed": seed}
        for label, columns, setting in [("all_features", list(range(len(features))), params),
                                        ("numeric_only", numeric, params),
                                        (f"only_{args.single_feature}", [single], dict(max_depth=10))]:
            row[f"{label}_train_f1"], row[f"{label}_test_f1"] = fit_scores(
                setting, X_fit[:, columns], y_fit, X_test[:, columns], y_test)
        row[f"{args.single_feature}_codes_test"] = len(np.unique(np.round(X_test[:, single], 6)))
        row[f"{args.single_feature}_codes_train"] = len(np.unique(X_fit[:, single]))
        rows.append(row)
        print(f"seed {seed}: all features train/test {row['all_features_train_f1']:.3f}/{row['all_features_test_f1']:.3f}, "
              f"numeric only {row['numeric_only_train_f1']:.3f}/{row['numeric_only_test_f1']:.3f}, "
              f"only {args.single_feature} {row[f'only_{args.single_feature}_train_f1']:.3f}/"
              f"{row[f'only_{args.single_feature}_test_f1']:.3f}", flush=True)

    runs = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS_DIR / "loo_leakage.csv", index=False)

    def cell(column):
        return f"{runs[column].mean():.3f} ± {runs[column].std(ddof=0):.3f}"

    single = f"only_{args.single_feature}"
    lines = [
        f"### Leave-one-out encoding and CART's train F1 ({name.capitalize()}, {len(runs)} trials)",
        "",
        "| CART fitted on | Train F1 | Test F1 |",
        "|---|---|---|",
        f"| All features (tuned settings) | {cell('all_features_train_f1')} | {cell('all_features_test_f1')} |",
        f"| Numeric features only (tuned settings) | {cell('numeric_only_train_f1')} | {cell('numeric_only_test_f1')} |",
        f"| `{args.single_feature}` only (depth 10) | {cell(single + '_train_f1')} | {cell(single + '_test_f1')} |",
        "",
        f"`{args.single_feature}` has {runs[f'{args.single_feature}_codes_test'].iloc[0]} codes in the test data but "
        f"{runs[f'{args.single_feature}_codes_train'].mean():.0f} distinct codes in the training data on average "
        "(leave-one-out codes differ by label, and SMOTE interpolates between them). A two-valued feature cannot "
        "legitimately give a train F1 far above its test F1, so the gap comes from the encoding, not from the tree.",
    ]
    (RESULTS_DIR / "loo_leakage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Baseline settings (CART and a random forest reference) and the paper's CART results.

Sources:
- How CART is trained: the authors' evaluate_gdt (experiments_paper_gradtree/utilities/utilities_GDT.py).
  CART is sklearn's DecisionTreeClassifier fitted on train + valid (it needs no validation split),
  with random_state 42 in every trial.
- Tuned settings: paper Table 15 (found by the authors' random search with cross-validation).
- Default settings: the authors' GDTs-*-DEFAULT.ipynb, which set max_depth to the default GradTree
  depth (10) and leave everything else at sklearn's defaults.
"""

from model.myat.gradtree.config import DATASETS, MODEL_SEED, SEEDS  # same datasets, splits and model seed as GradTree

__all__ = ["DATASETS", "SEEDS", "MODEL_SEED", "MODELS", "SETTINGS", "baseline_settings", "PAPER", "SEARCH_SPACE"]

MODELS = ["cart", "random_forest"]

CART_TUNED = {  # paper Table 15; max_features None and ccp_alpha 0.0 for all five datasets
    "adult": dict(max_depth=10, criterion="entropy", min_samples_leaf=10, min_samples_split=2),
    "echocardiogram": dict(max_depth=9, criterion="entropy", min_samples_leaf=1, min_samples_split=5),
    "iris": dict(max_depth=8, criterion="entropy", min_samples_leaf=5, min_samples_split=10),
    "spambase": dict(max_depth=10, criterion="gini", min_samples_leaf=1, min_samples_split=2),
    "wine": dict(max_depth=9, criterion="gini", min_samples_leaf=1, min_samples_split=5),
}
CART_DEFAULT = dict(max_depth=10)

# Not in the paper: a tree ensemble as an upper reference for accuracy (proposal section 4.1.2).
# Unlimited depth, so it is not interpretable; it shows what a single tree gives up.
RANDOM_FOREST_DEFAULT = dict(n_estimators=100)

# Which settings exist for each model. 'search' is our own random search with the paper's budget
# (search.py); 'search_equal' is the search with GradTree's budget and protocol (search.py --equal-budget).
SETTINGS = {
    "cart": ["tuned", "default", "search", "search_equal"],
    "random_forest": ["default"],
}

# Our random search for CART. The paper does not list its search ranges, so this space contains
# every value its Table 15 chose (for all 36 datasets) plus values in between. max_depth stops at 10,
# the largest GradTree depth, so both models are limited to trees of the same depth.
SEARCH_SPACE = {
    "max_depth": list(range(1, 11)),
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt", "log2"],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "min_samples_split": [2, 5, 10, 20, 50],
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4],
}
SEARCH_TRIALS = 300  # "The number of trials (300) was equal for each approach" (Appendix C)
SEARCH_FOLDS = 5


def baseline_settings(model: str, name: str, settings: str, search_results: dict | None = None) -> dict:
    """Keyword arguments for the sklearn model (random_state is added by train.py)."""
    if settings not in SETTINGS[model]:
        raise ValueError(f"{model} has no {settings!r} settings; choose from {SETTINGS[model]}")
    if model == "random_forest":
        return dict(RANDOM_FOREST_DEFAULT)
    if settings == "tuned":
        return dict(CART_TUNED[name])
    if settings == "default":
        return dict(CART_DEFAULT)
    if not search_results or name not in search_results:
        flag = " --equal-budget" if settings == "search_equal" else ""
        raise FileNotFoundError(f"No search result for {name}. Run: python -m model.myat.baselines.search{flag} --datasets {name}")
    return dict(search_results[name]["params"])


# CART in the paper, mean and std over 10 trials.
# test_f1: macro F1 on test (tuned: Tables 1-2, default: Table 7); train_f1: Table 8; tree_size: Table 9;
# fit_seconds: Table 10. Tables 8-10 are reported for the tuned settings only.
PAPER = {
    "cart": {
        "tuned": {
            "adult": dict(test_f1=(0.771, 0.011), train_f1=(0.973, 0.001), tree_size=(156.6, 23.08), fit_seconds=(0.07, 0.0)),
            "echocardiogram": dict(test_f1=(0.555, 0.111), train_f1=(0.981, 0.009), tree_size=(36.2, 5.81), fit_seconds=(0.00, 0.0)),
            "iris": dict(test_f1=(0.937, 0.046), train_f1=(0.973, 0.010), tree_size=(9.4, 1.50), fit_seconds=(0.00, 0.0)),
            "spambase": dict(test_f1=(0.917, 0.011), train_f1=(0.965, 0.003), tree_size=(201.0, 14.64), fit_seconds=(0.04, 0.0)),
            "wine": dict(test_f1=(0.907, 0.042), train_f1=(1.000, 0.000), tree_size=(13.4, 2.33), fit_seconds=(0.00, 0.0)),
        },
        "default": {
            "adult": dict(test_f1=(0.773, 0.008)),
            "echocardiogram": dict(test_f1=(0.559, 0.075)),
            "iris": dict(test_f1=(0.937, 0.036)),
            "spambase": dict(test_f1=(0.917, 0.011)),
            "wine": dict(test_f1=(0.907, 0.042)),
        },
    },
}

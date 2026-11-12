"""GradTree settings and the paper's reported results for our five datasets.

Sources:
- Shared training settings: the authors' experiment notebooks (experiments_paper_gradtree/GDTs-*-HPO.ipynb).
- Tuned settings: paper Table 12, identical to the authors' published HPO results
  (experiments_paper_gradtree/evaluation_results_HPO/hpo_best/gdt/*.csv). These produced Tables 1-2.
- Default settings: the authors' GDTs-BINARY-DEFAULT.ipynb and GDTs-MULTI-DEFAULT.ipynb (paper Table 7).
"""

DATASETS = ["adult", "echocardiogram", "iris", "spambase", "wine"]
SEEDS = [42 + i for i in range(10)]  # data seed = 42 + trial, 10 trials
MODEL_SEED = 42                      # the official code initialises the model with 42 in every trial
MULTI_CLASS = {"iris", "wine"}

# Paper Appendix C: no tuning of batch size or epochs; early stopping with patience 200 and 3 random restarts.
TRAINING = dict(
    batch_size=128,
    epochs=10_000,
    restarts=3,
    early_stopping_epochs=200,
    prune_threshold=1,  # prune branches that no training sample reaches
    focal_gamma=3.0,    # "a focal factor of 3" (Appendix C)
)

TUNED = {
    "adult": dict(depth=8, lr_index=0.05, lr_values=0.005, lr_leaf=0.05, loss="crossentropy", poly_epsilon=5),
    "echocardiogram": dict(depth=8, lr_index=0.01, lr_values=0.05, lr_leaf=0.1, loss="crossentropy", poly_epsilon=5),
    "spambase": dict(depth=10, lr_index=0.005, lr_values=0.01, lr_leaf=0.01, loss="crossentropy"),
    "iris": dict(depth=7, lr_index=0.005, lr_values=0.005, lr_leaf=0.05, loss="crossentropy"),
    "wine": dict(depth=10, lr_index=0.01, lr_values=0.05, lr_leaf=0.01, loss="focal"),
}

DEFAULT_BINARY = dict(depth=10, lr_index=0.05, lr_values=0.01, lr_leaf=0.005, loss="crossentropy", restarts=5)
DEFAULT_MULTI = dict(depth=10, lr_index=0.05, lr_values=0.01, lr_leaf=0.05, loss="focal")

# GradTree in the paper, mean and std over 10 trials.
# test_f1: macro F1 on test (tuned: Tables 1-2, default: Table 7); train_f1: Table 8; tree_size: Table 9.
PAPER = {
    "tuned": {
        "adult": dict(test_f1=(0.743, 0.034), train_f1=(0.875, 0.050), tree_size=(128.0, 37.19)),
        "echocardiogram": dict(test_f1=(0.658, 0.113), train_f1=(0.827, 0.088), tree_size=(43.2, 12.82)),
        "iris": dict(test_f1=(0.938, 0.057), train_f1=(0.978, 0.017), tree_size=(14.4, 2.54)),
        "spambase": dict(test_f1=(0.903, 0.025), train_f1=(0.905, 0.020), tree_size=(120.2, 37.01)),
        "wine": dict(test_f1=(0.933, 0.031), train_f1=(0.995, 0.009), tree_size=(24.0, 4.12)),
    },
    "default": {
        "adult": dict(test_f1=(0.749, 0.028)),
        "echocardiogram": dict(test_f1=(0.648, 0.130)),
        "iris": dict(test_f1=(0.938, 0.039)),
        "spambase": dict(test_f1=(0.893, 0.015)),
        "wine": dict(test_f1=(0.895, 0.035)),
    },
}


def gradtree_settings(name: str, settings: str = "tuned") -> dict:
    """Keyword arguments for GradTreeClassifier: 'tuned' (Tables 1-2) or 'default' (Table 7)."""
    if settings == "tuned":
        params = TUNED[name]
    elif settings == "default":
        params = DEFAULT_MULTI if name in MULTI_CLASS else DEFAULT_BINARY
    else:
        raise ValueError(f"Unknown settings {settings!r}")
    return {**TRAINING, **params}

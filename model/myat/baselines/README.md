# Baselines: CART and a random forest

CART is the paper's greedy baseline (sklearn's `DecisionTreeClassifier`). The random forest is not in the paper. We add it as an upper reference for accuracy, since a forest of 100 deep trees is not interpretable (proposal section 4.1.2).

| File | What it does |
|---|---|
| `config.py` | The CART settings (paper's tuned and default), our search space, the random forest settings and the paper's CART results |
| `search.py` | Our own random search with cross-validation for CART (proposal section 4.1.3) |
| `train.py` | Trains a baseline on the same 10 trials as GradTree and compares it with the paper |

The baselines use the same splits as GradTree (`model/myat/gradtree/data.py`), so see [gradtree/README.md](../gradtree/README.md) for creating them.

## How the baselines are trained

The setup is the same as the official experiments (`evaluate_gdt` in the authors' `utilities_GDT.py`):

- **Training data:** CART needs no validation split for early stopping, so it is fitted on train + valid. Train + valid is the same 80% of the data in our 64/16/20 split and the official 60/20/20 split, so CART sees exactly the paper's training data.
- **Train metrics:** "train F1" is measured on train + valid, the data the model was fitted on.
- **Seeds:** `random_state` is 42 in every trial; only the data split changes (seeds 42-51).
- **Tree size:** internal nodes plus leaves. CART needs no pruning step, since sklearn never creates empty branches.

## Run

From the repository root:

```
python -m model.myat.baselines.train                                   # CART, paper's tuned settings (Table 15)
python -m model.myat.baselines.train --settings default                # CART, default settings (Table 7)
python -m model.myat.baselines.search                                  # our random search (a few minutes)
python -m model.myat.baselines.train --settings search                 # CART with the settings it found
python -m model.myat.baselines.train --model random_forest --settings default
```

Results go to `results/<model>_<settings>/` with the same files as GradTree's: `runs.csv`, `summary.csv`, `settings.json` and `predictions/` (gitignored). The search writes `results/search/best_params.json` and every setting it tried to `results/search/<dataset>.csv`.

## Settings

| Settings | CART parameters |
|---|---|
| `tuned` | Paper Table 15 (the authors' random search) |
| `default` | `max_depth=10` (GradTree's default depth), everything else sklearn's default, as in the authors' default notebooks |
| `search` | Our random search: 300 settings (the paper's budget, Appendix C), 5-fold stratified CV, macro F1. It runs on train + valid of the first trial (seed 42) and never touches the test data. The space contains every value in the paper's Table 15; `max_depth` stops at 10, GradTree's largest depth. |

The random forest uses sklearn's defaults: 100 trees, unlimited depth.

## Results (10 trials, test macro F1, mean ± std)

| Dataset | CART tuned | Paper | CART default | Paper | CART search (ours) | Random forest |
|---|---|---|---|---|---|---|
| Adult | 0.766 ± 0.011 | 0.771 ± 0.011 | 0.773 ± 0.008 | 0.773 ± 0.008 | 0.764 ± 0.013 | 0.785 ± 0.006 |
| Echocardiogram | 0.538 ± 0.114 | 0.555 ± 0.111 | 0.527 ± 0.079 | 0.559 ± 0.075 | 0.592 ± 0.113 | 0.622 ± 0.080 |
| Iris | 0.937 ± 0.046 | 0.937 ± 0.046 | 0.940 ± 0.037 | 0.937 ± 0.036 | 0.934 ± 0.045 | 0.940 ± 0.040 |
| Spambase | 0.917 ± 0.010 | 0.917 ± 0.011 | 0.917 ± 0.010 | 0.917 ± 0.011 | 0.914 ± 0.008 | 0.952 ± 0.007 |
| Wine | 0.903 ± 0.025 | 0.907 ± 0.042 | 0.907 ± 0.038 | 0.907 ± 0.042 | 0.940 ± 0.043 | 0.982 ± 0.016 |

Tree sizes with the tuned settings match the paper's Table 9: Adult 158.4 (paper 156.6), Echocardiogram 36.0 (36.2), Iris 9.4 (9.4), Spambase 201.0 (201.0), Wine 15.2 (13.4).

What this shows:

- **CART reproduces the paper almost exactly.** Iris and Spambase are identical, which confirms that our 80/20 train/test splits are the authors'. The small Wine difference fits our Wine having 13 features (the official loader drops Proline).
- **Our search beats the paper's settings on Echocardiogram (+0.054) and Wine (+0.037)** and is within noise elsewhere.
- **Our search's cross-validation scores are optimistic on two datasets.**
  - Adult scores 0.977 in CV but 0.764 on test. Leave-one-out encoding leaks each training row's label into its categorical codes (see `analysis/leakage.py`), and CV runs inside that encoded training data.
  - Echocardiogram's CV is also optimistic, because SMOTE was applied before the folds were cut.
  - The official search has the same issue, since it also cross-validates on train + valid.

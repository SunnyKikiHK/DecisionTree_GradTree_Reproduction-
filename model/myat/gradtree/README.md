# GradTree in PyTorch (Myat)

A reproduction of GradTree trained on our preprocessed datasets.

| File | What it does |
|---|---|
| `gradtree.py` | The model: entmax feature selection with a straight-through hardmax, hard splits with a straight-through gradient, the tree pass (Algorithm 1) and the loss |
| `classifier.py` | `GradTreeClassifier`: training with Adam, early stopping on the validation loss, random restarts, then post-hoc pruning |
| `tree.py` | `HardTree`: the trained model as an ordinary decision tree, with the official pruning, node counts and `describe()` to print the rules |
| `config.py` | Training settings, the tuned settings per dataset and the paper's results |
| `data.py` | Loads `preprocess/<name>_output/seed_<seed>/` |
| `train.py` | Runs every dataset and seed, and compares the results with the paper |
| `search.py` | Our own random search for GradTree's settings (proposal section 4.1.3) |

## 1. Create the splits

Training uses all 10 trials (seeds 42-51), so each preprocessing notebook must be run with `SAVE_ALL_SEEDS = True`:

| Dataset | Notebook | Seed splits in the repo? |
|---|---|---|
| Echocardiogram, Iris | `preprocess/echocardiogram_iris_preprocessing.ipynb` | Yes, committed |
| Spambase, Wine | `preprocess/spambase_wine_preprocessing.ipynb` | No, gitignored: run the notebook |
| Adult | `preprocess/adult_preprocessing.ipynb` | No, gitignored (about 120 MB): run the notebook |

If a split is missing, `train.py` stops before training and names the notebook to run.

## 2. Train

From the repository root, after `pip install -r requirements.txt`:

```
python -m model.myat.gradtree.train                                  # 5 datasets x 10 seeds, the paper's tuned settings
python -m model.myat.gradtree.train --datasets iris wine --jobs 4    # some datasets, 4 trials in parallel
python -m model.myat.gradtree.train --settings default               # the paper's default settings (Table 7)
python -m model.myat.gradtree.search --jobs 6                        # our own search (see Settings)
python -m model.myat.gradtree.train --settings search                # the settings it found
python -m model.myat.gradtree.train --settings default --resume      # continue: skip trials already in runs.csv
python -m model.myat.gradtree.train --datasets wine --seeds 42 --epochs 50 --restarts 0   # quick test
```

Results go to `model/myat/gradtree/<settings>/` (`tuned/`, `default/`, `search/`):

- `runs.csv`: one row per dataset and seed, with macro F1 and accuracy on train, valid and test, the tree size after pruning, the epochs and the runtime.
- `summary.csv`: the mean and std over the seeds, next to the paper's numbers.
- `predictions/<dataset>_seed_<seed>.csv`: test labels, predictions and probabilities, for error analysis. These are gitignored; training recreates them.
- `settings.json`: the settings used.

Running only some datasets or seeds updates those rows and keeps the rest. `runs.csv` and `summary.csv` are saved after every finished trial, so a stopped run loses nothing; `--resume` continues it.

To use the model directly:

```python
from model.myat.gradtree import GradTreeClassifier
from model.myat.gradtree.config import gradtree_settings
from model.myat.gradtree.data import load_split

split = load_split("wine", seed=42)
clf = GradTreeClassifier(**gradtree_settings("wine"), n_classes=split["n_classes"])
clf.fit(split["X_train"], split["y_train"], split["X_valid"], split["y_valid"])
clf.predict(split["X_test"])
print(clf.tree_.describe(split["feature_names"]))
```

## Settings

Shared by every dataset (paper Appendix C and the authors' experiment notebooks): batch size 128, up to 10,000 epochs, early stopping after 200 epochs without a lower validation loss, 3 random restarts (the run with the lowest validation loss is kept), and pruning of branches that no training sample reaches. The model seed is 42 in every trial; only the data split changes.

Tuned settings (paper Table 12, the same as the authors' `hpo_best/gdt/*.csv`):

| Dataset | Depth | lr index | lr values | lr leaf | Loss |
|---|---|---|---|---|---|
| Adult | 8 | 0.05 | 0.005 | 0.05 | cross-entropy + Poly-1 (ε = 5) |
| Echocardiogram | 8 | 0.01 | 0.05 | 0.1 | cross-entropy + Poly-1 (ε = 5) |
| Spambase | 10 | 0.005 | 0.01 | 0.01 | cross-entropy |
| Iris | 7 | 0.005 | 0.005 | 0.05 | cross-entropy |
| Wine | 10 | 0.01 | 0.05 | 0.01 | focal (γ = 3) |

Our own search (`search.py`, results in `hpo/`):

- **What is searched:** depth 1-10, the three learning rates, the loss (cross-entropy or focal) and the Poly-1 term (none, ε = 2 or ε = 5). Every value in the paper's Table 12 is included.
- **Budget:** 50 random settings per dataset. The paper used 300, but one GradTree fit takes minutes on our CPU. CART gets the same 50 (`baselines/search.py --equal-budget`), so the two models are tuned with equal budgets.
- **Scoring:** each setting is scored by macro F1 on the first trial (seed 42), without its test split, in the way GradTree is finally trained:
  - Adult and Spambase: train on train, stop early on valid, score on valid.
  - Echocardiogram, Iris and Wine: their validation splits have only 21-28 rows, so they use the same 5 cross-validation folds as CART, holding out 10% of each fold's training part for early stopping.
- **Restarts:** none while searching (one run per setting). The final 10-trial evaluation uses the usual 3 restarts.

`hpo/fits.csv` holds every fit, `hpo/<dataset>.csv` every setting ranked, and `hpo/best_params.json` the winners.

## Where this differs from the official code

- **Optimizer:** the official code wraps Adam in TensorFlow Addons' stochastic weight averaging, but never copies the averaged weights into the model. Its trees are therefore trained with plain Adam, which is what this code uses. The paper describes weight averaging over 5 checkpoints, but the official code never applies it.
- **Poly-1 loss:** the official code computes p_t for the Poly-1 term from the already-activated output (a sigmoid of the sigmoid). This code does the same, because the tuned ε = 5 was chosen with it. See `gradtree_loss`.
- **Speed:** the tree pass builds the root-to-leaf products level by level instead of per leaf and depth. The values and gradients are the same; it is just faster.
- **Wine features:** the paper's Wine has 12 features, because the official loader drops Proline. Our preprocessing keeps all 13.
- **Splits:** our splits are 64/16/20 (the paper's text); the official code uses 60/20/20.

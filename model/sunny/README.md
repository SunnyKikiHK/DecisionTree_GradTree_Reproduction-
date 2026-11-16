# GradTree in PyTorch

A modular PyTorch implementation of **GradTree: Learning Axis-Aligned
Decision Trees with Gradient Descent** (Marton et al., 2024,
[arXiv:2305.03515](https://arxiv.org/abs/2305.03515)).

The implementation follows the paper's dense representation and the official
TensorFlow code (`experiments_paper_gradtree/utilities/GDT.py`).  It trains a
complete binary tree with gradient descent and then extracts an ordinary
axis-aligned decision tree for interpretation/evaluation.

For a detailed file-by-file explanation, see
[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## Module map (implementation order)

| Step | Module | What it does |
|---|---|---|
| 1 | [`config.py`](config.py) | Hyperparameters, validation, paper-default factory. |
| 2 | [`entmax.py`](entmax.py) | Wraps the `entmax` package (`entmax15`, `sparsemax`) and adds straight-through hardmax/rounding. |
| 3 | [`tree_structure.py`](tree_structure.py) | Constant path tables: internal node index and left/right branch per leaf/depth (Equations 4-5). |
| 4 | [`losses.py`](losses.py) | Cross-entropy, focal loss and PolyLoss for binary/multi-class. |
| 5 | [`gradtree.py`](gradtree.py) | The model: `I`, `T`, leaf logits, and the vectorised tree pass (Algorithm 1). |
| 6 | [`metrics.py`](metrics.py) | Accuracy and macro F1 without a mandatory scikit-learn dependency. |
| 7 | [`pruning.py`](pruning.py) | Hard decision-tree extraction and post-hoc pruning. |
| 8 | [`trainer.py`](trainer.py) | Adam with per-group learning rates, mini-batches, early stopping, restarts, checkpoints. |
| 9 | [`data.py`](data.py) | Loads the CSV splits written by the preprocessing notebooks. |
| 10 | [`train.py`](train.py) | Command-line training/evaluation entry point. |
| 11 | [`selftest.py`](selftest.py) | Gradcheck and end-to-end smoke tests. |

## The math in one page

For depth `d`, the model has `M = 2**d - 1` internal nodes and `L = 2**d`
leaves.  It learns:

```text
I in R^(M x V)    feature-selection scores
T in R^(M x V)    one threshold per node and feature
L in R^(L)        binary leaf logits, or R^(L x C) for C classes
```

The forward pass is Algorithm 1:

```text
A       = entmax(beta * I)                      # soft feature distribution
I_hard  = ST-hardmax(A)                         # one-hot in forward, identity grad

s1      = sum_v T[path_node, v] * I_hard[path_node, v]   # selected threshold
s2      = sum_v x[v]           * I_hard[path_node, v]   # selected feature
s       = sigmoid(s1 - s2)                      # soft split score
s_hard  = ST-round(s)                           # 0/1 in forward, sigmoid grad

p_leaf  = product over depth of the branch probabilities   # (batch, L)
logits  = p_leaf @ leaf_logits                  # convex combination of leaf logits
```

`entmax` makes `I` sparse, the straight-through hardmax makes the forward
pass a true axis-aligned tree, and the straight-through rounding lets the
sigmoid derivative flow to `T`.  The long-form derivations are in the paper
(Sections 3.1-3.4) and in the research folder's
`explanation_equation_6.md` / `explanation_I_matrix.md`.

## Quick start

The preprocessing notebooks write one CSV per split.  From the repository
root:

```bash
# Quick run on the default Iris split
python -m model.sunny.train --dataset iris --depth 4 --epochs 200

# Train trial seed 43 (requires preprocess/iris_output/seed_43/)
python -m model.sunny.train --dataset iris --trial-seed 43

# Adult, paper-style configuration
python -m model.sunny.train --dataset adult --depth 10 --epochs 10000 \
    --patience 200 --restarts 5 --save model/sunny/checkpoints/adult.pt
```

The CLI prints train/valid/test metrics for the differentiable model and,
by default, for the post-hoc-pruned hard tree.  `--print-tree` prints the
hard tree; `--no-prune` skips pruning.

The implementation was tested with Python 3.12 and PyTorch 2.3.1 (CPU).  If
the project virtualenv uses a Python version for which no compatible torch
wheel is available, create a separate Python 3.12/3.13 environment rather
than mixing environments.  For a CPU-only install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> The current `adult_output/` files were produced with the old
> `SEED = 20770802`.  Re-run `preprocess/adult_preprocessing.ipynb` if you
> want the top-level split to be trial 0 (seed 42) and `seed_42` to
> `seed_51`; the updated notebook saves all ten trials.

## Data expectations

[`data.py`](data.py) assumes the preprocessing convention used throughout
this repository:

- `<dataset>_train.csv`, `<dataset>_valid.csv`, `<dataset>_test.csv`;
- features first, target last;
- labels already encoded as `0..K-1`;
- features already quantile-transformed;
- the training split already SMOTE-balanced when applicable.

For the 10-trial experiments, pass `--trial-seed 42` ... `--trial-seed 51`.
Iris and Echocardiogram already have `seed_*` folders.  Spambase and Wine
were saved with `SAVE_ALL_SEEDS = False`, so re-run their preprocessing with
`SAVE_ALL_SEEDS = True` first.  Adult now saves all seeds when re-run.

The official experiment code uses **data seed `42 + trial`** and a fixed
**model seed `42`** for every dataset.  In this CLI, `--trial-seed` selects
the data split, while `--random-seed 42` keeps model initialisation fixed.

## Hyperparameters

`GradTreeConfig` defaults are practical for a first run:

```text
depth=5, lr_index=0.05, lr_values=0.01, lr_leaf=0.005,
batch_size=128, epochs=500, patience=50, restarts=0
```

The official default notebook uses `depth=10`, `epochs=10000`,
`patience=200`, `restarts=5`, and SWA.  Use
`GradTreeConfig.paper_default()` or the corresponding CLI flags for that
configuration.  The paper's per-dataset HPO depth is 7-10 (Appendix C,
Table 12).

## Implementation notes and differences from the TensorFlow code

- **Straight-through estimator.** `hardmax_st` and `round_st` use
  `x + (hard(x) - x).detach()`, exactly the paper's forward-hard /
  backward-soft convention.
- **Sparse activations.** `entmax15` and `sparsemax` come from the
  MIT-licensed [`entmax`](https://github.com/deep-spin/entmax) package,
  which provides PyTorch autograd.  The self-test `gradcheck`s the wrappers.
  Only the GradTree-specific straight-through estimators live in
  `entmax.py`.
- **Threshold sign.** The official code (and Algorithm 1) uses
  `sigmoid(T_selected - x_selected)`, so the hard tree routes
  `x <= threshold` left.  Equation 6 in the paper writes the opposite sign;
  we follow the official code and Algorithm 1.
- **Boundary ties.** The forward pass uses `torch.round`, which rounds 0.5
  to 0 (banker's rounding), while the hard tree uses `x <= threshold` for
  the left branch.  They can disagree only when a feature equals the
  threshold exactly.
- **Dense tree.** Training always uses the complete tree.  Post-hoc pruning
  is applied only to the extracted hard tree; the PyTorch parameters remain
  full-size.
- **Output activation.** `--output-activation` controls `predict_proba` for
  multi-class models.  Training always uses softmax cross-entropy (the
  paper's main configuration); the option is for prediction-time ablations.
- **SWA.** Stochastic weight averaging is not implemented.  The returned
  model is the best epoch of the best restart.  This is the main optimisation
  detail still missing relative to the official code.
- **Regression / DNDT baselines.** Only classification is implemented.

## Self-tests

```bash
python -m model.sunny.selftest
```

The self-test checks the entmax/sparsemax Jacobians with `gradcheck`, the
path tables, binary/multi-class forward shapes and gradients, agreement
between the differentiable model and the extracted hard tree, pruning,
losses, metrics, checkpoint round-trips, and a small end-to-end overfitting
run.

## Checkpoints

`GradTreeTrainer.save(path, extra=...)` stores the `state_dict`, the config,
the feature count/class count, and optional metadata (dataset, metrics,
feature names).  Reload with:

```python
from model.sunny.trainer import load_model_from_checkpoint

model, config, extra = load_model_from_checkpoint("model/sunny/checkpoints/adult.pt")
```

## Extending the implementation

Good next steps, roughly in order of value:

1. **SWA** — average parameter snapshots over the last `k` epochs, as in the
   official code.
2. **HPO** — wrap `GradTreeTrainer` in a random search over
   `GradTreeConfig` (the paper searched 300 configurations).
3. **Multi-seed evaluation** — loop over the 10 trial folders and report
   mean ± std macro F1, matching the paper's tables.
4. **Tree export** — serialize `HardDecisionTree.to_dict()` for error
   analysis or plotting.
5. **Regression** — add a regression head and MSE loss if needed.

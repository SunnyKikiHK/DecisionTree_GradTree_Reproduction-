# GradTree PyTorch: File-by-File Walkthrough

This guide explains what every Python file in `model/sunny/` does, what its
key functions receive and return, and how the files connect. It is meant to
be read alongside the code, not as a replacement for the docstrings.

If you only want to run the model, see the Quick Start in
[README.md](README.md). If you want to understand or modify it, read this
guide in the order below.

---

## 1. The big picture

```text
preprocess/<dataset>_output/*.csv
        │
        ▼
data.py                 load train/valid/test arrays
        │
        ▼
train.py                build config + model + trainer, run experiment
        │
        ├── config.py          hyperparameters
        ├── gradtree.py        the model
        │     ├── entmax.py          sparse activation + ST estimators
        │     └── tree_structure.py  path tables (Eq. 4-5)
        ├── losses.py          CE / focal / PolyLoss
        ├── trainer.py         Adam, early stopping, restarts, checkpoints
        │     └── metrics.py         accuracy / macro F1
        └── pruning.py         hard tree extraction + post-hoc pruning
```

`selftest.py` exercises the whole stack. `__init__.py` re-exports the public
API.

### Shape conventions

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `V` | number of input features |
| `d` | tree depth |
| `M = 2**d - 1` | number of internal nodes |
| `L = 2**d` | number of leaves |
| `C` | number of classes |

The three learned tensors are:

```text
I  (split_index)      (M, V)
T  (split_threshold)  (M, V)
L  (leaf_logits)      (L,)      for binary
                      (L, C)    for multi-class
```

The constant path tables are:

```text
path_identifier       (L, d)    # 0 = left, 1 = right
internal_node_index   (L, d)    # breadth-first internal node index
```

---

## 2. File-by-file

### 2.1 `__init__.py` (package exports)

**Role:** makes the package convenient to import. The parent
`model/__init__.py` is only a package marker; the exports live in
`model/sunny/__init__.py`.

It re-exports `GradTree`, `GradTreeConfig`, `GradTreeTrainer`,
`GradTreeLoss`, `HardDecisionTree`, the entmax/straight-through helpers,
the metrics and `load_model_from_checkpoint`.

It deliberately does **not** import `data.py`, because `data.py` needs
pandas while the core model only needs PyTorch and NumPy. This keeps
`python -m model.sunny.selftest` runnable in a minimal environment.

---

### 2.2 `config.py` — hyperparameters

**Role:** one validated dataclass holding every training choice.

The main fields:

```python
GradTreeConfig(
    depth=5,
    lr_index=0.05, lr_values=0.01, lr_leaf=0.005,
    batch_size=128, epochs=500, patience=50, restarts=0,
    activation="entmax", beta=1.0, dropout=0.0, init="glorot",
    loss="crossentropy", focal_gamma=2.0, focal_alpha=0.5,
    poly_epsilon=2.0, output_activation="softmax",
    prune=True, prune_threshold=1,
    device="auto", seed=42, verbose=True, log_every=50,
)
```

Key points:

- `lr_index`, `lr_values`, `lr_leaf` are the three separate Adam learning
  rates from the paper. `I`, `T` and the leaves are updated at different
  rates.
- `patience=0` disables early stopping (the trainer treats it as
  `epochs`).
- `restarts` is the number of restarts **after** the first run. The
  official code uses 5, i.e. six runs total.
- `activation` is the feature-selection activation: `entmax`, `softmax` or
  `sparsemax`.
- `init` selects `glorot` (paper Algorithm 2), `normal`, `uniform` or
  `data_aware`.
- `loss` is `crossentropy`, `focal` or `poly`.
- `output_activation` only changes multi-class `predict_proba`; training
  always uses softmax cross-entropy.
- `__post_init__` rejects invalid combinations early (depth < 1, negative
  learning rates, unknown activation names, etc.).
- `GradTreeConfig.paper_default()` returns the official notebook setting:
  depth 10, 10,000 epochs, patience 200, 5 restarts, learning rates
  0.05/0.01/0.005. SWA is *not* included.
- `to_dict()` is used for printing and checkpoint metadata.

---

### 2.3 `entmax.py` — sparse activations and straight-through estimators

**Role:** wraps the sparse activations from the MIT-licensed
[`entmax`](https://github.com/deep-spin/entmax) package and adds the two
GradTree-specific straight-through operators.

#### `entmax15(z)` and `sparsemax(z)`

Thin last-dimension wrappers around the library:

```python
from entmax import entmax15 as _entmax15, sparsemax as _sparsemax

def entmax15(z):
    return _entmax15(z, dim=-1)

def sparsemax(z):
    return _sparsemax(z, dim=-1)
```

`entmax15` is the paper's `alpha = 1.5` sparse projection:

```text
entmax_1.5(z) = 0.25 * relu(z - tau) ** 2,   sum = 1
```

It produces exact zeros for low-scoring features, which is what allows the
hardmax to choose a single feature per node. `sparsemax` is the
Euclidean-projection alternative used in the ablation. The library provides
the forward and backward passes; the self-test runs
`torch.autograd.gradcheck` on both wrappers.

Install the model dependencies with `pip install -r model/sunny/requirements.txt`
from the repository root (or just `pip install entmax` if torch is already
installed). The focused list is in
[requirements.txt](requirements.txt).

#### `hardmax_st(x)`

Straight-through hardmax:

```python
hard = one_hot(argmax(x))
return x + (hard - x).detach()
```

Forward value: one-hot. Backward: identity with respect to `x`. This is how
the model chooses exactly one feature per internal node while still letting
gradients reach `I`.

#### `round_st(x)`

Straight-through rounding:

```python
return x + (round(x) - x).detach()
```

Forward value: 0 or 1. Backward: identity, so the sigmoid derivative is used
for the split threshold.

#### `apply_activation(z, kind)`

Small dispatcher used by the model and trainer:

```python
softmax  -> torch.softmax(z, dim=-1)
entmax   -> entmax15(z)
sparsemax-> sparsemax(z)
```

---

### 2.4 `tree_structure.py` — constant path tables

**Role:** precomputes, for every leaf and every depth, the internal node on
that leaf's path and whether the leaf takes the left or right branch.

For leaf `l` (0-based) and depth `j` (1-based):

```text
internal_node_index[l, j-1] = 2**(j-1) + floor(l / 2**(d - (j-1))) - 1   # Eq. 4
path_identifier[l, j-1]     = floor(l / 2**(d - j)) % 2                   # Eq. 5
```

`make_tree_structure(depth)` returns a frozen `TreeStructure` dataclass with
`num_internal`, `num_leaves`, `path_identifier` and
`internal_node_index`.

Why this exists: the paper evaluates the whole tree in one batch. Instead of
recursively walking each sample down the tree, the model gathers the
parameters for every leaf's path into `(L, d, V)` tensors and multiplies
probabilities. The tables never change for a fixed depth, so the model
registers them as non-persistent buffers.

---

### 2.5 `gradtree.py` — the model

**Role:** defines the learnable parameters and the full forward pass
(Algorithm 1).

#### Constructor

```python
GradTree(
    num_features, num_classes, depth=3,
    activation="entmax", beta=1.0, dropout=0.0,
    init="glorot", output_activation="softmax", seed=None,
)
```

It creates:

```python
split_index      (M, V)     # I
split_threshold  (M, V)     # T
leaf_logits      (L,) or (L, C)
```

and registers the path tables.

#### `reset_parameters(seed=None)`

Initialises `I`, `T` and leaves. `init="glorot"` uses the paper's bounds:

```text
I, T ~ U(-sqrt(6/(M+V)), +sqrt(6/(M+V)))
L    ~ U(-sqrt(6/(L+C)), +sqrt(6/(L+C)))   # C=1 for binary
```

`normal`, `uniform` and `data_aware` are alternatives for ablation studies.
The trainer calls this again at every restart.

#### `initialize_thresholds_from_data(X)`

For `init="data_aware"`, replaces every `T[node, feature]` with a value
randomly drawn from that feature's training column. This mirrors the
official `dataAware` initialiser.

#### `_feature_distribution()`

```python
scores = dropout(split_index)          # training only
return activation(beta * scores)       # (M, V)
```

#### `forward(x)` — the core

Input `x`: `(B, V)`. Output: `(B,)` for binary, `(B, C)` for multi-class.

Step by step:

1. **Feature choice**

   ```python
   A = entmax(beta * I)          # (M, V), soft
   I_hard = hardmax_st(A)        # (M, V), one-hot in forward
   ```

2. **Gather per-leaf path parameters**

   ```python
   idx = internal_node_index             # (L, d)
   I_sel = I_hard[idx]                   # (L, d, V)
   T_sel = split_threshold[idx]          # (L, d, V)
   ```

3. **Split score**

   ```python
   threshold_term = (T_sel * I_sel).sum(-1)          # (L, d)
   feature_term   = einsum("bv,ldv->bld", x, I_sel)  # (B, L, d)
   s_soft = sigmoid(threshold_term - feature_term)   # (B, L, d)
   s_hard = round_st(s_soft)                         # (B, L, d)
   ```

   The sign is `T - x`, matching Algorithm 1 and the official code. The
   hard tree therefore routes `x <= threshold` left.

4. **Path probability**

   ```python
   path = path_identifier                      # (L, d)
   p = prod((1 - path) * s_hard
            + path * (1 - s_hard), dim=2)      # (B, L)
   ```

   In the forward pass `s_hard` is 0/1, so exactly one leaf has probability
   1 for each sample.

5. **Leaf output**

   ```python
   binary:      logits = sum_l p[b, l] * leaf_logits[l]
   multi-class: logits = p @ leaf_logits
   ```

Optional helpers:

- `predict_proba(X)` — sigmoid for binary; softmax/entmax/sparsemax for
  multi-class.
- `predict(X)` — hard labels.
- `feature_indices` — `argmax(I)` for every internal node.
- `hard_tree(prune_X=None, min_samples=1)` — converts the learned
  parameters into a `HardDecisionTree` and optionally prunes it.
- `num_parameters` — parameter count.

Important interpretation: **the forward pass is already a hard tree**. The
entmax distribution and the sigmoid are used to produce gradients; in the
forward pass the feature is one-hot and the branch is 0/1.

---

### 2.6 `losses.py` — training objectives

**Role:** `GradTreeLoss(num_classes, kind=...)` turns logits and labels into
a scalar loss.

Supported losses:

| `kind` | Binary | Multi-class |
|---|---|---|
| `crossentropy` | `binary_cross_entropy_with_logits` | `cross_entropy` |
| `focal` | alpha-balanced focal BCE | focal CE without alpha |
| `poly` | `CE + epsilon * (1 - p_true)` | same |

For binary models the logits are `(B,)`, so BCE with logits is used directly
instead of applying sigmoid first. This is numerically equivalent to the
official `binary_crossentropy(y, sigmoid(logits))`.

The focal and PolyLoss formulas are the paper's optional HPO variants; the
default main configuration is plain cross-entropy.

---

### 2.7 `metrics.py` — evaluation

**Role:** accuracy, confusion matrix and macro F1 using NumPy only.

- `accuracy_score(y_true, y_pred)` — fraction correct.
- `confusion_matrix(y_true, y_pred, num_classes=None)` — rows are true
  classes, columns predicted.
- `macro_f1_score(y_true, y_pred, num_classes=None)` — F1 for every class,
  then the unweighted mean. Classes absent from both prediction and truth
  still contribute F1=0 when `num_classes` is supplied, matching the
  official `F1Score(average="macro", num_classes=C)`.

Why not scikit-learn? The core package then runs in a minimal PyTorch
environment. The preprocessing notebooks already use scikit-learn, but the
model does not need it.

---

### 2.8 `pruning.py` — hard tree and post-hoc pruning

**Role:** turns the learned tensors into an ordinary axis-aligned decision
tree and applies the paper's pruning rules.

#### `Leaf` and `InternalNode`

Simple tree nodes:

- `Leaf(value, num_samples)` stores logits.
- `InternalNode(feature, threshold, left, right, num_samples)` stores a
  split.

#### `HardDecisionTree(feature, threshold, leaf_values)`

Inputs are the arrays extracted from the model:

```python
feature   = argmax(split_index, dim=1)                  # (M,)
threshold = split_threshold[range(M), feature]          # (M,)
leaf_values = leaf_logits                               # (L,) or (L, C)
```

Internal nodes are numbered breadth-first from 0; node `>= M` is leaf
`node - M`. The constructor validates the complete-tree relationship
`num_leaves == num_internal + 1`.

Methods:

- `leaf_values_for(X)` — walks the tree for each row and returns the leaf
  logits. It uses a stack of `(node, row_indices)` pairs, so it works on both
  the full and the pruned tree.
- `predict_proba(X)` — sigmoid for binary, softmax for multi-class.
- `predict(X)` — thresholded logits / argmax.
- `prune(X, min_samples=1)` — bottom-up pruning:
  - if both children receive fewer than `min_samples` rows, merge the node
    into a leaf whose value is the average of the two child values;
  - if only one child is too small, replace the node by the other child's
    subtree;
  - otherwise keep the split.
  A value in `(0, 1)` for `min_samples` is interpreted as a fraction of the
  dataset, as in the official code.
- `count_nodes()`, `size`, `summary()` — tree-size statistics.
- `to_dict()` — JSON-friendly nested representation.
- `print_tree()` — text representation for inspection.

Important: pruning is applied to the **extracted hard tree**, not to the
PyTorch parameters. The differentiable model keeps its complete tree. This
lets you compare model and pruned-tree metrics, which the CLI reports
separately.

---

### 2.9 `trainer.py` — training and inference loop

**Role:** fits a `GradTree` with the paper's optimisation recipe and provides
batched prediction/evaluation/checkpointing.

#### Setup

```python
trainer = GradTreeTrainer(model, config)
```

- Resolves `device="auto"` to CUDA or CPU.
- Moves the model there.
- Builds `GradTreeLoss` from the config.
- Builds one Adam optimiser with three parameter groups:

```python
[
  {"params": [split_index],      "lr": lr_index},
  {"params": [split_threshold],  "lr": lr_values},
  {"params": [leaf_logits],      "lr": lr_leaf},
]
```

This is equivalent to the official code's three parameter-group optimisers
(Adam in its default constructor) because the groups do not share state.

#### `fit(X_train, y_train, X_valid=None, y_valid=None)`

For each restart:

1. Seed everything with `config.seed + restart`.
2. Re-initialise the model (`reset_parameters`) and the optimiser.
3. Call `_fit_once`.
4. Keep the best state by validation loss (or training loss when no
   validation data is supplied).
5. After all restarts, load the globally best state and return a
   `TrainResult`.

`TrainResult` contains the best restart, best epoch, monitored loss, the
loss curves of that restart, and a short summary of every restart.

#### `_fit_once`

- Caps the batch size at half the split, matching the official code:
  `min(config.batch_size, ceil(n/2))`.
- Shuffles once per epoch with a deterministic local generator.
- Runs mini-batches, calls `loss.backward()` and `optimizer.step()`.
- Computes validation loss after each epoch.
- Saves a CPU copy of the state whenever the monitored loss improves.
- Stops after `patience` epochs without improvement.
- Restores the best state.

#### Prediction/evaluation

- `predict_logits(X)` — batched, returns CPU logits.
- `predict_proba(X)` — sigmoid or softmax/entmax/sparsemax.
- `predict(X)` — hard labels.
- `evaluate(X, y)` — returns `loss`, `accuracy`, `macro_f1`.

#### Checkpointing

- `save(path, extra=None)` writes `state_dict`, config, feature/class counts
  and optional metadata.
- `load_model_from_checkpoint(path)` rebuilds the model and config and loads
  the weights.

---

### 2.10 `data.py` — CSV loading

**Role:** loads the outputs of the preprocessing notebooks.

`load_dataset(dataset, data_dir=None, seed=None)` reads:

```text
preprocess/<dataset>_output/<dataset>_{train,valid,test}.csv
preprocess/<dataset>_output/seed_<seed>/<dataset>_{train,valid,test}.csv
```

It returns a `DatasetSplits` dataclass containing the six arrays plus
`feature_names` and `target_name`.

Assumptions, matching the preprocessing notebooks:

- features first, target last;
- labels already encoded as `0..K-1`;
- features already quantile-transformed;
- the training split already SMOTE-balanced where applicable.

`infer_num_classes(*targets)` returns `max(label) + 1` across the splits.
`DatasetSplits.num_features` and `.num_classes` expose those values.

If a `seed_<n>` folder is missing, the loader raises a message explaining
that the preprocessing notebook must be re-run with
`SAVE_ALL_SEEDS = True`.

---

### 2.11 `train.py` — command-line experiment

**Role:** the user-facing entry point.

```bash
python -m model.sunny.train --dataset iris --depth 4 --epochs 100
```

`build_parser()` defines all CLI flags: dataset, trial seed, depth, learning
rates, batch size, epochs, patience, restarts, activation, beta, dropout,
init, loss, pruning, device, random seed, checkpoint path and verbosity.

`main()`:

1. Builds a `GradTreeConfig`.
2. Loads the dataset.
3. Infers the number of classes.
4. Builds `GradTree` and `GradTreeTrainer`.
5. Calls `fit`.
6. Prints train/valid/test metrics for the differentiable model.
7. Extracts and prunes the hard tree, then prints its metrics (and the tree
   itself with `--print-tree`).
8. Optionally saves a checkpoint.

The distinction between the two seed flags is intentional:

- `--trial-seed` selects the **data split** (`seed_42` … `seed_51`).
- `--random-seed` selects the **model initialisation**. The official
  experiments use data seed `42 + trial` and model seed `42`.

---

### 2.12 `selftest.py` — verification

**Role:** a dependency-light test suite runnable with `python -m
model.sunny.selftest`.

It checks:

1. `entmax15` and `sparsemax` outputs and their Jacobians via
   `torch.autograd.gradcheck`.
2. Straight-through hardmax and rounding.
3. The path tables against the closed-form equations.
4. Binary and multi-class forward shapes and finite gradients for `I`, `T`
   and the leaves.
5. Agreement between the differentiable model's hard predictions and the
   extracted `HardDecisionTree`.
6. Pruning produces a smaller valid tree.
7. Loss values match PyTorch's reference cross-entropy.
8. Accuracy and macro F1 on a hand-computed example.
9. A small two-blob overfitting run reaches high accuracy.
10. A save/load checkpoint round-trip preserves predictions.

Run it after any change to the math or model; it catches shape, sign and
gradient mistakes quickly.

---

## 3. One complete forward pass, step by step

Using `B=4`, `V=5`, `d=3`:

| Step | Code | Shape |
|---|---|---|
| Input | `x` | `(4, 5)` |
| Feature scores | `split_index` | `(7, 5)` |
| Soft distribution | `entmax(beta * I)` | `(7, 5)` |
| Hard features | `hardmax_st(...)` | `(7, 5)` |
| Path indices | `internal_node_index` | `(8, 3)` |
| Selected features | `I_hard[idx]` | `(8, 3, 5)` |
| Selected thresholds | `T[idx]` | `(8, 3, 5)` |
| Threshold term | `(T_sel * I_sel).sum(-1)` | `(8, 3)` |
| Feature term | `einsum("bv,ldv->bld", x, I_sel)` | `(4, 8, 3)` |
| Split scores | `sigmoid(T - x)` | `(4, 8, 3)` |
| Hard routing | `round_st(...)` | `(4, 8, 3)` |
| Path probabilities | product over depth | `(4, 8)` |
| Binary logits | weighted sum of leaf logits | `(4,)` |
| Multi-class logits | `p @ leaf_logits` | `(4, C)` |

The forward values are hard: one feature per node and one branch per depth.
The backward pass uses the `entmax` package's exact Jacobian and the
sigmoid derivative.

---

## 4. Mapping to the paper and official code

| Paper | Official TensorFlow | This implementation |
|---|---|---|
| `I` | `split_index_array` | `GradTree.split_index` |
| `T` | `split_values` | `GradTree.split_threshold` |
| `L` | `leaf_classes_array` | `GradTree.leaf_logits` |
| entmax | `entmax15` | `entmax` package (`entmax15`) |
| hardmax ST | `tfa.seq2seq.hardmax` + stop-gradient | `entmax.hardmax_st` |
| Equation 7 rounding | `tf.round` + stop-gradient | `entmax.round_st` |
| Algorithm 1 | `GDT.forward` | `GradTree.forward` |
| post-hoc pruning | `DecisionTree.prune_tree` | `HardDecisionTree.prune` |

---

## 5. Where to change things

| Want to change | Edit |
|---|---|
| Add a hyperparameter | `config.py`, then the CLI in `train.py` |
| Add an activation | `entmax.py` (`apply_activation`), validation lists |
| Add a loss | `losses.py`, validation in `config.py`, CLI choices |
| Change the forward pass | `gradtree.py` (`forward`) |
| Change pruning rules | `pruning.py` (`HardDecisionTree.prune`) |
| Add SWA / another optimiser | `trainer.py` (`fit`, `_fit_once`) |
| Change CSV layout | `data.py` (`load_dataset`) |
| Add a metric | `metrics.py` and `trainer.evaluate` |

### Known gaps

- Stochastic weight averaging (SWA) is not implemented. The returned model
  is the best epoch of the best restart.
- Regression is not implemented.
- `output_activation` affects `predict_proba`; training uses softmax
  cross-entropy.
- The forward pass and hard tree can disagree when a feature equals its
  threshold exactly, because the forward uses `torch.round` (0.5 → 0) while
  the hard tree uses `x <= threshold` for the left branch.

---

## 6. Running the checks

```bash
# All self-tests
python -m model.sunny.selftest

# Quick real-data smoke run
python -m model.sunny.train --dataset iris --depth 4 --epochs 100

# Ten-trial-style run: data seed 43, model seed 42
python -m model.sunny.train --dataset iris --trial-seed 43 --random-seed 42
```

The README has the installation notes and the paper-default commands.

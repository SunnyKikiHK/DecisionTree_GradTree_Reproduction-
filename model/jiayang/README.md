# GradTree reproduction (Jiayang)

PyTorch reproduction of **GradTree: Learning Axis-Aligned Decision Trees with Gradient Descent**
(Marton, Lüdtke, Bartelt and Stuckenschmidt, AAAI 2024, [arXiv 2305.03515](https://arxiv.org/abs/2305.03515)),
with the CART baseline and the paper's feature-selection ablation, on Adult, Echocardiogram,
Spambase, Iris and Wine.

The paper is followed first; where it is silent, values follow the official code
([s-marton/GradTree](https://github.com/s-marton/GradTree)).

## Files

| File | What it does |
|---|---|
| `gradtree.py` | GradTree model: dense tree representation, straight-through entmax, Eq. 3-7 tree pass, Adam per parameter group, weight averaging, early stopping, restarts, post-hoc pruning. Also the ablation variants (Gumbel-softmax, temperature annealing). |
| `train.py` | Trains GradTree on the 10 trials (split seeds 42-51) with the paper's tuned hyperparameters (Table 12); resumable; parallel trials. |
| `cart.py` | CART baseline (sklearn) with the paper's tuned hyperparameters (Table 15); GradTree vs CART comparison. |
| `ablation.py` | Builds the ablation table (ST entmax vs ST Gumbel-softmax vs temperature annealing) next to the paper's Table 5. |
| `results/` | All results (CSV) and training logs. |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Our results were produced with Python 3.11, PyTorch 2.15.0 (CPU), scikit-learn 1.9.1,
pandas 3.1.0, NumPy 2.4.6 and SciPy 1.17.1. (`requirements.txt` pins NumPy 2.5.4, which needs
Python 3.12 or newer; on Python 3.11 install NumPy 2.4.x instead.) No GPU is needed.

## Reproduce

All commands run from the repository root.

**1. Data splits.** Run the three notebooks in `preprocess/` with `SAVE_ALL_SEEDS = True`
(the default). Each writes `preprocess/<dataset>_output/seed_<seed>/<dataset>_{train,valid,test}.csv`
for seeds 42-51: 64/16/20 train/valid/test split, quantile transform (`output_distribution='normal'`)
fitted on the training split, SMOTE on the training split only when the minority class share is at
most 0.25 / (classes - 1), as in the paper.

**2. GradTree, main results (Tables 1, 2, 8, 9).**

```bash
python model/jiayang/train.py --datasets adult echocardiogram spambase iris wine --jobs 4
```

Writes `results/runs.csv` (one row per trial, saved after every trial) and `results/summary.csv`.
Trials already in `runs.csv` are skipped, so the same command resumes an interrupted run
(`--rerun` trains them again). Took about 4 hours on a 4-core laptop CPU; Adult and Spambase are
about 95% of it, the other three datasets take minutes.

**3. CART baseline and comparison.**

```bash
python model/jiayang/cart.py
```

Seconds. Writes `results/cart_runs.csv`, `results/cart_summary.csv` and `results/comparison.csv`.

**4. Ablation (Table 5).**

```bash
python model/jiayang/train.py --datasets echocardiogram iris wine --index-activation gumbel --jobs 4
python model/jiayang/train.py --datasets echocardiogram iris wine --index-activation annealing --jobs 4
python model/jiayang/ablation.py
```

About 30 minutes. Writes `results/runs_gumbel.csv`, `results/runs_annealing.csv`, their summaries
and `results/ablation.csv`.

## Implementation choices

| Component | Choice | Source |
|---|---|---|
| Tree representation | Dense `I`, `T` (2^d - 1 x features), `L` (2^d x classes) | Paper, Section 3.1 |
| Split feature | entmax-1.5, then straight-through hardmax | Paper, Section 3.3, Algorithm 1 |
| Split | `sigmoid(sum(iota * tau) - sum(iota * x))`, straight-through rounding; `S = 1` goes left (`x < threshold`) | Paper, Eq. 6-7 |
| Tree pass | Product along each root-to-leaf path (Eq. 3-5), computed level by level (same values and gradients as the per-leaf product, 2-9x faster) | Paper, Algorithm 1 |
| Initialization | Glorot uniform | Paper, Algorithm 2 |
| Optimizer | Adam, one learning rate per parameter group (Table 12), eps = 1e-7 | Paper; eps from TensorFlow's default |
| Loss | Cross-entropy, focal (gamma = 3) or Poly-1, per Table 12 | Paper, Table 12 |
| Batch size, epochs | min(128, ceil(n / 2)), up to 10,000 epochs | Official code |
| Early stopping | Validation loss, patience 200 | Paper (patience from official code) |
| Weight averaging | Mean of the last 5 epoch checkpoints | Paper (count from official code) |
| Restarts | 3 restarts (4 runs, seeds 42-45), best validation loss kept | Paper / official code |
| Pruning | Remove branches no training sample reaches | Paper, Section 4.1 |
| CART | `DecisionTreeClassifier`, Table 15 hyperparameters, fitted on train + valid | Paper, Appendix C; official code |
| Gumbel-softmax | `F.gumbel_softmax(hard=True)`, tau = 1; hard argmax for validation and test | Jang et al. 2016 (not in paper or official code) |
| Temperature annealing | entmax(I / tau), tau from 1 to 0.01 in log space over 4000 steps, then hard | NODE-GAM, Chang et al. 2021 (not in paper or official code) |

Metric: macro F1 on the test split, mean and standard deviation over 10 trials (split seeds 42-51,
model seed 42). Tree size: internal + leaf nodes after pruning.

## Results

### GradTree and CART (10 trials, test macro F1)

| Dataset | GradTree (ours) | GradTree (paper) | CART (ours) | CART (paper) | GradTree better in |
|---|---|---|---|---|---|
| Adult | 0.750 ± 0.016 | 0.743 ± 0.034 | **0.766** ± 0.011 | 0.771 ± 0.011 | 2 / 10 trials |
| Echocardiogram | **0.598** ± 0.108 | 0.658 ± 0.113 | 0.538 ± 0.114 | 0.555 ± 0.111 | 6 / 10 |
| Spambase | 0.904 ± 0.014 | 0.903 ± 0.025 | **0.917** ± 0.010 | 0.917 ± 0.011 | 2 / 10 |
| Iris | 0.914 ± 0.072 | 0.938 ± 0.057 | **0.937** ± 0.046 | 0.937 ± 0.046 | 3 / 10 |
| Wine | **0.929** ± 0.032 | 0.933 ± 0.031 | 0.903 ± 0.025 | 0.907 ± 0.042 | 8 / 10 |

| Dataset | GradTree size (ours / paper) | CART size (ours / paper) | GradTree train F1 (ours / paper) | CART train F1 (ours / paper) |
|---|---|---|---|---|
| Adult | 152.4 / 128.0 | 158.4 / 156.6 | 0.857 / 0.875 | 0.979 / 0.973 |
| Echocardiogram | 50.4 / 43.2 | 36.0 / 36.2 | 0.907 / 0.827 | 0.977 / 0.981 |
| Spambase | 157.2 / 120.2 | 201.0 / 201.0 | 0.912 / 0.905 | 0.965 / 0.965 |
| Iris | 16.4 / 14.4 | 9.4 / 9.4 | 0.970 / 0.978 | 0.973 / 0.973 |
| Wine | 22.8 / 24.0 | 15.2 / 13.4 | 0.996 / 0.995 | 0.996 / 1.000 |

- GradTree's test F1 is within one standard deviation of the paper on all 5 datasets.
- CART matches the paper almost exactly (identical on Spambase and Iris), which also confirms that
  our splits match the paper's.
- The paper's winner is reproduced on 4 of 5 datasets: GradTree wins on Echocardiogram and Wine,
  CART on Adult and Spambase. Iris is a tie in the paper (0.938 vs 0.937); ours goes to CART.
- GradTree overfits less than CART (smaller train-test gap), as in the paper.
- Our GradTrees are 20-30% larger than the paper's on Adult and Spambase.

### Ablation: split feature selection (test macro F1, 10 trials)

| Dataset | ST entmax (ours / paper) | ST Gumbel-softmax (ours / paper) | Temperature annealing (ours / paper) |
|---|---|---|---|
| Echocardiogram | 0.598 / 0.658 | 0.623 / 0.574 | 0.558 / 0.619 |
| Iris | 0.914 / 0.938 | 0.908 / 0.743 | 0.916 / 0.930 |
| Wine | 0.929 / 0.933 | 0.941 / 0.565 | 0.891 / 0.881 |

- ST entmax beats temperature annealing on Echocardiogram and Wine, a tie on Iris, as in the paper.
- ST Gumbel-softmax does **not** collapse in our runs, unlike the paper. The paper does not say how its
  Gumbel variant is evaluated and the official code does not include it; we evaluate with the
  noise-free argmax and select by validation loss with weight averaging, which keeps it stable.
- With the paper's early stopping, runs stop after about 600-900 steps, so annealing only reaches a
  temperature of about 0.4, not 0.01.

## Not reproduced

- DNDT, GeneticTree and DL8.5 baselines, default-hyperparameter runs (Table 7), runtime (Table 10).
- Hyperparameter search: the search ranges are not published; we use the paper's tuned values.
- Averages and ranks (MRR / MRD) over the paper's 36 datasets: we use 5.
- Ablation on Adult and Spambase (compute time).

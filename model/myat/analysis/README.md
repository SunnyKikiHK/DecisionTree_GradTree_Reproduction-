# Analysis

Compares GradTree (`model/myat/gradtree`) with the baselines (`model/myat/baselines`) and covers the analyses in proposal section 4.1.5:

- the comparison itself;
- the effect of depth;
- the feature-selection ablation;
- the tree figures;
- a check on the preprocessing.

| Script | What it does | Output |
|---|---|---|
| `compare.py` | GradTree vs CART on the same 10 trials: test F1, wins, Wilcoxon test, tree size, overfitting, time, next to the paper | `results/comparison_<settings>.{csv,md}`, `figures/comparison_<settings>.png` |
| `depth.py` | Both models at maximum depths 1-10 with their other settings tuned | `results/depth_*.csv`, `figures/depth.png` |
| `ablation.py` | GradTree with four ways of choosing a node's feature (paper Table 4) | `results/ablation_*.csv`, `figures/ablation.png` |
| `trees.py` | A GradTree and a CART tree from the same trial, drawn side by side (like paper Figure 1) | `figures/trees_*.png`, `results/trees_*.txt` |
| `leakage.py` | How leave-one-out encoding lets CART memorize Adult's training labels | `results/loo_leakage.{csv,md}` |
| `variants.py`, `common.py` | The ablation's GradTree variants; shared paths, chart style and run helpers | |

## Run

From the repository root, after training GradTree and the baselines (`python -m model.myat.gradtree.train`, then `python -m model.myat.baselines.train` for each setting):

```
python -m model.myat.analysis.compare
python -m model.myat.analysis.leakage
python -m model.myat.analysis.trees                                 # Echocardiogram, seed 42, depth 3
python -m model.myat.analysis.trees --dataset iris --depth tuned
python -m model.myat.analysis.depth --jobs 5                        # ~15-20 min: CART everywhere, GradTree on the 3 small datasets
python -m model.myat.analysis.ablation --jobs 5                     # ~15-20 min: Echocardiogram, Iris, Wine
```

`depth.py` and `ablation.py` save each finished run straight away, and a rerun skips the runs already saved. So you can stop them and continue later, or add datasets: for example, `--gradtree-datasets spambase` or `--datasets spambase`. Adult and Spambase take about 5-30 and 15 minutes per GradTree run on a CPU, which is why they are not in the defaults; the saved results include them (`--datasets echocardiogram iris wine adult spambase` and `--gradtree-datasets echocardiogram iris wine adult spambase`). `--plot-only` redraws the figures from the saved runs.

Figures use one colour per model throughout: GradTree blue, CART orange, random forest green. Hollow diamonds are the paper's numbers.

## Findings so far

### 1. GradTree vs CART (tuned settings): mixed on our splits

From `results/comparison_tuned.md`:

| Dataset | GradTree | CART | Paper G / C | Wins G / C (of 10) | Wilcoxon p |
|---|---|---|---|---|---|
| Adult | 0.737 ± 0.034 | **0.766 ± 0.011** | 0.743 / 0.771 | 2 / 8 | 0.037 |
| Echocardiogram | **0.606 ± 0.108** | 0.538 ± 0.114 | 0.658 / 0.555 | 6 / 2 | 0.078 |
| Iris | 0.902 ± 0.075 | **0.937 ± 0.046** | 0.938 / 0.937 | 1 / 6 | 0.047 |
| Spambase | 0.904 ± 0.012 | **0.917 ± 0.010** | 0.903 / 0.917 | 3 / 7 | 0.105 |
| Wine | **0.939 ± 0.056** | 0.903 ± 0.025 | 0.933 / 0.907 | 8 / 2 | 0.084 |

- **Accuracy.** GradTree wins Echocardiogram and Wine and CART wins Adult, Iris and Spambase. The paper had GradTree ahead on Iris too, by 0.001. Only Adult and Iris differ at p < 0.05, and with 10 trials no difference is strong evidence.
- **Tree size.** GradTree's trees are smaller where the data is large (Adult 120 vs 158 nodes, Spambase 124 vs 201) and larger on the small datasets. This is the paper's pattern.
- **Overfitting.** GradTree's train-test gap is smaller on 4 of 5 datasets, all but Iris, as the paper claims. For Adult, though, see finding 2.
- **Time.** CART trains in under 0.1 s. GradTree takes 25 s (Iris) to 32 min (Adult) on our CPU, including its 3 restarts.

### 2. Leave-one-out encoding inflates train scores on Adult

From `results/loo_leakage.md`. The preprocessing (like the paper's) encodes each category by the mean label of the *other* training rows in that category. Within a category, rows with label 1 therefore get a slightly different code than rows with label 0. A deep tree can read the label off the code, while every test row in a category gets the same code.

| CART fitted on | Train F1 | Test F1 |
|---|---|---|
| All features | 0.979 | 0.766 |
| Numeric features only | 0.786 | 0.736 |
| `sex` only (2 categories) | 0.966 | 0.431 |

A two-valued feature cannot honestly give 0.966 train F1. So CART's train F1 on Adult, and the paper's 0.973 for CART, mostly measure this leak rather than the model's overfitting. Cross-validation on the encoded training data is optimistic for the same reason (our CART search scored 0.977 in CV). GradTree's train F1 on Adult is much lower (0.867), so it exploits the leak less, if at all; we have not measured that directly.

### 3. Depth: shallow trees are enough on the small datasets, but not on Spambase

From `figures/depth.png` (GradTree at depths 2, 4, 6, 8 and 10 on all five datasets, CART at 1-10):

- **CART** stops growing by depth 4 on Iris (9.4 nodes) and Wine (15.2 nodes), and keeps growing on Adult and Spambase. On Echocardiogram its test F1 falls from 0.629 (depth 3) to 0.531 (depth 10): it overfits the small training set (105 real rows).
- **GradTree** at depth 2-4 is as good as or better than at its tuned depth, with smaller trees:

  | Dataset | Shallow GradTree | GradTree at tuned depth |
  |---|---|---|
  | Echocardiogram | 0.650 at depth 4, 17 nodes | 0.606 at depth 8, 49 nodes |
  | Wine | 0.948 at depth 4, 14 nodes | 0.939 at depth 10, 23 nodes |
  | Iris | 0.925 at depth 2 | 0.902 at depth 7 |

  The paper's tuned depths may not carry over to our splits.
- **Adult:** GradTree peaks at depth 6 (0.752, 52 nodes), above its tuned depth 8 (0.737, 120 nodes) and depth 10 (0.731, 196 nodes). At shallow depths it is far better than CART: 0.714 against 0.445 at depth 2, with 7 nodes each. CART's shallow Adult trees reach 0.90-0.96 train F1 but 0.44-0.58 test F1, the leave-one-out leak of finding 2: its first splits read the label off the encoded categories. CART only catches up at depth 7 and above.
- **Spambase:** both models keep improving with depth, and CART is ahead at every depth from 4 up (depth 4: 0.891 vs 0.843; depth 10: 0.917 vs 0.904). With 57 features, GradTree needs its full tuned depth 10 here.

### 4. Feature-selection ablation: the paper's gap is not reproduced

From `results/ablation_summary.csv`, test macro F1 over 10 trials on all five datasets, with the paper's Table 5 in brackets:

| Dataset | ST entmax (GradTree) | ST softmax | ST Gumbel-softmax | Temperature annealing |
|---|---|---|---|---|
| Adult | 0.737 (0.743) | 0.737 | 0.740 (0.583) | 0.748 (0.737) |
| Echocardiogram | 0.606 (0.658) | 0.586 | 0.608 (0.574) | 0.598 (0.619) |
| Iris | 0.902 (0.938) | 0.930 | 0.903 (0.743) | 0.934 (0.930) |
| Spambase | 0.904 (0.903) | 0.904 | 0.907 (0.541) | 0.897 (0.875) |
| Wine | 0.939 (0.933) | 0.922 | 0.921 (0.565) | 0.911 (0.881) |

- **Variants:** all four are within about one standard deviation of each other on every dataset, including the two large ones. The paper's claim that entmax matters most when there are many features (Spambase has 57) is not reproduced: on Spambase all four score 0.897-0.907.
- **Entmax and annealing** match the paper.
- **Gumbel:** the paper's ST Gumbel-softmax is far worse than ours on every dataset except Echocardiogram (Adult 0.583 vs 0.740, Spambase 0.541 vs 0.907).
  - The paper does not describe its alternatives' settings. Ours uses temperature 1 and makes the hard, noise-free choice when evaluating (see `variants.py`).
  - If the paper kept the Gumbel noise at prediction time, that alone could explain the gap, but we cannot check this.

### 5. Tree figures

`figures/trees_echocardiogram_seed42_depth3.png` shows depth-3 trees on Echocardiogram, like paper Figure 1. GradTree gets test F1 0.679 against CART's 0.625, with the same number of nodes (13). The splits differ from the root's children down: CART splits on `pericardial-effusion` twice, while GradTree uses `fractional-shortening` and `age-at-heart-attack`. GradTree's pruning keeps a split whose two leaves predict the same class, because it only removes branches that no training row reaches.

## Status

Done since the first version of this README (results in `results/`, figures in `figures/`):

- GradTree with default settings (`comparison_default.md`) and with our own search (`comparison_search.md`, 50 settings per dataset, the same budget as `cart_search_equal`).
- Error analysis of individual predictions (`errors_<settings>.md`).
- Depth study and ablation for GradTree on Adult and Spambase (findings 3 and 4).

Not covered:

- **Mean reciprocal rank / mean relative difference across datasets (paper Table 2 summary rows):** all the inputs are in `results/comparison_*.csv`, but they are not computed yet.
- **The paper's other baselines** (DNDT, GeneticTree, DL8.5): not implemented. We compare with CART and a random forest only.

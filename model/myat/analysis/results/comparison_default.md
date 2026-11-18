### GradTree vs CART: default settings, 10 trials

| Dataset | GradTree | CART | Paper GradTree | Paper CART | Wins G / C | Wilcoxon p | Random forest |
|---|---|---|---|---|---|---|---|
| Adult | 0.763 ± 0.013 | **0.773 ± 0.008** | 0.749 | 0.773 | 2 / 8 | 0.064 | 0.785 |
| Echocardiogram | **0.683 ± 0.126** | 0.527 ± 0.079 | 0.648 | 0.559 | 9 / 1 | 0.006 | 0.622 |
| Iris | 0.907 ± 0.068 | **0.940 ± 0.037** | 0.938 | 0.937 | 3 / 5 | 0.195 | 0.940 |
| Spambase | 0.886 ± 0.018 | **0.917 ± 0.010** | 0.893 | 0.917 | 1 / 9 | 0.004 | 0.952 |
| Wine | **0.912 ± 0.039** | 0.907 ± 0.038 | 0.895 | 0.907 | 4 / 5 | 1.000 | 0.982 |
| **Mean** | 0.830 | 0.813 | 0.825 | 0.819 | | | 0.856 |

Test macro F1, mean ± std. Wins: trials (of 10) where that model's test F1 is higher. Wilcoxon signed-rank test on the paired test F1s; with 10 trials the smallest possible p is 0.002.

| Dataset | Tree size G / C | Paper tree size G / C | Train F1 G / C | Train-test gap G / C | Seconds G / C |
|---|---|---|---|---|---|
| Adult | 138.4 / 260.0 | – / – | 0.843 / 0.982 | 0.080 / 0.209 | 7781.9 / 0.082 |
| Echocardiogram | 23.4 / 47.2 | – / – | 0.863 / 1.000 | 0.180 / 0.473 | 52.6 / 0.001 |
| Iris | 19.8 / 15.0 | – / – | 0.970 / 1.000 | 0.063 / 0.060 | 38.1 / 0.001 |
| Spambase | 73.8 / 201.0 | – / – | 0.896 / 0.965 | 0.010 / 0.049 | 1522.2 / 0.038 |
| Wine | 48.8 / 16.2 | – / – | 0.991 / 1.000 | 0.079 / 0.093 | 37.1 / 0.002 |

G = GradTree, C = CART. Tree size counts internal and leaf nodes after pruning. CART's train F1 is measured on train + valid, the data it is fitted on. Times are on a CPU and include GradTree's restarts.

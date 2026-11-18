### GradTree vs CART: our own search (50 settings each), 10 trials

| Dataset | GradTree | CART | Paper GradTree | Paper CART | Wins G / C | Wilcoxon p | Random forest |
|---|---|---|---|---|---|---|---|
| Adult | **0.757 ± 0.016** | 0.757 ± 0.012 | – | – | 5 / 5 | 0.770 | 0.785 |
| Echocardiogram | **0.660 ± 0.078** | 0.584 ± 0.113 | – | – | 6 / 2 | 0.078 | 0.622 |
| Iris | 0.920 ± 0.041 | **0.934 ± 0.045** | – | – | 2 / 5 | 0.156 | 0.940 |
| Spambase | 0.881 ± 0.014 | **0.904 ± 0.010** | – | – | 0 / 10 | 0.002 | 0.952 |
| Wine | **0.934 ± 0.057** | 0.923 ± 0.029 | – | – | 6 / 4 | 0.557 | 0.982 |
| **Mean** | 0.831 | 0.820 | – | – | | | 0.856 |

Test macro F1, mean ± std. Wins: trials (of 10) where that model's test F1 is higher. Wilcoxon signed-rank test on the paired test F1s; with 10 trials the smallest possible p is 0.002.

| Dataset | Tree size G / C | Paper tree size G / C | Train F1 G / C | Train-test gap G / C | Seconds G / C |
|---|---|---|---|---|---|
| Adult | 59.4 / 90.8 | – / – | 0.759 / 0.978 | 0.002 / 0.221 | 3696.4 / 0.074 |
| Echocardiogram | 61.2 / 11.0 | – / – | 0.927 / 0.841 | 0.267 / 0.257 | 20.9 / 0.001 |
| Iris | 11.2 / 7.2 | – / – | 0.956 / 0.972 | 0.036 / 0.037 | 15.5 / 0.002 |
| Spambase | 71.6 / 89.2 | – / – | 0.894 / 0.928 | 0.013 / 0.024 | 1002.8 / 0.028 |
| Wine | 15.6 / 12.8 | – / – | 0.986 / 0.996 | 0.052 / 0.073 | 32.0 / 0.002 |

G = GradTree, C = CART. Tree size counts internal and leaf nodes after pruning. CART's train F1 is measured on train + valid, the data it is fitted on. Times are on a CPU and include GradTree's restarts.

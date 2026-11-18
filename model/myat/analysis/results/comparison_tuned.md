### GradTree vs CART: paper's tuned settings, 10 trials

| Dataset | GradTree | CART | Paper GradTree | Paper CART | Wins G / C | Wilcoxon p | Random forest |
|---|---|---|---|---|---|---|---|
| Adult | 0.737 ± 0.034 | **0.766 ± 0.011** | 0.743 | 0.771 | 2 / 8 | 0.037 | 0.785 |
| Echocardiogram | **0.606 ± 0.108** | 0.538 ± 0.114 | 0.658 | 0.555 | 6 / 2 | 0.078 | 0.622 |
| Iris | 0.902 ± 0.075 | **0.937 ± 0.046** | 0.938 | 0.937 | 1 / 6 | 0.047 | 0.940 |
| Spambase | 0.904 ± 0.012 | **0.917 ± 0.010** | 0.903 | 0.917 | 3 / 7 | 0.105 | 0.952 |
| Wine | **0.939 ± 0.056** | 0.903 ± 0.025 | 0.933 | 0.907 | 8 / 2 | 0.084 | 0.982 |
| **Mean** | 0.818 | 0.812 | 0.835 | 0.817 | | | 0.856 |

Test macro F1, mean ± std. Wins: trials (of 10) where that model's test F1 is higher. Wilcoxon signed-rank test on the paired test F1s; with 10 trials the smallest possible p is 0.002.

| Dataset | Tree size G / C | Paper tree size G / C | Train F1 G / C | Train-test gap G / C | Seconds G / C |
|---|---|---|---|---|---|
| Adult | 119.8 / 158.4 | 128.0 / 156.6 | 0.867 / 0.979 | 0.131 / 0.213 | 1931.6 / 0.080 |
| Echocardiogram | 48.6 / 36.0 | 43.2 / 36.2 | 0.851 / 0.977 | 0.245 / 0.439 | 27.0 / 0.001 |
| Iris | 15.6 / 9.4 | 14.4 / 9.4 | 0.977 / 0.973 | 0.075 / 0.036 | 24.6 / 0.001 |
| Spambase | 124.2 / 201.0 | 120.2 / 201.0 | 0.911 / 0.965 | 0.007 / 0.049 | 984.4 / 0.036 |
| Wine | 22.6 / 15.2 | 24.0 / 13.4 | 0.991 / 0.996 | 0.051 / 0.093 | 60.4 / 0.001 |

G = GradTree, C = CART. Tree size counts internal and leaf nodes after pruning. CART's train F1 is measured on train + valid, the data it is fitted on. Times are on a CPU and include GradTree's restarts.

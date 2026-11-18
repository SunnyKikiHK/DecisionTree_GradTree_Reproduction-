### Where GradTree and CART go wrong (default settings, test rows of all 10 trials)

| Dataset | Test rows | Error G / C / RF | Both wrong | Only G wrong | Only C wrong | G's errors RF also misses | C's errors RF also misses |
|---|---|---|---|---|---|---|---|
| Adult | 65130 | 16.8% / 15.3% / 14.8% | 10.5% | 6.2% | 4.8% | 60.5% | 70.5% |
| Echocardiogram | 270 | 23.3% / 31.1% / 23.0% | 16.7% | 6.7% | 14.4% | 61.9% | 64.3% |
| Iris | 300 | 9.0% / 5.7% / 5.7% | 4.7% | 4.3% | 1.0% | 51.9% | 82.4% |
| Spambase | 9210 | 10.8% / 7.9% / 4.6% | 4.2% | 6.6% | 3.8% | 30.4% | 44.6% |
| Wine | 360 | 8.3% / 9.4% / 1.9% | 2.8% | 5.6% | 6.7% | 16.7% | 14.7% |

Error = share of test rows predicted wrongly (G = GradTree, C = CART, RF = random forest).

| Dataset | Class | Share of test | Recall G / C / RF | Precision G / C |
|---|---|---|---|---|
| Adult | <=50K | 75.8% | 0.905 / 0.933 / 0.932 | 0.878 / 0.873 |
| Adult | >50K | 24.2% | 0.607 / 0.576 / 0.603 | 0.670 / 0.734 |
| Echocardiogram | alive-at-1 = 0 | 77.0% | 0.822 / 0.817 / 0.909 | 0.868 / 0.787 |
| Echocardiogram | alive-at-1 = 1 | 23.0% | 0.581 / 0.258 / 0.306 | 0.493 / 0.296 |
| Iris | Iris-setosa | 33.7% | 0.960 / 1.000 / 1.000 | 0.960 / 1.000 |
| Iris | Iris-versicolor | 29.3% | 0.830 / 0.909 / 0.920 | 0.859 / 0.899 |
| Iris | Iris-virginica | 37.0% | 0.928 / 0.919 / 0.910 | 0.904 / 0.927 |
| Spambase | class 0 | 60.0% | 0.924 / 0.947 / 0.971 | 0.899 / 0.923 |
| Spambase | class 1 | 40.0% | 0.844 / 0.882 / 0.929 | 0.881 / 0.917 |
| Wine | class 1 | 33.3% | 0.950 / 0.917 / 0.992 | 0.927 / 0.902 |
| Wine | class 2 | 39.2% | 0.872 / 0.872 / 0.957 | 0.918 / 0.891 |
| Wine | class 3 | 27.5% | 0.939 / 0.939 / 1.000 | 0.903 / 0.930 |

| Dataset | Confidence when right G / C | Confidence when wrong G / C | Wrong with confidence ≥ 0.9, G / C |
|---|---|---|---|
| Adult | 0.848 / 0.900 | 0.679 / 0.747 | 8.2% / 23.6% |
| Echocardiogram | 0.828 / 1.000 | 0.829 / 1.000 | 15.9% / 100.0% |
| Iris | 0.855 / 1.000 | 0.802 / 1.000 | 18.5% / 100.0% |
| Spambase | 0.908 / 0.971 | 0.762 / 0.940 | 26.3% / 76.1% |
| Wine | 0.857 / 1.000 | 0.809 / 1.000 | 36.7% / 100.0% |

Confidence = the predicted probability of the chosen class (GradTree: from its leaf logits; CART: the class share in its leaf).

### Where GradTree and CART go wrong (search settings, test rows of all 10 trials)

| Dataset | Test rows | Error G / C / RF | Both wrong | Only G wrong | Only C wrong | G's errors RF also misses | C's errors RF also misses |
|---|---|---|---|---|---|---|---|
| Adult | 65130 | 18.1% / 15.7% / 14.8% | 10.6% | 7.5% | 5.1% | 54.2% | 68.0% |
| Echocardiogram | 270 | 25.6% / 27.0% / 23.0% | 15.6% | 10.0% | 11.5% | 44.9% | 69.9% |
| Iris | 300 | 7.7% / 6.3% / 5.7% | 5.3% | 2.3% | 1.0% | 65.2% | 73.7% |
| Spambase | 9210 | 11.4% / 9.1% / 4.6% | 4.6% | 6.9% | 4.6% | 27.6% | 40.1% |
| Wine | 360 | 6.4% / 7.8% / 1.9% | 3.1% | 3.3% | 4.7% | 26.1% | 17.9% |

Error = share of test rows predicted wrongly (G = GradTree, C = CART, RF = random forest).

| Dataset | Class | Share of test | Recall G / C / RF | Precision G / C |
|---|---|---|---|---|
| Adult | <=50K | 75.8% | 0.872 / 0.949 / 0.932 | 0.887 / 0.859 |
| Adult | >50K | 24.2% | 0.653 / 0.513 / 0.603 | 0.621 / 0.761 |
| Echocardiogram | alive-at-1 = 0 | 77.0% | 0.798 / 0.846 / 0.909 | 0.860 / 0.811 |
| Echocardiogram | alive-at-1 = 1 | 23.0% | 0.565 / 0.339 / 0.306 | 0.455 / 0.396 |
| Iris | Iris-setosa | 33.7% | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 |
| Iris | Iris-versicolor | 29.3% | 0.864 / 0.932 / 0.920 | 0.874 / 0.863 |
| Iris | Iris-virginica | 37.0% | 0.901 / 0.883 / 0.910 | 0.893 / 0.942 |
| Spambase | class 0 | 60.0% | 0.897 / 0.949 / 0.971 | 0.911 / 0.903 |
| Spambase | class 1 | 40.0% | 0.869 / 0.848 / 0.929 | 0.849 / 0.918 |
| Wine | class 1 | 33.3% | 0.967 / 0.942 / 0.992 | 0.959 / 0.934 |
| Wine | class 2 | 39.2% | 0.901 / 0.908 / 0.957 | 0.934 / 0.901 |
| Wine | class 3 | 27.5% | 0.949 / 0.919 / 1.000 | 0.913 / 0.938 |

| Dataset | Confidence when right G / C | Confidence when wrong G / C | Wrong with confidence ≥ 0.9, G / C |
|---|---|---|---|
| Adult | 0.862 / 0.874 | 0.726 / 0.715 | 12.5% / 11.0% |
| Echocardiogram | 0.893 / 0.863 | 0.876 / 0.808 | 65.2% / 32.9% |
| Iris | 0.973 / 0.980 | 0.867 / 0.833 | 65.2% / 57.9% |
| Spambase | 0.907 / 0.942 | 0.768 / 0.818 | 27.4% / 51.5% |
| Wine | 0.983 / 0.996 | 0.964 / 0.940 | 95.7% / 85.7% |

Confidence = the predicted probability of the chosen class (GradTree: from its leaf logits; CART: the class share in its leaf).

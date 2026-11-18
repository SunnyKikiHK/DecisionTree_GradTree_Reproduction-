### Where GradTree and CART go wrong (tuned settings, test rows of all 10 trials)

| Dataset | Test rows | Error G / C / RF | Both wrong | Only G wrong | Only C wrong | G's errors RF also misses | C's errors RF also misses |
|---|---|---|---|---|---|---|---|
| Adult | 65130 | 17.5% / 15.4% / 14.8% | 11.0% | 6.5% | 4.3% | 57.8% | 70.9% |
| Echocardiogram | 270 | 27.0% / 30.4% / 23.0% | 17.8% | 9.3% | 12.6% | 61.6% | 64.6% |
| Iris | 300 | 9.3% / 6.0% / 5.7% | 4.3% | 5.0% | 1.7% | 53.6% | 77.8% |
| Spambase | 9210 | 9.1% / 7.9% / 4.6% | 4.1% | 5.0% | 3.8% | 35.6% | 44.6% |
| Wine | 360 | 5.8% / 9.7% / 1.9% | 2.2% | 3.6% | 7.5% | 23.8% | 14.3% |

Error = share of test rows predicted wrongly (G = GradTree, C = CART, RF = random forest).

| Dataset | Class | Share of test | Recall G / C / RF | Precision G / C |
|---|---|---|---|---|
| Adult | <=50K | 75.8% | 0.920 / 0.944 / 0.932 | 0.859 / 0.865 |
| Adult | >50K | 24.2% | 0.526 / 0.541 / 0.603 | 0.679 / 0.755 |
| Echocardiogram | alive-at-1 = 0 | 77.0% | 0.832 / 0.827 / 0.909 | 0.820 / 0.789 |
| Echocardiogram | alive-at-1 = 1 | 23.0% | 0.387 / 0.258 / 0.306 | 0.407 / 0.308 |
| Iris | Iris-setosa | 33.7% | 0.960 / 1.000 / 1.000 | 0.990 / 1.000 |
| Iris | Iris-versicolor | 29.3% | 0.852 / 0.932 / 0.920 | 0.833 / 0.872 |
| Iris | Iris-virginica | 37.0% | 0.901 / 0.892 / 0.910 | 0.893 / 0.943 |
| Spambase | Class = 0 | 60.0% | 0.938 / 0.947 / 0.971 | 0.912 / 0.923 |
| Spambase | Class = 1 | 40.0% | 0.864 / 0.882 / 0.929 | 0.903 / 0.917 |
| Wine | class 1 | 33.3% | 0.975 / 0.908 / 0.992 | 0.944 / 0.893 |
| Wine | class 2 | 39.2% | 0.908 / 0.872 / 0.957 | 0.941 / 0.891 |
| Wine | class 3 | 27.5% | 0.949 / 0.939 / 1.000 | 0.940 / 0.930 |

| Dataset | Confidence when right G / C | Confidence when wrong G / C | Wrong with confidence ≥ 0.9, G / C |
|---|---|---|---|
| Adult | 0.890 / 0.887 | 0.744 / 0.706 | 17.0% / 12.2% |
| Echocardiogram | 0.918 / 0.992 | 0.878 / 0.951 | 63.0% / 85.4% |
| Iris | 0.985 / 0.984 | 0.957 / 0.893 | 89.3% / 72.2% |
| Spambase | 0.919 / 0.971 | 0.777 / 0.940 | 24.5% / 76.1% |
| Wine | 0.813 / 0.998 | 0.758 / 0.933 | 0.0% / 82.9% |

Confidence = the predicted probability of the chosen class (GradTree: from its leaf logits; CART: the class share in its leaf).

### Leave-one-out encoding and CART's train F1 (Adult, 10 trials)

| CART fitted on | Train F1 | Test F1 |
|---|---|---|
| All features (tuned settings) | 0.979 ± 0.001 | 0.766 ± 0.011 |
| Numeric features only (tuned settings) | 0.786 ± 0.002 | 0.736 ± 0.006 |
| `sex` only (depth 10) | 0.966 ± 0.001 | 0.431 ± 0.001 |

`sex` has 2 codes in the test data but 167 distinct codes in the training data on average (leave-one-out codes differ by label, and SMOTE interpolates between them). A two-valued feature cannot legitimately give a train F1 far above its test F1, so the gap comes from the encoding, not from the tree.

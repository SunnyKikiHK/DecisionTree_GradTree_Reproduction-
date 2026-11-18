"""Small NumPy metrics (avoiding a mandatory scikit-learn dependency).

The preprocessing notebooks already use scikit-learn, but keeping the core
model package dependent only on NumPy/PyTorch makes it easy to run the
self-test in a minimal environment.  The definitions match the metrics used
by the official code: accuracy and macro-averaged F1.
"""

from __future__ import annotations

import numpy as np


def _as_1d_int(values) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int64)
    return array


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of correctly classified examples."""
    true = _as_1d_int(y_true)
    pred = _as_1d_int(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        return 0.0
    return float(np.mean(true == pred))


def confusion_matrix(y_true, y_pred, num_classes: int | None = None) -> np.ndarray:
    """Confusion matrix with rows = true class, columns = predicted class."""
    true = _as_1d_int(y_true)
    pred = _as_1d_int(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if num_classes is None:
        num_classes = int(max(true.max(initial=0), pred.max(initial=0))) + 1
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def macro_f1_score(y_true, y_pred, num_classes: int | None = None) -> float:
    """Macro-averaged F1 over ``num_classes`` classes.

    Classes with no true and no predicted examples contribute F1=0.  This
    matches the official code's ``F1Score(average='macro', num_classes=C)``
    metric, which averages over all known classes.
    """
    true = _as_1d_int(y_true)
    pred = _as_1d_int(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if num_classes is None:
        num_classes = int(max(true.max(initial=0), pred.max(initial=0))) + 1
    if num_classes < 1:
        return 0.0

    matrix = confusion_matrix(true, pred, num_classes=num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(f1.mean())

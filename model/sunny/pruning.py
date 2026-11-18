"""Hard decision-tree extraction and post-hoc pruning.

GradTree is trained as a complete binary tree.  At any point the learned
parameters can be converted into an ordinary axis-aligned decision tree:

* each internal node uses ``argmax(I[node])`` as its feature;
* its threshold is ``T[node, feature]``;
* each leaf stores its learned logit vector.

The official implementation then prunes branches that receive fewer than
``min_samples`` training examples.  We implement the same pruning rules:

* if both children of a node receive too few samples, merge the node into a
  leaf (the two child leaf logits are averaged);
* if only one child receives too few samples, replace the node by the other
  child's subtree;
* otherwise keep the split.

The code in this module uses NumPy only, so a trained tree can be inspected
without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Leaf:
    """A leaf storing logits (binary: scalar; multi-class: vector)."""

    value: np.ndarray
    num_samples: int = 0

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=np.float64)


@dataclass
class InternalNode:
    """An axis-aligned split: go left when ``x[feature] <= threshold``."""

    feature: int
    threshold: float
    left: "Node"
    right: "Node"
    num_samples: int = 0


Node = Union[Leaf, InternalNode]


class HardDecisionTree:
    """An ordinary decision tree extracted from a trained GradTree.

    Parameters
    ----------
    feature:
        ``(2**depth - 1,)`` integer feature index for every internal node.
    threshold:
        ``(2**depth - 1,)`` threshold for every internal node.
    leaf_values:
        ``(2**depth,)`` logits for binary classification, or
        ``(2**depth, num_classes)`` logits for multi-class classification.
    num_features:
        Optional number of input features.  When given, ``feature`` is checked
        against it; feature indices are indices into ``X``, not into the node
        array, so they are unrelated to ``2**depth - 1``.

    Internal nodes are numbered breadth-first from 0.  A node index
    ``>= 2**depth - 1`` denotes leaf ``index - (2**depth - 1)``.
    """

    def __init__(self, feature, threshold, leaf_values, num_features=None) -> None:
        self.feature = np.asarray(feature, dtype=np.int64).reshape(-1)
        self.threshold = np.asarray(threshold, dtype=np.float64).reshape(-1)
        self.leaf_values = np.asarray(leaf_values, dtype=np.float64)

        if self.feature.shape != self.threshold.shape:
            raise ValueError("feature and threshold must have the same shape")
        num_internal = int(self.feature.size)
        if num_internal < 1:
            raise ValueError("the tree must have at least one internal node")
        if self.leaf_values.ndim not in (1, 2):
            raise ValueError("leaf_values must be 1-D (binary) or 2-D (multi-class)")
        num_leaves = int(self.leaf_values.shape[0])
        if num_leaves != num_internal + 1:
            raise ValueError(
                "a complete binary tree must satisfy num_leaves == num_internal + 1; "
                f"got {num_leaves} leaves and {num_internal} internal nodes"
            )
        depth = int(round(np.log2(num_leaves)))
        if 2**depth != num_leaves:
            raise ValueError(f"num_leaves must be a power of two, got {num_leaves}")
        if num_internal != 2**depth - 1:
            raise ValueError(
                f"expected {2**depth - 1} internal nodes for depth {depth}, "
                f"got {num_internal}"
            )
        if self.leaf_values.ndim == 2 and self.leaf_values.shape[1] < 2:
            raise ValueError("multi-class leaf_values must have at least two columns")

        self.depth = depth
        self.num_internal = num_internal
        self.num_leaves = num_leaves
        self.num_classes = 2 if self.leaf_values.ndim == 1 else int(self.leaf_values.shape[1])

        # ``feature`` holds *input feature* indices (0 .. num_features - 1), not
        # node indices, so it must be validated against the width of X.  A
        # shallow tree has fewer internal nodes than the data has features
        # (depth 2 -> 3 nodes), so comparing against num_internal rejects valid
        # trees for any dataset with more than 2**depth - 1 features.
        if self.feature.min() < 0:
            raise ValueError(
                f"feature indices must be non-negative, got {int(self.feature.min())}"
            )
        if num_features is not None and self.feature.max() >= int(num_features):
            raise ValueError(
                f"feature index {int(self.feature.max())} is outside the "
                f"{int(num_features)} input features"
            )
        self.root: Node = self._build(0)

    # Construction
    def _build(self, node_index: int) -> Node:
        if node_index >= self.num_internal:
            return Leaf(self.leaf_values[node_index - self.num_internal].copy())
        return InternalNode(
            feature=int(self.feature[node_index]),
            threshold=float(self.threshold[node_index]),
            left=self._build(2 * node_index + 1),
            right=self._build(2 * node_index + 2),
        )

    # Inference
    def leaf_values_for(self, X) -> np.ndarray:
        """Return the leaf logits reached by every row of ``X``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        n = X.shape[0]
        out_shape = (n,) + self.leaf_values.shape[1:] # binary: (n, ), multi-class: (n, C)
        out = np.empty(out_shape, dtype=np.float64)
        if n == 0:
            return out

        stack: list[tuple[Node, np.ndarray]] = [(self.root, np.arange(n))]
        while stack:
            node, indices = stack.pop()
            if indices.size == 0:
                continue
            if isinstance(node, Leaf):
                out[indices] = node.value
                continue
            values = X[indices, node.feature]
            go_left = values <= node.threshold
            stack.append((node.left, indices[go_left]))
            stack.append((node.right, indices[~go_left]))
        return out

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities.

        Binary: sigmoid of the leaf logit.
        Multi-class: softmax over the leaf logits.
        """
        values = self.leaf_values_for(X)
        if values.ndim == 1:
            # Numerically stable logistic.
            probabilities = np.empty_like(values)
            non_negative = values >= 0.0
            probabilities[non_negative] = 1.0 / (
                1.0 + np.exp(-values[non_negative])
            )
            exp_values = np.exp(values[~non_negative])
            probabilities[~non_negative] = exp_values / (1.0 + exp_values)
            return probabilities
        shifted = values - values.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Hard class predictions."""
        values = self.leaf_values_for(X)
        if values.ndim == 1:
            return (values > 0.0).astype(np.int64)
        return values.argmax(axis=1).astype(np.int64)

    # Pruning
    def prune(self, X, min_samples: int | float = 1) -> "HardDecisionTree":
        """Prune branches with fewer than ``min_samples`` training examples.

        X: training features, shape (n_samples, F). It is used only to count how many rows reach each branch.
        min_samples: minimum branch size

        A value in ``(0, 1)`` is treated as a fraction of the dataset size,
        matching the official implementation.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        n = X.shape[0]

        if min_samples < 1:
            min_samples = max(1, int(round(n * float(min_samples))))
        else: # min_samples == 1 only remove empty branches
            min_samples = max(1, int(min_samples))

        def build(node_index: int, indices: np.ndarray) -> Node:
            """
            indices (m) are the positions of the training rows that reach this node.
            """
            if node_index >= self.num_internal:
                # node_index - N converts global node number to a leaf index
                # len(indices) for recording how many training samples reached that leaf.
                return Leaf(self.leaf_values[node_index - self.num_internal].copy(), len(indices))

            feature = int(self.feature[node_index]) #(), the feature chosen in this node
            threshold = float(self.threshold[node_index]) #()
            values = X[indices, feature] # (m,) get feature of x at indices, X.shape = (n_samples, F)
            go_left = values <= threshold
            left_indices = indices[go_left] # (n_left,)
            right_indices = indices[~go_left] # (m_right,)

            left = build(2 * node_index + 1, left_indices)
            right = build(2 * node_index + 2, right_indices)

            left_too_small = len(left_indices) < min_samples
            right_too_small = len(right_indices) < min_samples

            if left_too_small and right_too_small:
                if not (isinstance(left, Leaf) and isinstance(right, Leaf)):
                    raise RuntimeError(
                        "pruning reached a node whose children are not both leaves; "
                        "this should not happen with a bottom-up recursion"
                    )
                return Leaf(0.5 * (left.value + right.value), len(indices)) # cut here
            if left_too_small:
                return right
            if right_too_small:
                return left
            return InternalNode(feature, threshold, left, right, len(indices))

        self.root = build(0, np.arange(n))
        return self

    # Inspection
    def count_nodes(self) -> tuple[int, int]:
        """Return ``(number_of_internal_nodes, number_of_leaves)`` after pruning."""
        internal = 0
        leaves = 0
        stack: list[Node] = [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, Leaf):
                leaves += 1
            else:
                internal += 1
                stack.append(node.left)
                stack.append(node.right)
        return internal, leaves

    @property
    def size(self) -> int:
        internal, leaves = self.count_nodes()
        return internal + leaves

    def summary(self) -> dict[str, int]:
        internal, leaves = self.count_nodes()

        def max_depth(node: Node, depth: int = 0) -> int:
            if isinstance(node, Leaf):
                return depth
            return max(max_depth(node.left, depth + 1), max_depth(node.right, depth + 1))

        return {
            "full_depth": self.depth,
            "max_depth": max_depth(self.root),
            "num_classes": self.num_classes,
            "internal_nodes": internal,
            "leaves": leaves,
            "total_nodes": internal + leaves,
        }

    def to_dict(self) -> dict:
        """Export the (possibly pruned) tree as nested dictionaries."""

        def convert(node: Node) -> dict:
            if isinstance(node, Leaf):
                value = node.value
                return {
                    "type": "leaf",
                    "num_samples": int(node.num_samples),
                    "value": float(value) if value.ndim == 0 or value.size == 1 else value.tolist(),
                }
            return {
                "type": "internal",
                "feature": int(node.feature),
                "threshold": float(node.threshold),
                "num_samples": int(node.num_samples),
                "left": convert(node.left),
                "right": convert(node.right),
            }

        return convert(self.root)

    def print_tree(self, max_depth: int | None = None) -> str:
        """Return a human-readable text representation of the tree."""
        lines: list[str] = []

        def visit(node: Node, depth: int) -> None:
            indent = "  " * depth
            if isinstance(node, Leaf):
                lines.append(f"{indent}leaf: logits={node.value.tolist()}")
                return
            lines.append(
                f"{indent}if x[{node.feature}] <= {node.threshold:.6g}:"
            )
            if max_depth is not None and depth >= max_depth:
                lines.append(f"{indent}  ...")
                return
            visit(node.left, depth + 1)
            lines.append(f"{indent}else:")
            visit(node.right, depth + 1)

        visit(self.root, 0)
        return "\n".join(lines)

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            f"HardDecisionTree(max_depth={summary['max_depth']}, "
            f"internal_nodes={summary['internal_nodes']}, leaves={summary['leaves']}, "
            f"num_classes={summary['num_classes']})"
        )

"""The trained GradTree as an ordinary decision tree, with the official post-hoc pruning
(DecisionTree in experiments_paper_gradtree/utilities/GDT.py)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Node:
    feature: int = -1
    threshold: float = 0.0
    left: Node | None = None
    right: Node | None = None
    value: np.ndarray | None = None  # leaf logits; None for internal nodes
    n_samples: int = 0               # training samples reaching the node (set by prune)

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class HardTree:
    """A sample goes left when x[feature] < threshold, exactly like GradTree's forward pass."""

    def __init__(self, features: np.ndarray, thresholds: np.ndarray, leaf_values: np.ndarray) -> None:
        n_internal = len(features)

        def build(index: int) -> Node:
            if index >= n_internal:
                return Node(value=np.asarray(leaf_values[index - n_internal], dtype=np.float64))
            return Node(int(features[index]), float(thresholds[index]),
                        build(2 * index + 1), build(2 * index + 2))

        self.root = build(0)
        self.value_shape = np.shape(leaf_values[0])

    @classmethod
    def from_model(cls, model) -> HardTree:
        with torch.no_grad():
            features = model.selected_features().argmax(dim=1)
            thresholds = model.split_values.gather(1, features.unsqueeze(1)).squeeze(1)
            return cls(features.cpu().numpy(), thresholds.cpu().numpy(), model.leaf_values.cpu().numpy())

    def leaf_logits(self, X: np.ndarray) -> np.ndarray:
        """Logits of the leaf each row reaches: (n,) for 2 classes, (n, C) otherwise."""
        out = np.empty((len(X),) + self.value_shape)
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if node.is_leaf:
                out[rows] = node.value
                continue
            left = X[rows, node.feature] < node.threshold
            stack.append((node.left, rows[left]))
            stack.append((node.right, rows[~left]))
        return out

    def prune(self, X: np.ndarray, min_samples: float = 1) -> None:
        """Remove branches that fewer than min_samples training rows reach (official prunePostHoc).

        A node whose left branch is empty is replaced by its right child, and vice versa, so test
        samples that would land in an empty region follow the populated branch instead.
        """
        if min_samples < 1:
            min_samples = max(1, len(X) * min_samples)
        self._count(self.root, X)
        self._prune(self.root, X, min_samples)

    def _rows_left(self, node: Node, X: np.ndarray) -> np.ndarray:
        return X[:, node.feature] < node.threshold

    def _count(self, node: Node, X: np.ndarray) -> None:
        node.n_samples = len(X)
        if node.is_leaf:
            return
        left = self._rows_left(node, X)
        self._count(node.left, X[left])
        self._count(node.right, X[~left])

    def _prune(self, node: Node, X: np.ndarray, min_samples: float) -> None:
        """X holds the training rows that reach node."""
        if node.is_leaf:
            return
        left = self._rows_left(node, X)
        self._prune(node.left, X[left], min_samples)
        self._prune(node.right, X[~left], min_samples)

        left_empty = node.left.n_samples < min_samples
        right_empty = node.right.n_samples < min_samples
        if left_empty and right_empty:
            # Empty subtrees have already collapsed into leaves (children are pruned first)
            node.value = np.mean([node.left.value, node.right.value], axis=0)
            node.left = node.right = None
        elif left_empty:
            self._replace(node, node.right)
        elif right_empty:
            self._replace(node, node.left)
        else:
            return
        # The official code re-counts the whole tree here; only this subtree can have changed
        self._count(node, X)

    @staticmethod
    def _replace(node: Node, child: Node) -> None:
        node.feature, node.threshold = child.feature, child.threshold
        node.left, node.right, node.value = child.left, child.right, child.value

    def count_nodes(self) -> tuple[int, int]:
        """(internal nodes, leaves)."""
        internal = leaves = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves += 1
            else:
                internal += 1
                stack += [node.left, node.right]
        return internal, leaves

    def describe(self, feature_names: list[str] | None = None, class_names: list[str] | None = None) -> str:
        """The tree as nested if/else rules. Thresholds are in the preprocessed (quantile) scale."""
        lines = []

        def label(value: np.ndarray) -> str:
            index = int(value > 0) if np.ndim(value) == 0 else int(np.argmax(value))
            return class_names[index] if class_names else str(index)

        def walk(node: Node, indent: str) -> None:
            if node.is_leaf:
                lines.append(f"{indent}class {label(node.value)}  ({node.n_samples} training samples)")
                return
            name = feature_names[node.feature] if feature_names else f"x[{node.feature}]"
            lines.append(f"{indent}if {name} < {node.threshold:.4g}:")
            walk(node.left, indent + "    ")
            lines.append(f"{indent}else:")
            walk(node.right, indent + "    ")

        walk(self.root, "")
        return "\n".join(lines)

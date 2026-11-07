"""The GradTree model in PyTorch.

Parameters (paper notation in parentheses)
------------------------------------------
``split_index``       (I)  shape (2**d - 1, n_features)
    Real-valued feature scores for every internal node.  The forward pass
    applies entmax (or softmax/sparsemax), then a straight-through hardmax.

``split_threshold``   (T)  shape (2**d - 1, n_features)
    One candidate threshold per internal node and feature.  Only the column
    selected by ``I`` is used by a node.

``leaf_logits``       (L)  shape (2**d,) for binary, or (2**d, n_classes)
    Class logits stored at every leaf.

Shape symbols used in the forward-pass comments:
    d = tree depth
    N = number of internal nodes (2**d - 1)
    F = number of features
    B = batch size
    P = number of leaves (2**d)

The ``L`` in ``leaf_logits (L)`` is the paper's notation for the leaf
parameter, not the leaf count.

The forward pass follows Algorithm 1 of the paper:

1. ``I_hard = ST-hardmax(activation(beta * I))``
2. for every leaf and depth, gather the internal node and branch direction
3. ``s = sigmoid(T_selected - x_selected)``  (hard-routed with ST rounding)
4. multiply the branch probabilities along each root-to-leaf path
5. return the path-weighted leaf logits

The model is a hard axis-aligned tree in the forward pass; entmax and the
sigmoid only provide gradients in the backward pass.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from .entmax import apply_activation, hardmax_st, round_st
from .tree_structure import make_tree_structure


class GradTree(nn.Module):
    """A differentiable, fully grown, axis-aligned decision tree.

    Parameters
    ----------
    num_features:
        Number of input features.
    num_classes:
        Number of classes.  ``2`` uses a single output logit and binary
        cross-entropy; ``> 2`` uses one logit per class.
    depth:
        Tree depth.  The model has ``2**depth - 1`` internal nodes and
        ``2**depth`` leaves.
    activation:
        Feature-selection activation: ``"entmax"`` (default), ``"softmax"``
        or ``"sparsemax"``.
    beta:
        Scale applied to ``I`` before the activation.
    dropout:
        Dropout rate on the feature-selection scores during training.
    init:
        Initialisation scheme: ``"glorot"`` (paper Algorithm 2),
        ``"normal"``, ``"uniform"`` or ``"data_aware"``.  With
        ``"data_aware"``, call :meth:`initialize_thresholds_from_data` after
        construction (the trainer does this automatically).
    output_activation:
        Probability transform used by :meth:`predict_proba` for multi-class
        models.  It does not change :meth:`predict`, because softmax,
        entmax and sparsemax preserve the argmax.
    seed:
        Seed used by :meth:`reset_parameters`.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        depth: int = 3,
        activation: str = "entmax",
        beta: float = 1.0,
        dropout: float = 0.0,
        init: str = "glorot", # how the three learnable tensors are initialized before training.
        output_activation: str = "softmax", # For leaf 
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if num_features < 1:
            raise ValueError("num_features must be >= 1")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 (regression is not implemented)")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if activation not in {"entmax", "softmax", "sparsemax"}:
            raise ValueError(f"Unknown activation {activation!r}")
        if output_activation not in {"softmax", "entmax", "sparsemax"}:
            raise ValueError(f"Unknown output_activation {output_activation!r}")
        if init not in {"glorot", "normal", "uniform", "data_aware"}:
            raise ValueError(f"Unknown init {init!r}")
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.binary = self.num_classes == 2
        self.depth = int(depth)
        self.activation = activation
        self.beta = float(beta)
        self.dropout = float(dropout)
        self.init = init
        self.output_activation = output_activation
        self.seed = seed

        structure = make_tree_structure(self.depth)
        self.num_internal = structure.num_internal
        self.num_leaves = structure.num_leaves

        # Constant path tables; persistent=False keeps them out of checkpoints
        # they are rebuilt from `depth` in __init__, so dont need to save in checkpoint
        self.register_buffer(
            "path_identifier", structure.path_identifier, persistent=False
        )
        self.register_buffer(
            "internal_node_index", structure.internal_node_index, persistent=False
        )

        leaf_shape = (
            (self.num_leaves,)
            if self.binary
            else (self.num_leaves, self.num_classes)
        )
        self.split_index = nn.Parameter(
            torch.empty(self.num_internal, self.num_features) # (N, F)
        )
        self.split_threshold = nn.Parameter(
            torch.empty(self.num_internal, self.num_features) # (N, F)
        )
        self.leaf_logits = nn.Parameter(torch.empty(*leaf_shape)) # (P, C) / # (P)

        self.reset_parameters(seed=seed)

    # Initialisation
    def reset_parameters(self, seed: int | None = None) -> None:
        """Re-initialise all parameters, used at 1. construction AND 2. on restarts)."""
        if seed is not None:
            torch.manual_seed(int(seed))

        if self.init == "glorot": # According to paper's algorithm 2
            bound_index = math.sqrt(6.0 / (self.num_internal + self.num_features))
            bound_leaf = math.sqrt(
                6.0 / (self.num_leaves + (1 if self.binary else self.num_classes))
            )
            nn.init.uniform_(self.split_index, -bound_index, bound_index) # uniform distribution between [-bound_index, bound_index]
            nn.init.uniform_(self.split_threshold, -bound_index, bound_index)
            nn.init.uniform_(self.leaf_logits, -bound_leaf, bound_leaf)
        elif self.init == "normal": # Thresholds start around zero, which is reasonable because the preprocessing quantile-transforms features to roughly Normal(0, 1)
            nn.init.normal_(self.split_index, mean=0.0, std=1.0)
            nn.init.normal_(self.split_threshold, mean=0.0, std=1.0)
            nn.init.normal_(self.leaf_logits, mean=0.0, std=1.0)
        elif self.init == "uniform": 
            nn.init.uniform_(self.split_index, -1.0, 1.0)
            nn.init.uniform_(self.split_threshold, -1.0, 1.0)
            nn.init.uniform_(self.leaf_logits, -1.0, 1.0)
        elif self.init == "data_aware":
            # The official dataAware initialiser uses normal I/L and uniform
            # T; thresholds are then replaced by values from the training set.
            nn.init.normal_(self.split_index, mean=0.0, std=1.0)
            nn.init.uniform_(self.split_threshold, 0.0, 1.0)
            nn.init.normal_(self.leaf_logits, mean=0.0, std=1.0)
        else:  # pragma: no cover - validated in __init__
            raise ValueError(f"Unknown init {self.init!r}")

    @torch.no_grad()
    def initialize_thresholds_from_data(self, X) -> None:
        """Set every threshold to a random value from the training data.

        For every (internal node, feature) pair we draw one row uniformly at
        random and copy that feature's value, matching the official
        ``dataAware`` initialiser.
        """
        if torch.is_tensor(X):
            values = X.detach().to(
                device=self.split_threshold.device, dtype=self.split_threshold.dtype
            )
        else:
            values = torch.as_tensor(
                np.asarray(X),
                dtype=self.split_threshold.dtype,
                device=self.split_threshold.device,
            )
        if values.ndim != 2 or values.shape[1] != self.num_features:
            raise ValueError(
                f"X must have shape (n, {self.num_features}), got {tuple(values.shape)}"
            )
        if values.shape[0] == 0:
            raise ValueError("cannot initialise thresholds from an empty dataset")

        row_indices = torch.randint(
            0,
            values.shape[0],
            (self.num_internal, self.num_features),
            device=values.device,
        ) # (N, F) -> each entry is a random row index in [0, n_samples - 1]
        column_indices = (
            torch.arange(self.num_features, device=values.device)
            .unsqueeze(0)
            .expand(self.num_internal, -1)
        ) # (F,) -> (1, F) -> (N, F)

        # result[n, f] = values[row_indices[n, f], f]
        #  row_indices[n,f] indicates the row sample that we use, 
        # and f is the feature column of the row sample data that used
        self.split_threshold.copy_(values[row_indices, column_indices])

    # Forward pass
    def _feature_distribution(self) -> Tensor: 
        """Soft distribution over features for every internal node (N, F)."""
        scores = self.split_index # (N, F)
        if self.dropout > 0.0 and self.training:
            scores = functional.dropout(scores, p=self.dropout, training=True)
        return apply_activation(self.beta * scores, self.activation) # become probability-like distribution (N, F)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits.

        Binary: shape ``(B,)``.  Multi-class: ``(B, num_classes)``.
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(
                np.asarray(x),
                dtype=self.split_index.dtype,
                device=self.split_index.device,
            )
        else:
            x = x.to(device=self.split_index.device, dtype=self.split_index.dtype)

        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"x must have shape (B, {self.num_features}), got {tuple(x.shape)}"
            )

        # 1. Soft feature distribution -> straight-through one-hot features.
        feature_distribution = self._feature_distribution()  # (N, F)
        # feature_one_hot has one row per internal node, each one is one hot vector with 1 as the selected feature 
        feature_one_hot = hardmax_st(feature_distribution)  # (N, F) 

        # 2. Gather the parameters belonging to every leaf's path.
        index = self.internal_node_index  # (P, d), internel path (size: depth) to reach each leaf
        # Each entry index[p, j] selects one whole row of feature_one_hot, i.e. node transform to the feature selection of the node
        selected_features = feature_one_hot[index]  # (P, d, F)
        selected_thresholds = self.split_threshold[index]  # (P, d, F)

        # 3. Split scores s = sigmoid(tau_selected - x_selected).
        threshold_term = (selected_thresholds * selected_features).sum(dim=-1)  # (P, d), get the selected feature's threshold of each leaf
        feature_term = torch.einsum(
            "bf,ldf->bld", x, selected_features
        )  # (B, P, d), for each data, get the selected feature val in the path of each leaf
        split_soft = torch.sigmoid(threshold_term.unsqueeze(0) - feature_term) # from algorithm 1
        split_hard = round_st(split_soft)  # (B, P, d), straight-through 0/1 routing

        # 4. Probability of each root-to-leaf path.
        path = self.path_identifier.unsqueeze(0)  # (1, P, d), from data.py path_rows
        path_prob = torch.prod( # e.g. path_prob for one sample = [0, 1, 0, 0]
            (1.0 - path) * split_hard + path * (1.0 - split_hard), # alg 1 equ 3
            dim=2,
        )  # (B, P)

        # 5. Path-weighted leaf logits.
        if self.binary:
            return (path_prob * self.leaf_logits.unsqueeze(0)).sum(dim=-1) # (B, P) * (1, P)
        # For each sample b and class c, the model computes the class-c logit as a weighted combination of that class’s logit at every leaf.
        return path_prob @ self.leaf_logits # (B, P) @ (P, C) , only 1 leaf is included for each sample data

    # Inference helpers
    def _output_probabilities(self, logits: Tensor) -> Tensor:
        if self.binary:
            return torch.sigmoid(logits)
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        return apply_activation(logits, self.output_activation)

    @torch.no_grad()
    def predict_proba(self, X) -> Tensor:
        """Class probabilities for ``X`` (not batched; use the trainer for large data)."""
        self.eval()
        logits = self.forward(X)
        return self._output_probabilities(logits)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        """Hard class predictions for ``X`` (not batched)."""
        proba = self.predict_proba(X)
        if proba.numel() == 0:
            return np.empty(0, dtype=np.int64)
        if self.binary:
            return (proba > 0.5).long().cpu().numpy()
        return proba.argmax(dim=-1).cpu().numpy()

    # Hard tree extraction
    @property
    def feature_indices(self) -> Tensor:
        """Selected feature index for every internal node (argmax of I)."""
        return self.split_index.detach().argmax(dim=1)

    def hard_tree(self, prune_X=None, min_samples: int | float = 1):
        """Extract a :class:`~model.sunny.pruning.HardDecisionTree`.

        Parameters
        ----------
        prune_X:
            Training features used to count samples and prune the tree.  If
            ``None``, the returned tree is the full, unpruned tree.
        min_samples:
            Minimum branch size for pruning (see ``HardDecisionTree.prune``).
        """
        from .pruning import HardDecisionTree

        if prune_X is not None and torch.is_tensor(prune_X):
            prune_X = prune_X.detach().cpu().numpy()

        feature = self.feature_indices.cpu().numpy()
        thresholds = self.split_threshold.detach().cpu().numpy()
        threshold = thresholds[np.arange(self.num_internal), feature]
        leaf_values = self.leaf_logits.detach().cpu().numpy()

        tree = HardDecisionTree(feature, threshold, leaf_values)
        if prune_X is not None:
            tree.prune(prune_X, min_samples=min_samples)
        return tree

    # Introspection
    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, num_classes={self.num_classes}, "
            f"depth={self.depth}, internal_nodes={self.num_internal}, "
            f"leaves={self.num_leaves}, activation={self.activation!r}, "
            f"beta={self.beta}, dropout={self.dropout}, init={self.init!r}, "
            f"parameters={self.num_parameters}"
        )

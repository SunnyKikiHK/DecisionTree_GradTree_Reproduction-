"""GradTree with other ways of choosing each node's feature, for the ablation (paper Table 4).

GradTree picks a node's feature with entmax followed by a straight-through hardmax. The paper
compares this with two alternatives from the literature; we add plain softmax to isolate the
effect of entmax's sparsity:

    entmax     entmax15 + straight-through hardmax (GradTree, unchanged)
    softmax    softmax + straight-through hardmax
    gumbel     straight-through Gumbel-softmax (Jang et al., 2016): Gumbel noise is added to the
               scores before a softmax with temperature tau, then the straight-through hardmax
    annealing  temperature annealing (Chang et al., 2021): entmax15(scores / T) with no hardmax,
               where T falls from 1 to min_temperature over anneal_epochs epochs, so the soft
               choice gradually becomes one-hot

The paper does not give the settings of the two alternatives, so tau, anneal_epochs and
min_temperature are our choices. Every variant evaluates (validation loss, the extracted tree,
predictions) with the hard argmax of the scores, so all four are compared as the same kind of
hard, axis-aligned tree.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from model.myat.gradtree.classifier import GradTreeClassifier
from model.myat.gradtree.gradtree import GradTree, entmax15, hardmax_st

FEATURE_SELECTION = ["entmax", "softmax", "gumbel", "annealing"]


class FeatureSelectionGradTree(GradTree):
    def __init__(self, n_features: int, n_classes: int, depth: int, method: str, steps_per_epoch: int = 1,
                 gumbel_tau: float = 1.0, anneal_epochs: int = 200, min_temperature: float = 0.01) -> None:
        super().__init__(n_features, n_classes, depth)
        if method not in FEATURE_SELECTION:
            raise ValueError(f"Unknown feature selection {method!r}")
        self.method = method
        self.steps_per_epoch = steps_per_epoch
        self.gumbel_tau = gumbel_tau
        self.anneal_epochs = anneal_epochs
        self.min_temperature = min_temperature

    @torch.no_grad()
    def reset_parameters(self, seed: int | None = None) -> None:
        super().reset_parameters(seed)
        self.steps = 0  # training steps since this restart, for the annealing schedule
        self.noise = torch.Generator().manual_seed(seed if seed is not None else 0)  # Gumbel noise

    def temperature(self) -> float:
        """1 at the start, falling geometrically to min_temperature after anneal_epochs epochs."""
        progress = min(1.0, (self.steps // self.steps_per_epoch) / self.anneal_epochs)
        return self.min_temperature ** progress

    def selected_features(self) -> Tensor:
        scores = self.beta * self.split_index
        if not (self.training and torch.is_grad_enabled()):
            # Validation, tree extraction and prediction: the hard choice, without noise
            return F.one_hot(scores.argmax(dim=-1), scores.shape[-1]).to(scores.dtype)

        self.steps += 1
        if self.method == "entmax":
            return hardmax_st(entmax15(scores))
        if self.method == "softmax":
            return hardmax_st(torch.softmax(scores, dim=-1))
        if self.method == "gumbel":
            uniform = torch.rand(scores.shape, generator=self.noise).to(scores.device).clamp(1e-10, 1 - 1e-7)
            gumbel = -torch.log(-torch.log(uniform))  # standard Gumbel noise
            return hardmax_st(torch.softmax((scores + gumbel) / self.gumbel_tau, dim=-1))
        return entmax15(scores / self.temperature())  # annealing: soft during training


class FeatureSelectionClassifier(GradTreeClassifier):
    """GradTreeClassifier whose trees use another feature selection. 'entmax' is plain GradTree."""

    def __init__(self, feature_selection: str = "entmax", gumbel_tau: float = 1.0, anneal_epochs: int = 200,
                 min_temperature: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_selection = feature_selection
        self.gumbel_tau = gumbel_tau
        self.anneal_epochs = anneal_epochs
        self.min_temperature = min_temperature

    def fit(self, X_train, y_train, X_valid, y_valid):
        batch_size = min(self.batch_size, math.ceil(len(X_train) / 2))  # as in GradTreeClassifier.fit
        self._steps_per_epoch = math.ceil(len(X_train) / batch_size)
        return super().fit(X_train, y_train, X_valid, y_valid)

    def _make_model(self, n_features: int) -> GradTree:
        if self.feature_selection == "entmax":
            return super()._make_model(n_features)
        return FeatureSelectionGradTree(
            n_features, self.n_classes_, self.depth, self.feature_selection, self._steps_per_epoch,
            self.gumbel_tau, self.anneal_epochs, self.min_temperature,
        )

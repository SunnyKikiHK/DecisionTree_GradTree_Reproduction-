"""Hyperparameters for the PyTorch GradTree implementation.

The defaults below follow the official TensorFlow implementation's
``GDTs-BINARY-DEFAULT.ipynb`` configuration where possible, but use fewer
epochs / restarts so a first run on the course datasets is practical.  Use
:meth:`GradTreeConfig.paper_default` for the full paper configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass
class GradTreeConfig:
    """Configuration for :class:`model.sunny.gradtree.GradTree`.

    Parameters
    ----------
    depth:
        Depth of the complete binary tree.  A tree of depth ``d`` has
        ``2**d - 1`` internal nodes and ``2**d`` leaves.
    lr_index, lr_values, lr_leaf:
        Adam learning rates for the feature-selection matrix ``I``, the
        threshold matrix ``T`` and the leaf logits ``L``.  The paper uses
        separate rates for the three parameter groups.
    batch_size:
        Mini-batch size.  The official default configuration uses 128.
    epochs:
        Maximum number of epochs per restart.
    patience:
        Early-stopping patience, measured in epochs without an improvement
        in validation loss (or training loss when no validation set is
        supplied).  ``0`` disables early stopping.
    restarts:
        Number of random restarts *after* the first run.  The official code
        uses ``restarts=5`` (six runs in total).
    activation:
        Feature-selection activation: ``"entmax"`` (paper default),
        ``"softmax"`` or ``"sparsemax"``.
    beta:
        Scaling factor applied to ``I`` before the activation.  The official
        code calls this ``split_index_activation_beta`` and defaults to 1.0.
    dropout:
        Dropout rate applied to ``I`` during training (before the activation).
    init:
        Parameter initialisation: ``"glorot"`` (paper Algorithm 2),
        ``"normal"``, ``"uniform"`` or ``"data_aware"``.  With
        ``"data_aware"`` the trainer replaces the thresholds with random
        values sampled from the training features.
    loss:
        ``"crossentropy"`` (default), ``"focal"`` or ``"poly"``.
    focal_gamma, focal_alpha:
        Focal-loss parameters.  The official binary default uses
        ``alpha=0.5`` and ``gamma=2`` (the notebook uses 3).
    poly_epsilon:
        Epsilon for PolyLoss (``epsilon=2`` in the paper).
    output_activation:
        Activation used by ``predict_proba`` for multi-class models:
        ``"softmax"`` (default), ``"entmax"`` or ``"sparsemax"``.  It does
        not affect ``predict`` because all three preserve the argmax.  The
        training loss uses softmax cross-entropy (the paper's default); this
        option is intended for prediction-time ablations.
    prune:
        Whether to build and post-hoc prune the hard decision tree after
        training.
    prune_threshold:
        Minimum number of training samples required for a branch.  ``1``
        matches the paper's pruning threshold.  A value in ``(0, 1)`` is
        interpreted as a fraction of the training set.
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"`` (or any valid torch device string).
    seed:
        Base random seed.  Restart ``r`` uses ``seed + r``.
    verbose, log_every:
        Progress-printing controls for the trainer.
    """

    depth: int = 5
    lr_index: float = 0.05
    lr_values: float = 0.01
    lr_leaf: float = 0.005

    batch_size: int = 128
    epochs: int = 500
    patience: int = 50
    restarts: int = 0

    activation: str = "entmax"
    beta: float = 1.0
    dropout: float = 0.0
    init: str = "glorot"

    loss: str = "crossentropy"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    poly_epsilon: float = 2.0

    output_activation: str = "softmax"

    prune: bool = True
    prune_threshold: int | float = 1

    device: str = "auto"
    seed: int = 42
    verbose: bool = True
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if self.restarts < 0:
            raise ValueError("restarts must be >= 0")
        if self.activation not in {"entmax", "softmax", "sparsemax"}:
            raise ValueError(
                "activation must be 'entmax', 'softmax' or 'sparsemax', "
                f"got {self.activation!r}"
            )
        if self.output_activation not in {"softmax", "entmax", "sparsemax"}:
            raise ValueError(
                "output_activation must be 'softmax', 'entmax' or 'sparsemax', "
                f"got {self.output_activation!r}"
            )
        if self.init not in {"glorot", "normal", "uniform", "data_aware"}:
            raise ValueError(
                "init must be 'glorot', 'normal', 'uniform' or 'data_aware', "
                f"got {self.init!r}"
            )
        if self.loss not in {"crossentropy", "focal", "poly"}:
            raise ValueError(
                "loss must be 'crossentropy', 'focal' or 'poly', "
                f"got {self.loss!r}"
            )
        for name in ("lr_index", "lr_values", "lr_leaf"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.prune_threshold < 0:
            raise ValueError("prune_threshold must be >= 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def paper_default(cls, **overrides: Any) -> "GradTreeConfig":
        """Return the official default-notebook configuration.

        The notebook uses depth 10, batch size 128, 10,000 epochs, patience
        200 and five restarts, with Adam + stochastic weight averaging.  SWA
        is not implemented here; everything else is reproduced.  Any keyword
        argument overrides the corresponding field.
        """
        base = cls(
            depth=10,
            lr_index=0.05,
            lr_values=0.01,
            lr_leaf=0.005,
            batch_size=128,
            epochs=10_000,
            patience=200,
            restarts=5,
            activation="entmax",
            init="glorot",
            loss="crossentropy",
            prune=True,
            prune_threshold=1,
        )
        return replace(base, **overrides) if overrides else base

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of the configuration."""
        return asdict(self)

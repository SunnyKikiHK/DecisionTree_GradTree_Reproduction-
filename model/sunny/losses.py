"""Loss functions used by GradTree.

The official implementation trains binary models with binary cross-entropy
and multi-class models with categorical cross-entropy.  It also optionally
uses focal loss or PolyLoss during hyperparameter search.  All three are
implemented here so the same model can reproduce the paper's variants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GradTreeLoss(nn.Module):
    """Cross-entropy, focal loss or PolyLoss for binary / multi-class logits.

    Parameters
    ----------
    num_classes:
        Number of classes.  ``2`` means the model outputs one logit per
        example (binary); values greater than 2 mean ``(batch, classes)``
        logits.
    kind:
        ``"crossentropy"``, ``"focal"`` or ``"poly"``.
    focal_gamma, focal_alpha:
        Focal-loss parameters.  ``focal_alpha`` is only used for binary
        classification; the official multi-class focal loss has no alpha.
    poly_epsilon:
        PolyLoss epsilon.
    """

    def __init__(
        self,
        num_classes: int,
        kind: str = "crossentropy",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.5,
        poly_epsilon: float = 2.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if kind not in {"crossentropy", "focal", "poly"}:
            raise ValueError(f"Unknown loss {kind!r}")
        self.num_classes = num_classes
        self.kind = kind
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.poly_epsilon = poly_epsilon

    @property
    def binary(self) -> bool:
        return self.num_classes == 2

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.binary:
            return self._binary(logits, targets)
        return self._multiclass(logits, targets)

    def _binary(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if logits.ndim != 1:
            raise ValueError(
                f"binary logits must have shape (batch,), got {tuple(logits.shape)}"
            )
        target = targets.to(dtype=logits.dtype).reshape(-1)
        if target.shape != logits.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match "
                f"logits shape {tuple(logits.shape)}"
            )

        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.kind == "crossentropy":
            return ce.mean()

        prob = torch.sigmoid(logits)
        prob_true = torch.where(target > 0.5, prob, 1.0 - prob)
        if self.kind == "focal":
            alpha_true = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
            return (alpha_true * (1.0 - prob_true).pow(self.focal_gamma) * ce).mean()

        # PolyLoss: the cross-entropy plus epsilon * (1 - p_true).
        return (ce + self.poly_epsilon * (1.0 - prob_true)).mean()

    def _multiclass(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(
                f"multi-class logits must have shape (batch, classes), got {tuple(logits.shape)}"
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} logits per example, got {logits.shape[1]}"
            )
        target = targets.long().reshape(-1)
        if target.shape[0] != logits.shape[0]:
            raise ValueError("targets and logits have different batch sizes")

        ce = F.cross_entropy(logits, target, reduction="none")
        if self.kind == "crossentropy":
            return ce.mean()

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob_true = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        prob_true = log_prob_true.exp()
        if self.kind == "focal":
            return ((1.0 - prob_true).pow(self.focal_gamma) * ce).mean()

        return (ce + self.poly_epsilon * (1.0 - prob_true)).mean()

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, kind={self.kind!r}, "
            f"focal_gamma={self.focal_gamma}, focal_alpha={self.focal_alpha}, "
            f"poly_epsilon={self.poly_epsilon}"
        )

"""Sparse activations and straight-through estimators used by GradTree.

``entmax15`` and ``sparsemax`` come from the MIT-licensed ``entmax`` package
(https://github.com/deep-spin/entmax), which provides PyTorch implementations
with autograd support. This module wraps them so the rest of the code uses a
single last-dimension interface.

The straight-through hardmax and rounding are GradTree-specific and are kept
here because the ``entmax`` package does not provide them.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from entmax import entmax15 as _entmax15
    from entmax import sparsemax as _sparsemax
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "GradTree requires the 'entmax' package. "
        "Install it with `pip install entmax`."
    ) from exc


def entmax15(z: Tensor) -> Tensor:
    """1.5-entmax over the last dimension (from the ``entmax`` package)."""
    return _entmax15(z, dim=-1)


def sparsemax(z: Tensor) -> Tensor:
    """Sparsemax over the last dimension (from the ``entmax`` package)."""
    return _sparsemax(z, dim=-1)


def apply_activation(z: Tensor, kind: str) -> Tensor:
    """Apply ``softmax``, ``entmax`` or ``sparsemax`` over the last dimension."""
    if kind == "softmax":
        return torch.softmax(z, dim=-1)
    if kind == "entmax":
        return entmax15(z)
    if kind == "sparsemax":
        return sparsemax(z)
    raise ValueError(f"Unknown activation {kind!r}")


def hardmax_st(x: Tensor) -> Tensor:
    """Straight-through hardmax.

    Forward: one-hot argmax.
    Backward: identity with respect to ``x``.
    """
    hard = torch.zeros_like(x) # (N, F)
    hard.scatter_(-1, x.argmax(dim=-1, keepdim=True), 1.0) # argmax -> (1,N) -> (N, F)
    return x + (hard - x).detach() # do not return hard directly, so upstream graident flows straight into x


def round_st(x: Tensor) -> Tensor:
    """Straight-through rounding to the nearest integer.

    Forward: ``round(x)``.
    Backward: identity with respect to ``x`` (so the sigmoid derivative is
    used in the backward pass, as in the paper).
    """
    return x + (torch.round(x) - x).detach()

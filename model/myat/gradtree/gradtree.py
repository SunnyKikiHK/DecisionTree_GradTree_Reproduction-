"""GradTree (Marton et al., AAAI 2024) in PyTorch: the tree, its forward pass and its loss.

Shapes: d = depth, N = 2**d - 1 internal nodes, P = 2**d leaves, F = features, C = classes, B = batch.

Parameters (paper Section 3, official experiments_paper_gradtree/utilities/GDT.py):
    split_index   I  (N, F)  feature scores of every internal node
    split_values  T  (N, F)  one threshold per internal node and feature; a node only uses the one
                             of the feature it selects
    leaf_values   L  (P,) for 2 classes, (P, C) otherwise: the logits stored at every leaf

Internal nodes are numbered breadth-first from the root (0), so the children of node n are 2n+1 and
2n+2. Leaves are numbered left to right. A sample goes left when x[feature] < threshold.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _Entmax15(torch.autograd.Function):
    """1.5-entmax over the last dimension (Peters et al., 2019), with its closed-form gradient."""

    @staticmethod
    def forward(ctx, z: Tensor) -> Tensor:
        z = z / 2
        z = z - z.max(dim=-1, keepdim=True).values
        z_sorted = torch.sort(z, dim=-1, descending=True).values
        rho = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype, device=z.device)
        mean = z_sorted.cumsum(dim=-1) / rho
        mean_sq = (z_sorted ** 2).cumsum(dim=-1) / rho
        delta = (1 - rho * (mean_sq - mean ** 2)) / rho
        tau = mean - torch.sqrt(torch.clamp(delta, min=0))
        support = (tau <= z_sorted).sum(dim=-1, keepdim=True)
        p = torch.clamp(z - tau.gather(-1, support - 1), min=0) ** 2
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        (p,) = ctx.saved_tensors
        sqrt_p = p.sqrt()
        grad_z = grad * sqrt_p
        return grad_z - sqrt_p * grad_z.sum(dim=-1, keepdim=True) / sqrt_p.sum(dim=-1, keepdim=True)


def entmax15(z: Tensor) -> Tensor:
    """Sparse alternative to softmax: most entries are exactly 0 (paper Section 3.2)."""
    return _Entmax15.apply(z)


def hardmax_st(p: Tensor) -> Tensor:
    """One-hot argmax in the forward pass, identity gradient in the backward pass."""
    hard = F.one_hot(p.argmax(dim=-1), p.shape[-1]).to(p.dtype)
    return p + (hard - p).detach()


class GradTree(nn.Module):
    """A complete, hard, axis-aligned decision tree whose parameters are learned by gradient descent."""

    def __init__(self, n_features: int, n_classes: int, depth: int = 10, beta: float = 1.0) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.depth = depth
        self.beta = beta  # scale of I before entmax; 1.0 in every official setting
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth

        leaf_shape = (self.n_leaves,) if n_classes == 2 else (self.n_leaves, n_classes)
        self.split_index = nn.Parameter(torch.empty(self.n_internal, n_features))
        self.split_values = nn.Parameter(torch.empty(self.n_internal, n_features))
        self.leaf_values = nn.Parameter(torch.empty(leaf_shape))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, seed: int | None = None) -> None:
        """Glorot-uniform initialisation with Keras' fan sizes, as the official GlorotUniform setting."""
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        for param in self.parameters():
            fan_in, fan_out = param.shape if param.dim() == 2 else (param.shape[0], param.shape[0])
            limit = math.sqrt(6 / (fan_in + fan_out))
            param.copy_(torch.empty(param.shape).uniform_(-limit, limit, generator=generator))

    def selected_features(self) -> Tensor:
        """One-hot feature choice of every internal node, (N, F)."""
        return hardmax_st(entmax15(self.beta * self.split_index))

    def forward(self, x: Tensor) -> Tensor:
        """Logits of the leaf each sample reaches: (B,) for 2 classes, (B, C) otherwise.

        Algorithm 1 multiplies the split decisions along every root-to-leaf path (Eq. 3-5). Here
        the product is built level by level: a child gets its parent's value times go_left (left
        child) or 1 - go_left (right child). After d levels this is the same product, with the same
        values and gradients, but shared path prefixes are computed once instead of per leaf.
        """
        features = self.selected_features()                        # (N, F)
        thresholds = (self.split_values * features).sum(dim=1)     # (N,)
        diff = thresholds - x @ features.T                         # (B, N): threshold - tested value
        soft = torch.sigmoid(diff)
        # round(sigmoid(diff)) with a straight-through gradient. (diff > 0) is that rounding without
        # float32 error near 0, so the extracted tree routes every sample exactly like this model.
        go_left = soft + ((diff > 0).to(soft.dtype) - soft).detach()

        reached = torch.ones(x.shape[0], 1, dtype=soft.dtype, device=x.device)
        for level in range(self.depth):
            s = go_left[:, 2 ** level - 1:2 ** (level + 1) - 1]   # nodes of this level, left to right
            reached = torch.stack([reached * s, reached * (1 - s)], dim=2).flatten(1)
        return reached @ self.leaf_values                          # reached: (B, P), one-hot


def gradtree_loss(
    logits: Tensor,
    y: Tensor,
    n_classes: int,
    loss: str = "crossentropy",
    focal_gamma: float = 3.0,
    poly_epsilon: float | None = None,
) -> Tensor:
    """Mean loss over the batch, computed as in the official code.

    loss:          'crossentropy', or 'focal' for (1 - p_t)**gamma * cross-entropy (the official
                   'binary_focal_crossentropy', which is the categorical focal loss for > 2 classes).
    poly_epsilon:  if set, adds epsilon * (1 - p_t) (Poly-1 loss, Leng et al. 2022).
    """
    if n_classes == 2:
        ce = F.binary_cross_entropy_with_logits(logits, y.to(logits.dtype), reduction="none")
        p = torch.sigmoid(logits)
        p_true = torch.where(y == 1, p, 1 - p)
    else:
        log_p = F.log_softmax(logits, dim=1)
        ce = -log_p.gather(1, y.unsqueeze(1)).squeeze(1)
        p_true = torch.exp(-ce)

    if loss == "crossentropy":
        per_sample = ce
    elif loss == "focal":
        per_sample = (1 - p_true) ** focal_gamma * ce
    else:
        raise ValueError(f"Unknown loss {loss!r}")

    if poly_epsilon:
        # The official code computes p_t for this term from the probabilities a second time
        # (sigmoid/softmax of the already activated output). We keep that, because the tuned
        # epsilon (5 for Adult and Echocardiogram) was chosen with it.
        if n_classes == 2:
            q = torch.sigmoid(p)
            p_poly = torch.where(y == 1, q, 1 - q)
        else:
            p_poly = torch.softmax(log_p.exp(), dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
        per_sample = per_sample + poly_epsilon * (1 - p_poly)

    return per_sample.mean()

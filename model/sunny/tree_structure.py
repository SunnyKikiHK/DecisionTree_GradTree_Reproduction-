"""Static path tables for a complete binary tree.

GradTree processes the whole tree in a single batch.  For every leaf ``l``
and every depth ``j = 1..d`` it needs:

    internal_node_index[l, j-1]  -- the internal node on leaf l's path
                                   (Equation 4 in the paper)
    path_identifier[l, j-1]      -- 0 for a left branch, 1 for a right branch
                                   (Equation 5 in the paper)

Both tables are CONSTANTS for a fixed depth, so the model registers them as
non-persistent buffers.  Internal nodes are numbered breadth-first starting
at 0 (i.e. Internal nodes are numbered breadth-first: root 0, then level 2 is 1, 2, level 3 is 3, 4, 5, 6, etc.)
, leaves are numbered left-to-right starting at 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TreeStructure:
    """Constant tensors describing every root-to-leaf path."""

    depth: int
    num_internal: int
    num_leaves: int
    path_identifier: Tensor  # (num_leaves, depth), float32, values 0/1
    internal_node_index: Tensor  # (num_leaves, depth), int64


def make_tree_structure(depth: int, device: torch.device | str | None = None) -> TreeStructure:
    """Build the path tables for a complete binary tree of the given depth.

    Examples
    --------
    >>> s = make_tree_structure(2)
    >>> s.num_internal, s.num_leaves # 2**2-1 = 3
    (3, 4)
    >>> s.path_identifier.tolist()
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    >>> s.internal_node_index.tolist()
    [[0, 1], [0, 1], [0, 2], [0, 2]]
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    num_internal = 2**depth - 1
    num_leaves = 2**depth
    powers = [2**k for k in range(depth + 1)] # loopup table for 2**k

    path_rows: list[list[float]] = []
    internal_rows: list[list[int]] = []
    for leaf in range(num_leaves): # leaf: numbered 0 ... num_leaves-1 from left to right
        path_row: list[float] = []
        internal_row: list[int] = []
        for j in range(1, depth + 1): # j: depth level, where j=1 is root and j=depth of deepest internal node level
            path_row.append( # equation 5:  indicates whether the left (p = 0) or the right branch (p = 1) was taken at the internal node preceding a leaf node l at a certain depth j.
                float((leaf // powers[depth - j]) % 2)
            ) 
            internal_row.append( # equation 4: the internal node preceding a leaf node l at a certain depth j 
                powers[j - 1] + leaf // powers[depth - (j - 1)] - 1 
            )
        path_rows.append(path_row)
        internal_rows.append(internal_row)

    return TreeStructure(
        depth=depth,
        num_internal=num_internal, # 2**d - 1
        num_leaves=num_leaves, # 2**d
        path_identifier=torch.tensor(path_rows, dtype=torch.float32, device=device), # convert nested list to tensor
        internal_node_index=torch.tensor(internal_rows, dtype=torch.long, device=device), # convert nested list to tensor 
    )

"""A modular PyTorch implementation of GradTree.

Reference:
    Marton, S., Luedtke, S., Bartelt, C., & Stuckenschmidt, H. (2024).
    GradTree: Learning Axis-Aligned Decision Trees with Gradient Descent.
    https://arxiv.org/abs/2305.03515
"""

from .config import GradTreeConfig
from .entmax import apply_activation, entmax15, hardmax_st, round_st, sparsemax
from .gradtree import GradTree
from .losses import GradTreeLoss
from .metrics import accuracy_score, confusion_matrix, macro_f1_score
from .pruning import HardDecisionTree
from .trainer import GradTreeTrainer, TrainResult, load_model_from_checkpoint

__all__ = [
    "GradTreeConfig",
    "GradTree",
    "GradTreeLoss",
    "GradTreeTrainer",
    "TrainResult",
    "HardDecisionTree",
    "load_model_from_checkpoint",
    "apply_activation",
    "entmax15",
    "sparsemax",
    "hardmax_st",
    "round_st",
    "accuracy_score",
    "macro_f1_score",
    "confusion_matrix",
]

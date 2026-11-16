"""Training loop for GradTree.

The paper trains the three parameter groups with Adam, using a separate
learning rate for each group, mini-batches, early stopping on validation
loss and random restarts.  This module implements those pieces.

Stochastic weight averaging (SWA), which the official implementation also
uses, is not implemented here; the parameters returned are the best epoch of
the best restart.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import GradTreeConfig
from .entmax import apply_activation
from .gradtree import GradTree
from .losses import GradTreeLoss
from .metrics import accuracy_score, macro_f1_score


def resolve_device(device: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    """Summary of a completed training run."""

    best_restart: int
    best_epoch: int
    best_monitor_loss: float
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    restarts: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"TrainResult(restart={self.best_restart}, epoch={self.best_epoch}, "
            f"monitor_loss={self.best_monitor_loss:.6f})"
        )


class GradTreeTrainer:
    """Fit a :class:`GradTree` with the paper's optimisation recipe."""

    def __init__(self, model: GradTree, config: GradTreeConfig) -> None:
        self.model = model
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)
        self.loss_fn = GradTreeLoss(
            num_classes=model.num_classes,
            kind=config.loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            poly_epsilon=config.poly_epsilon,
        )
        self.optimizer = self._make_optimizer()
        self.history: TrainResult | None = None

    # Setup helpers
    def _make_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            [
                {"params": [self.model.split_index], "lr": self.config.lr_index},
                {"params": [self.model.split_threshold], "lr": self.config.lr_values},
                {"params": [self.model.leaf_logits], "lr": self.config.lr_leaf},
            ]
        )

    def _as_features(self, X) -> Tensor:
        if torch.is_tensor(X):
            return X.detach().to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

    def _as_targets(self, y) -> Tensor:
        if torch.is_tensor(y):
            return y.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        return torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device).reshape(-1)

    def _effective_batch_size(self, n: int) -> int:
        """Cap the batch size at half the split, matching the official code."""
        return max(1, min(self.config.batch_size, math.ceil(n / 2)))

    # Public API
    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
    ) -> TrainResult:
        """Train the model, optionally with a validation set for early stopping.

        The best restart is selected by validation loss when validation data
        is supplied, otherwise by training loss.
        """
        X_train = self._as_features(X_train)
        y_train = self._as_targets(y_train)
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train have different numbers of rows")
        if X_train.shape[0] == 0:
            raise ValueError("X_train is empty")

        if (X_valid is None) != (y_valid is None):
            raise ValueError("pass both X_valid and y_valid, or neither")
        if X_valid is not None:
            X_valid = self._as_features(X_valid)
            y_valid = self._as_targets(y_valid)
            if X_valid.shape[0] != y_valid.shape[0]:
                raise ValueError("X_valid and y_valid have different numbers of rows")

        best_state: dict[str, Tensor] | None = None
        best_monitor = float("inf")
        best_restart = 0
        best_epoch = 0
        best_train_losses: list[float] = []
        best_val_losses: list[float] = []
        restart_summaries: list[dict[str, Any]] = []

        set_seed(self.config.seed)
        for restart in range(self.config.restarts + 1):
            seed = self.config.seed + restart
            set_seed(seed)
            self.model.reset_parameters(seed=seed)
            if self.config.init == "data_aware":
                self.model.initialize_thresholds_from_data(X_train)
            self.optimizer = self._make_optimizer()

            state, epoch, monitor, train_losses, val_losses = self._fit_once(
                X_train, y_train, X_valid, y_valid, seed
            )
            restart_summaries.append(
                {
                    "restart": restart,
                    "seed": seed,
                    "best_epoch": epoch,
                    "monitor_loss": monitor,
                }
            )

            if self.config.verbose:
                print(
                    f"[restart {restart}] best epoch={epoch} "
                    f"monitor_loss={monitor:.6f}"
                )

            if monitor < best_monitor:
                best_monitor = monitor
                best_state = state
                best_restart = restart
                best_epoch = epoch
                best_train_losses = train_losses
                best_val_losses = val_losses

        if best_state is None:
            # Degenerate case (e.g. NaN loss in every run): keep the current
            # parameters rather than leaving an untrained model weighted.
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            best_epoch = len(best_train_losses)

        self.model.load_state_dict(best_state)
        self.history = TrainResult(
            best_restart=best_restart,
            best_epoch=best_epoch,
            best_monitor_loss=best_monitor,
            train_losses=best_train_losses,
            val_losses=best_val_losses,
            restarts=restart_summaries,
        )
        return self.history

    def _fit_once(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_valid: Tensor | None,
        y_valid: Tensor | None,
        seed: int,
    ) -> tuple[dict[str, Tensor], int, float, list[float], list[float]]:
        n = X_train.shape[0]
        batch_size = self._effective_batch_size(n)
        patience = self.config.patience if self.config.patience > 0 else self.config.epochs
        log_every = max(1, self.config.log_every)

        best_monitor = float("inf")
        best_state: dict[str, Tensor] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
            order = torch.randperm(n, generator=generator).to(self.device)

            total_loss = 0.0
            for start in range(0, n, batch_size):
                indices = order[start : start + batch_size]
                features = X_train[indices]
                targets = y_train[indices]

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(features)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach()) * indices.numel()

            train_loss = total_loss / n
            train_losses.append(train_loss)

            if X_valid is not None and y_valid is not None:
                val_loss: float | None = self._loss_on(X_valid, y_valid)
                val_losses.append(val_loss)
                monitor = val_loss
            else:
                val_loss = None
                val_losses.append(float("nan"))
                monitor = train_loss

            if monitor < best_monitor - 1e-12:
                best_monitor = monitor
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if self.config.verbose and (
                epoch == 1
                or epoch % log_every == 0
                or epoch == self.config.epochs
                or epochs_without_improvement >= patience
            ):
                message = f"  epoch {epoch:5d}/{self.config.epochs}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    message += f", val_loss={val_loss:.6f}"
                print(message)

            if epochs_without_improvement >= patience:
                break

        if best_state is None:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            best_epoch = len(train_losses)
            best_monitor = train_losses[-1] if train_losses else float("inf")

        return best_state, best_epoch, float(best_monitor), train_losses, val_losses

    # Evaluation
    def _loss_on(self, X: Tensor, y: Tensor) -> float:
        self.model.eval()
        n = X.shape[0]
        if n == 0:
            return float("nan")
        batch_size = self._effective_batch_size(n)
        total_loss = 0.0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                features = X[start : start + batch_size]
                targets = y[start : start + batch_size]
                loss = self.loss_fn(self.model(features), targets)
                total_loss += float(loss) * features.shape[0]
        return total_loss / n

    def predict_logits(self, X) -> Tensor:
        """Return logits on CPU, in input order."""
        features = self._as_features(X)
        self.model.eval()
        outputs: list[Tensor] = []
        batch_size = self._effective_batch_size(features.shape[0])
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                outputs.append(
                    self.model(features[start : start + batch_size]).detach().cpu()
                )
        if not outputs:
            return torch.empty(0)
        return torch.cat(outputs, dim=0)

    def predict_proba(self, X) -> Tensor:
        """Class probabilities on CPU."""
        logits = self.predict_logits(X)
        if self.model.binary:
            return torch.sigmoid(logits)
        if self.config.output_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        return apply_activation(logits, self.config.output_activation)

    def predict(self, X) -> np.ndarray:
        """Hard class predictions as a NumPy array."""
        proba = self.predict_proba(X)
        if proba.numel() == 0:
            return np.empty(0, dtype=np.int64)
        if self.model.binary:
            return (proba > 0.5).long().numpy()
        return proba.argmax(dim=-1).numpy()

    def evaluate(self, X, y) -> dict[str, float]:
        """Return loss, accuracy and macro F1 for a split."""
        features = self._as_features(X)
        targets = self._as_targets(y)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("X and y have different numbers of rows")

        loss = self._loss_on(features, targets)
        predictions = self.predict(X)
        truth = targets.cpu().numpy()
        return {
            "loss": float(loss),
            "accuracy": accuracy_score(truth, predictions),
            "macro_f1": macro_f1_score(
                truth, predictions, num_classes=self.model.num_classes
            ),
        }

    # Checkpointing
    def save(self, path, extra: dict[str, Any] | None = None) -> Path:
        """Save the model, its config and optional metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "num_features": self.model.num_features,
            "num_classes": self.model.num_classes,
            "extra": extra or {},
        }
        torch.save(payload, path)
        return path


def load_model_from_checkpoint(
    path, map_location="cpu"
) -> tuple[GradTree, GradTreeConfig, dict[str, Any]]:
    """Recreate a model and config from a checkpoint written by ``GradTreeTrainer.save``."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = GradTreeConfig(**payload["config"])
    model = GradTree(
        num_features=int(payload["num_features"]),
        num_classes=int(payload["num_classes"]),
        depth=config.depth,
        activation=config.activation,
        beta=config.beta,
        dropout=config.dropout,
        init=config.init,
        output_activation=config.output_activation,
        seed=config.seed,
    )
    model.load_state_dict(payload["state_dict"])
    return model, config, payload.get("extra", {})

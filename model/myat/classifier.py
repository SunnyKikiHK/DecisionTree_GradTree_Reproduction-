"""Training GradTree the way the official experiments do (GDT.fit in experiments_paper_gradtree/utilities/GDT.py)."""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from .gradtree import GradTree, gradtree_loss
from .tree import HardTree


class GradTreeClassifier:
    """fit() trains on the training split with early stopping on the validation split, then predicts
    with the pruned hard tree. Defaults are the paper's binary default setting; config.py has the
    tuned settings per dataset.

    The official code wraps Adam in a stochastic weight averaging optimizer but never copies the
    averaged weights into the model, so plain Adam reproduces its training.
    """

    def __init__(
        self,
        depth: int = 10,
        lr_index: float = 0.05,
        lr_values: float = 0.01,
        lr_leaf: float = 0.005,
        loss: str = "crossentropy",
        focal_gamma: float = 3.0,
        poly_epsilon: float | None = None,
        batch_size: int = 128,
        epochs: int = 10_000,
        restarts: int = 3,
        early_stopping_epochs: int = 200,
        prune_threshold: float = 1,
        n_classes: int | None = None,
        seed: int = 42,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.depth = depth
        self.lr_index = lr_index
        self.lr_values = lr_values
        self.lr_leaf = lr_leaf
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.poly_epsilon = poly_epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.restarts = restarts
        self.early_stopping_epochs = early_stopping_epochs
        self.prune_threshold = prune_threshold
        self.n_classes = n_classes
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def fit(self, X_train, y_train, X_valid, y_valid) -> GradTreeClassifier:
        start = time.perf_counter()
        X_train_np = np.asarray(X_train, dtype=np.float32)
        X_train, y_train = self._tensors(X_train_np, y_train)
        X_valid, y_valid = self._tensors(X_valid, y_valid)
        self.n_classes_ = self.n_classes or int(max(y_train.max(), y_valid.max())) + 1
        batch_size = min(self.batch_size, math.ceil(len(X_train) / 2))  # as in the official fit

        model = GradTree(X_train.shape[1], self.n_classes_, self.depth).to(self.device)
        best_state, best_loss = None, math.inf
        self.history_ = []
        for restart in range(self.restarts + 1):
            # The official code adds the restart number to the seed each time: 42, 43, 45, 48, ...
            run_seed = self.seed + restart * (restart + 1) // 2
            model.reset_parameters(run_seed)
            optimizer = torch.optim.Adam(
                [
                    {"params": [model.split_index], "lr": self.lr_index},
                    {"params": [model.split_values], "lr": self.lr_values},
                    {"params": [model.leaf_values], "lr": self.lr_leaf},
                ],
                eps=1e-7,  # Keras' Adam default
            )
            run_loss, run_state, epochs = self._train_run(
                model, optimizer, X_train, y_train, X_valid, y_valid, batch_size, run_seed
            )
            self.history_.append({"restart": restart, "seed": run_seed, "epochs": epochs, "best_valid_loss": run_loss})
            if self.verbose:
                print(f"restart {restart}: {epochs} epochs, best valid loss {run_loss:.4f}")
            if best_state is None or run_loss < best_loss:
                best_loss, best_state, self.best_restart_ = run_loss, run_state, restart

        model.load_state_dict(best_state)
        self.model_ = model
        self.best_valid_loss_ = best_loss
        self.tree_ = HardTree.from_model(model)
        if self.prune_threshold:
            self.tree_.prune(X_train_np, self.prune_threshold)
        self.fit_seconds_ = time.perf_counter() - start
        return self

    def _train_run(self, model, optimizer, X_train, y_train, X_valid, y_valid, batch_size, run_seed):
        """One run from a fresh initialisation; returns (best valid loss, its parameters, epochs run)."""
        best_loss, best_state, waited = math.inf, None, 0
        for epoch in range(self.epochs):
            model.train()
            order = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(run_seed + epoch))
            for begin in range(0, len(order), batch_size):
                batch = order[begin:begin + batch_size].to(self.device)
                loss = self._loss(model(X_train[batch]), y_train[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_loss = self._valid_loss(model, X_valid, y_valid, batch_size)
            if valid_loss < best_loss:
                best_loss, best_state, waited = valid_loss, copy.deepcopy(model.state_dict()), 0
            else:
                waited += 1
                if waited >= self.early_stopping_epochs:
                    break
        return best_loss, best_state, epoch + 1

    @torch.no_grad()
    def _valid_loss(self, model, X_valid, y_valid, batch_size) -> float:
        """Mean of the per-batch losses, as the official test loop computes it."""
        model.eval()
        losses = [
            self._loss(model(X_valid[begin:begin + batch_size]), y_valid[begin:begin + batch_size]).item()
            for begin in range(0, len(X_valid), batch_size)
        ]
        return float(np.mean(losses))

    def _loss(self, logits, y):
        return gradtree_loss(logits, y, self.n_classes_, self.loss, self.focal_gamma, self.poly_epsilon)

    def _tensors(self, X, y):
        X = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        y = torch.tensor(np.asarray(y, dtype=np.int64), device=self.device)
        return X, y

    def predict_proba(self, X) -> np.ndarray:
        logits = self.tree_.leaf_logits(np.asarray(X, dtype=np.float32))
        if self.n_classes_ == 2:
            p = 1 / (1 + np.exp(-logits))
            return np.column_stack([1 - p, p])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

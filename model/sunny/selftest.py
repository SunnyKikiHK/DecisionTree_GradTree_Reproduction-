"""Dependency-light self-tests for the PyTorch GradTree implementation.

Run from the repository root::

    python -m model.sunny.selftest

The script checks the entmax Jacobian, the path-table construction, forward
shapes and gradients, hard-tree consistency, pruning, losses, metrics and a
small end-to-end overfitting run.  It needs only PyTorch and NumPy for the
core checks; the checkpoint test uses a temporary directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import GradTreeConfig
from .entmax import entmax15, hardmax_st, round_st, sparsemax
from .gradtree import GradTree
from .losses import GradTreeLoss
from .metrics import accuracy_score, macro_f1_score
from .pruning import HardDecisionTree
from .trainer import GradTreeTrainer, load_model_from_checkpoint
from .tree_structure import make_tree_structure


def test_entmax_and_sparsemax() -> None:
    torch.manual_seed(0)
    z = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    p = entmax15(z)
    assert torch.allclose(p.sum(dim=-1), torch.ones(3, dtype=torch.double), atol=1e-9)
    assert (p >= 0).all()
    assert torch.autograd.gradcheck(
        lambda value: entmax15(value), (z,), eps=1e-6, atol=1e-5, rtol=1e-4
    )

    z2 = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    p2 = sparsemax(z2)
    assert torch.allclose(p2.sum(dim=-1), torch.ones(3, dtype=torch.double), atol=1e-9)
    assert (p2 >= 0).all()
    assert torch.autograd.gradcheck(
        lambda value: sparsemax(value), (z2,), eps=1e-6, atol=1e-5, rtol=1e-4
    )

    x = torch.randn(4, 6, requires_grad=True)
    hard = hardmax_st(x)
    assert torch.allclose(hard.sum(dim=-1), torch.ones(4))
    hard.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    r = torch.rand(10, requires_grad=True)
    rounded = round_st(r)
    assert ((rounded == 0) | (rounded == 1)).all()
    rounded.sum().backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()
    print("  entmax/sparsemax/ST checks passed")


def test_tree_structure() -> None:
    structure = make_tree_structure(2)
    assert structure.num_internal == 3
    assert structure.num_leaves == 4
    assert structure.path_identifier.tolist() == [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    assert structure.internal_node_index.tolist() == [
        [0, 1],
        [0, 1],
        [0, 2],
        [0, 2],
    ]

    depth = 4
    structure = make_tree_structure(depth)
    assert structure.path_identifier.shape == (2**depth, depth)
    assert structure.internal_node_index.shape == (2**depth, depth)
    assert structure.path_identifier.sum(dim=0).tolist() == [2 ** (depth - 1)] * depth
    assert structure.internal_node_index[:, 0].eq(0).all()
    for j in range(depth):
        counts = torch.bincount(
            structure.internal_node_index[:, j], minlength=structure.num_internal
        )
        # At depth j+1 (0-based column j) the active nodes are the breadth-first
        # block [2**j - 1, 2**(j+1) - 1), and each occurs 2**(depth-j) times.
        start = 2**j - 1
        end = 2 ** (j + 1) - 1
        assert counts[:start].sum().item() == 0
        assert counts[end:].sum().item() == 0
        assert counts[start:end].eq(2 ** (depth - j)).all()
    print("  tree-structure checks passed")


def test_model_shapes_and_gradients() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, 5)

    binary = GradTree(num_features=5, num_classes=2, depth=3, seed=0)
    logits = binary(x)
    assert logits.shape == (7,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.float32)
    )
    loss.backward()
    for name in ("split_index", "split_threshold", "leaf_logits"):
        grad = getattr(binary, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name

    multi = GradTree(num_features=5, num_classes=4, depth=3, seed=0)
    logits = multi(x)
    assert logits.shape == (7, 4)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 1, 2, 3, 0, 1, 2])
    )
    loss.backward()
    for name in ("split_index", "split_threshold", "leaf_logits"):
        grad = getattr(multi, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
    print("  model shape/gradient checks passed")


def test_hard_tree_consistency() -> None:
    torch.manual_seed(1)
    model = GradTree(num_features=4, num_classes=3, depth=3, init="normal", seed=1)
    X = torch.randn(200, 4)
    soft_predictions = model.predict(X)
    tree = model.hard_tree()
    tree_predictions = tree.predict(X.numpy())
    agreement = float((soft_predictions == tree_predictions).mean())
    assert agreement == 1.0, f"hard tree agreed on only {agreement:.4f} of rows"

    pruned = model.hard_tree(prune_X=X.numpy(), min_samples=5)
    assert pruned.size <= tree.size
    assert pruned.predict(X.numpy()).shape == (200,)
    print("  hard-tree consistency/pruning checks passed")


def test_losses_and_metrics() -> None:
    binary_loss = GradTreeLoss(2, kind="crossentropy")
    logits = torch.tensor([2.0, -1.0], dtype=torch.float64)
    targets = torch.tensor([1, 0], dtype=torch.float64)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(binary_loss(logits, targets), expected)

    multi_loss = GradTreeLoss(3, kind="crossentropy")
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    targets = torch.tensor([0, 2])
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(multi_loss(logits, targets), expected)

    # Focal and PolyLoss must be finite and differentiable.
    for kind in ("focal", "poly"):
        loss = GradTreeLoss(2, kind=kind)(torch.randn(5, requires_grad=True), torch.randint(0, 2, (5,)))
        assert torch.isfinite(loss)

    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 1])
    assert accuracy_score(y_true, y_pred) == 0.5
    assert abs(macro_f1_score(y_true, y_pred, num_classes=3) - 0.5) < 1e-12
    print("  loss/metric checks passed")


def test_end_to_end_overfit() -> None:
    torch.manual_seed(0)
    n = 120
    X0 = torch.randn(n, 2) * 0.6 + torch.tensor([-2.0, -2.0])
    X1 = torch.randn(n, 2) * 0.6 + torch.tensor([2.0, 2.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    # Simple train/valid split
    order = torch.randperm(2 * n, generator=torch.Generator().manual_seed(0))
    train_idx, valid_idx = order[: 3 * n // 4], order[3 * n // 4 :]

    config = GradTreeConfig(
        depth=3,
        lr_index=0.1,
        lr_values=0.1,
        lr_leaf=0.1,
        batch_size=32,
        epochs=250,
        patience=0,
        restarts=0,
        verbose=False,
        device="cpu",
        prune=True,
    )
    model = GradTree(num_features=2, num_classes=2, depth=config.depth, seed=0)
    trainer = GradTreeTrainer(model, config)
    trainer.fit(X[train_idx], y[train_idx], X[valid_idx], y[valid_idx])
    accuracy = trainer.evaluate(X, y)["accuracy"]
    assert accuracy > 0.95, f"overfit test reached only {accuracy:.4f} accuracy"

    tree = model.hard_tree(prune_X=X[train_idx].numpy(), min_samples=1)
    tree_accuracy = accuracy_score(y.numpy(), tree.predict(X.numpy()))
    assert tree_accuracy > 0.95
    print(f"  end-to-end overfit passed (accuracy={accuracy:.4f})")


def test_checkpoint_roundtrip() -> None:
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    config = GradTreeConfig(
        depth=2, epochs=3, batch_size=8, patience=0, verbose=False, device="cpu"
    )
    model = GradTree(num_features=3, num_classes=2, depth=2, seed=0)
    trainer = GradTreeTrainer(model, config)
    trainer.fit(X, torch.randint(0, 2, (20,)))
    before = trainer.predict_proba(X)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        trainer.save(path, extra={"test": True})
        loaded_model, loaded_config, extra = load_model_from_checkpoint(path)
        loaded_model.to("cpu")
        after = loaded_model.predict_proba(X)
        assert torch.allclose(before, after, atol=1e-6)
        assert loaded_config.depth == 2 and extra["test"] is True
    print("  checkpoint round-trip passed")


def main() -> None:
    tests = [
        test_entmax_and_sparsemax,
        test_tree_structure,
        test_model_shapes_and_gradients,
        test_hard_tree_consistency,
        test_losses_and_metrics,
        test_end_to_end_overfit,
        test_checkpoint_roundtrip,
    ]
    print("Running GradTree self-tests")
    for test in tests:
        test()
    print("All self-tests passed")


if __name__ == "__main__":
    main()

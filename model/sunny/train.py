"""Command-line entry point for training GradTree.

Examples
--------
Run a quick model on the default (trial 0) Iris split::

    python -m model.sunny.train --dataset iris --depth 4 --epochs 200

Train trial seed 43::

    python -m model.sunny.train --dataset iris --trial-seed 43

Use the paper's default configuration on Adult::

    python -m model.sunny.train --dataset adult --depth 10 --epochs 10000 \
        --patience 200 --restarts 5 --save model/sunny/checkpoints/adult.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import GradTreeConfig
from .data import DEFAULT_PREPROCESS_DIR, load_dataset
from .gradtree import GradTree
from .metrics import accuracy_score, macro_f1_score
from .trainer import GradTreeTrainer, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch GradTree on the preprocessed CSV splits."
    )
    parser.add_argument("--dataset", required=True, help="e.g. iris, wine, adult, spambase")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_PREPROCESS_DIR),
        help="directory containing <dataset>_output/ (default: repo/preprocess)",
    )
    parser.add_argument(
        "--trial-seed",
        type=int,
        default=None,
        help="load preprocess/<dataset>_output/seed_<n>/ instead of the top-level split",
    )

    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr-index", type=float, default=0.05)
    parser.add_argument("--lr-values", type=float, default=0.01)
    parser.add_argument("--lr-leaf", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=0)

    parser.add_argument(
        "--activation",
        choices=["entmax", "softmax", "sparsemax"],
        default="entmax",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--init",
        choices=["glorot", "normal", "uniform", "data_aware"],
        default="glorot",
    )
    parser.add_argument(
        "--loss", choices=["crossentropy", "focal", "poly"], default="crossentropy"
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--poly-epsilon", type=float, default=2.0)
    parser.add_argument(
        "--output-activation",
        choices=["softmax", "entmax", "sparsemax"],
        default="softmax",
    )

    parser.add_argument("--no-prune", action="store_true", help="skip post-hoc pruning")
    parser.add_argument("--prune-threshold", type=float, default=1.0)
    parser.add_argument("--print-tree", action="store_true")

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--save", default=None, help="checkpoint path")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(
        f"{title:8s} loss={metrics['loss']:.6f} "
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = GradTreeConfig(
        depth=args.depth,
        lr_index=args.lr_index,
        lr_values=args.lr_values,
        lr_leaf=args.lr_leaf,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        restarts=args.restarts,
        activation=args.activation,
        beta=args.beta,
        dropout=args.dropout,
        init=args.init,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        poly_epsilon=args.poly_epsilon,
        output_activation=args.output_activation,
        prune=not args.no_prune,
        prune_threshold=int(args.prune_threshold)
        if float(args.prune_threshold).is_integer()
        else args.prune_threshold,
        device=args.device,
        seed=args.random_seed,
        verbose=not args.quiet,
    )

    set_seed(config.seed)
    try:
        splits = load_dataset(
            args.dataset,
            data_dir=Path(args.data_dir),
            seed=args.trial_seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Could not load dataset: {error}")
        return 1

    num_classes = splits.num_classes
    if num_classes == 2:
        unique = set(np.unique(splits.y_train).tolist())
        if not unique.issubset({0, 1}):
            print(f"Binary dataset must use labels 0/1, found {sorted(unique)}")
            return 1

    trial_label = ""
    if args.trial_seed is not None:
        trial_label = f" (trial seed {args.trial_seed})"
    print(f"Dataset: {args.dataset}{trial_label}")
    print(json.dumps(splits.describe(), indent=2))

    model = GradTree(
        num_features=splits.num_features,
        num_classes=num_classes,
        depth=config.depth,
        activation=config.activation,
        beta=config.beta,
        dropout=config.dropout,
        init=config.init,
        output_activation=config.output_activation,
        seed=config.seed,
    )
    trainer = GradTreeTrainer(model, config)
    print(model)
    print(f"Device: {trainer.device}")
    print(f"Config: {json.dumps(config.to_dict(), indent=2)}")

    result = trainer.fit(
        splits.X_train,
        splits.y_train,
        splits.X_valid,
        splits.y_valid,
    )
    print(f"Training finished: {result}")

    print("\nDifferentiable model metrics")
    metrics: dict[str, dict[str, float]] = {}
    for name, X, y in (
        ("train", splits.X_train, splits.y_train),
        ("valid", splits.X_valid, splits.y_valid),
        ("test", splits.X_test, splits.y_test),
    ):
        metrics[name] = trainer.evaluate(X, y)
        _print_metrics(name, metrics[name])

    tree_metrics: dict[str, dict[str, float]] = {}
    tree = None
    if config.prune:
        tree = model.hard_tree(
            prune_X=splits.X_train, min_samples=config.prune_threshold
        )
        print(f"\nHard decision tree: {tree}")
        for name, X, y in (
            ("train", splits.X_train, splits.y_train),
            ("valid", splits.X_valid, splits.y_valid),
            ("test", splits.X_test, splits.y_test),
        ):
            predictions = tree.predict(X)
            tree_metrics[name] = {
                "accuracy": accuracy_score(y, predictions),
                "macro_f1": macro_f1_score(y, predictions, num_classes=num_classes),
            }
            print(
                f"{name:8s} accuracy={tree_metrics[name]['accuracy']:.4f} "
                f"macro_f1={tree_metrics[name]['macro_f1']:.4f}"
            )
        if args.print_tree:
            print("\n" + tree.print_tree())

    if args.save:
        extra = {
            "dataset": args.dataset,
            "trial_seed": args.trial_seed,
            "feature_names": splits.feature_names,
            "target_name": splits.target_name,
            "metrics": metrics,
            "hard_tree_metrics": tree_metrics,
            "train_result": {
                "best_restart": result.best_restart,
                "best_epoch": result.best_epoch,
                "best_monitor_loss": result.best_monitor_loss,
                "restarts": result.restarts,
            },
        }
        saved_path = trainer.save(args.save, extra=extra)
        print(f"\nSaved checkpoint to {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

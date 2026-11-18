"""Parallel driver for the GradTree experiments.

One *trial* is a full sequential training run for a fixed
``(experiment, dataset, seed, depth, activation)``.  Trials are independent:
they share no parameters, no data and no random state, so several of them can
run at the same time in separate processes.

The training body lives here rather than in ``train_all.ipynb`` because
``ProcessPoolExecutor`` sends the worker function to the child process by
reference.  A function defined in a notebook cell cannot be imported by the
child; a function in a module can.  This is exactly why ``model.myat.train``
can use ``--jobs`` and a notebook-defined function cannot.

Typical use from the notebook::

    from model.sunny.experiments import run_tasks

    rows = run_tasks(
        tasks,
        runtime=dict(protocol="paper", epochs=200, patience=20, restarts=0,
                     batch_size=128, device="cpu", threads=1, model_seed=42),
        jobs=4,
        on_result=append_result,
    )

Notes
-----
- With ``jobs > 1`` each worker gets ``runtime["threads"]`` PyTorch threads.
  Give each worker one thread and run one worker per core; four processes with
  eight OpenMP threads each is slower than one process with eight threads.
- Workers are started with ``spawn``: a fresh interpreter per worker.  That is
  safe in a notebook (kernels are threaded; forking one can deadlock), on
  Windows, and with CUDA (a CUDA context must never be forked).  The import
  cost is paid once per worker, not once per trial.  Override with
  ``runtime["start_method"]``; ``"fork"`` is slightly faster from a plain
  script on Linux.
- On a single GPU the workers share the device rather than each getting one.
  That still helps when the workload is launch-bound, but the speedup is much
  smaller than the number of workers.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .config import GradTreeConfig
from .data import load_dataset
from .gradtree import GradTree
from .metrics import macro_f1_score
from .trainer import GradTreeTrainer

# Dataset list and the paper's per-dataset settings (Appendix C, Table 12).
DATASETS = ["adult", "echocardiogram", "iris", "spambase", "wine"]

DATASET_LABELS = {
    "adult": "Adult",
    "echocardiogram": "Echocardiogram",
    "iris": "Iris",
    "spambase": "Spambase",
    "wine": "Wine",
}

PAPER_SETTINGS = {
    "adult": dict(
        depth=8, lr_index=0.050, lr_values=0.005, lr_leaf=0.050,
        loss="poly", poly_epsilon=5,
    ),
    "echocardiogram": dict(
        depth=8, lr_index=0.010, lr_values=0.050, lr_leaf=0.100,
        loss="poly", poly_epsilon=5,
    ),
    "iris": dict(
        depth=7, lr_index=0.005, lr_values=0.005, lr_leaf=0.05,
        loss="crossentropy", poly_epsilon=2,
    ),
    "spambase": dict(
        depth=10, lr_index=0.005, lr_values=0.010, lr_leaf=0.010,
        loss="crossentropy", poly_epsilon=2,
    ),
    "wine": dict(
        depth=10, lr_index=0.010, lr_values=0.050, lr_leaf=0.01,
        loss="focal", poly_epsilon=2, focal_gamma=3.0,
    ),
}

# Paper reference values: (mean, std over the 10 trials).
PAPER_TEST_F1 = {
    "adult": (0.743, 0.034),
    "echocardiogram": (0.658, 0.113),
    "iris": (0.938, 0.057),
    "spambase": (0.903, 0.025),
    "wine": (0.933, 0.031),
}

PAPER_TREE_SIZE = {
    "adult": (128.00, 37.19),
    "echocardiogram": (43.20, 12.82),
    "iris": (14.40, 2.54),
    "spambase": (120.20, 37.01),
    "wine": (24.00, 4.12),
}


def resolve_depth(task: dict[str, Any]) -> int:
    """Use the task's depth, or the paper's tuned depth when it is ``None``."""
    settings = PAPER_SETTINGS[task["dataset"]]
    return settings["depth"] if task.get("depth") is None else int(task["depth"])


def probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Return an ``(n, num_classes)`` probability matrix.

    Binary models emit a single sigmoid column, so the complement is added to
    make the ranking well defined for every dataset.
    """
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
        positive = array.reshape(-1)
        return np.stack([1.0 - positive, positive], axis=1)
    return array


def rank_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    """Return ``(MRR, MRD)`` for one split.

    The classes are ranked per row by predicted probability.  The true class
    gets a 1-based competition rank: 1 plus the number of classes scored
    strictly higher.

    - ``MRR`` (mean reciprocal rank) = mean of ``1 / rank``; higher is better.
    - ``MRD`` (mean rank difference) = mean of ``rank - 1``: how many classes
      are ranked above the true one, so 0 is perfect and lower is better.

    For binary data both are determined by accuracy: a correct row has rank 1
    and a wrong row rank 2.  With more than two classes, a wrong prediction
    still scores partially when the true class was close to the top.
    """
    matrix = probability_matrix(probabilities)
    targets = np.asarray(targets).reshape(-1)
    true_probability = matrix[np.arange(targets.shape[0]), targets]
    ranks = 1 + (matrix > true_probability[:, None]).sum(axis=1)
    return float(np.mean(1.0 / ranks)), float(np.mean(ranks - 1.0))


def _numpy_probabilities(trainer: GradTreeTrainer, X) -> np.ndarray:
    """Predict class probabilities for a split and return them as NumPy."""
    probabilities = trainer.predict_proba(X)
    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def run_trial(task: dict[str, Any], settings: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate one trial; returns a row for the results CSV.

    Every argument is a plain dictionary so it can be sent to a worker
    process.  The row keys match the notebook's ``RESULT_COLUMNS``.
    """
    threads = int(runtime.get("threads", 1) or 1)
    if threads > 0:
        torch.set_num_threads(threads)

    experiment = str(task.get("experiment", "main"))
    dataset = str(task["dataset"])
    seed = int(task["seed"])
    activation = str(task.get("activation", "entmax"))
    depth = resolve_depth(task)

    splits = load_dataset(dataset, seed=seed)
    config = GradTreeConfig(
        depth=depth,
        lr_index=settings["lr_index"],
        lr_values=settings["lr_values"],
        lr_leaf=settings["lr_leaf"],
        batch_size=int(runtime.get("batch_size", 128)),
        epochs=int(runtime["epochs"]),
        patience=int(runtime["patience"]),
        restarts=int(runtime["restarts"]),
        activation=activation,
        beta=1.0,
        dropout=0.0,
        init="glorot",
        loss=settings["loss"],
        focal_gamma=settings.get("focal_gamma", 3.0),
        focal_alpha=0.5,
        poly_epsilon=settings.get("poly_epsilon", 2.0),
        output_activation="softmax",
        prune=True,
        prune_threshold=1,
        device=str(runtime.get("device", "auto")),
        seed=int(runtime.get("model_seed", 42)),
        verbose=False,
    )

    model = GradTree(
        num_features=splits.num_features,
        num_classes=splits.num_classes,
        depth=config.depth,
        activation=config.activation,
        beta=config.beta,
        dropout=config.dropout,
        init=config.init,
        output_activation=config.output_activation,
        seed=config.seed,
    )
    trainer = GradTreeTrainer(model, config)

    started = time.perf_counter()
    train_result = trainer.fit(
        splits.X_train, splits.y_train, splits.X_valid, splits.y_valid
    )
    fit_seconds = time.perf_counter() - started

    train_metrics = trainer.evaluate(splits.X_train, splits.y_train)
    valid_metrics = trainer.evaluate(splits.X_valid, splits.y_valid)
    test_metrics = trainer.evaluate(splits.X_test, splits.y_test)

    # Ranking metrics from the predicted class probabilities, on both splits,
    # so the train-vs-test gap can be read off for each metric.
    train_probabilities = _numpy_probabilities(trainer, splits.X_train)
    test_probabilities = _numpy_probabilities(trainer, splits.X_test)
    mrr_train, mrd_train = rank_metrics(train_probabilities, splits.y_train)
    mrr_test, mrd_test = rank_metrics(test_probabilities, splits.y_test)

    predictions_dir = runtime.get("predictions_dir")
    if predictions_dir and experiment == "main":
        directory = Path(predictions_dir)
        directory.mkdir(parents=True, exist_ok=True)
        matrix = probability_matrix(test_probabilities)
        predictions = pd.DataFrame(
            {"y_true": splits.y_test, "y_pred": matrix.argmax(axis=1)}
        )
        for index in range(matrix.shape[1]):
            predictions[f"proba_{index}"] = matrix[:, index]
        predictions.to_csv(directory / f"{dataset}_seed_{seed}.csv", index=False)

    tree = model.hard_tree(prune_X=splits.X_train, min_samples=config.prune_threshold)
    tree_internal, tree_leaves = tree.count_nodes()
    tree_test_f1 = macro_f1_score(
        splits.y_test,
        tree.predict(splits.X_test),
        num_classes=splits.num_classes,
    )

    return {
        "protocol": runtime.get("protocol", ""),
        "experiment": experiment,
        "dataset": dataset,
        "seed": seed,
        "depth": depth,
        "activation": activation,
        "loss": settings["loss"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "valid_macro_f1": valid_metrics["macro_f1"],
        "train_macro_f1": train_metrics["macro_f1"],
        "overfitting": train_metrics["macro_f1"] - test_metrics["macro_f1"],
        "tree_nodes": tree_internal + tree_leaves,
        "tree_internal": tree_internal,
        "tree_leaves": tree_leaves,
        "tree_test_macro_f1": tree_test_f1,
        "best_epoch": train_result.best_epoch,
        "best_restart": train_result.best_restart,
        "best_monitor_loss": train_result.best_monitor_loss,
        "mrr_train": mrr_train,
        "mrr_test": mrr_test,
        "mrd_train": mrd_train,
        "mrd_test": mrd_test,
        "fit_seconds": fit_seconds,
        "config_json": json.dumps(config.to_dict()),
    }


def _start_method(runtime: dict[str, Any]) -> str:
    """Pick a process start method that is safe wherever this runs.

    ``spawn`` starts a fresh interpreter per worker, which is safe inside a
    notebook (Jupyter kernels are threaded, and forking a threaded process can
    deadlock), safe on Windows, and safe with CUDA (a CUDA context must never
    be forked).  It costs a few seconds of imports per worker, paid once per
    worker, not once per trial.

    Override with ``runtime["start_method"]`` -- ``"fork"`` is faster from a
    plain script on Linux.
    """
    requested = runtime.get("start_method")
    if requested:
        return str(requested)
    available = multiprocessing.get_all_start_methods()
    return "spawn" if "spawn" in available else available[0]


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    runtime: dict[str, Any],
    jobs: int = 1,
    on_start: Callable[[int, int, dict[str, Any]], None] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    on_error: Callable[[dict[str, Any], BaseException], None] | None = None,
) -> list[dict[str, Any]]:
    """Run ``tasks`` sequentially (``jobs == 1``) or in ``jobs`` processes.

    The callbacks always run in the *calling* process, so they may touch
    notebook state, pandas DataFrames and the results CSV.  ``on_result`` fires
    as soon as a row is available, which is what makes the CSV cache survive an
    interrupted run.
    """
    tasks = list(tasks)
    if not tasks:
        return []

    results: list[dict[str, Any]] = []

    if jobs <= 1:
        for index, task in enumerate(tasks, 1):
            if on_start is not None:
                on_start(index, len(tasks), task)
            try:
                row = run_trial(task, PAPER_SETTINGS[task["dataset"]], runtime)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                if on_error is not None:
                    on_error(task, exc)
                continue
            results.append(row)
            if on_result is not None:
                on_result(row)
        return results

    method = _start_method(runtime)
    context = multiprocessing.get_context(method)
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=context) as pool:
        futures = {
            pool.submit(run_trial, task, PAPER_SETTINGS[task["dataset"]], runtime): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                if on_error is not None:
                    on_error(task, exc)
                continue
            results.append(row)
            if on_result is not None:
                on_result(row)
    return results

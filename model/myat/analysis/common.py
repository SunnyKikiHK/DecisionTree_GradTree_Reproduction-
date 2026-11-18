"""Shared paths, chart style and trial runners for the analysis scripts."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from model.myat.gradtree.config import MODEL_SEED
from model.myat.gradtree.data import load_split

from .variants import FeatureSelectionClassifier

MYAT = Path(__file__).resolve().parents[1]  # model/myat
RESULTS_DIR = MYAT / "analysis" / "results"
FIGURES_DIR = MYAT / "analysis" / "figures"
GRADTREE_RESULTS = MYAT / "gradtree"            # gradtree/<settings>/runs.csv
BASELINE_RESULTS = MYAT / "baselines" / "results"

LABELS = {"adult": "Adult", "echocardiogram": "Echocardiogram", "iris": "Iris", "spambase": "Spambase", "wine": "Wine"}
MODEL_LABELS = {"gradtree": "GradTree", "cart": "CART", "random_forest": "Random forest"}

# Chart colors: the dataviz skill's reference palette (light mode), checked for colour-blind
# separation as a set of three. A model keeps its colour in every figure.
COLORS = {"gradtree": "#2a78d6", "cart": "#eb6834", "random_forest": "#1baf7a"}
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def use_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    })


def plain_log_ticks(axis) -> None:
    """Label a log axis 5, 10, 20, 50, 100, ... instead of 10^1, 10^2."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    axis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))


def save_figure(fig, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run_gradtree(dataset: str, seed: int, settings: dict, preprocess_dir: Path, feature_selection: str = "entmax",
                 threads: int = 1, **extra) -> dict:
    """Train GradTree on one trial and return its metrics. extra is copied into the row (e.g. depth)."""
    torch.set_num_threads(threads)
    split = load_split(dataset, seed, preprocess_dir)
    clf = FeatureSelectionClassifier(feature_selection=feature_selection, **settings,
                                     n_classes=split["n_classes"], seed=MODEL_SEED)
    clf.fit(split["X_train"], split["y_train"], split["X_valid"], split["y_valid"])
    internal, leaves = clf.tree_.count_nodes()
    row = {"model": "gradtree", "dataset": dataset, "seed": seed, "feature_selection": feature_selection, **extra}
    for part in ["train", "test"]:
        pred = clf.predict(split[f"X_{part}"])
        row[f"f1_{part}"] = f1_score(split[f"y_{part}"], pred, average="macro")
        row[f"accuracy_{part}"] = accuracy_score(split[f"y_{part}"], pred)
    row.update(total_nodes=internal + leaves, epochs=sum(run["epochs"] for run in clf.history_),
               fit_seconds=clf.fit_seconds_)
    return row


def append_row(path: Path, row: dict) -> None:
    """Append one row to a CSV, in the column order of its header if it already exists."""
    frame = pd.DataFrame([row])
    if path.exists():
        frame = frame.reindex(columns=pd.read_csv(path, nrows=0).columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def finished_keys(path: Path, key: list[str]) -> set[str]:
    if not path.exists():
        return set()
    done = pd.read_csv(path)
    return set(done[key].astype(str).agg("|".join, axis=1)) if len(done) else set()


def reuse_main_runs(tasks: list[dict], out_path: Path, key: list[str]) -> int:
    """Copy runs of the main experiment (gradtree/tuned/runs.csv) for tasks that train exactly the
    same model (tuned settings, entmax feature selection, same split), instead of training them
    again. Training is deterministic, so the copied rows are what the task would produce."""
    from model.myat.gradtree.config import gradtree_settings

    main_path = GRADTREE_RESULTS / "tuned" / "runs.csv"
    if not main_path.exists():
        return 0
    main = pd.read_csv(main_path).set_index(["dataset", "seed"])
    finished, copied = finished_keys(out_path, key), 0
    for task in tasks:
        same_model = (task.get("feature_selection", "entmax") == "entmax"
                      and task["settings"] == gradtree_settings(task["dataset"], "tuned"))
        if not same_model or (task["dataset"], task["seed"]) not in main.index:
            continue
        if "|".join(str(task[k]) for k in key) in finished:
            continue
        run = main.loc[(task["dataset"], task["seed"])]
        extra = {k: v for k, v in task.items() if k not in ("dataset", "seed", "settings", "preprocess_dir", "feature_selection")}
        row = {"model": "gradtree", "dataset": task["dataset"], "seed": task["seed"], "feature_selection": "entmax", **extra}
        row.update({column: run[column] for column in ["f1_train", "accuracy_train", "f1_test", "accuracy_test",
                                                      "total_nodes", "epochs", "fit_seconds"]})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        append_row(out_path, row)
        copied += 1
    if copied:
        print(f"Copied {copied} identical runs from {main_path}", flush=True)
    return copied


def run_tasks(tasks: list[dict], fn, jobs: int, out_path: Path, key: list[str]) -> pd.DataFrame:
    """Run fn(**task) for every task not yet in out_path, saving each finished row straight away,
    so an interrupted run continues where it stopped. Returns every row in out_path."""
    finished = finished_keys(out_path, key)
    todo = [task for task in tasks if "|".join(str(task[k]) for k in key) not in finished]
    print(f"{len(tasks) - len(todo)} of {len(tasks)} runs already in {out_path.name}; {len(todo)} to go", flush=True)

    def write(row: dict) -> None:
        append_row(out_path, row)
        shown = ", ".join(f"{k} {row[k]}" for k in key)
        print(f"{shown}: test F1 {row['f1_test']:.3f}, {row['total_nodes']} nodes, {row['fit_seconds']:.0f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if jobs <= 1:
        threads = max(1, (os.cpu_count() or 2) // 2)
        for task in todo:
            write(fn(**task, threads=threads) if fn is run_gradtree else fn(**task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(fn, **task) for task in todo]
            for future in as_completed(futures):
                write(future.result())
    return pd.read_csv(out_path)


def mean_std(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} ± {values.std(ddof=0):.3f}"

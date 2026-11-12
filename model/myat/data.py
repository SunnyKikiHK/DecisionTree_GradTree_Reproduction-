"""Loading the splits saved by the notebooks in preprocess/ (features first, target last)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PREPROCESS_DIR = Path(__file__).resolve().parents[2] / "preprocess"

NOTEBOOKS = {
    "adult": "adult_preprocessing.ipynb",
    "echocardiogram": "echocardiogram_iris_preprocessing.ipynb",
    "iris": "echocardiogram_iris_preprocessing.ipynb",
    "spambase": "spambase_wine_preprocessing.ipynb",
    "wine": "spambase_wine_preprocessing.ipynb",
}


def split_dir(name: str, seed: int, preprocess_dir: Path = PREPROCESS_DIR) -> Path:
    """Folder holding one trial's split: <name>_output/seed_<seed>/."""
    output = Path(preprocess_dir) / f"{name}_output"
    seed_dir = output / f"seed_{seed}"
    if (seed_dir / f"{name}_train.csv").exists():
        return seed_dir
    # The top-level files hold a single split; use them only if their summary says it is this seed
    summary = output / f"{name}_preprocessing_summary.json"
    if (output / f"{name}_train.csv").exists() and summary.exists():
        if json.loads(summary.read_text(encoding="utf-8")).get("random_seed") == seed:
            return output
    raise FileNotFoundError(
        f"No {name} split for seed {seed} in {output}. "
        f"Run preprocess/{NOTEBOOKS[name]} with SAVE_ALL_SEEDS = True to create it."
    )


def load_split(name: str, seed: int, preprocess_dir: Path = PREPROCESS_DIR) -> dict:
    folder = split_dir(name, seed, preprocess_dir)
    split = {"folder": folder}
    for part in ["train", "valid", "test"]:
        frame = pd.read_csv(folder / f"{name}_{part}.csv")
        split[f"X_{part}"] = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
        split[f"y_{part}"] = frame.iloc[:, -1].to_numpy(dtype=np.int64)
    split["feature_names"] = list(frame.columns[:-1])
    split["n_classes"] = int(max(split[f"y_{part}"].max() for part in ["train", "valid", "test"])) + 1
    return split

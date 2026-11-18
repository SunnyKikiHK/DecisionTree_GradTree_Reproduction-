"""Load the CSV splits produced by the preprocessing notebooks.

Every preprocessing notebook writes:

    preprocess/<dataset>_output/<dataset>_{train,valid,test}.csv

and, when ``SAVE_ALL_SEEDS = True``, one folder per trial:

    preprocess/<dataset>_output/seed_<seed>/<dataset>_{train,valid,test}.csv

The target is the last column, and labels are already encoded to 0..K-1
(binary datasets use 0/1).  Features are already quantile-transformed and
the training split is already SMOTE-balanced, so this loader does no
additional preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESS_DIR = PROJECT_ROOT / "preprocess"


@dataclass
class DatasetSplits:
    """The three arrays and their column metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str

    @property
    def num_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return infer_num_classes(self.y_train, self.y_valid, self.y_test)

    def describe(self) -> dict[str, object]:
        return {
            "features": self.num_features,
            "classes": self.num_classes,
            "target": self.target_name,
            "train_rows": int(self.X_train.shape[0]),
            "valid_rows": int(self.X_valid.shape[0]),
            "test_rows": int(self.X_test.shape[0]),
        }


def infer_num_classes(*targets: Sequence[int]) -> int:
    """Infer ``max(label) + 1`` across one or more target arrays."""
    maximum = -1
    for target in targets:
        array = np.asarray(target).reshape(-1)
        if array.size:
            maximum = max(maximum, int(array.max()))
    if maximum < 1:
        raise ValueError(
            "could not infer at least two classes; labels must be encoded as 0..K-1"
        )
    return maximum + 1


def _split_path(
    dataset: str, part: str, data_dir: Path, seed: int | None
) -> Path:
    base = data_dir / f"{dataset}_output"
    if seed is not None:
        base = base / f"seed_{seed}"
    return base / f"{dataset}_{part}.csv"


def load_dataset(
    dataset: str,
    data_dir: str | Path | None = None,
    seed: int | None = None,
) -> DatasetSplits:
    """Load train/valid/test CSVs for ``dataset``.

    Parameters
    ----------
    dataset:
        Dataset name, e.g. ``"iris"`` or ``"adult"``.
    data_dir:
        Directory containing the ``<dataset>_output`` folders.  Defaults to
        ``<repo>/preprocess``.
    seed:
        If given, load ``seed_<seed>/`` instead of the top-level trial-0
        split.  An informative error is raised when that folder does not
        exist (for example when the preprocessing notebook was run with
        ``SAVE_ALL_SEEDS = False``).

    Returns
    -------
    DatasetSplits
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_PREPROCESS_DIR
    frames: dict[str, pd.DataFrame] = {}
    for part in ("train", "valid", "test"):
        path = _split_path(dataset, part, data_dir, seed)
        if not path.exists():
            hint = ""
            if seed is not None:
                hint = (
                    f"\nThe seed folder {path.parent} does not exist. "
                    "Re-run the preprocessing notebook with SAVE_ALL_SEEDS = True, "
                    "or omit `seed` to use the top-level trial-0 split."
                )
            raise FileNotFoundError(f"Missing split file: {path}{hint}")
        frames[part] = pd.read_csv(path)

    train_columns = list(frames["train"].columns)
    if len(train_columns) < 2:
        raise ValueError("the CSV must contain at least one feature and a target")
    for part, frame in frames.items():
        if list(frame.columns) != train_columns:
            raise ValueError(
                f"{part} columns do not match the training split; "
                "re-run the preprocessing notebook"
            )

    feature_names = train_columns[:-1]
    target_name = train_columns[-1]

    def features(part: str) -> np.ndarray:
        return frames[part].iloc[:, :-1].to_numpy(dtype=np.float32)

    def target(part: str) -> np.ndarray:
        values = frames[part].iloc[:, -1].to_numpy()
        if not np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.int64)
        return values.astype(np.int64, copy=False)

    splits = DatasetSplits(
        X_train=features("train"),
        y_train=target("train"),
        X_valid=features("valid"),
        y_valid=target("valid"),
        X_test=features("test"),
        y_test=target("test"),
        feature_names=list(feature_names),
        target_name=target_name,
    )

    for name, values in (
        ("train", splits.y_train),
        ("valid", splits.y_valid),
        ("test", splits.y_test),
    ):
        if values.size and values.min() < 0:
            raise ValueError(f"{name} labels must be non-negative, got {values.min()}")
    return splits

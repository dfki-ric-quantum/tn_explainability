import pickle
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

from tn.data.loader import load_dataset
from tn.data.prepare import _scale


class TrainingKFoldIterator:
    """Iterator for a Training fold"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        shuffle: bool,
        seed: int,
        include_test: bool = False,
    ) -> None:
        """The constructor.

        Parameters
        ----------
        X: np.ndarray,
            data to iterate over
        y: np.ndarray,
            labels, 0 = inliers, > 0 different types of outliers
        n_splits: int,
            Number of folds
        shuffle: bool,
            Shuffle data within fold
        seed: int,
            Random seed
        include_test: bool = False,
            If True, also provide test set for each fold, else only the training set
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.include_test = include_test

        self.skf_it = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed
        ).split(X, y)

    def __iter__(self) -> "TrainingKFoldIterator":
        """Return the iterator"""
        return self

    def __next__(self) -> tuple[np.ndarray, ...]:
        """Return next fold"""
        train_idx, test_idx = next(self.skf_it)

        if self.include_test:
            return (
                self.X[train_idx],
                self.X[test_idx],
                self.y[train_idx],
                self.y[test_idx],
            )
        else:
            return self.X[train_idx], self.y[train_idx]


class TrainingKFold:
    """Utility class to create a K-fold for the training/evaluation."""

    def __init__(
        self,
        dataset: str,
        contamination: float,
        n_splits: int,
        shuffle: bool,
        scaler: str,
        include_test: bool,
        seed: int,
        path_prefix: Optional[str] = None,
    ) -> None:
        """The constructor.

        Parameters
        ----------
        dataset: str,
            Name of the dataset to load
        contamination: float,
            Percentage of outliers to contaminate the data with
        n_splits: int,
            Number of splits in the k-fold
        shuffle: bool,
            Shuffle data within a fold
        scaler: str,
            The preprocessing scaler to use:
            * maxabs: scale in [-1, 1]
            * minmax: scale in [0, 1]
            * standard: scale to standard normal distribution
            * none: no preprocessing
        include_test: bool,
            Include test data in each iteration
        seed: int,
            random seed
        path_prefix: Optional[str] = None,
            A path prefix relative to the project root for data loading
        """
        self.dataset = dataset
        self.contamination = contamination
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.include_test = include_test
        self.path_prefix = path_prefix

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if dataset == "ecg5000" and path_prefix is not None:
            x, a = load_dataset(
                dataset, ds_path=path_prefix + "/data/datasets/ecg/ecg.csv"
            )
        else:
            x, a = load_dataset(dataset)

        n_total_outliers = int(contamination * x.shape[0] / (1 - contamination))
        n_data_outliers = min(a.shape[0], n_total_outliers // 2)

        if path_prefix is not None:
            gen_anomaly_path = (
                f"{path_prefix}/data/datasets/anomalies/{dataset}_gen.pickle"
            )
        else:
            gen_anomaly_path = f"data/datasets/anomalies/{dataset}_gen.pickle"

        with open(gen_anomaly_path, "rb") as gen_file:
            gen_data = pickle.load(gen_file)

        n_gen_outliers = (n_total_outliers - n_data_outliers) // len(gen_data.keys())

        self.X = np.vstack(
            [
                x,
                self.rng.choice(a, size=n_data_outliers, replace=False),
                *[
                    self.rng.choice(gen_data[k], size=n_gen_outliers, replace=False)
                    for k in gen_data.keys()
                ],
            ]
        )
        self.X = _scale(self.X, scaler)

        self.y = np.concatenate(
            [
                np.zeros(x.shape[0]),
                np.ones(n_data_outliers),
                *[
                    np.ones(n_gen_outliers) * (n + 2)
                    for n in range(len(gen_data.keys()))
                ],
            ]
        )

    def split(self) -> TrainingKFoldIterator:
        """Split the dataset into folds."""
        return TrainingKFoldIterator(
            self.X, self.y, self.n_splits, self.shuffle, self.seed, self.include_test
        )

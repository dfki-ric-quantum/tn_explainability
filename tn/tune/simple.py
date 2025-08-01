from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import optuna
from ml_collections import config_dict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from tn.data.loader import load_dataset
from tn.data.prepare import prepare_dataset
from tn.ml.simple import SVM, IsolationForest, SimpleModelBase


def get_objective(
    estimator: str,
    dataset: str,
    seed: int,
    n_splits: int,
    contamination: float = 0.05,
    n_jobs: Optional[int] = None,
) -> type["BaseObjective"]:
    """Factory function to create the tuning objective for simple sklearn models.

    Parameters
    ----------
    estimator: str
        The sklearn estimator to use. currently supported are 'svm' for a one
        class SVM and 'if' for the isolation forest algorithm.
    dataset: str
        Name of the dataset to tune on
    seed: int
        Random seed
    n_splits: int
        Number of random subsets of the dataset the MPS should be trained on.
    contamination: float = 0.05
        Percentage of outlier contamination in the dataset
    n_jobs: Optional[int] = None
        Number of jobs to use for the training. None means 1, -1 uses all CPUs.
    """
    match estimator:
        case "svm":
            return SVMObjective(dataset, seed, n_splits, contamination, n_jobs)
        case "if":
            return IFObjective(dataset, seed, n_splits, contamination, n_jobs)
        case _:
            raise ValueError(f"No objective for estimator: {estimator}.")


class BaseObjective(ABC):
    def __init__(
        self,
        dataset: str,
        seed: int,
        n_splits: int,
        contamination: float = 0.05,
        n_jobs: Optional[int] = None,
    ) -> None:
        """The constructor

        Parameters
        ----------
        dataset: str
            Name of the dataset to tune on
        seed: int
            Random seed
        n_splits: int
            Number of random subsets of the dataset the MPS should be trained on.
        contamination: float = 0.05
            Percentage of outlier contamination in the dataset
        n_jobs: Optional[int] = None
            Number of jobs to use for the training. None means 1, -1 uses all CPUs.
        """

        self.rng = np.random.default_rng(seed)
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.contamination = contamination

        inliers, outliers = load_dataset(dataset)
        self.x, self.y, _, _ = prepare_dataset(
            inliers,
            outliers,
            scaler="minmax",
            random_state=self.rng,
            contamination=contamination,
        )

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Call the tuning objective.

        Parameters
        ----------
        trial: optuna.trial.Trial
            The current trial

        Returns
        -------
        Mean ROC AUC score over the splits.
        """

        cfg = self._get_config(trial)
        auc_scores = []

        spliter = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2)

        # Note: we ignore the "test" set, since optuna does not support pruning over
        # multiple objectives
        for fold_idx, (train_idxs, _) in enumerate(spliter.split(self.x, self.y)):
            model = self._build_model(cfg)
            model.fit(self.x[train_idxs])
            scores = model.score(self.x[train_idxs])
            auc_score = roc_auc_score(self.y[train_idxs], scores)
            trial.report(auc_score, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

            auc_scores.append(auc_score)

        return np.mean(auc_scores)

    @abstractmethod
    def _get_config(self, trial: optuna.trial.Trial) -> config_dict.FrozenConfigDict:
        """Suggest trial config

        Parameters
        ----------
        trial: optuna.trial.Trial
            The current trial

        Returns
        -------
        The configuration for model/training
        """
        pass

    @abstractmethod
    def _build_model(self, cfg: config_dict.FrozenConfigDict) -> type[SimpleModelBase]:
        """Build model based on configuration

        Parameters
        ----------
        cfg: config_dict.FrozenConfigDict
            The model configuration

        Returns
        -------
        The model, ready for training.
        """
        pass


class SVMObjective(BaseObjective):
    def _get_config(self, trial: optuna.trial.Trial) -> config_dict.FrozenConfigDict:
        """Suggest trial config

        Parameters
        ----------
        trial: optuna.trial.Trial
            The current trial

        Returns
        -------
        The configuration for model/training
        """
        cfg = config_dict.ConfigDict()
        cfg.kernel = "rbf"
        cfg.gamma = trial.suggest_float("gamma", low=0.1, high=5.0)
        cfg.n_components = trial.suggest_int("n_components", low=100, high=1000)
        cfg.nu = self.contamination
        return config_dict.FrozenConfigDict(cfg)

    def _build_model(self, cfg: config_dict.FrozenConfigDict) -> type[SimpleModelBase]:
        """Build model based on configuration

        Parameters
        ----------
        cfg: config_dict.FrozenConfigDict
            The model configuration

        Returns
        -------
        The model, ready for training.
        """
        return SVM(cfg, n_jobs=self.n_jobs)


class IFObjective(BaseObjective):
    def _get_config(self, trial: optuna.trial.Trial) -> config_dict.FrozenConfigDict:
        """Suggest trial config

        Parameters
        ----------
        trial: optuna.trial.Trial
            The current trial

        Returns
        -------
        The configuration for model/training
        """
        cfg = config_dict.ConfigDict()
        cfg.n_estimators = trial.suggest_int("n_estimators", low=50, high=500)
        cfg.max_samples = trial.suggest_float("max_samples", low=0.1, high=1.0)
        return config_dict.FrozenConfigDict(cfg)

    def _build_model(self, cfg: config_dict.FrozenConfigDict) -> type[SimpleModelBase]:
        """Build model based on configuration

        Parameters
        ----------
        cfg: config_dict.FrozenConfigDict
            The model configuration

        Returns
        -------
        The model, ready for training.
        """
        return IsolationForest(cfg, n_jobs=self.n_jobs)

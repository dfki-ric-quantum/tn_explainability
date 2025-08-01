from abc import ABC, abstractmethod

import jax
import numpy as np
import optuna
from flax.training import train_state
from ml_collections import config_dict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from tn.data.loader import load_dataset
from tn.data.prepare import prepare_dataset
from tn.ml.ae.train import score_ae, train_ae


def get_objective(
    dataset: str,
    seed: int,
    n_splits: int,
    contamination: float = 0.05,
) -> type["BaseObjective"]:
    """Factory function to create the tuning objective an auto encoder.

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
    """
    return AEObjective(dataset, seed, n_splits, contamination)


class BaseObjective(ABC):
    def __init__(
        self,
        dataset: str,
        seed: int,
        n_splits: int,
        contamination: float = 0.05,
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
        """

        self.rng = np.random.default_rng(seed)
        self.n_splits = n_splits
        self.contamination = contamination

        inliers, outliers = load_dataset(dataset)

        self.x, self.y, _, _ = prepare_dataset(
            inliers,
            outliers,
            scaler="minmax",
            random_state=self.rng,
            contamination=self.contamination,
        )

        self.key = jax.random.key(seed)

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

        for fold_idx, (train_idxs, _) in enumerate(spliter.split(self.x, self.y)):
            state = self._train(self.x[train_idxs], cfg)
            scores = self._score(self.x[train_idxs], cfg, state)
            auc_score = roc_auc_score(self.y[train_idxs], scores)
            trial.report(auc_score, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

            auc_scores.append(auc_score)

        return np.mean(auc_scores)

    @abstractmethod
    def _train(
        self, x: np.ndarray, cfg: config_dict.FrozenConfigDict
    ) -> train_state.TrainState:
        """Train model.

        Parameters
        ----------
        x: np.ndarray
            The data to train on
        cfg: config_dict.FrozenConfigDict
            model/training config

        Returns
        -------
        The train state.
        """
        pass

    @abstractmethod
    def _score(
        self,
        x: np.ndarray,
        cfg: config_dict.FrozenConfigDict,
        state: train_state.TrainState,
    ) -> jax.typing.ArrayLike:
        """Score the model.

        Parameters
        ----------
        x: np.ndarray
            The data to train on
        cfg: config_dict.FrozenConfigDict
            model/training config
        state: train_state.TrainState
            The training state of the model

        Returns
        -------
        Score for each sample
        """
        pass

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
        cfg.out_dim = self.x.shape[1]
        cfg.activation = trial.suggest_categorical(
            "activation", choices=["relu", "elu", "leaky_relu"]
        )
        cfg.batch_size = trial.suggest_categorical(
            "batch_size", choices=[32, 64, 128, 256]
        )
        cfg.n_epochs = 10
        cfg.lrate = trial.suggest_float("lrate", low=0.001, high=0.01)

        first_layer = trial.suggest_categorical("first_layer", [32, 64, 128, 256])
        cfg.layers = [first_layer, first_layer // 2]

        cfg.n_latent = int(
            self.x.shape[1] * trial.suggest_float("latent_ration", low=0.1, high=0.5)
        )

        return config_dict.FrozenConfigDict(cfg)


class AEObjective(BaseObjective):
    def _train(
        self, x: np.ndarray, cfg: config_dict.FrozenConfigDict
    ) -> train_state.TrainState:
        """Train model.

        Parameters
        ----------
        x: np.ndarray
            The data to train on
        cfg: config_dict.FrozenConfigDict
            model/training config

        Returns
        -------
        The train state.
        """
        self.key, subkey = jax.random.split(self.key)
        _, state = train_ae(x, cfg=cfg, random_key=subkey)

        return state

    def _score(
        self,
        x: np.ndarray,
        cfg: config_dict.FrozenConfigDict,
        state: train_state.TrainState,
    ) -> jax.typing.ArrayLike:
        """Score the model.

        Parameters
        ----------
        x: np.ndarray
            The data to train on
        cfg: config_dict.FrozenConfigDict
            model/training config
        state: train_state.TrainState
            The training state of the model

        Returns
        -------
        Score for each sample
        """
        return score_ae(state.params, x, cfg)

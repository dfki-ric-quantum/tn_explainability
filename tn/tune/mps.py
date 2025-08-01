import jax
import numpy as np
import optuna
from ml_collections import config_dict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from tn.data.loader import load_dataset
from tn.data.prepare import prepare_dataset
from tn.encoding import get_encoder
from tn.mps.metrics import NLLFunctor
from tn.mps.mps import random_mps
from tn.mps.train import train_mps


def get_hyper_params(trial: optuna.trial.Trial) -> dict:
    """Get hyper-parameters. Note: The range of the parameters was first narrowed
    in several coarse hyper-parameter searches, to reduce the overall search space.

    Parameters
    ----------
    trial: optuna.trial.Trial
        The current trial

    Returns
    -------
    A dictonary containing all the hyper-parameters.
    """
    return dict(
        phys_dim=trial.suggest_int(name="phys_dim", low=3, high=6),
        batch_size=trial.suggest_categorical(name="batch_size", choices=[64, 128, 256]),
        lrate=trial.suggest_categorical(name="lrate", choices=[0.001, 0.01]),
        max_bond=trial.suggest_categorical(name="max_bond", choices=[10, 20, 30]),
        sigma_thresh=trial.suggest_categorical(
            name="sigma_thresh", choices=[0.01, 0.05]
        ),
    )


def get_config(hyper_params: dict) -> config_dict.FrozenConfigDict:
    """Turn hyper-parameters into MPS train/model config.

    Parameters
    ----------
    hyper_params: dict
        Dictonary containing the trial hyper-parameters

    Returns
    -------
    The config dictonary
    """
    cfg = config_dict.ConfigDict()

    cfg.bond_dim = 2
    cfg.max_bond = hyper_params["max_bond"]
    cfg.sigma_thresh = hyper_params["sigma_thresh"]
    cfg.d = hyper_params["phys_dim"]
    cfg.encoder = "legendre"
    cfg.encoder_kwargs = {"shifted": True, "p": list(range(cfg.d))}
    cfg.n_sweeps = 1
    cfg.lrate = hyper_params["lrate"]
    cfg.batch_size = hyper_params["batch_size"]

    return config_dict.FrozenConfigDict(cfg)


class Objective:
    """Hyper-parameter tuning objective for MPS training."""

    def __init__(
        self, dataset: str, seed: int, n_splits: int, contamination: float = 0.05
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
        self.key = jax.random.key(seed)
        self.n_splits = n_splits

        inliers, outliers = load_dataset(dataset)
        self.x, self.y, _, _ = prepare_dataset(
            inliers,
            outliers,
            scaler="minmax",
            random_state=self.rng,
            contamination=contamination,
        )

        self.n_sites = self.x.shape[1]

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

        hyper_params = get_hyper_params(trial)
        cfg = get_config(hyper_params=hyper_params)

        encoder = get_encoder(cfg.encoder, **cfg.encoder_kwargs)
        x = encoder(self.x)

        auc_scores = []

        spliter = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2)

        # Note: we ignore the "test" set, since optuna does not support pruning over
        # multiple objectives
        for fold_idx, (train_idxs, _) in enumerate(spliter.split(x, self.y)):
            self.key, subkey = jax.random.split(self.key)
            mps = random_mps(self.n_sites, bond_dim=2, random_key=subkey, d=cfg.d)
            mps.left_canonicalize()

            train_mps(mps, x, cfg=cfg)

            loss_fn = NLLFunctor(mps.shapes, mps.d)
            scores = loss_fn(x[train_idxs], mps)
            auc_score = roc_auc_score(self.y[train_idxs], scores)
            trial.report(auc_score, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

            auc_scores.append(auc_score)

        return np.mean(auc_scores)

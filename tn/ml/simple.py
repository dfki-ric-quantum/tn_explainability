from abc import ABC
from typing import Optional

import numpy as np
from ml_collections import config_dict
from sklearn.ensemble import IsolationForest as IFBase
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline


class SimpleModelBase(ABC):
    """Wrapper base class for simple sklearn models."""

    def __init__(self, cfg: config_dict.FrozenConfigDict) -> None:
        """The constructor.

        Parameters
        ----------
        cfg: config_dict.FrozenConfigDict
            The model configuration
        """
        self.cfg = cfg

    def fit(self, x: np.ndarray) -> None:
        """Fit model to data.

        Parameters
        ----------
        x: np.ndarray
            The data to fit to, shape (n_samples, n_features)
        """
        self.base.fit(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score data.

        Note: we invert the sign here, as sklearn scores outliers low and inliers
        high, where we do the opposite (to be better compatible with loss based scoring)

        Parameters
        ----------
        x: np.ndarray
            The data to score to, shape (n_samples, n_features)

        Returns
        -------
        The score for each sample, shape (n_samples,)

        """
        return -self.base.decision_function(x)


class SVM(SimpleModelBase):
    """Wrapper for One-class SVM."""

    def __init__(
        self, cfg: config_dict.FrozenConfigDict, n_jobs: Optional[int] = None
    ) -> None:
        """The constructor.

        Parameters
        ----------
        cfg: config_dict.FrozenConfigDict
            The model configuration
        n_jobs: Optional[int] = None
            Number of jobs to use for the training. None means 1, -1 uses all CPUs.
        """
        super().__init__(cfg)

        self.base = make_pipeline(
            Nystroem(
                kernel=cfg.kernel,
                gamma=cfg.gamma,
                n_components=cfg.n_components,
                n_jobs=n_jobs,
            ),
            SGDOneClassSVM(nu=cfg.nu),
        )


class IsolationForest(SimpleModelBase):
    """Wrapper for the Isolation Forest algorithm."""

    def __init__(
        self, cfg: config_dict.FrozenConfigDict, n_jobs: Optional[int] = None
    ) -> None:
        """The constructor.

        Parameters
        ----------
        cfg: config_dict.FrozenConfigDict
            The model configuration
        n_jobs: Optional[int] = None
            Number of jobs to use for the training. None means 1, -1 uses all CPUs.
        """
        super().__init__(cfg)
        self.base = IFBase(
            n_estimators=cfg.n_estimators, max_samples=cfg.max_samples, n_jobs=n_jobs
        )

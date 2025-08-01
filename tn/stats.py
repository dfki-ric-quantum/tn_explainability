from typing import Optional

import jax
import numpy as np
from scipy.stats import pearsonr

_EPS = 1e-5


def _mi_hist_marginal_bins(N: int) -> int:
    """Optimal number of bins for a low-bias historgram based mutual informatin estimation.

    The function computes the number of bins for the marginal entropy estimation in a low-bias
    mutual information estimator [1].

    [1] Hacine-Gharbi, A. et al. "Low Bias Histogram-Based Estimation of Mutual Information
    for Feature Selection." (2012).

    Parameters
    ----------
    N: int
        Number of samples for which to estimate the marginal entropy H(X)

    Returns
    -------
    The number of bins for the histogram based estimator.
    """
    xi = np.power(8 + 324 * N + 12 * np.sqrt(36 * N + 729 * N**2), 1 / 3)
    return int(xi / 6 + 2 / (3 * xi) + 1 / 3)


def _mi_hist_joint_bins(N: int, corr: float) -> int:
    """Optimal number of bins for a low-bias historgram based mutual informatin estimation.

    The function computes the number of bins for the joint entropy estimation in a low-bias
    mutual information estimator [1].

    [1] Hacine-Gharbi, A. et al. "A Binning Formula of Bi-histogram for Joint Entropy
    Estimation Using Mean Square Error Minimization." (2018)

    Parameters
    ----------
    N: int
        Number of samples for which to estimate the joint entropy H(X, Y)
    corr: float
        Correlation corr(X, Y)

    Returns
    -------
    The number of bins for the histogram based estimator.
    """
    return int(np.sqrt(1 / np.sqrt(2) * (1 + np.sqrt(1 + 24 * N / (1 - corr**2)))))


def mutual_information(
    data: jax.typing.ArrayLike, corr: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate all-to-all feature mutual information.

    The low-bias estimator that makes no assumptions about the underlying probability distribution,
    probability densities are estimated with a histogram based approach using optimal bin-sizes for
    marginal and joint estimates.

    Parameters
    ----------
    data: jax.typing.ArrayLike
        The data to estimate the mutual information on, shape (n_samples, n_features)
    corr: Optional[np.ndarry]
        The all-to-all feature correlations. They are needed to compute the bin sizes for
        the joint probability histograms. If not provided, they'll be computed and returned
        together with the mutual information. Shape (n_features, n_features)

    Returns
    -------
    Mutual information estimate, shape (n_features, n_features) and all-to-all feature
    correlation, shape (n_features, n_features). If no correlations were provided, these are the
    all-to-all Pearson correlation coefficients, otherwise the ones provided.

    """
    if corr is None:
        corr = all_to_all_correlation(data)

    n_samples, n_features = data.shape

    n_marginal_bins = _mi_hist_marginal_bins(n_samples)
    marginal_entropies = np.empty(n_features, dtype=float)

    for i in range(n_features):
        histogram = np.histogram(data[:, i], bins=n_marginal_bins)[0]
        probs = histogram / n_samples + _EPS

        marginal_entropies[i] = -np.sum(probs * np.log(probs))

    joint_entropies = np.empty((n_features, n_features), dtype=float)

    for i in range(n_features):
        joint_entropies[i, i] = marginal_entropies[i]
        for j in range(i + 1, n_features):
            n_bins = _mi_hist_joint_bins(n_samples, corr[i, j])

            histogram = np.histogram2d(data[:, i], data[:, j], bins=n_bins)[0]
            probs = histogram / n_samples + _EPS

            joint_entropies[i, j] = -np.sum(probs * np.log(probs))
            joint_entropies[j, i] = joint_entropies[i, j]

    tiled_marginals = np.tile(marginal_entropies, (n_features, 1))
    mutual_information = tiled_marginals + tiled_marginals.T - joint_entropies

    return mutual_information, corr


def all_to_all_correlation(data: jax.typing.ArrayLike) -> np.ndarray:
    """All-to-all linear feature correlation within a data set.

    Computes the Pearson correlation coefficient for all combinations of features.

    Parameters
    ----------
    data: jax.typing.ArrayLike
        The data set, shape (n_samples, n_features)

    Returns
    -------
    The correlations in a symmetric matrix of shape (n_features, n_features)
    """
    n_features = data.shape[1]

    pcorr = np.empty((n_features, n_features), dtype=float)

    for i in range(n_features):
        pcorr[i, i] = 1.0

        for j in range(i + 1, n_features):
            pcorr[i, j] = pearsonr(data[:, i], data[:, j]).statistic
            pcorr[j, i] = pcorr[i, j]

    return pcorr

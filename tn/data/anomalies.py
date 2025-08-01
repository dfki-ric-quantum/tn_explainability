import numpy as np
import pandas as pd
from copulas.univariate import GaussianKDE
from sklearn.mixture import GaussianMixture


def generate_all_anomalies(
    X: np.ndarray,
    n_anomalies: int,
    alpha: int = 5,
    percentage: float = 0.1,
    seed: int = 42,
) -> dict:
    """Returns all types of anomalies for a given dataset

    Parameters
    ----------
    X : np.ndarray
        Data without outliers to create outliers from
    n_anomalies : int
        Number of anomalies to create
    alpha : int, optional
        Scaling parameter for covariance and mean, by default 5
    percentage : float, optional
        Scaling parameter for global anomaly, by default 0.1
    seed : int, optional
        Seed for random number generation, by default 42

    Returns
    -------
    dict
        Returns a dict containing the anomalies for each type
    """
    anomaly_types = ["local", "cluster", "dependency", "global"]
    all_anomalies = []

    for type in anomaly_types:
        all_anomalies.append(
            generate_anomaly(X, type, n_anomalies, alpha, percentage, seed)
        )

    return dict(zip(anomaly_types, all_anomalies))


def generate_anomaly(
    X: np.ndarray,
    anomaly_type: str,
    n_anomalies: int,
    alpha: int = 5,
    percentage: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generates anomalies for given anomaly type and data

    Parameters
    ----------
    X : np.ndarray
        Data without outliers
    anomaly_type : str
        Type of anomaly
    n_anomalies : int
        Number of anomalies
    alpha : int, optional
        Scaling parameter for mean and covariance, by default 5
    percentage : float, optional
        Scaling param for global anomaly, by default 0.1
    seed : int, optional
        Random number Seed, by default 42

    Returns
    -------
    np.ndarray
        Generated anomalies

    Raises
    ------
    NotImplementedError
        If anomaly type not implemented
    """
    print(f"Generating artificial anomalies for type {anomaly_type}")

    if anomaly_type not in ["local", "cluster", "dependency", "global"]:
        raise NotImplementedError

    if anomaly_type in ["local", "cluster"]:
        bic_metric = []
        n_components = list(np.arange(1, 10))

        for n_c in n_components:
            gm = GaussianMixture(n_components=n_c, random_state=seed).fit(X)
            bic_metric.append(gm.bic(X))

        best = n_components[np.argmin(bic_metric)]
        gm = GaussianMixture(n_components=best, random_state=seed).fit(X)

    if anomaly_type == "local":
        gm.covariances_ = alpha * gm.covariances_
        anomalies = gm.sample(n_anomalies)[0]

    if anomaly_type == "cluster":
        gm.means_ = alpha * gm.means_
        anomalies = gm.sample(n_anomalies)[0]

    if anomaly_type == "dependency":
        anomalies = np.zeros((n_anomalies, X.shape[1]))
        for i in range(X.shape[1]):
            kde = GaussianKDE()
            kde.fit(X[:, i])
            anomalies[:, i] = kde.sample(n_anomalies)

    if anomaly_type == "global":
        anomalies = []
        for i in range(X.shape[1]):
            min = np.min(X[:, i]) * (1 + percentage)
            max = np.max(X[:, i]) * (1 + percentage)
            anomalies.append(np.random.uniform(min, max, size=n_anomalies))
        anomalies = np.array(anomalies).T

    return anomalies

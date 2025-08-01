import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from tn.data.anomalies import generate_anomaly


def _random_state(random_state: int | np.random.Generator) -> np.random.Generator:
    """Return random number generator for a seed, the generator else."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    else:
        return np.random.default_rng(random_state)


def _scale(data: np.ndarray, scaler: str) -> np.ndarray:
    """Scale dataset."""
    match scaler:
        case "maxabs":
            return MaxAbsScaler().fit_transform(data)
        case "minmax":
            return MinMaxScaler().fit_transform(data)
        case "standard":
            return StandardScaler().fit_transform(data)
        case "none":
            return data
        case _:
            raise ValueError(f"Scaler {scaler} is not implemented.")


def prepare_dataset(
    inliers: np.ndarray,
    outliers: np.ndarray,
    scaler: str,
    random_state: int | np.random.Generator,
    contamination: float = 0.05,
    anomaly_types: list[str] = None,
    anomaly_contamination: float = 0.0,
    alpha: int = 5,
    percentage: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare dataset for outlier detection training.

    Parameters
    ----------
    inliers: np.ndarray,
        The inlier data, shape (n_samples, n_features)
    outliers: np.ndarray,
        the outlier data, shape (n_samples, n_features)
    scaler: str,
        The preprocessing scaler to use:
        * maxabs: scale in [-1, 1]
        * minmax: scale in [0, 1]
        * standard: scale to standard normal distribution
        * none: no preprocessing
    random_state: int | np.random.Generator,
        Random state to use for sampling outliers, either a seed or a random number
        generator
    contamination: float = 0.05,
        Percentage of contamination with outliers in the resulting data set
    anomaly_types: list[str] = None
        Types of artificial anomalies to generate for the given data
    anomaly_contamination: float = 0.0
        Percentage of contamination with artificial anomalies in the resulting data
        set. If several types are given, they are distributed equally over anomaly
        contamination.
    alpha: int = 5
        Scaling parameter for covariance and mean for artificial outliers.
    percentage: float = 0.1
        Scaling parameter for artifical global anomalies.
    seed: int = 42
        Seed for random number generation in anomaly generation.
    Returns
    -------
    The data set, labels (0 -> inliers, 1 -> outliers), an index set of the outliers
    mixed into the dataset and the generated anomalies.
    """
    rng = _random_state(random_state)

    n_inliers = inliers.shape[0]
    n_outliers = int((contamination * n_inliers) / (1 - contamination))

    replace = n_outliers > outliers.shape[0]

    outlier_idxs = rng.choice(
        np.arange(0, outliers.shape[0]), n_outliers, replace=replace
    )

    outliers_to_add = outliers[outlier_idxs]

    generated_anomalies = []
    if anomaly_types is not None and anomaly_contamination > 0:
        cont_per_type = anomaly_contamination / len(anomaly_types)
        n_anomalies = int((cont_per_type * n_inliers) / (1 - cont_per_type))

        for type in anomaly_types:
            generated_anomalies.append(
                generate_anomaly(inliers, type, n_anomalies, alpha, percentage, seed)
            )

        generated_anomalies = np.vstack(generated_anomalies)

    if len(generated_anomalies) > 0:
        outliers_to_add = np.vstack([outliers_to_add, generated_anomalies])

    x = np.vstack([inliers, outliers_to_add])
    x = _scale(x, scaler)
    y = np.concatenate([np.zeros(n_inliers), np.ones(outliers_to_add.shape[0])])

    return x, y, outlier_idxs, generated_anomalies

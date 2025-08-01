import numpy as np
import pandas as pd

from tn.data.fetch import fetch


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Load the specified dataset with its default settings.

    Parameters
    ----------
    name: str
        Name of the dataset

    Returns
    -------
    The regular data and the anomalies
    """
    match name:
        case "satellite":
            return load_satellite()
        case "spambase":
            return load_spambase()
        case "ecg5000":
            return load_ecg(**kwargs)
        case _:
            raise ValueError(f"No such dataset: {name}")


def load_satellite(
    anomaly_classes: list[int] = [2, 4, 5]
) -> tuple[np.ndarray, np.ndarray]:
    """Loads the statlog (Landsat Satellite) dataset from the UCI Machine Learning repository.

    Parameters
    ----------
    anomaly_classes: list[int]] = [2,4,5]
        A single label or a list of labels to consider anomalies.

    Returns
    -------
    The the data and anomalies from the given anomly_classes.
    """
    satellite = fetch(repo_id=146)

    X = satellite.data.features.to_numpy()
    y = satellite.data.targets.to_numpy().reshape(-1)

    assert np.all(
        np.isin(np.array(anomaly_classes), np.unique(y))
    ), "Anomalies out of class range"

    anomaly_mask = np.isin(y, np.array(anomaly_classes))
    data_mask = np.invert(anomaly_mask)

    return X[data_mask], X[anomaly_mask]


def load_spambase() -> tuple[np.ndarray, np.ndarray]:
    """Loads the spambase dataset from the UCI Machine Learning repository.

    Returns
    -------
    The the data and anomalies.
    """
    spambase = fetch(repo_id=94)
    X = spambase.data.features.to_numpy()
    y = spambase.data.targets.to_numpy().reshape(-1)

    return X[y == 0], X[y == 1]


def load_ecg(
    ds_path: str = "data/datasets/ecg/ecg.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the ECG5000 dataset.

    Parameters
    ----------
    ds_path: str
        Path to the dataset file

    Returns
    -------
    The data and anonalies.
    """
    ecg = pd.read_csv(ds_path, header=None)

    X = ecg.values[:, 0:-1]
    y = ecg.values[:, -1]

    return X[y == 1], X[y == 0]

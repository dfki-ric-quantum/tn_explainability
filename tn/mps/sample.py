import numpy as np

from tn.mps import MPS


def sample_binary(mps: MPS, rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
    """Create full samples from the MPS

    Parameters
    ----------
    mps: MPS
        The MPS to sample from
    rng: np.random.Generator
        Random generator to sample with
    n_samples: int = 1
        The number of samples to create

    Returns
    -------
    An np.ndarray of shape (n_samples, mps.n_sites) with each row being one of the
    requested samples.
    """
    rand = rng.uniform(size=(n_samples, mps.n_sites))
    out = np.empty(shape=(n_samples, mps.n_sites), dtype=np.int32)
    x = np.array([1.0])

    for sample in range(n_samples):
        for idx in reversed(range(mps.n_sites)):
            xa = np.matmul(mps[idx][:, 1, :], x)
            sq_norm = (np.linalg.norm(xa) / np.linalg.norm(x)) ** 2

            if rand[sample, idx] < sq_norm:
                out[sample, idx] = 1
                x = xa
            else:
                out[sample, idx] = 0
                x = np.matmul(mps[idx][:, 0, :], x)

    return out


def sample_binary_partial(
    partial: np.ndarray, mask: np.ndarray, mps: MPS, rng: np.random.Generator
) -> np.ndarray:
    """Create full samples from the MPS

    Parameters
    ----------
    partial: np.ndarray
        Partial samples to be completed
    mask: np.ndarray
        Booleans of the same shape as partial. True if that index needs to be sampled
        False if it should be kept.
    mps: MPS
        The MPS to sample from
    rng: np.random.Generator
        Random generator to sample with

    Returns
    -------
    An np.ndarray the same shape as partial with each row being one of the
    requested samples.
    """
    out = np.copy(partial)
    rand = rng.uniform(size=out.shape)

    for sample in range(out.shape[0]):
        x = np.array([1.0])

        for idx in reversed(range(mps.n_sites)):
            if mask[sample, idx]:
                xa = np.matmul(mps[idx][:, 1, :], x)
                sq_norm = (np.linalg.norm(xa) / np.linalg.norm(x)) ** 2

                if rand[sample, idx] < sq_norm:
                    out[sample, idx] = 1
                    x = xa
                else:
                    out[sample, idx] = 0
                    x = np.matmul(mps[idx][:, 0, :], x)
            else:
                x = np.matmul(mps[idx][:, out[sample, idx], :], x)

    return out

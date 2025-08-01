from typing import Callable, Optional

import jax
import jax.numpy as jnp
from scipy.integrate import quad

_EPS = 1e-5


def von_neumann_entropy(density_matrix: jax.typing.ArrayLike) -> jax.Array:
    """Compute the von Neumann entropy of a (reduced) density matrix.

    Instead of

        S = -tr(rho * log(rho))

    we compute

        S = -sum e*log(e)

    for all eigenvalues e of rho, since jax.scipy doesn't have a matrix logarithm yet. As rho is
    hermitian, this is fast (and faster than e.g. using scipy.linalg.logm).

    Parameters
    ----------
    density_matrix: jax.typing.ArrayLike
        The density matrix

    Returns
    -------
    The von Neumann Entropy (relative to base e)
    """
    evals, _ = jnp.linalg.eigh(density_matrix)
    evals = jnp.where(jnp.abs(evals) < 0.0, _EPS, evals)
    return -jnp.dot(evals, jnp.log(jnp.abs(evals + _EPS)))


class SingleMarginalPDF:
    """A single site marginal probability density function from a reduced density matrix."""

    def __init__(
        self, rdm: jax.typing.ArrayLike, encoder: Callable, low: float, high: float
    ) -> None:
        """The constructor

        Parameters
        ----------
        rdm: jax.typing.ArrayLike
            The reduced density matrix for a single site
        encoder: Callable
            The encoder to encode the data into a feature vector
        low: float
            Lower bound of the data
        high: float
            Upper bound of the data
        """
        self.rdm = rdm
        self.low = low
        self.high = high
        self.encoder = encoder
        self.d = quad(self._q, self.low, self.high, full_output=1)[0]

        self._mu: Optional[float] = None
        self._var: Optional[float] = None

    def __call__(self, x: float) -> float:
        """Compute probability density for a value

        Parameters
        ----------
        x: float
            The value to compute the probability density for

        Returns
        -------
        The probability density f(x)
        """
        return self._q(x) / self.d

    @property
    def expected_value(self) -> float:
        """Expected value of the PDF. Lazy evaluation."""
        if self._mu is None:
            self._mu = quad(lambda x: x * self(x), self.low, self.high, full_output=1)[
                0
            ]

        return self._mu

    @property
    def variance(self) -> float:
        """Variance of the PDF. Lazy evaluation."""
        if self._var is None:
            mu = self.expected_value
            self._var = quad(
                lambda x: (x - mu) ** 2 * self(x), self.low, self.high, full_output=1
            )[0]

        return self._var

    def _q(self, x: float) -> float:
        """Unormalized probability density value."""
        v = self.encoder.encode_single(x)
        return jnp.trace(self.rdm @ jnp.outer(v, v))

import math

import jax
import jax.numpy as jnp

he = [
    jnp.array([1]),
    jnp.array([1, 0]),
    jnp.array([1, 0, -1]),
    jnp.array([1, 0, -3, 0]),
    jnp.array([1, 0, -6, 0, 3]),
    jnp.array([1, 0, -10, 0, 15, 0]),
    jnp.array([1, 0, -15, 0, 45, 0, -15]),
    jnp.array([1, 0, -21, 0, 105, 0, -105, 0]),
    jnp.array([1, 0, -28, 0, 210, 0, -420, 0, 105]),
    jnp.array([1, 0, -36, 0, 378, 0, -1260, 0, 945, 0]),
]


def hermite_polynomial(n: int, x: jax.typing.ArrayLike) -> jax.Array:
    """Evaluate the n-th 'probabilistic' Hermite polynomial.

    Parameters
    ----------
    n: int
        The n-th polynomial
    x: jax.typing.ArrayLike
        The input to evaluate the polynomial on

    Returns
    -------
    The evaluation result, same shape as x
    """
    return jnp.polyval(he[n], x)


class IsoHermitePolynomial:
    """Isometrized Hermite Polynomial.

    The object is a callable of the n-th "probabilistic" Hermite polynomial, which is
    isometrized by scaling it by the square root of its weight functions:

        w(x) = e^(-x^2/2)

    and a normalization constant:

        N = √(1/(√(2*π)) * n!)

    Hence any two of these polynomials fulfill the following isometrie conditions:

        (1)  ∫ he_m(x)*he_n(x)dx = δ(m,n)        (orthonormal)

    with δ(m,n) being the Kronecker delta.

        (2)  Σ he_i(x)*he_i(x') = Π(x, x')       (kernel function)

    with ∫ Π(x, x')Π(x', x'')dx' = Π(x, x'').
    """

    def __init__(self, n: int) -> None:
        """The constructor

        Parameters
        ----------
        n: int
            the n-th Hermite polynomial
        """
        self.n = n
        self.normalization = jnp.sqrt(1 / (jnp.sqrt(2 * jnp.pi) * math.factorial(n)))

    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        """Call the polynomial.

        Parameters
        ----------
        x: jax.typing.ArrayLike
            Input to evaluate the kernel on, element-wise.

        Returns
        -------
        The evaluation result, same shape as x
        """
        return hermite_polynomial(self.n, x) * self._weight_term(x) * self.normalization

    def _weight_term(self, x: float) -> float:
        """Square root of the weight term, relative to which the Hermite Polynomials are
        orthogonal."""
        return jnp.exp(-(x**2) / 4)


class HermitePolyEncoder:
    """Encode real valued data with Hermite Polynomials."""

    RANGE = (-jnp.inf, jnp.inf)

    def __init__(self, p: list[int]) -> None:
        """The constructor

        Parameters
        ----------
        p: list[int]
            List of indeces for the polynomial to be used for encoding. The number
            of indices corresponds to the dimensionality of output state(s).
        """
        self.encoder = [IsoHermitePolynomial(n) for n in p]

        # These are callable
        self.encode_sample = jax.vmap(self.encode_single)
        self.encode_samples = jax.vmap(self.encode_sample)

    def encode_single(self, x: jax.typing.ArrayLike | float) -> jax.Array:
        """Encodes a single value.

        Parameters
        ----------
        value: jax.typing.ArrayLike | float
            The value to encode, either as numpy/jax array or float.

        Returns
        -------
        The encoded value as array of shape (len(p),)
        """
        state = jnp.array([enc(x) for enc in self.encoder])
        return state / jnp.linalg.norm(state)

    def __call__(self, samples: jax.typing.ArrayLike) -> jax.Array:
        """Convinience wrapper for `encode_samples` to make the object callable.

        Encodes a set of samples.

        Parameters
        ----------
        samples: jax.typing.ArrayLike
            Samples to encode, shape (n_samples, n_features)

        Returns
        -------
        The encoded states, shape (n_samples, n_features, len(p))
        """
        return self.encode_samples(samples)

import jax
import jax.numpy as jnp

# fmt: off
la = [
    (1, jnp.array([1])),
    (1, jnp.array([-1, 1])),
    (1/2, jnp.array([1, -4, 2])),
    (1/6, jnp.array([-1, 9, -18, 6])),
    (1/24, jnp.array([1, -16, 72, -96, 24])),
    (1/120, jnp.array([-1, 25, -200, 600, -600, 120])),
    (1/720, jnp.array([1, -36, 450, -2400, 5400, -4320, 720])),
    (1/5040, jnp.array([-1, 49, -882, 7350, -29400, 52920, -35280, 5040])),
    (1/40320, jnp.array([1, -64, 1568, -18816, 117600, -376320, 564480, -322560, 40320]),),
]
# fmt: on


def laguerre_polynomial(n: int, x: jax.typing.ArrayLike) -> jax.Array:
    """Evaluate the n-th Laguerre polynomial.

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
    scale, coeffs = la[n]
    coeffs *= scale

    return jnp.polyval(coeffs, x)


class IsoLaguerrePolynomial:
    """Isometrized Laguerre Polynomial.

    The object is a callable of the n-th Laguerre polynomial, which is
    isometrized by scaling it by the square root of its weight functions:

        w(x) = e^-x

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
            the n-th Laguerre polynomial
        """
        self.n = n

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
        return laguerre_polynomial(self.n, x) * self._weight_term(x)

    def _weight_term(self, x: float) -> float:
        """Square root of the weight term, relative to which the Laguerre Polynomials are
        orthogonal."""
        return jnp.exp(-x/2)


class LaguerrePolyEncoder:
    """Encode real valued data with Laguerre Polynomials."""

    RANGE = (0.0, jnp.inf)

    def __init__(self, p: list[int]) -> None:
        """The constructor

        Parameters
        ----------
        p: list[int]
            List of indeces for the polynomial to be used for encoding. The number
            of indices corresponds to the dimensionality of output state(s).
        """
        self.encoder = [IsoLaguerrePolynomial(n) for n in p]

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

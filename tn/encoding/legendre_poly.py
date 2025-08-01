import jax
import jax.numpy as jnp

le = [
    (1, jnp.array([1])),
    (1, jnp.array([1, 0])),
    (1 / 2, jnp.array([3, 0, -1])),
    (1 / 2, jnp.array([5, 0, -3, 0])),
    (1 / 8, jnp.array([35, 0, -30, 0, 3])),
    (1 / 8, jnp.array([63, 0, -70, 0, 15, 0])),
    (1 / 16, jnp.array([231, 0, -315, 0, 105, 0, -5])),
    (1 / 16, jnp.array([429, 0, -693, 0, 315, 0, -35, 0])),
    (1 / 128, jnp.array([6435, 0, -12012, 0, 6930, 0, -1260, 0, 35])),
    (1 / 128, jnp.array([12155, 0, -25740, 0, 18018, 0, -4620, 0, 315, 0])),
]


def legendre_polynomial(
    n: int, x: jax.typing.ArrayLike, shifted: bool = False
) -> jax.Array:
    """Evaluate the n-th Legendre polynomial.

    Parameters
    ----------
    n: int
        The n-th polynomial
    x: jax.typing.ArrayLike
        The input to evaluate the polynomial on
    shifted: bool = False
        If False, evaluate the regular Legendre polynomial Pn(x), if
        True the shifted version Sn(x), given by

            Sn(x) = Pn(2x-1)

        is evaluated.

    Returns
    -------
    The evaluation result, same shape as x
    """
    scale, coeffs = le[n]
    coeffs *= scale

    if shifted:
        return jnp.polyval(coeffs, 2 * x - 1)
    else:
        return jnp.polyval(coeffs, x)


class IsoLegendrePolynomial:
    """Isometrized Legendre Polynomial.

    The object is a callable of the n-th Legendre polynomial, which is
    isometrized by scaling it by a normalization constant. For the regular polynomial
    the normalization constant is:

        N = √((2n+1)/2)

    For the shifted version:

        N = √(2n+1)

    Hence any two of these polynomials fulfill the following isometrie conditions:

        (1)  ∫ he_m(x)*he_n(x)dx = δ(m,n)        (orthonormal)

    with δ(m,n) being the Kronecker delta.

        (2)  Σ he_i(x)*he_i(x') = Π(x, x')       (kernel function)

    with ∫ Π(x, x')Π(x', x'')dx' = Π(x, x'').
    """

    def __init__(self, n: int, shifted: bool = False) -> None:
        """The constructor

        Parameters
        ----------
        n: int
            the n-th Legendre polynomial
        shifted: bool = False
            If False, evaluate the regular Legendre polynomial Pn(x), if
            True the shifted version Sn(x).
        """
        self.n = n
        self.shifted = shifted

        if shifted:
            self.normalization = jnp.sqrt(2 * n + 1)
        else:
            self.normalization = jnp.sqrt((2 * n + 1) / 2)

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
        return legendre_polynomial(self.n, x, self.shifted) * self.normalization


class LegendrePolyEncoder:
    """Encode real valued data with Legendre Polynomials."""

    def __init__(self, p: list[int], shifted: bool = False) -> None:
        """The constructor

        Parameters
        ----------
        p: list[int]
            List of indeces for the polynomial to be used for encoding. The number
            of indices corresponds to the dimensionality of output state(s).
        shifted: bool = False
            If False, evaluate the regular Legendre polynomial Pn(x), if
            True the shifted version Sn(x).
        """
        self.shifted = shifted
        self.encoder = [IsoLegendrePolynomial(n, shifted) for n in p]

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

    @property
    def RANGE(self) -> tuple[float, float]:
        if self.shifted:
            return (0., 1.)
        else:
            return (-1., 1.)

import jax
import jax.numpy as jnp


class FourierBasisFunction:
    """Fourier Basis function of the form:

        f(x;n) = 1/√2 * (e^(iπnx) + e^(-iπnx))

    where n is an integer frequency > 0 (note: using negative frequency will yield the
    same function due to the symmetric exponents) and 1/√2 acts as a normalization factor
    to create an orthonormal basis over [0,1]. The symmetric exponents ensure the result is
    real valued.
    """

    def __init__(self, freq: int) -> None:
        """The constructor

        Parameters
        ----------
        freq: int
            The integer frequency > 0.
        """
        self.freq = freq

        if freq == 0:
            self.normalization = 1 / 2
        else:
            self.normalization = 1 / jnp.sqrt(2)

    def __call__(self, x: jax.typing.ArrayLike | float) -> jax.Array:
        """Call the function on input data.

        Parameters
        ----------
        x: jax.typing.ArrayLike | float
            The data to call the function on.

        Returns
        -------
        The evaluated result, same shape as x.
        """
        exponent = jax.lax.complex(0.0, self.freq * jnp.pi * x)

        return self.normalization * jnp.real(jnp.exp(exponent) + jnp.exp(-exponent))


class FourierBasisEncoder:
    """Encode real valued data in a Fourier Basis."""

    RANGE = (0.0, 1.0)

    def __init__(self, freqs: list[int]) -> None:
        """The constructor

        Parameters
        ----------
        freqs: list[int]
            List of integer frequencies > 0 to be used for the basis functions.
        """
        self.encoder = [FourierBasisFunction(f) for f in freqs]

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

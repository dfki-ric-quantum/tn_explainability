import jax
import jax.numpy as jnp

class BasisEncoder:
    """Encodes discrete/integer data into the `dims` basis states of a `dims` level system.

    The data passed to the encoder has to be a) Integers and b.) the distance between the largest
    value and 0 needs to be covered by `dims`. This isn't checked and within the callers
    responsibility. Binary data would e.g., be encoded into a two level system such that:

    0 -> (1, 0) and 1 -> (0, 1)

    whereas data with three possible values would be encoded as:

    0 -> (1, 0, 0)
    1 -> (0, 1, 0)
    2 -> (0, 0, 1)

    """

    def __init__(self, dims: int = 2) -> None:
        """The constructor.

        Parameters
        ----------
        dims: int = 2
            Dimension of the basis to encode into. Needs to be >= 2.
        """
        self.dims = dims
        self.basis = jnp.eye(dims)

        # Note: These are callable
        self.encode_sample = jax.vmap(self.encode_single)
        self.encode_samples = jax.vmap(self.encode_sample)

    def encode_single(self, value: jax.typing.ArrayLike | int) -> jax.Array:
        """Encode single value.

        Parameters
        ----------
        value: jax.typing.ArrayLike | int
            numpy/jax containing a single integer or an integer

        Returns
        -------
        The encoded basis state.
        """
        return self.basis[value]

    def __call__(self, samples: jax.typing.ArrayLike) -> jax.Array:
        """Convinience wrapper for `encode_samples` to make the object callable.

        Encodes a set of samples of some discrete dataset into the `dims` basis states.

        Parameters
        ----------
        samples: jax.typing.ArrayLike
            Samples to encode, shape (n_samples, n_features)

        Returns
        -------
        The encoded basis states, shape (n_samples, n_features, dims)
        """
        return self.encode_samples(samples)


class AngleEncoder:
    """Angular encoding onto qubits. This encoding works on real-valued data in the range [-1,1] and
    otherwise isn't an isometrie. It can only encode onto two dimensional vectors. Encoding:

    x -> (cos(π * x/2), sin(π * x/2))

    """
    RANGE = (-1, 1)

    def __init__(self) -> None:
        """The constructor."""

        # These are callable
        self.encode_sample = jax.vmap(self.encode_single)
        self.encode_samples = jax.vmap(self.encode_sample)

    def encode_single(self, value: jax.typing.ArrayLike | float) -> jax.Array:
        """Encodes a single value.

        Parameters
        ----------
        value: jax.typing.ArrayLike | float
            The value to encode, either as numpy/jax array or float.

        Returns
        -------
        The encoded value as array of shape (2,)
        """
        return jnp.array([jnp.cos(jnp.pi * value/2), jnp.sin(jnp.pi * value/2)])

    def __call__(self, samples: jax.typing.ArrayLike) -> jax.Array:
        """Convinience wrapper for `encode_samples` to make the object callable.

        Encodes a set of samples.

        Parameters
        ----------
        samples: jax.typing.ArrayLike
            Samples to encode, shape (n_samples, n_features)

        Returns
        -------
        The encoded states, shape (n_samples, n_features, 2)
        """
        return self.encode_samples(samples)


class DiscretizedBasisEncoder:
    """The encoder discretizes real valued data of any range into |bins|+1 integers first,
    before encoding them into a |bins|+1 dimensional basis via the BasisEncoder.

    A real value x is first discretized into i such that.

    bins[i-1] <= x < bins[i]

    If values in x are beyond the bounds of bins, 0 or |bins| is returned as appropriate. Afterwards
    i is encoded into the i-th basis state of the |bins|+1 dimensional basis.
    """

    def __init__(self, bins: tuple[float,...]) -> None:
        """The constructor.

        Parameters
        ----------
        bins: tuple[float,...]
            A monotonic sequence of the boundaries for the bins.
        """
        self.bins = jnp.array(bins)
        self.dims = len(bins) + 1
        self.basis_encoder = BasisEncoder(dims=self.dims)

    def encode_single(self, value: jax.typing.ArrayLike | float) -> jax.Array:
        """Encode a single value.

        Parameters
        ----------
        value: jax.typing.ArrayLike | float
            The value to encode, either numpy/jax array or float.

        Returns
        -------
        The encoded state, shape (len(bins)+1,)
        """
        int_value = jnp.digitize(value, self.bins)
        return self.basis_encoder.encode_single(int_value)

    def encode_sample(self, sample: jax.typing.ArrayLike) -> jax.Array:
        """Encode a sample of multiple features.

        Parameters
        ----------
        sample: jax.typing.ArrayLike
            The sample to encode, shape (n_features,)

        Returns
        -------
        The encoded states, shape (n_features, len(bins)+1)
        """
        int_sample = jnp.digitize(sample, self.bins)
        return self.basis_encoder.encode_sample(int_sample)

    def encode_samples(self, samples: jax.typing.ArrayLike) -> jax.Array:
        """Encode multiple sample of multiple features.

        Parameters
        ----------
        samples: jax.typing.ArrayLike
            The sample to encode, shape (n_samples, n_features)

        Returns
        -------
        The encoded states, shape (n_samples, n_features, len(bins)+1)
        """
        int_samples = jnp.digitize(samples, self.bins)
        return self.basis_encoder.encode_samples(int_samples)

    def __call__(self, samples: jax.typing.ArrayLike) -> jax.Array:
        """Convinience wrapper for `encode_samples` to make the object callable.

        Encodes a set of samples.

        Parameters
        ----------
        samples: jax.typing.ArrayLike
            Samples to encode, shape (n_samples, n_features)

        Returns
        -------
        The encoded states, shape (n_samples, n_features, len(bins)+1)
        """
        return self.encode_samples(samples)

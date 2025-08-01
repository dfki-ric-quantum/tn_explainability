import jax
from flax import linen as nn
from ml_collections import config_dict

from tn.ml.common import activation


class Encoder(nn.Module):
    """Encoder part of the Autoenoder.

    Parameters
    ----------
    n_latent: int
        Number of units in the latent space
    activation: str
        Activation function
    layers: list[int]
        List of number of units per hidden layer, from input to last layer before
        latent space.
    """

    n_latent: int
    activation: str
    layers: list[int]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Call the encoder.

        Parameters
        ----------
        x: jax.Array
            Data to encode

        Returns
        -------
        The latent space representation of the data
        """
        act = activation(self.activation)

        for n_units in self.layers:
            x = nn.Dense(n_units)(x)
            x = act(x)
        x = nn.Dense(self.n_latent)(x)

        return x


class Decoder(nn.Module):
    """The decoder part of the Autoencoder

    Parameters
    ----------
    out_dim: int
        Output dimension
    activation: str
        activation function
    layers: list[int]
        List of number of units per hidden layer, from input to last layer before
        latent space.
    """

    out_dim: int
    activation: str
    layers: list[int]

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        """Call decoder.

        Parameters
        ----------
        z: jax.Array
            latent space representation

        Returns
        -------
        The decoded data
        """
        act = activation(self.activation)

        for n_units in reversed(self.layers):
            z = nn.Dense(n_units)(z)
            z = act(z)
        z = nn.Dense(self.out_dim)(z)

        return z


class Autoencoder(nn.Module):
    """Autoencoder.

    Parameters
    ----------
    cfg: config_dict.FrozenConfigDict
        The autoencoder configuration
    """

    cfg: config_dict.FrozenConfigDict

    def setup(self):
        """Setup the AE."""
        self.encoder = Encoder(self.cfg.n_latent, self.cfg.activation, self.cfg.layers)
        self.decoder = Decoder(self.cfg.out_dim, self.cfg.activation, self.cfg.layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode/Decode data.

        Parameters
        ----------
        x: jax.Array
            data to encode and then decode.

        Returns
        -------
        Reconstruction of the passed data
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat


def ae(cfg: config_dict.FrozenConfigDict) -> Autoencoder:
    """Wrapper to create Autoencoder from configuration."""
    return Autoencoder(cfg)

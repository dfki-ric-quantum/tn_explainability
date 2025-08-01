from typing import Any

from tn.encoding.fourier import FourierBasisEncoder
from tn.encoding.hermite_poly import HermitePolyEncoder
from tn.encoding.laguerre_poly import LaguerrePolyEncoder
from tn.encoding.legendre_poly import LegendrePolyEncoder
from tn.encoding.simple import AngleEncoder, BasisEncoder, DiscretizedBasisEncoder


_ENC = {
    "basis": BasisEncoder,
    "angle": AngleEncoder,
    "discrete": DiscretizedBasisEncoder,
    "fourier": FourierBasisEncoder,
    "hermite": HermitePolyEncoder,
    "laguerre": LaguerrePolyEncoder,
    "legendre": LegendrePolyEncoder,
}


def get_encoder(encoder_name: str, **kwargs) -> Any:
    """Get encoder by name.

    Parameters
    ----------
    encoder_name: str
        The name of the encoder, see the keys of _ENC
    **kwargs:
        Keyword argumenst for the encoder

    Returns
    -------
    The initialized encoder.
    """
    enc = _ENC[encoder_name]

    if encoder_name == 'angle':
        return enc()
    else:
        return enc(**kwargs)

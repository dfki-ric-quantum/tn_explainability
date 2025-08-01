from typing import Callable

from flax import linen as nn

_ACTIVATION = {
    'elu': nn.elu,
    'leaky_relu': nn.leaky_relu,
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'softmax': nn.softmax,
    'tanh': nn.tanh,
}

def activation(act: str) -> Callable:
    return _ACTIVATION[act]

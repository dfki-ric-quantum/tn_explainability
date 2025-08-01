from dataclasses import dataclass
from typing import Any, Optional

import jax
import optax


@dataclass
class OptState:
    """Stores the shape of the contracted tensor and the optimizer state for a bond."""

    shape: Optional[tuple[int, ...]] = None
    state: Optional[Any] = None


class MPSOptimizer:
    """Simple wrapper around optax to use various optimizers for the MPS training.

    Optimizers in optax track there state in an external object, which has dependency
    on the shape of the parameters to optimize. During the MPS training, we neither
    update all parameters at once, nor do we have fixed parameter shapes, due to changing
    bond dimensions.

    This class holds the state and shape per bond and re-initializes the optimitzer state
    for a bond, whenever the shape changes.
    """

    def __init__(
        self, base_optimizer: optax.GradientTransformation, n_bonds: int
    ) -> None:
        """The constructor.

        Parameters
        ----------
        base_optimizer: optax.GradientTransformation
            The optax optimizer to use, e.g. optax.adam
        n_bonds: int
            Number of bonds in the MPS
        """
        self.n_bonds = n_bonds
        self.base_optimizer = base_optimizer

        self.state = [OptState() for _ in range(n_bonds)]

    def update(
        self, tensor: jax.Array, gradient: jax.Array, bond_idx: int
    ) -> optax.Params:
        """Gradient update of a contracted order-4 tensor.

        Parameters
        ----------
        tensor: jax.Array
            The contracted order-4 tensor
        gradient: jax.Array
            Gradient for the tensor, same shape
        bond_idx: int
            The index of the bond to update

        Returns
        -------
        The updated tensor.
        """
        if tensor.shape != self.state[bond_idx].shape:
            self.state[bond_idx].shape = tensor.shape
            self.state[bond_idx].state = self.base_optimizer.init(tensor)

        updates, self.state[bond_idx].state = self.base_optimizer.update(
            gradient, self.state[bond_idx].state
        )

        return optax.apply_updates(tensor, updates)


class TreeOptimizer:
    """Simple wrapper around optax to use various optimizers for the Tree Tensor Network
    training.

    Optimizers in optax track there state in an external object, which has dependency
    on the shape of the parameters to optimize. During the TTN training, we neither
    update all parameters at once, nor do we have fixed parameter shapes, due to changing
    bond dimensions.

    This class holds the state and shape per bond and re-initializes the optimitzer state
    for a bond, whenever the shape changes.
    """

    def __init__(self, base_optimizer: optax.GradientTransformation) -> None:
        """The constructor.

        Parameters
        ----------
        base_optimizer: optax.GradientTransformation
            The optax optimizer to use, e.g. optax.adam
        """
        self.base_optimizer = base_optimizer
        self.state = dict()

    def update(self, tensor: jax.Array, gradient: jax.Array, cidx: int, pidx: int) -> optax.Params:
        """Gradient update of a contracted tensor in the tree.

        Parameters
        ----------
        tensor: jax.Array
            The contracted order-4 tensor
        gradient: jax.Array
            Gradient for the tensor, same shape
        cidx: int
            Index of the child node
        pidx: int
            Index of the parent node

        Returns
        -------
        The updated tensor.
        """
        state = self.state.get((cidx, pidx), OptState())

        if tensor.shape != state.shape:
            state.shape = tensor.shape
            state.state = self.base_optimizer.init(tensor)
            self.state[(cidx, pidx)] = state

        updates, self.state[(cidx, pidx)].state = self.base_optimizer.update(
            gradient, self.state[(cidx, pidx)].state
        )

        return optax.apply_updates(tensor, updates)

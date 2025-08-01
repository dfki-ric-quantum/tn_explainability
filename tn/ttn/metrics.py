import jax
import jax.numpy as jnp

from tn.ttn.contraction import build_psi_contraction_expr
from tn.ttn.tree import Tree
from tn.ttn.types import NodeShapes


class PsiFunctor:
    """Functor to compute psi(v) for a given Tree Tensor Network

    The functor pre-computes an optimized contraction path and compiles it to XLA via
    jax.jit. If the tree's bond dimensions change (e.g. during training), the functor has
    to be rebuild.
    """

    def __init__(self, node_shapes: NodeShapes, leaf_mask: list[bool], d: int):
        """The constructor.

        Parameters
        ----------
        node_shapes: NodeShapes
            The node  shapes n BFS order to build the functor for. As long as the
            bond dimension don't change the functor will keep working and be optimal.
        leaf_mask: list[bool]
            Mask for the legs that are connected to leaves.
        d: int
            Physical bond dimension, 2 for qubits >=2 for qudits
        """
        self.expr = build_psi_contraction_expr(node_shapes, leaf_mask, d)

    def __call__(self, states: jax.Array, tree: Tree) -> jax.Array:
        """Compute the psi(v) on a batch of states.

        Parameters
        ----------
        states: jax.Array
            Batch of N states. Shape (N, mps_sites, 2)
        tree: Tree
            The Tree Tensor Network

        Returns
        -------
        The psi(v) on the batch
        """
        return jax.vmap(self._psi, in_axes=(0, None))(states, tree)

    def _psi(self, state: jax.Array, tree: Tree) -> jax.Array:
        """Compute psi(v) for a state

        Parameters
        ----------
        state: jax.Array
            State to compute psi(v) for
        tree: Tree
            The Tree Tensor Network

        Returns
        -------
        psi(v)
        """
        return jnp.squeeze(self.expr(*tree.tensors, *state, backend="jax"))


class ProbFunctor:
    """Functor to compute P(v) for a given Tree Tensor Network

    The functor pre-computes an optimized contraction path and compiles it to XLA via
    jax.jit. If the tree's bond dimensions change (e.g. during training), the functor has
    to be rebuild.
    """

    def __init__(self, node_shapes: NodeShapes, leaf_mask: list[bool], d: int):
        """The constructor.

        Parameters
        ----------
        node_shapes: NodeShapes
            The node  shapes n BFS order to build the functor for. As long as the
            bond dimension don't change the functor will keep working and be optimal.
        leaf_mask: list[bool]
            Mask for the legs that are connected to leaves.
        d: int
            Physical bond dimension, 2 for qubits >=2 for qudits
        """
        self.psi = PsiFunctor(node_shapes, leaf_mask, d)

    def __call__(self, states: jax.Array, tree: Tree) -> jax.Array:
        """Compute the P(v) on a batch of states.

        Parameters
        ----------
        states: jax.Array
            Batch of N states. Shape (N, mps_sites, 2)
        tree: Tree
            The Tree Tensor Network

        Returns
        -------
        The P(v) on the batch
        """
        return jnp.square(jnp.abs(self.psi(states, tree)))


class NLLFunctor:
    """Functor to compute the Negative-Log-Likelihood loss for a given Tree Tensor Network.

    The functor pre-computes an optimized contraction path and compiles it to XLA via
    jax.jit. If the tree's bond dimensions change (e.g. during training), the functor has
    to be rebuild.
    """

    def __init__(self, node_shapes: NodeShapes, leaf_mask: list[bool], d: int):
        """The constructor.

        Parameters
        ----------
        node_shapes: NodeShapes
            The node  shapes n BFS order to build the functor for. As long as the
            bond dimension don't change the functor will keep working and be optimal.
        leaf_mask: list[bool]
            Mask for the legs that are connected to leaves.
        d: int
            Physical bond dimension, 2 for qubits >=2 for qudits
        """
        self.prob = ProbFunctor(node_shapes, leaf_mask, d)

    def __call__(self, states: jax.Array, tree: Tree) -> jax.Array:
        """Compute the NLL on a batch of states.

        Parameters
        ----------
        states: jax.Array
            Batch of N states. Shape (N, mps_sites, 2)
        tree: Tree
            The Tree Tensor Network

        Returns
        -------
        The NLL on the batch
        """
        return -jnp.log(self.prob(states, tree))

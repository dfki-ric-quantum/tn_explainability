import jax
import jax.numpy as jnp

from tn.mps import MPS, MPSShapes
from tn.mps.contraction import build_psi_contraction_expr


class PsiFunctor:
    """Functor to compute psi(v) for a given MPS

    The functor pre-computes an optimized contraction path and compiles it to XLA via
    jax.jit. If the MPS bond dimensions change (e.g. during training), the functor has
    to be rebuild.
    """

    def __init__(self, mps_shapes: MPSShapes, d: int = 2):
        """The constructor.

        Parameters
        ----------
        mps_shapes: MPSShapes
            The MPS  shapes to build the functor for. As long as the bond dimension don't
            change the functor will keep working and be optimal.
        d: int
            Physical bond dimension, 2 for qubits >=2 for qudits
        """
        self.expr = build_psi_contraction_expr(mps_shapes, d)

    def __call__(self, states: jax.Array, mps: MPS) -> jax.Array:
        """Compute the psi(v) on a batch of states.

        Parameters
        ----------
        states: jax.Array
            Batch of N states. Shape (N, mps_sites, 2)
        mps: MPS
            The Matrix Product State

        Returns
        -------
        The psi(v) on the batch
        """
        return jax.vmap(self._psi, in_axes=(0, None))(states, mps)

    def _psi(self, state: jax.Array, mps: MPS) -> jax.Array:
        """Compute psi(v) for a state

        Parameters
        ----------
        state: jax.Array
            State to compute psi(v) for
        mps: MPS
            The Matrix Product State

        Returns
        -------
        psi(v)
        """
        return jnp.trace(self.expr(*mps, *state, backend="jax"))


class ProbFunctor:
    """Functor to compute P(v) for a given MPS

    The functor pre-computes an optimized contraction path and compiles it to XLA via
    jax.jit. If the MPS bond dimensions change (e.g. during training), the functor has
    to be rebuild.
    """

    def __init__(self, mps_shapes: MPSShapes, d: int = 2):
        """The constructor.

        Parameters
        ----------
        mps_shapes: MPSShapes
            The MPS  shapes to build the functor for. As long as the bond dimension don't
            change the functor will keep working and be optimal.
        d: int
            Physical bond dimension, 2 for qubits >=2 for qudits
        """
        self.psi = PsiFunctor(mps_shapes, d)

    def __call__(self, states: jax.Array, mps: MPS) -> jax.Array:
        """Compute the P(v) on a batch of states.

        Parameters
        ----------
        states: jax.Array
            Batch of N states. Shape (N, mps_sites, 2)
        mps: MPS
            The Matrix Product State

        Returns
        -------
        The P(v) on the batch
        """
        return jnp.square(jnp.abs(self.psi(states, mps)))


class NLLFunctor:
    """Functor to compute the Negative-Log-Likelihood loss for a given MPS.

    The functor pre-computes an optimized contraction path and compiles it to XLA via
    jax.jit. If the MPS bond dimensions change (e.g. during training), the functor has
    to be rebuild.
    """

    def __init__(self, mps_shapes: MPSShapes, d: int = 2):
        """The constructor.

        Parameters
        ----------
        mps_shapes: MPSShapes
            The MPS  shapes to build the functor for. As long as the bond dimension don't
            change the functor will keep working and be optimal.
        d: int
            Physical bond dimension, 2 for qubits >=2 for qudits
        """
        self.prob = ProbFunctor(mps_shapes, d)

    def __call__(self, states: jax.Array, mps: MPS) -> jax.Array:
        """Compute the elementwise NLL on a batch of states.

        Parameters
        ----------
        states: jax.Array
            Batch of N states. Shape (N, mps_sites, 2)
        mps: MPS
            The Matrix Product State

        Returns
        -------
        The elementwise NLL on the batch
        """
        return -jnp.log(self.prob(states, mps))

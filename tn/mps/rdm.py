import jax
import jax.numpy as jnp

from tn.mps.contraction import build_crdm_contraction_expr, build_rdm_contraction_expr
from tn.mps import MPS

def conditional_reduced_density_matrix(
    mps: MPS, sub_system: list[int], cond_vals: list[int], x: jax.Array
) -> jax.Array:
    """Compute the conditioned reduced density matrix of a sub-system conditioned by given sites.

    Parameters
    ----------
    mps: MPS
        The matrix product state
    sub_system: list[int]
        List of site indices for which to compute the reduced density matrix.
    cond_vals: list[int]
        List of site indices where values are conditioned to a specific value.

    Returns
    -------
    The reduced density matrix of shape (d^(len(sub_system)), d^(len(sub_system)))
    """
    assert len(sub_system) == len(set(sub_system)), "Duplicate sites in sub-system"
    assert len(cond_vals) == len(set(cond_vals)), "Duplicate sites in conditional sites"
    assert set(cond_vals).isdisjoint(
        set(sub_system)
    ), "Sub system and conditioned indices cannot share the same indices"

    sub_system.sort()
    cond_vals.sort()
    lidx = min((sub_system[0], cond_vals[0]))
    ridx = max((sub_system[-1], cond_vals[-1]))
    n_dim = mps.d ** len(sub_system)

    mps.mixed_canonicalize(lidx)
    expr = build_crdm_contraction_expr(mps.shapes, sub_system, cond_vals, d=mps.d)

    crdm = expr(
        *mps[lidx : ridx + 1],
        *[jnp.conjugate(t) for t in mps[lidx : ridx + 1]],
        *[jnp.outer(x[0, i, :], x[0, i, :]) for i in cond_vals],
        backend="jax"
    )

    crdm = crdm.reshape((n_dim, n_dim))

    return crdm / jnp.trace(crdm)


def reduced_density_matrix(mps: MPS, sub_system: list[int]) -> jax.Array:
    """Compute the reduced density matrix of a sub-system.

    Computes rho_X = tr_Y(rho), where X and Y are a bi-partition of the entire system rho
    encoded by the MPS. X is the sub-system to consider and tr_Y is the partial trace with respect
    to Y.

    Parameters
    ----------
    mps: MPS
        The matrix product state
    sub_system: list[int]
        List of site indices for which to compute the reduced density matrix.

    Returns
    -------
    The reduced density matrix of shape (d^(len(sub_system)), d^(len(sub_system)))
    """
    assert len(sub_system) == len(set(sub_system)), "Duplicate sites in sub-system"

    sub_system.sort()
    lidx = sub_system[0]
    ridx = sub_system[-1]
    n_dim = mps.d ** len(sub_system)

    assert ridx < mps.n_sites, "Sub-system is out of range"

    mps.mixed_canonicalize(lidx)
    expr = build_rdm_contraction_expr(mps.shapes, sub_system)

    rdm = expr(
        *mps[lidx : ridx + 1],
        *[jnp.conjugate(t) for t in mps[lidx : ridx + 1]],
        backend="jax"
    )

    return rdm.reshape((n_dim, n_dim))

import jax
import jax.numpy as jnp

from tn.ops import sv, us


@jax.jit
def two_site_svd(
    lhs: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute SVD on the contraction of two order-3 tensors.

    Parameters
    ----------
    lhs: jax.Array
        The left order-3 tensor in the contraction
    rhs: jax.Array
        The right order-3 tensor in the contraction

    Returns
    -------
    U, S, and V, the result of the SVD.
    """
    contracted = jnp.einsum("abc,cde->abde", lhs, rhs).reshape(
        (lhs.shape[0] * lhs.shape[1], rhs.shape[1] * rhs.shape[2])
    )
    return jnp.linalg.svd(contracted, full_matrices=False)


@jax.jit
def svd_decompose(tensor: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Decompose order-4 tensor into two order-3 tensors via SVD."""
    lbond, lphys, rphys, rbond = tensor.shape

    return jnp.linalg.svd(
        tensor.reshape((lphys * lbond, rphys * rbond)), full_matrices=False
    )


def svd_normalize_left(lhs: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Contract and decompose two order-3 tensors.

    As a result, the left tensor (lhs) will be left-normalized, the right tensor (rhs)
    will have unit norm.

    Parameters
    ----------
    lhs: jax.Array
        The left order-3 tensor in the contraction
    rhs: jax.Array
        The right order-3 tensor in the contraction

    Returns
    -------
    updated left and right order-3 tensor, after decomposition/normalization.

    """
    lbond, lphys = lhs.shape[0], lhs.shape[1]
    rbond, rphys = rhs.shape[2], rhs.shape[1]

    u, s, v = two_site_svd(lhs, rhs)
    bond = s.shape[0]

    lres = u[:, :bond]
    rres = sv(jnp.diag(s[:bond]), v[:bond,])

    return lres.reshape((lbond, lphys, bond)), rres.reshape((bond, rphys, rbond))


def svd_normalize_right(lhs: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Contract and decompose two order-3 tensors.

    As a result, the right tensor (rhs) will be right-normalized, the left tensor (lhs)
    will have unit norm.

    Parameters
    ----------
    lhs: jax.Array
        The left order-3 tensor in the contraction
    rhs: jax.Array
        The right order-3 tensor in the contraction

    Returns
    -------
    updated left and right order-3 tensor, after decomposition/normalization.

    """
    lbond, lphys = lhs.shape[0], lhs.shape[1]
    rbond, rphys = rhs.shape[2], rhs.shape[1]

    u, s, v = two_site_svd(lhs, rhs)
    bond = s.shape[0]

    lres = us(u[:, :bond], jnp.diag(s[:bond]))
    rres = v[:bond, :]

    return lres.reshape((lbond, lphys, bond)), rres.reshape((bond, rphys, rbond))


def trunc_svd_normalize_left(
    lhs: jax.Array, rhs: jax.Array, max_bond: int, sigma_thresh: float
) -> tuple[jax.Array, jax.Array]:
    """Contract and decompose two order-3 tensors and truncate singular values.

    As a result, the left tensor (lhs) will be left-normalized, the right tensor (rhs)
    will have unit norm. During decomposition, the singular values will be truncated, regularizing
    the dimensionality of the common bond of the two tensors.

    Parameters
    ----------
    lhs: jax.Array
        The left order-3 tensor in the contraction
    rhs: jax.Array
        The right order-3 tensor in the contraction
    max_bond: int
        Upper bound on the common bond after decomposition
    sigma_thresh: float
        Lower bound for the singular value truncation

    Returns
    -------
    updated left and right order-3 tensor, after decomposition/normalization.

    """
    lbond, lphys = lhs.shape[0], lhs.shape[1]
    rbond, rphys = rhs.shape[2], rhs.shape[1]

    u, s, v = two_site_svd(lhs, rhs)

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    lres = u[:, :bond]
    rres = sv(jnp.diag(s[:bond]), v[:bond,])

    return lres.reshape((lbond, lphys, bond)), rres.reshape((bond, rphys, rbond))


def trunc_svd_normalize_right(
    lhs: jax.Array, rhs: jax.Array, max_bond: int, sigma_thresh: float
) -> tuple[jax.Array, jax.Array]:
    """Contract and decompose two order-3 tensors and truncate singular values.

    As a result, the right tensor (rhs) will be right-normalized, the left tensor (lhs)
    will have unit norm. During decomposition, the singular values will be truncated, regularizing
    the dimensionality of the common bond of the two tensors.

    Parameters
    ----------
    lhs: jax.Array
        The left order-3 tensor in the contraction
    rhs: jax.Array
        The right order-3 tensor in the contraction
    max_bond: int
        Upper bound on the common bond after decomposition
    sigma_thresh: float
        Lower bound for the singular value truncation

    Returns
    -------
    updated left and right order-3 tensor, after decomposition/normalization.

    """
    lbond, lphys = lhs.shape[0], lhs.shape[1]
    rbond, rphys = rhs.shape[2], rhs.shape[1]

    u, s, v = two_site_svd(lhs, rhs)

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    lres = us(u[:, :bond], jnp.diag(s[:bond]))
    rres = v[:bond, :]

    return lres.reshape((lbond, lphys, bond)), rres.reshape((bond, rphys, rbond))


def trunc_svd_decomp_normalize_left(
    tensor: jax.Array, max_bond: int, sigma_thresh: float
) -> tuple[jax.Array, jax.Array]:
    """Decompose order-4 tensor into two order-3 tensors and truncate singular values.

    As a result, the left tensor (lhs) will be left-normalized, the right tensor (rhs)
    will have unit norm. During decomposition, the singular values will be truncated, regularizing
    the dimensionality of the common bond of the two tensors.

    Parameters
    ----------
    tensor: jax.Array
        The order-4 tensor to decompose
    max_bond: int
        Upper bound on the common bond after decomposition
    sigma_thresh: float
        Lower bound for the singular value truncation

    Returns
    -------
    updated left and right order-3 tensor, after decomposition/normalization.

    """
    lbond, lphys, rphys, rbond = tensor.shape

    u, s, v = svd_decompose(tensor)

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    lres = u[:, :bond]
    rres = sv(jnp.diag(s[:bond]), v[:bond,])

    return lres.reshape((lbond, lphys, bond)), rres.reshape((bond, rphys, rbond))


def trunc_svd_decomp_normalize_right(
    tensor: jax.Array, max_bond: int, sigma_thresh: float
) -> tuple[jax.Array, jax.Array]:
    """Decompose order-4 tensor into two order-3 tensors and truncate singular values.

    As a result, the right tensor (rhs) will be right-normalized, the left tensor (lhs)
    will have unit norm. During decomposition, the singular values will be truncated, regularizing
    the dimensionality of the common bond of the two tensors.

    Parameters
    ----------
    tensor: jax.Array
        The order-4 tensor to decompose
    max_bond: int
        Upper bound on the common bond after decomposition
    sigma_thresh: float
        Lower bound for the singular value truncation

    Returns
    -------
    updated left and right order-3 tensor, after decomposition/normalization.

    """
    lbond, lphys, rphys, rbond = tensor.shape

    u, s, v = svd_decompose(tensor)

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    lres = us(u[:, :bond], jnp.diag(s[:bond]))
    rres = v[:bond, :]

    return lres.reshape((lbond, lphys, bond)), rres.reshape((bond, rphys, rbond))

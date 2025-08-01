"""Low level contraction and decomposition operations for tensor tree networks.

There are three groups of low level contractions/decompositions:

1. Contraction of a leaf with its parent:

                     |                          |
                     P                          P
    From the left:  / \       From the right:  / \
                   L                              L
                   |                              |

2. Contraction of an inner node with its non-root parent:

                     |                          |
                     P                          P
    From the left:  / \       From the right:  / \
                   N                              N
                  / \                            / \

3. Contraction of an inner node with the root node:

                     R                          R
    From the left:  / \       From the right:  / \
                   N                              N
                  / \                            / \

Here the following conventions are assumed:

* Only nodes with a physical bonds are considered leaves.
* If an inner node has only one child, the other bond dimension is 1.
* Leafs are order-2 tensors of shape (physical_bond, upper_bond)
* Inner nodes are order-3 tensors of shape (left-lower bond, upper bond, right lower-bond)
* The root node is an order-2 tensor of shape (left-lower bond, right lower bond)
* All contractions are expressed bottom-up
* After any contraction, the indices are ordered from the left-bottom most one in clock-wise order

"""

import jax
import jax.numpy as jnp

from tn.ops import sv, us


@jax.jit
def contract_left_leaf(leaf: jax.Array, parent: jax.Array) -> jax.Array:
    """Contract left leaf with its parent node.

    Parameters
    ----------
    leaf: jax.Array
        The order-2 leaf tensor
    parent: jax.Array
        The order-3 parent node

    Returns
    -------
    The contracted order-3 tensor, index order (physical-bond, top bond, right bond)

    """
    return jnp.einsum("ab,bcd->acd", leaf, parent)


@jax.jit
def contract_right_leaf(leaf: jax.Array, parent: jax.Array) -> jax.Array:
    """Contract right leaf with its parent node.

    Parameters
    ----------
    leaf: jax.Array
        The order-2 leaf tensor
    parent: jax.Array
        The order-3 parent node

    Returns
    -------
    The contracted order-3 tensor, index order (left bond, top bond, physical bond)

    """
    return jnp.einsum("ab,dcb->dca", leaf, parent)


@jax.jit
def contract_left_inner_node(child: jax.Array, parent: jax.Array) -> jax.Array:
    """Contract left inner node with its non-root parent node.

    Parameters
    ----------
    child: jax.Array
        The order-3 child inner node tensor.
    parent: jax.Array
        The order-3 parent inner node tensor.

    Returns
    -------
    The contracted order-4 tensor, index order (child left, child right, parent top,
    parent right)
    """
    return jnp.einsum("abc,bde->acde", child, parent)


@jax.jit
def contract_right_inner_node(child: jax.Array, parent: jax.Array) -> jax.Array:
    """Contract right inner node with its non-root parent node.

    Parameters
    ----------
    child: jax.Array
        The order-3 child inner node tensor.
    parent: jax.Array
        The order-3 parent inner node tensor.

    Returns
    -------
    The contracted order-4 tensor, index order (parent left, parent top, child left,
    child right)
    """
    return jnp.einsum("abc,edb->edac", child, parent)


@jax.jit
def contract_left_with_root(child: jax.Array, root: jax.Array) -> jax.Array:
    """Contract left inner node with the tree's root.

    Parameters
    ----------
    child: jax.Array
        The order-3 child inner node tensor.
    root: jax.Array
        The order-2 root node tensor.

    Returns
    -------
    The contracted order-3 tensor, index order (child left, child right, root right)
    """
    return jnp.einsum("abc,bd->acd", child, root)


@jax.jit
def contract_right_with_root(child: jax.Array, root: jax.Array) -> jax.Array:
    """Contract right inner node with the tree's root.

    Parameters
    ----------
    child: jax.Array
        The order-3 child inner node tensor.
    root: jax.Array
        The order-2 root node tensor.

    Returns
    -------
    The contracted order-3 tensor, index order (left root, left child, right child)
    """
    return jnp.einsum("abc,db->dac", child, root)


def decomp_left_leaf_up(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate left leaf from its parent and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (physical bond, top bond, right
        bond)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The left leaf (unitary) and its parent (norm 1).
    """
    pbond, tbond, rbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((pbond, tbond * rbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    parent = sv(jnp.diag(s[:bond]), v[:bond, :])

    return u[:, :bond], parent.reshape((bond, tbond, rbond))


def decomp_left_leaf_down(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate left leaf from its parent and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (physical bond, top bond, right bond)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The left leaf (norm 1) and its parent (right normalized).
    """
    pbond, tbond, rbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((pbond, tbond * rbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    leaf = us(u[:, :bond], jnp.diag(s[:bond]))

    return leaf, v[:bond, :].reshape((bond, tbond, rbond))


def decomp_right_leaf_up(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate right leaf from its parent and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (left bond, top bond, physical bond)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The right leaf (unitary) and its parent (norm 1).
    """
    lbond, tbond, pbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((lbond * tbond, pbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    parent = us(u[:, :bond], jnp.diag(s[:bond]))

    return v[:bond, :].T, parent.reshape((lbond, tbond, bond))


def decomp_right_leaf_down(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate right leaf from its parent and push singular values down.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (left bond, top bond, physical bond)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The right leaf (norm 1) and its parent (left normalized).
    """
    lbond, tbond, pbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((lbond * tbond, pbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    leaf = sv(jnp.diag(s[:bond]), v[:bond, :]).T

    return leaf, u[:, :bond].reshape((lbond, tbond, bond))


def decomp_left_inner_node_up(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate left inner node from its parent and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-4 tensor, index order (child left, child right, parent top, parent right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The left child (upper normalized) and its parent (norm 1).
    """
    clbond, crbond, ptbond, prbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((clbond * crbond, ptbond * prbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    child = u[:, :bond].reshape((clbond, crbond, bond))
    child = jnp.transpose(child, (0, 2, 1))

    parent = sv(jnp.diag(s[:bond]), v[:bond, :]).reshape((bond, ptbond, prbond))

    return child, parent


def decomp_left_inner_node_down(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate left inner node from its parent and push singular values down.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-4 tensor, index order (child left, child right, parent top, parent right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The left child (norm 1) and its parent (right normalized).
    """
    clbond, crbond, ptbond, prbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((clbond * crbond, ptbond * prbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    child = us(u[:, :bond], jnp.diag(s[:bond])).reshape((clbond, crbond, bond))
    child = jnp.transpose(child, (0, 2, 1))

    return child, v[:bond, :].reshape((bond, ptbond, prbond))


def decomp_right_inner_node_up(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate right inner node from its parent and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-4 tensor, index order (parent left, parent top, child left, child right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The right child (upper normalized) and its parent (norm 1).
    """
    plbond, ptbond, clbond, crbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((plbond * ptbond, clbond * crbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    parent = us(u[:, :bond], jnp.diag(s[:bond])).reshape((plbond, ptbond, bond))
    child = v[:bond, :].reshape((bond, clbond, crbond))
    child = jnp.transpose(child, (1, 0, 2))

    return child , parent


def decomp_right_inner_node_down(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate right inner node from its parent and push singular values down.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-4 tensor, index order (parent left, parent top, child left, child right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The right child (norm 1) and its parent (left normalized).
    """
    plbond, ptbond, clbond, crbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((plbond * ptbond, clbond * crbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    child = sv(jnp.diag(s[:bond]), v[:bond, :]).reshape((bond, clbond, crbond))
    child = jnp.transpose(child, (1, 0, 2))

    return child, u[:, :bond].reshape((plbond, ptbond, bond))


def decomp_left_from_root_up(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate left inner node from root and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (child left, child right, root right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The order-3 child tensor (upper normalized) and the order-2 root tensor (norm 1)
    """
    clbond, crbond, rrbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((clbond * crbond, rrbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    root = sv(jnp.diag(s[:bond]), v[:bond, :])
    child = jnp.transpose(u[:, :bond].reshape((clbond, crbond, bond)), (0, 2, 1))

    return child, root


def decomp_left_from_root_down(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate left inner node from root and push singular values down.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (child left, child right, root right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The order-3 child tensor (norm 1) and the order-2 root tensor (unitary)
    """
    clbond, crbond, rrbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((clbond * crbond, rrbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    child = us(u[:, :bond], jnp.diag(s[:bond])).reshape((clbond, crbond, bond))
    child = jnp.transpose(child, (0, 2, 1))

    return child, v[:bond, :]


def decomp_right_from_root_up(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate right inner node from root and push singular values up.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (root right, child left, child right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The order-3 child tensor (upper normalized) and the order-2 root tensor (norm 1)
    """
    rlbond, clbond, crbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((rlbond, clbond * crbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    root = us(u[:, :bond], jnp.diag(s[:bond]))
    child = v[:bond, :].reshape((bond, clbond, crbond))
    child = jnp.transpose(child, (1, 0, 2))

    return child, root


def decomp_right_from_root_down(
    contracted: jax.Array, max_bond: int = 1000, sigma_thresh: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """SVD decompose & truncate right inner node from root and push singular values down.

    Parameters
    ----------
    contracted: jax.Array
        Contracted order-3 tensor, index order (root right, child left, child right)
    max_bond: int = 1000
        Upper bound on the common bond after decomposition
    sigma_thresh: float = 0.0
        Lower bound for the singular value truncation

    Returns
    -------
    The order-3 child tensor (norm 1) and the order-2 root tensor (unitary)
    """
    rlbond, clbond, crbond = contracted.shape

    u, s, v = jnp.linalg.svd(contracted.reshape((rlbond, clbond * crbond)))

    s = jnp.where(s > sigma_thresh, s, 0.0)
    bond = jnp.minimum(jnp.count_nonzero(s), max_bond)

    child = sv(jnp.diag(s[:bond]), v[:bond, :]).reshape((bond, clbond, crbond))
    child = jnp.transpose(child, (1, 0, 2))

    return child, u[:, :bond]

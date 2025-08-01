from collections import deque
from itertools import compress, count
from typing import Callable

import opt_einsum as oe

from tn.ttn.types import NodeShapes


def _full_node_labels(n_nodes: int) -> tuple[list[str], list[str]]:
    """Build full edge labels for all nodes of the tree in BFS order. Label order is:

                    Inner node:             Root:
                    -----------             -----

                       | b                  a   b
                       o                    --o--
                    a / \ c

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree

    Returns
    -------
    List of strings, each containing the labels of a node and a list of the legs of the
    lowest nodes.
    """
    es = []
    n = count()

    root_l = oe.get_symbol(next(n))
    root_r = oe.get_symbol(next(n))

    es.append(f"{root_l}{root_r}")

    legs = deque([root_l, root_r])
    nodes = 2

    while nodes < n_nodes:
        for idx in range(nodes):
            head = legs.popleft()
            left = oe.get_symbol(next(n))
            right = oe.get_symbol(next(n))
            legs.append(left)
            legs.append(right)
            es.append(f"{left}{head}{right}")
        nodes *= 2

    return es, list(legs)


def get_psi_einsum_str(n_nodes: int, leaf_mask: list[bool]) -> str:
    """Get the Einstein sum string for psi(v).

    Compiles the Einstein sum string for the following contraction:

                 +-#-+
                /     \
               #       #
              / \     / \
             #   #   #   #
            / \ / \ / \ / \
            o o o o o o o o

    Where `#` are the nodes of the Tree Tensor network and `o` the vectors
    encoding data.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree
    leaf_mask: list[bool]
        Mask for the legs that are connected to leaves.

    Returns
    -------
    The Einstein sum string for the contraction.
    """
    es, legs = _full_node_labels(n_nodes)
    es += list(compress(legs, leaf_mask))

    return ",".join(es)


def get_psic_einsum_str(
    n_nodes: int, cidx: int, pidx: int, leaf_mask: list[bool]
) -> str:
    """Get the Einstein sum string for psi(v) with two nodes contracted.

    Compiles the Einstein sum string for the following contraction:

                 +-#-+
                /     \
               #       X
              / \     / X
             #   #   #   X
            / \ / \ / \ / \
            o o o o o o o o

    Where `#` are the nodes of the Tree Tensor network, `o` the vectors
    encoding data and the three `X` is the contracted tensor.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree
    cidx: int
        Index of the child node in the contraction.
    pidx: int
        Index of the parent node in the contraction.
    leaf_mask: list[bool]
        Mask for the legs that are connected to leaves.

    Returns
    -------
    The Einstein sum string for the contraction.
    """

    def get_constr(cstr, pstr, cidx):
        cpart = cstr[0] + cstr[2]

        if cidx % 2 == 0:
            return pstr[:-1] + cpart
        else:
            return cpart + pstr[1:]

    es, legs = _full_node_labels(n_nodes)

    cstr = es.pop(cidx)
    pstr = es.pop(pidx)
    constr = get_constr(cstr, pstr, cidx)
    es.append(constr)

    es += list(compress(legs, leaf_mask))

    return ",".join(es)


def build_psi_contraction_expr(
    node_shapes: NodeShapes, leaf_mask: list[bool], d: int
) -> Callable:
    """Pre-compute, optimize and compile the contraction expression for psi(v).

    Pre-computes the expression for:

                 +-#-+
                /     \
               #       #
              / \     / \
             #   #   #   #
            / \ / \ / \ / \
            o o o o o o o o

    Where `#` are the nodes of the Tree Tensor network and `o` the vectors
    encoding data.

    Parameters
    ----------
    node_shapes: NodeShapes
        Shapes of the node tensors
    leaf_mask: list[bool]
        Mask for the legs that are connected to leaves.
    d: int
        Physical bond dimension

    Returns
    -------
    The pre-computed contraction expression
    """
    n_nodes = len(node_shapes)
    n_features = leaf_mask.count(True)

    shapes = node_shapes + [(d,)] * n_features

    es_str = get_psi_einsum_str(n_nodes, leaf_mask)

    return oe.contract_expression(es_str, *shapes, memory_limit=-1)


def build_psic_contraction_expr(
    node_shapes: NodeShapes,
    contracted_shape: tuple[int, ...],
    cidx: int,
    pidx: int,
    leaf_mask: list[bool],
    d: int,
) -> Callable:
    """Pre-compute, optimize and compile the contraction expression for psi(v) with two
    nodes contracted.

    Pre-computes the expression for:

                 +-#-+
                /     \
               #       X
              / \     / X
             #   #   #   X
            / \ / \ / \ / \
            o o o o o o o o

    Where `#` are the nodes of the Tree Tensor network, `o` the vectors
    encoding data and the three `X` is the contracted tensor.

    Parameters
    ----------
    node_shapes: NodeShapes
        Shapes of the node tensors
    contracted_shape: tuple[int, ...],
        Shape of the contracted tensor
    cidx: int
        Index of the child node in the contraction.
    pidx: int
        Index of the parent node in the contraction.
    leaf_mask: list[bool]
        Mask for the legs that are connected to leaves.
    d: int
        Physical bond dimension

    Returns
    -------
    The Einstein sum string for the contraction.
    """
    n_nodes = len(node_shapes)
    n_features = leaf_mask.count(True)

    shapes = (
        [shape for idx, shape in enumerate(node_shapes) if idx not in [cidx, pidx]]
        + [contracted_shape]
        + [(d,)] * n_features
    )

    es_str = get_psic_einsum_str(n_nodes, cidx, pidx, leaf_mask)

    return oe.contract_expression(es_str, *shapes, memory_limit=-1)

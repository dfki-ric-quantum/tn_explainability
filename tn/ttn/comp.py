import jax

from tn.ttn.ops import (
    contract_left_inner_node,
    contract_left_with_root,
    contract_right_inner_node,
    contract_right_with_root,
    decomp_left_from_root_down,
    decomp_left_from_root_up,
    decomp_left_inner_node_down,
    decomp_left_inner_node_up,
    decomp_right_from_root_down,
    decomp_right_from_root_up,
    decomp_right_inner_node_down,
    decomp_right_inner_node_up,
)
from tn.ttn.types import Node, NodeKind, NodeSide, NodeState, TDir


def contract(child: Node, parent: Node) -> jax.Array:
    """Contract child node with its parent.

    Parameters
    ----------
    child: Node
        The child node
    parent: Node
        The parent node

    Returns
    -------
    The contracted tensor. Shape depends on the kind of contracted nodes.

    """
    match (child.kind, child.side, parent.kind):
        case (NodeKind.INNER, NodeSide.LEFT, NodeKind.INNER):
            return contract_left_inner_node(child.tensor, parent.tensor)
        case (NodeKind.INNER, NodeSide.RIGHT, NodeKind.INNER):
            return contract_right_inner_node(child.tensor, parent.tensor)
        case (NodeKind.INNER, NodeSide.LEFT, NodeKind.ROOT):
            return contract_left_with_root(child.tensor, parent.tensor)
        case (NodeKind.INNER, NodeSide.RIGHT, NodeKind.ROOT):
            return contract_right_with_root(child.tensor, parent.tensor)
        case _:
            raise RuntimeError("Invalid node combination for contraction")


def decomp(
    contracted: jax.Array,
    child_kind: NodeKind,
    child_side: NodeSide,
    parent_kind: NodeKind,
    direction: TDir,
    max_bond: int = 1000,
    sigma_thresh: float = 0.0,
) -> tuple[NodeState, NodeState, jax.Array, jax.Array]:
    """Decompose contracted tensor into node tensors

    Parameters
    ----------
    contracted: jax.Array,
        The contracted tensor, shape depends on the nodes that were contracted
    child_kind: NodeKind,
        Node kind of the child node
    child_side: NodeSide,
        Side of the child node
    parent_kind: NodeKind,
        Node kind of the parent node
    direction: TDir
        Direction of the decomposition
    max_bond: int = 1000,
        Maximum number of singular values to keep during decomposition
    sigma_thresh: float = 0.0,
        Lower bound for singular values to keep during decomposition

    Returns
    -------
    Normalization states of the decomposed tensors (child, parent) and the two
    tensors (child, parent)
    """

    if direction == TDir.UP:
        return decomp_up(
            contracted, child_kind, child_side, parent_kind, max_bond, sigma_thresh
        )
    else:
        return decomp_down(
            contracted, child_kind, child_side, parent_kind, max_bond, sigma_thresh
        )


def decomp_up(
    contracted: jax.Array,
    child_kind: NodeKind,
    child_side: NodeSide,
    parent_kind: NodeKind,
    max_bond: int = 1000,
    sigma_thresh: float = 0.0,
) -> tuple[NodeState, NodeState, jax.Array, jax.Array]:
    """Decompose contracted tensor into node tensor, pushing singular values up.

    Parameters
    ----------
    contracted: jax.Array,
        The contracted tensor, shape depends on the nodes that were contracted
    child_kind: NodeKind,
        Node kind of the child node
    child_side: NodeSide,
        Side of the child node
    parent_kind: NodeKind,
        Node kind of the parent node
    max_bond: int = 1000,
        Maximum number of singular values to keep during decomposition
    sigma_thresh: float = 0.0,
        Lower bound for singular values to keep during decomposition

    Returns
    -------
    Normalization states of the decomposed tensors (child, parent) and the two
    tensors (child, parent)
    """
    # fmt: off
    match (child_kind, child_side, parent_kind):
        case (NodeKind.INNER, NodeSide.LEFT, NodeKind.INNER):
            child, parent = decomp_left_inner_node_up(contracted, max_bond, sigma_thresh)
            return NodeState.UN, NodeState.MX, child, parent
        case (NodeKind.INNER, NodeSide.RIGHT, NodeKind.INNER):
            child, parent = decomp_right_inner_node_up(contracted, max_bond, sigma_thresh)
            return NodeState.UN, NodeState.MX, child, parent
        case (NodeKind.INNER, NodeSide.LEFT, NodeKind.ROOT):
            child, parent = decomp_left_from_root_up(contracted, max_bond, sigma_thresh)
            return NodeState.UN, NodeState.MX, child, parent
        case (NodeKind.INNER, NodeSide.RIGHT, NodeKind.ROOT):
            child, parent = decomp_right_from_root_up(contracted, max_bond, sigma_thresh)
            return NodeState.UN, NodeState.MX, child, parent
        case _:
            raise RuntimeError("Invalid node combination for decomposition")
    # fmt: on


def decomp_down(
    contracted: jax.Array,
    child_kind: NodeKind,
    child_side: NodeSide,
    parent_kind: NodeKind,
    max_bond: int = 1000,
    sigma_thresh: float = 0.0,
) -> tuple[NodeState, NodeState, jax.Array, jax.Array]:
    """Decompose contracted tensor into node tensor, pushing singular values down.

    Parameters
    ----------
    contracted: jax.Array,
        The contracted tensor, shape depends on the nodes that were contracted
    child_kind: NodeKind,
        Node kind of the child node
    child_side: NodeSide,
        Side of the child node
    parent_kind: NodeKind,
        Node kind of the parent node
    max_bond: int = 1000,
        Maximum number of singular values to keep during decomposition
    sigma_thresh: float = 0.0,
        Lower bound for singular values to keep during decomposition

    Returns
    -------
    Normalization states of the decomposed tensors (child, parent) and the two
    tensors (child, parent)
    """
    # fmt: off
    match (child_kind, child_side, parent_kind):
        case (NodeKind.INNER, NodeSide.LEFT, NodeKind.INNER):
            child, parent = decomp_left_inner_node_down(contracted, max_bond, sigma_thresh)
            return NodeState.MX, NodeState.RN, child, parent
        case (NodeKind.INNER, NodeSide.RIGHT, NodeKind.INNER):
            child, parent = decomp_right_inner_node_down(contracted, max_bond, sigma_thresh)
            return NodeState.MX, NodeState.LN, child, parent
        case (NodeKind.INNER, NodeSide.LEFT, NodeKind.ROOT):
            child, parent = decomp_left_from_root_down(contracted, max_bond, sigma_thresh)
            return NodeState.MX, NodeState.RN, child, parent
        case (NodeKind.INNER, NodeSide.RIGHT, NodeKind.ROOT):
            child, parent = decomp_right_from_root_down(contracted, max_bond, sigma_thresh)
            return NodeState.MX, NodeState.LN, child, parent
        case _:
            raise RuntimeError("Invalid node combination for decomposition")
    # fmt: on

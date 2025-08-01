from dataclasses import dataclass
from enum import Enum

import jax

"""Shape of a node tensor"""
NodeShapes = list[tuple[int, ...]]


class NodeKind(Enum):
    """Node kind in the tree, inner or root."""

    INNER = 1
    ROOT = 2


class NodeSide(Enum):
    """Direction from which a node connects to its parent."""

    LEFT = 1
    RIGHT = 2


class NodeState(Enum):
    """Normalization state of a node."""

    ANY = 1
    LN = 2
    RN = 3
    UN = 4
    MX = 5


@dataclass
class Node:
    """Node in the tree tensor network"""

    tensor: jax.Array
    kind: NodeKind
    side: NodeSide
    state: NodeState = NodeState.ANY


class TDir(Enum):
    """Direction of a traversal command."""

    UP = 1
    DOWN = 2


@dataclass
class TraversalCommand:
    """Traversal command: direction and indices of child and parent node."""

    direction: TDir
    cidx: int
    pidx: int


"""Tree traversal iterator return type

* child index
* parent index
* child node
* parent node
* Direction of decomposition

"""
TraversalIt = tuple[int, int, Node, Node, TDir]

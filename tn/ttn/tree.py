import pickle
from functools import reduce
from operator import iconcat
from typing import Optional

import jax
import numpy as np

from tn.ttn.comp import contract, decomp
from tn.ttn.traversal import (
    Traversal,
    _is_leaf,
    _is_left_node,
    _is_right_node,
    _left_leaf,
)
from tn.ttn.types import (
    Node,
    NodeKind,
    NodeShapes,
    NodeSide,
    NodeState,
    TDir,
    TraversalIt,
)


def _get_leaf_structure(n_leaves: int) -> list[int]:
    """Get leaf structure for a fixed number of leaves.

    Parameters
    ----------
    n_leaves: int
        Number of leaves

    Returns
    -------
    Under the assumption of a binary tree with equal depth of each leaf, the number of leaves
    in the final subtree, either 1 or 2. Length of the list is ceil(log2(n_leaves)).
    """

    def split_int(x: int) -> list[int]:
        val = x // 2
        return [val, val] if x % 2 == 0 else [val + 1, val]

    is_power_2 = (n_leaves & (n_leaves - 1) == 0) and n_leaves != 0

    lstruct = [n_leaves]
    steps = int(np.floor(np.log2(n_leaves)))

    if is_power_2:
        steps -= 1

    for _ in range(steps):
        lstruct = reduce(iconcat, [split_int(leaf) for leaf in lstruct], [])

    return lstruct


def _leaf_structure_to_mask(lstruct: list[int]) -> list[bool]:
    """Transform leaf structure into a boolean mask."""
    return reduce(
        iconcat, [[True, True] if ls == 2 else [True, False] for ls in lstruct], []
    )


def _leaf_mask_to_structure(lmask: list[bool]) -> list[int]:
    """Transform boolean mask into leaf structure."""
    ls = np.array(lmask[0::2], dtype=int) + np.array(lmask[1::2], dtype=int)
    return ls.tolist()


def _build_root(bond_dim: int, random_key: jax.Array) -> Node:
    """Build random root node.

    Parameters
    ----------
    bond_dim: int
        Bond dimension (left and right)
    random_key: jax.Array
        jax random key to sample with

    Returns
    -------
    The root node.
    """
    tensor = jax.random.uniform(
        random_key, shape=(bond_dim, bond_dim), minval=-1.0, maxval=1.0
    )
    return Node(tensor, NodeKind.ROOT, NodeSide.LEFT)


def _build_inner_node(
    shape: tuple[int, int, int], side: NodeSide, random_key: jax.Array
) -> Node:
    """Build random inner node.

    Parameters
    ----------
    shape: tuple[int, int, int]
        Shape of the inner node (lbond, tbond, rbond)
    side:
        Left or right
    random_key: jax.Array
        jax random key to sample with

    Returns
    -------
    The inner node.
    """
    tensor = jax.random.uniform(random_key, shape=shape, minval=-1.0, maxval=1.0)
    return Node(tensor, NodeKind.INNER, side)


def random_tree(data_dim: int, bond_dim: int, d: int, random_key: jax.Array) -> "Tree":
    """Build a random initialized tree tensor network.

    Parameters
    ----------
    data_dim: int
        Data dimensionality, corresponds to the number of data features.
    bond_dim: int
        Bond dimension for all but the physical bonds.
    d: int
        Physical bond dimension
    random_key: jax.Array
        jax random key to sample from

    Returns
    -------
    The (non-normalized) tree tensor network.
    """
    key, subkey = jax.random.split(random_key)
    nodes = [_build_root(bond_dim, subkey)]

    n_nodes = 2
    lstruct = _get_leaf_structure(data_dim)

    while n_nodes < len(lstruct):
        for idx in range(n_nodes):
            key, subkey = jax.random.split(key)
            side = NodeSide.LEFT if idx % 2 == 0 else NodeSide.RIGHT
            nodes.append(
                _build_inner_node((bond_dim, bond_dim, bond_dim), side, subkey)
            )
        n_nodes *= 2

    for idx, ls in enumerate(lstruct):
        key, subkey = jax.random.split(key)
        pshape = (d, bond_dim, d) if ls == 2 else (d, bond_dim, 1)
        pside = NodeSide.LEFT if idx % 2 == 0 else NodeSide.RIGHT

        nodes.append(_build_inner_node(pshape, pside, subkey))

    return Tree(nodes, data_dim, _leaf_structure_to_mask(lstruct), d)


def load_ttn(path: str) -> "Tree":
    """Loads an TTN from a file.

    Parameters
    ----------
    path: str
        Full path to the TTN file

    Returns
    -------
    The Tree
    """
    with open(path, "rb") as infile:
        data = pickle.load(infile)
        return Tree(**data)

    raise RuntimeError(f"Could not load {path}")


def save_ttn(ttn: "Tree", path: str) -> None:
    """Save TTN to a file.

    Parameters
    ----------
    ttn: tree
        The TTN to save
    path: str
        Path to save to. Note: All sub-folders need to exits.
    """
    with open(path, "wb") as outfile:
        pickle.dump(
            {
                "nodes": ttn.nodes,
                "data_dim": ttn.data_dim,
                "leaf_mask": ttn.leaf_mask,
                "d": ttn.d,
            },
            outfile,
        )
        return

    raise RuntimeError(f"Could not save to {path}")


class Tree:
    """Tree Tensor Network"""

    def __init__(
        self, nodes: list[Node], data_dim: int, leaf_mask: list[bool], d: int
    ) -> None:
        """The constructor.

        Parameters
        ----------
        nodes: list[Node]
            The nodes of the TTN in BFS order.
        data_dim: int
            Number of features in the data
        leaf_mask: list[bool]
            Mask indicating which legs contract with a data vector
        d: int
            physical bond dimension
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.data_dim = data_dim
        self.leaf_mask = leaf_mask
        self.d = d
        self.traversal = Traversal(n_nodes=self.n_nodes)

    def __iter__(self):
        """Return iterator on nodes in BFS order."""
        return iter(self.nodes)

    @property
    def shapes(self) -> NodeShapes:
        """Return the shapes of all node tensors in BFS order."""
        return [node.tensor.shape for node in self.nodes]

    @property
    def tensors(self) -> list[jax.Array]:
        """Return the tensors of all nodes in BFS order."""
        return [node.tensor for node in self.nodes]

    def feature_to_leaf_idx(self, feature_idx: int) -> int:
        """Get the leaf node index for a feature.

        Parameters
        ----------
        feature_idx: int
            The index of the feature

        Returns
        -------
        The leaf node index
        """
        lstruct = _leaf_mask_to_structure(self.leaf_mask)
        m = np.cumsum(lstruct) - 1

        if feature_idx > m[-1] or feature_idx < 0:
            raise IndexError("Feature index is out of range.")

        return np.argmax(m >= feature_idx) + int(self.n_nodes // 2)

    def leaf_feature_indices(self, node: int) -> tuple[int, Optional[int]]:
        """Return the feature indices for a given leaf node.

        Parameters
        ----------
        node: int
            The leaf node to return the connected feature indices for

        Returns
        -------
        The left feature index and either None, if there is no right feature index or the right
        feature index.
        """
        assert _is_leaf(node, self.n_nodes), f"{node} is not a leaf"

        node_off = node - _left_leaf(self.n_nodes)
        m = np.cumsum(self.leaf_mask) - 1
        lf = m[node_off * 2]
        rf = m[node_off * 2 + 1]

        if lf == rf:
            return lf, None
        else:
            return lf, rf

    def left_canonicalize(self) -> None:
        """Left canonicalize the TTN.

        After left canonicalization, the right most leaf is in no normalization state and has norm 1
        all nodes on the path from root to right most leaf are left normalized and all other nodes
        are upper normalized:

                        Root:            Right path:      All other:
                        -----            -----------      ----------

                                          +---+              |
                        --#--+            #   |              #
                             |   |       / \  |   |         / \    |
                             | = |         |  | = |         | |  = |
                             |   |       \ /  |   |         \ /    |
                        --#--+            #   |              #
                                          +---+              |
        """

        for iteration in self.traversal.left_canonicalize(self.nodes):
            cidx, pidx, child, parent, direction = iteration

            is_normalized = (direction == TDir.UP and child.state == NodeState.UN) or (
                direction == TDir.DOWN and parent.state == NodeState.LN
            )

            if is_normalized:
                continue

            bond = child.tensor.shape[1]

            contracted = contract(child, parent)
            self.update_nodes(
                cidx,
                pidx,
                *decomp(
                    contracted,
                    child.kind,
                    child.side,
                    parent.kind,
                    direction,
                    max_bond=bond,
                ),
            )

    def right_canonicalize(self) -> None:
        """Right canonicalize the TTN.

        After right canonicalization, the left most leaf is in no normalization state and has norm 1
        all nodes on the path from root to right most leaf are right normalized and all other nodes
        are upper normalized:

                        Root:            Left path:       All other:
                        -----            -----------      ----------

                                          +---+              |
                        +--#--            |   #              #
                        |        |        |  / \    |       / \    |
                        |      = |        |  |    = |       | |  = |
                        |        |        |  \ /    |       \ /    |
                        +--#--            |   #              #
                                          +---+              |
        """
        for iteration in self.traversal.right_canonicalize(self.nodes):
            cidx, pidx, child, parent, direction = iteration

            is_normalized = (direction == TDir.UP and child.state == NodeState.UN) or (
                direction == TDir.DOWN and parent.state == NodeState.RN
            )

            if is_normalized:
                continue

            bond = child.tensor.shape[1]

            contracted = contract(child, parent)
            self.update_nodes(
                cidx,
                pidx,
                *decomp(
                    contracted,
                    child.kind,
                    child.side,
                    parent.kind,
                    direction,
                    max_bond=bond,
                ),
            )

    def mixed_canonicalize(self, node: int) -> None:
        """Mixed canonicalize the TTN centering the specified leaf node.

        * The subtree left of `node` gets left-canonicalized
        * The subtree right of `node` gets right-canonicalized

        """
        cmd_iterator, down_path = self.traversal.mixed_canonicalize(node, self.nodes)

        for iteration in cmd_iterator:
            cidx, pidx, child, parent, direction = iteration

            is_norm_up = direction == TDir.UP and child.state == NodeState.UN
            is_norm_down_left = (
                direction == TDir.DOWN
                and _is_left_node(cidx)
                and parent.state == NodeState.RN
            )
            is_norm_down_right = (
                direction == TDir.DOWN
                and _is_right_node(cidx)
                and parent.state == NodeState.LN
            )

            if is_norm_up or is_norm_down_left or is_norm_down_right:
                continue

            bond = child.tensor.shape[1]

            contracted = contract(child, parent)
            self.update_nodes(
                cidx,
                pidx,
                *decomp(
                    contracted,
                    child.kind,
                    child.side,
                    parent.kind,
                    direction,
                    max_bond=bond,
                ),
            )

    def left_sweep(self) -> TraversalIt:
        """Return iterator from right to left."""
        return self.traversal.left_sweep(self.nodes)

    def right_sweep(self) -> TraversalIt:
        """Return iterator from left to right."""
        return self.traversal.right_sweep(self.nodes)

    def update_nodes(
        self,
        cidx: int,
        pidx: int,
        cstate: NodeState,
        pstate: NodeState,
        ctensor: jax.Array,
        ptensor: jax.Array,
    ) -> None:
        """Update child and parent node.

        Parameters
        ----------
        cidx: int
            Index of the child
        pidx: int
            Index of the parent
        cstate: NodeState
            Normalization state of the child
        pstate: NodeState
            Normalization state of the parent
        ctensor: jax.Array
            Child tensor
        ptensor: jax.Array
            Parent tensor
        """
        self.nodes[cidx].state = cstate
        self.nodes[cidx].tensor = ctensor
        self.nodes[pidx].state = pstate
        self.nodes[pidx].tensor = ptensor

from abc import ABC, abstractmethod
from collections import deque
from itertools import count

import jax
import jax.numpy as jnp
import opt_einsum as oe

from tn.ttn.traversal import _is_leaf, _left_child, _right_child, _root_to_leaf
from tn.ttn.tree import Tree


def _build_subtree(tree: Tree, sub_system: list[int]) -> list[int]:
    """Build the minimal subtree, that contains all sub-system feature indices.

    Parameters
    ----------
    tree: Tree
        The Tree tensor network
    sub_system: list[int]
        List of feature (not node) indices to include in the subtree

    Returns
    -------
    List of node indices of the subtree in BFS order
    """
    lnode = tree.feature_to_leaf_idx(sub_system[0])
    rnode = tree.feature_to_leaf_idx(sub_system[-1])

    lpath = _root_to_leaf(lnode, tree.n_nodes)
    rpath = _root_to_leaf(rnode, tree.n_nodes)

    offset = -1

    for ln, rn in zip(lpath, rpath):
        if ln == rn:
            offset += 1
        else:
            break

    lpath = lpath[offset:]
    rpath = rpath[offset:]

    subtree = []

    bidx = len(lpath) - 1

    while bidx >= 0:
        start = lpath[bidx]
        stop = rpath[bidx]
        subtree += list(range(start, stop + 1))
        bidx -= 1

    subtree.sort()

    return subtree


def reduced_density_matrix(tree: Tree, sub_system: list[int]) -> jax.Array:
    """Compute the reduced density matrix of a sub-system.

    Computes rho_X = tr_Y(rho), where X and Y are a bi-partition of the entire system rho
    encoded by the Tree Tensor Network. X is the sub-system to consider and tr_Y is the partial
    trace with respect to Y.

    Parameters
    ----------
    tree: Tree
        The Tree tensor network
    sub_system: list[int]
        List of feature (not node) indices to include in the subtree

    Returns
    -------
    The reduced density matrix of shape (d^(len(sub_system)), d^(len(sub_system)))
    """
    assert len(sub_system) == len(set(sub_system)), "Duplicate sites in sub-system"

    sub_system.sort()

    lnode = tree.feature_to_leaf_idx(sub_system[0])
    tree.mixed_canonicalize(lnode)

    rdm_expr = ReducedDensityMatrixExpr(tree, sub_system)
    n_dim = tree.d ** len(sub_system)

    rdm = rdm_expr(tree)

    return rdm.reshape(n_dim, n_dim)


def conditional_reduced_density_matrix(
    tree: Tree, sub_system: list[int], cond_indices: list[int], cond_states: jax.Array
) -> jax.Array:
    """Compute the conditioned reduced density matrix of a sub-system conditioned by given sites.

    Parameters
    ----------
    tree: Tree
        The Tree tensor network
    sub_system: list[int]
        List of feature (not node) indices to include in the subtree
    cond_indices: list[int]
        List of feature (not node) indices to condition the RDM on
    cond_states: jax.Array
        The states to condition on, shape (len(cond_indices), d)

    Returns
    -------
    The conditional reduced density matrix of shape (d^(len(sub_system)), d^(len(sub_system)))
    """
    assert len(sub_system) == len(set(sub_system)), "Duplicate sites in sub-system"
    assert len(cond_indices) == len(
        set(cond_indices)
    ), "Duplicate sites in conditional sites"
    assert set(cond_indices).isdisjoint(
        set(sub_system)
    ), "Sub system and conditioned indices cannot share the same indices"

    sub_system.sort()
    cond_indices.sort()

    min_idx = min(sub_system[0], cond_indices[0])
    lnode = tree.feature_to_leaf_idx(min_idx)
    tree.mixed_canonicalize(lnode)

    crdm_expr = CondReducedDensityMatrixExp(tree, sub_system, cond_indices)
    n_dim = tree.d ** len(sub_system)

    crdm = crdm_expr(tree, cond_states)
    crdm = crdm.reshape((n_dim, n_dim))

    return crdm / jnp.trace(crdm)


class RDMExprBase(ABC):
    """Base class for TTN reduced density matrix contractions"""

    def _build_einsum_str(self) -> None:
        """Build the Einstein sum string for the necessary contraction"""
        n = count()

        self._prepare_build()
        self._handle_root(n)

        for idx, node in enumerate(self.subtree[1:], start=1):
            uleft = oe.get_symbol(next(n))
            uhead = self.upper_legs.popleft()
            uright = oe.get_symbol(next(n))

            self.labels[idx] = f"{uleft}{uhead}{uright}"

            if _is_leaf(node, self.n_nodes):
                lleft, lright = self._handle_leaf(node, n, uleft, uright)
            else:
                lleft, lright = self._handle_node(node, n, uleft, uright)

            lhead = self.lower_legs.popleft()
            self.labels[idx + self.slen] = f"{lleft}{lhead}{lright}"

        self._compose_str()

    def _prepare_build(self) -> None:
        """Prepare the necessary object to build the einsum string."""
        self.labels = [None] * 2 * self.slen
        self.upper_open = []
        self.lower_open = []

    def _handle_root(self, n: count) -> None:
        """Handle root node (of the subtree) during einsum string creation.

        Parameters
        ----------
        n: count
            counter iterable for the index labels
        """
        rul = oe.get_symbol(next(n))
        rur = oe.get_symbol(next(n))
        rll = oe.get_symbol(next(n))
        rlr = oe.get_symbol(next(n))

        if self.subtree[0] == 0:
            self.labels[0] = f"{rul}{rur}"
            self.labels[self.slen] = f"{rll}{rlr}"
        else:
            rm = oe.get_symbol(next(n))

            self.labels[0] = f"{rul}{rm}{rur}"
            self.labels[self.slen] = f"{rll}{rm}{rlr}"

        self.upper_legs = deque([rul, rur])
        self.lower_legs = deque([rll, rlr])

    def _handle_node(
        self, node: int, n: count, uleft: str, uright: str
    ) -> tuple[str, str]:
        """Handle non-root/non-leaf node during einsum string creation.

        Parameters
        ----------
        node: int
            The node index
        n: count
            counter iterable for the index labels
        uleft: str
            label for the upper-subtree left bond
        uright: str
            label for the upper-subtree right bond

        Returns
        -------
        The labels for the corresponding lower-subtree left and right bond
        """
        if _left_child(node) in self.subtree:
            lleft = oe.get_symbol(next(n))
            self.upper_legs.append(uleft)
            self.lower_legs.append(lleft)
        else:
            lleft = uleft

        if _right_child(node) in self.subtree:
            lright = oe.get_symbol(next(n))
            self.upper_legs.append(uright)
            self.lower_legs.append(lright)
        else:
            lright = uright

        return lleft, lright

    @abstractmethod
    def _handle_leaf(
        self, node: int, n: count, uleft: str, uright: str
    ) -> tuple[str, str]:
        """Handle leaf node during einsum string creation.

        Parameters
        ----------
        node: int
            The node index
        n: count
            counter iterable for the index labels
        uleft: str
            label for the upper-subtree left bond
        uright: str
            label for the upper-subtree right bond

        Returns
        -------
        The labels for the corresponding lower-subtree left and right bond
        """
        pass

    @abstractmethod
    def _compose_str(self) -> None:
        """Compose the einsum string from collected labels."""
        pass


class ReducedDensityMatrixExpr(RDMExprBase):
    """Expression builder for the reduced density matrix of a TTN."""

    def __init__(self, tree: Tree, sub_system: list[int]) -> None:
        """The constructor

        Parameters
        ----------
        tree: Tree
            The Tree Tensor Network
        sub_system: list[int]
            Feature indices of the subsystem to consider
        """
        self.sub_system = sub_system

        self.subtree = _build_subtree(tree, sub_system)
        self.slen = len(self.subtree)
        self.n_nodes = tree.n_nodes

        self.node_features = {
            node: tree.leaf_feature_indices(node)
            for node in self.subtree
            if _is_leaf(node, self.n_nodes)
        }

        if self.slen == 1:
            lf, rf = self.node_features[self.subtree[0]]

            if len(sub_system) == 1:
                if sub_system[0] == lf:
                    self.einsum_str = "abc,dbc->ad"
                else:
                    self.einsum_str = "abc,abd->cd"
            else:
                self.einsum_str = "abc,dbe->caed"
        else:
            self._build_einsum_str()

        shapes = [shape for idx, shape in enumerate(tree.shapes) if idx in self.subtree]
        shapes += shapes

        self.expr = oe.contract_expression(self.einsum_str, *shapes, memory_limit=-1)

    def __call__(self, tree: Tree) -> jax.Array:
        """Execute the contraction expression.

        Parameters
        ----------
        tree: Tree
            The Tree Tensor Network

        Returns
        -------
        Non-reshaped conditional reduced desnity matrix.
        """
        tensors = [
            node.tensor for idx, node in enumerate(tree.nodes) if idx in self.subtree
        ]
        tensors += [jnp.conjugate(t) for t in tensors]

        return self.expr(*tensors, backend="jax")

    def _handle_leaf(
        self, node: int, n: count, uleft: str, uright: str
    ) -> tuple[str, str]:
        """Handle leaf node during einsum string creation.

        Parameters
        ----------
        node: int
            The node index
        n: count
            counter iterable for the index labels
        uleft: str
            label for the upper-subtree left bond
        uright: str
            label for the upper-subtree right bond

        Returns
        -------
        The labels for the corresponding lower-subtree left and right bond
        """
        lf, rf = self.node_features[node]

        if lf not in self.sub_system:
            lleft = uleft
        else:
            lleft = oe.get_symbol(next(n))
            self.upper_open.append(uleft)
            self.lower_open.append(lleft)

        if rf is not None and rf not in self.sub_system:
            lright = uright
        else:
            lright = oe.get_symbol(next(n))
            self.upper_open.append(uright)
            self.lower_open.append(lright)

        return lleft, lright

    def _compose_str(self) -> None:
        """Compose the einsum string from collected labels."""
        out = "".join(reversed(self.upper_open)) + "".join(reversed(self.lower_open))
        self.einsum_str = ",".join(self.labels) + "->" + out


class CondReducedDensityMatrixExp(RDMExprBase):
    """Expression builder for the conditional reduced density matrix of a TTN."""

    def __init__(
        self, tree: Tree, sub_system: list[int], cond_indices: list[int]
    ) -> None:
        """The constructor

        Parameters
        ----------
        tree: Tree
            The Tree Tensor Network
        sub_system: list[int]
            Feature indices of the subsystem to consider
        cond_indices: list[int]
            Feature indices to condition on
        """
        self.sub_system = sub_system
        self.cond_indices = cond_indices

        self.all_sub = sub_system + cond_indices
        self.all_sub.sort()

        self.subtree = _build_subtree(tree, self.all_sub)
        self.slen = len(self.subtree)
        self.n_nodes = tree.n_nodes

        self.node_features = {
            node: tree.leaf_feature_indices(node)
            for node in self.subtree
            if _is_leaf(node, self.n_nodes)
        }

        if self.slen == 1:
            lf, rf = self.node_features[self.subtree[0]]

            if sub_system[0] == lf:
                self.einsum_str = "abc,def,cf->ad"
            else:
                self.einsum_str = "abc,def,ad->cf"
        else:
            self._build_einsum_str()

        shapes = [shape for idx, shape in enumerate(tree.shapes) if idx in self.subtree]
        shapes += shapes
        shapes += [(tree.d, tree.d)] * len(cond_indices)

        self.expr = oe.contract_expression(self.einsum_str, *shapes, memory_limit=-1)

    def __call__(self, tree: Tree, cond_states: jax.Array) -> jax.Array:
        """Execute the contraction expression.

        Parameters
        ----------
        tree: Tree
            The Tree Tensor Network
        cond_states: jax.Array
            The states to condition on, shape (len(cond_indices), d)

        Returns
        -------
        Unornamlized and non-reshaped conditional reduced desnity matrix.
        """
        tensors = [
            node.tensor for idx, node in enumerate(tree.nodes) if idx in self.subtree
        ]
        tensors += [jnp.conjugate(t) for t in tensors]
        tensors += [jnp.outer(state, jnp.conjugate(state)) for state in cond_states]

        return self.expr(*tensors, backend="jax")

    def _prepare_build(self) -> None:
        """Prepare the necessary object to build the einsum string."""
        super()._prepare_build()
        self.cond_labels = []

    def _handle_leaf(
        self, node: int, n: count, uleft: str, uright: str
    ) -> tuple[str, str]:
        """Handle leaf node during einsum string creation.

        Parameters
        ----------
        node: int
            The node index
        n: count
            counter iterable for the index labels
        uleft: str
            label for the upper-subtree left bond
        uright: str
            label for the upper-subtree right bond

        Returns
        -------
        The labels for the corresponding lower-subtree left and right bond
        """
        lf, rf = self.node_features[node]

        if lf not in self.all_sub:
            lleft = uleft
        else:
            lleft = oe.get_symbol(next(n))

            if lf in self.cond_indices:
                self.cond_labels.append(f"{uleft}{lleft}")
            else:
                self.upper_open.append(uleft)
                self.lower_open.append(lleft)

        if rf is not None and rf not in self.all_sub:
            lright = uright
        else:
            lright = oe.get_symbol(next(n))

            if rf in self.cond_indices:
                self.cond_labels.append(f"{uright}{lright}")
            else:
                self.upper_open.append(uright)
                self.lower_open.append(lright)

        return lleft, lright

    def _compose_str(self) -> None:
        """Compose the einsum string from collected labels."""
        out = "".join(reversed(self.upper_open)) + "".join(reversed(self.lower_open))
        self.einsum_str = ",".join(self.labels + self.cond_labels) + "->" + out

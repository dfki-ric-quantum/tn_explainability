from tn.ttn.types import Node, TDir, TraversalCommand, TraversalIt


def _is_left_node(node: int) -> bool:
    """Check if node is connecting to parent from the left."""
    return node % 2 == 1


def _is_right_node(node: int) -> bool:
    """Check if node is connecting to parent from the right."""
    return node % 2 == 0


def _is_leaf(node: int, n_nodes: int) -> bool:
    """Check if node is leaf node."""
    return node >= _left_leaf(n_nodes) and node <= _right_leaf(n_nodes)


def _parent(node: int) -> int:
    """Get parent of a node."""
    if _is_left_node(node):
        return (node - 1) // 2
    else:
        return (node - 2) // 2


def _left_child(node: int) -> int:
    """Get left child of a node"""
    return 2 * node + 1


def _right_child(node: int) -> int:
    """Get right child of a node"""
    return 2 * node + 2


def _left_leaf(n_nodes: int) -> int:
    """Get left-most leaf of the tree."""
    return n_nodes // 2


def _right_leaf(n_nodes: int) -> int:
    """Get right-most leaf of the tree."""
    return n_nodes - 1


def _root_to_leaf(node: int, n_nodes: int) -> list[int]:
    """Return the path from the root node to a specified leaf.

    Parameters
    ----------
    node: int
        The leaf node
    n_nodes: int
        number of nodes in the tree

    Returns
    -------
    List of nodes to traverse from root to the leaf.
    """

    assert _is_leaf(node, n_nodes), f"Node {node} is not a leaf node"

    path = [node]

    while node != 0:
        parent = _parent(node)
        path.append(parent)
        node = parent

    return list(reversed(path))


def _traversal_cmd(node_from: int, node_to: int) -> TraversalCommand:
    """Create the traversal command to move singular values from `node_from` to `node_to`

    Parameters
    ----------
    node_from: int
        Node index from which to traverse
    node_to
        Node index to which to traverse

    Returns
    -------
    The traversal command.
    """
    if node_from > node_to:
        return TraversalCommand(TDir.UP, node_from, node_to)
    else:
        return TraversalCommand(TDir.DOWN, node_to, node_from)


def _neighbours_right(node: int, n_nodes: int) -> list[int]:
    """Get the neighbours of a node, in order (right child, left child, parent).

    Parameters
    ----------
    node: int
        The node's index
    n_nodes: int
        Total number of nodes in the tree

    Returns
    -------
    A list of the nodes neighbours, in the order specified above.
    """
    if node == 0:
        return [2, 1]

    is_leaf = node >= _left_leaf(n_nodes)
    parent = _parent(node)

    if is_leaf:
        return [parent]

    return [_right_child(node), _left_child(node), parent]


def _neighbours_left(node: int, n_nodes: int) -> list[int]:
    """Get the neighbours of a node, in order (left child, right child, parent).

    Parameters
    ----------
    node: int
        The node's index
    n_nodes: int
        Total number of nodes in the tree

    Returns
    -------
    A list of the nodes neighbours, in the order specified above.
    """
    if node == 0:
        return [1, 2]

    is_leaf = node >= _left_leaf(n_nodes)
    parent = _parent(node)

    if is_leaf:
        return [parent]

    return [_left_child(node), _right_child(node), parent]


def _dfs_sweep_cmd(
    node: int,
    final_node: int,
    n_nodes: int,
    corder: str,
    commands: list[TraversalCommand],
    cache: list[int] = [],
) -> None:
    """Depth-first traversal from either the left or right leaf node, building the sweep path.

    After the function was called with the start_node, it will recursively build up the path in
    `commands`. Note: The last command to push singular values up from the final node, is not
    included after the function was called!

    Parameters
    ----------
    node: int
        The start node
    final_node: int
        The final node of the traversal
    n_nodes: int
        total number of nodes in the tree
    corder: str
        Order in which to traverse childre, 'L' to follow left child first 'R' to follow right child
        first.
    commands: list[TraversalCommand]
        in-out parameter to store the commands for the sweep. should be an empty list, when called
    cache: list[int] = []
        recursion cache, to track already visited nodes.

    """
    cache.append(node)

    if corder == "L":
        neighbours = _neighbours_left(node, n_nodes)
    else:
        neighbours = _neighbours_right(node, n_nodes)

    for n in neighbours:
        if n not in cache:
            if n < _left_leaf(n_nodes):
                commands.append(_traversal_cmd(node, n))

            _dfs_sweep_cmd(n, final_node, n_nodes, corder, commands, cache)

            if final_node not in cache:
                commands.append(_traversal_cmd(n, node))


def _build_left_sweep(n_nodes: int) -> list[TraversalCommand]:
    """Build traversal path for left sweep (training bonds from right to left).

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree

    Returns
    -------
    List of traversal commands to execute for left sweep.
    """
    commands = []
    start_node = _right_leaf(n_nodes)
    final_node = _left_leaf(n_nodes)

    _dfs_sweep_cmd(start_node, final_node, n_nodes, "R", commands, [])

    commands.append(TraversalCommand(TDir.DOWN, final_node, _parent(final_node)))

    return commands


def _build_right_sweep(n_nodes: int) -> list[TraversalCommand]:
    """Build traversal path for right sweep (training bonds from left to right).

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree

    Returns
    -------
    List of traversal commands to execute for right sweep.
    """
    commands = []
    start_node = _left_leaf(n_nodes)
    final_node = _right_leaf(n_nodes)

    _dfs_sweep_cmd(start_node, final_node, n_nodes, "L", commands, [])

    commands.append(TraversalCommand(TDir.DOWN, final_node, _parent(final_node)))

    return commands


def _build_lc_path(n_nodes: int) -> list[TraversalCommand]:
    """Build traversal path for left canonicalization.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree

    Returns
    -------
    List of traversal commands to execute for left canonicalization.
    """
    start = _left_leaf(n_nodes)
    stop = _right_leaf(n_nodes)

    commands = []

    while start != 0:
        for nidx in range(start, stop):
            commands.append(TraversalCommand(TDir.UP, nidx, _parent(nidx)))

        start = _parent(start)
        stop = _parent(stop)

    node = 0

    while node != _right_leaf(n_nodes):
        child = _right_child(node)

        if child == n_nodes:
            child = _left_child(node)

        commands.append(TraversalCommand(TDir.DOWN, child, node))
        node = child

    return commands


def _build_rc_path(n_nodes: int) -> list[TraversalCommand]:
    """Build traversal path for right canonicalization.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the tree

    Returns
    -------
    List of traversal commands to execute for right canonicalization.
    """
    start = _right_leaf(n_nodes)
    stop = _left_leaf(n_nodes)

    commands = []

    while start != 0:
        for nidx in range(start, stop, -1):
            commands.append(TraversalCommand(TDir.UP, nidx, _parent(nidx)))

        start = _parent(start)
        stop = _parent(stop)

    node = 0

    while node != _left_leaf(n_nodes):
        child = _left_child(node)
        commands.append(TraversalCommand(TDir.DOWN, child, node))
        node = child

    return commands


def _build_mc_path(node: int, n_nodes: int) -> tuple[list[TraversalCommand], list[int]]:
    """Build traversal path for mixed canonicalization.

    Parameters
    ----------
    node: int
        The leaf node to center

    n_nodes: int
        Number of nodes in the tree

    Returns
    -------
    List of traversal commands to execute for mixed canonicalization and the path from the
    root node to the center leaf.
    """
    down_path = _root_to_leaf(node, n_nodes)

    start = _left_leaf(n_nodes)
    stop = _right_leaf(n_nodes)

    commands = []

    while start != 0:
        for nidx in range(start, stop + 1):
            if nidx not in down_path:
                commands.append(TraversalCommand(TDir.UP, nidx, _parent(nidx)))

        start = _parent(start)
        stop = _parent(stop)

    for child, parent in zip(down_path[1:], down_path):
        commands.append(TraversalCommand(TDir.DOWN, child, parent))

    return commands, down_path


class TraversalIterator:
    """Iterator for tree traversals."""

    def __init__(self, nodes: list[Node], path: list[TraversalCommand]) -> None:
        """The constructor.

        Parameters
        ----------
        nodes: list[Node]
            List of nodes in the tree in BFS order.
        path: list[TraversalCommand]
            The traversal path to iterate
        """
        self.nodes = nodes
        self.path_iter = iter(path)

    def __iter__(self) -> "TraversalIterator":
        """Return the iterator."""
        return self

    def __next__(self) -> TraversalIt:
        """Get next item in the traversal.

        Returns
        -------
        Indices of the child and parent node, the child and parent node and the direction
        of decomposition.
        """
        cmd = next(self.path_iter)

        return (
            cmd.cidx,
            cmd.pidx,
            self.nodes[cmd.cidx],
            self.nodes[cmd.pidx],
            cmd.direction,
        )


class Traversal:
    """Manages traversal paths for various actions on the TTN."""

    def __init__(self, n_nodes: int) -> None:
        """The constructor.

        Parameters
        ----------
        n_nodes: int
            Number of nodes in the tree
        """
        self.n_nodes = n_nodes

        self.lc_path = _build_lc_path(n_nodes)
        self.rc_path = _build_rc_path(n_nodes)
        self.left_sweep_path = _build_left_sweep(n_nodes)
        self.right_sweep_path = _build_right_sweep(n_nodes)

        left_leaf = _left_leaf(n_nodes)
        right_leaf = _right_leaf(n_nodes)

        self.mc_cache = {
            left_leaf: _build_mc_path(left_leaf, n_nodes),
            right_leaf: _build_mc_path(right_leaf, n_nodes),
        }

    def left_canonicalize(self, nodes: list[Node]) -> TraversalIterator:
        """Traverse the left canonicalization path."""
        return TraversalIterator(nodes, self.lc_path)

    def right_canonicalize(self, nodes: list[Node]) -> TraversalIterator:
        """Traverse the right canonicalization path."""
        return TraversalIterator(nodes, self.rc_path)

    def mixed_canonicalize(
        self, node: int, nodes: list[Node]
    ) -> tuple[TraversalIterator, list[int]]:
        """Traverse the mixed canonicalization path with a given center node.

        Parameters
        ----------
        node: int
            The leaf node to center
        nodes: list[Node]
            Nodes in the tree

        Returns
        -------
        List of traversal commands and list of node indices of the down path.
        """
        if node not in self.mc_cache:
            self.mc_cache[node] = _build_mc_path(node, self.n_nodes)

        cmd_path, down_path = self.mc_cache[node]

        return TraversalIterator(nodes, cmd_path), down_path

    def left_sweep(self, nodes: list[Node]) -> TraversalIterator:
        """Traverse the left sweep path."""
        return TraversalIterator(nodes, self.left_sweep_path)

    def right_sweep(self, nodes: list[Node]) -> TraversalIterator:
        """Traverse the right sweep path."""
        return TraversalIterator(nodes, self.right_sweep_path)

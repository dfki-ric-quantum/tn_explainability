import pickle

from tn.ttn.tree import Tree


def save_ttn(tree: Tree, path: str) -> None:
    """Save MPS to a file.

    Parameters
    ----------
    tree: Tree
        The Tree Tensor Network to save
    path: str
        Path to save to. Note: All sub-folders need to exits.
    """
    with open(path, "wb") as outfile:
        pickle.dump(
            {"nodes": tree.nodes, "leaf_mask": tree.leaf_mask, "d": tree.d}, outfile
        )
        return

    raise RuntimeError(f"Could not save to {path}")


def load_ttn(path: str) -> Tree:
    """Loads an Tree Tensor Network from a file.

    Parameters
    ----------
    path: str
        Full path to the MPS file

    Returns
    -------
    The Tree Tensor Network
    """
    with open(path, "rb") as infile:
        data = pickle.load(infile)

        return Tree(data["nodes"], data["leaf_mask"], data["d"])

    raise RuntimeError(f"Could not load {path}")

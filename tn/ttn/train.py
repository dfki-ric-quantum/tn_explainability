import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from ml_collections import config_dict
from sklearn.model_selection import train_test_split

from tn.data.batch import get_batches
from tn.optimizer import TreeOptimizer
from tn.ttn.comp import contract, decomp
from tn.ttn.contraction import build_psic_contraction_expr
from tn.ttn.metrics import NLLFunctor
from tn.ttn.tree import Tree


class PsiGradFunctor:
    """Functor to compute the data dependent part of the gradient during training.

    On initialization, the functor pre-computes and optimizes the contraction path

                 +-#-+
                /     \
               #       X
              / \     / X cn
             #   #   #   X
            / \ / \ / \ / \
            o o o o o o o o

    Where `#` are the nodes of the Tree Tensor network, `o` the vectors
    encoding data and the three `X` is the contracted tensor, marked by `cn`.

    As long as the shapes of all tensors remain unchanged (e.g., while iterating over batches), the
    functor can be reused.
    """

    def __init__(
        self, tree: Tree, contracted_shape: tuple[int, ...], cidx: int, pidx: int
    ) -> None:
        """The constructor:

        Parameters
        ----------
        tree: Tree
            The tree tensor network
        contracted_shape: tuple[int, ...]
            Shape of the contracted tensor.
        cidx: int
            Index of the child node of the contraction
        pidx: int
            Index of the parent node of the contraction
        """
        self.cidx = cidx
        self.pidx = pidx
        self.tensors = [
            node.tensor for nidx, node in enumerate(tree) if nidx not in [cidx, pidx]
        ]

        self.expr = build_psic_contraction_expr(
            tree.shapes, contracted_shape, cidx, pidx, tree.leaf_mask, tree.d
        )

    def __call__(self, states: jax.Array, contracted: jax.Array) -> jax.Array:
        """Compute the data dependent part of the NLL gradient.

        2/|D| * Sum psi'(v)/psi(v)

        Parameters
        ----------
        states: jax.Array
            The states to compute the gradient on, shape (n_samples, n_sites, d)
        contracted: jax.Array
            The contracted tensor.

        Returns
        -------
        The data dependent part of the NLL loss with respect to the contracted tensor.

        """
        return (2.0 / states.shape[0]) * jnp.sum(
            jax.vmap(self._grad, in_axes=(0, None))(states, contracted), axis=0
        )

    def _grad(self, state: jax.Array, contracted: jax.Array) -> jax.Array:
        """Compute the data dependent gradient on a single state.

        psi'(v)/psi(v)

        Parameters
        ----------
        state: jax.Array
            The state to compute the gradient on, shape (n_sites, d)
        contracted: jax.Array
            The contracted order-4 tensor.

        Returns
        -------
        The data dependent part of the NLL loss with respect to the contracted order-4 tensor.
        """
        psi_v, psi_d = jax.value_and_grad(self._psi)(contracted, state)
        return psi_d / psi_v

    def _psi(self, contracted, state: jax.Array) -> jax.Array:
        """Forward pass, compute psi(v).

        Parameters
        ----------
        contracted: jax.Array
            The contracted  tensor.
        state: jax.Array
            The state to compute the gradient on, shape (n_sites, d)

        Returns
        -------
        psi(v)
        """
        return jnp.squeeze(
            self.expr(
                *self.tensors,
                contracted,
                *state,
                backend="jax",
            )
        )


def evaluate(tree: Tree, x_train: jax.Array, x_val: jax.Array) -> dict:
    """Evaluate the MPS

    Parameters
    ----------
    tree: Tree
        The tree tensor network to evaluate
    x_train: jax.Array
        Training data
    x_val: jax.Array
        validation data

    Returns
    -------
    A dictonary containing training and validation loss.
    """
    loss_fn = NLLFunctor(tree.shapes, tree.leaf_mask, tree.d)

    return {
        "tl": loss_fn(x_train, tree).mean(),
        "vl": loss_fn(x_val, tree).mean(),
    }


def log_metrics(sweep: int, metrics: dict) -> None:
    """Log MPS metrics.

    Parameters
    ----------
    sweep: int
        The current training sweep
    metrics: dict
        dictonary containing the evaluation metrics
    """
    logging.info(
        "Sweep: %d, tl: %.4f, , vl: %.4f",
        sweep,
        metrics["tl"],
        metrics["vl"],
    )


def train_tree(
    tree: Tree,
    x: jax.Array,
    *,
    cfg: config_dict.FrozenConfigDict,
    validation_split: float = 0.2,
    patience: int = 2,
) -> list[dict]:
    """Example training routine for a generative Tree Tensor Network.

    This function implements the most basic training loop over a Tree Tensor Network to learn
    a generative model using stochastic gradient descent. The implementation follows the original
    paper[1] closely. It's main purpose:

    * Demonstrate the training loop with minimal overhead and customization
    * Provide a performance tuned baseline implementation
    * Act as template for an actual training loop
    * Provide a readily available facility to test training on data sets of interest.

    [1] Cheng et al., "Tree tensor networks for generative modeling" (2019).

    Parameters
    ----------
    tree: Tree,
        The initial Tree Tensor network to train on
    x: jax.Array
        The data to train on
    cfg: config_dict.FrozenConfigDict
        MPS/training configuration
    validation_split: float = 0.2
        Fraction of data to use for post-sweep evaluation
    patience: int = 2
        Number of sweeps to be patience for no further improvement, before
        stopping early.

    Returns
    -------
    Evaluation metrics (train/validation loss and avg. bond dimensions)
    Returns
    """
    eval_metrics = []

    tree.left_canonicalize()

    adam = optax.adam(cfg.lrate)
    optimizer = TreeOptimizer(adam)

    x_train, x_val = train_test_split(x, test_size=validation_split)

    logging.info("Starting TTN training...")
    logging.info("Computing initial metrics...")

    metrics = evaluate(tree, x_train, x_val)
    eval_metrics.append(metrics)
    log_metrics(sweep=0, metrics=metrics)

    early_stop = EarlyStopping(min_delta=0.01, patience=patience)
    early_stop.update(metrics["vl"])

    for sweep in range(cfg.n_sweeps):
        logging.info("Sweep %d - left sweep", sweep + 1)

        for iteration in tree.left_sweep():
            cidx, pidx, child, parent, direction = iteration

            contracted = contract(child, parent)
            psi_grad = PsiGradFunctor(tree, contracted.shape, cidx, pidx)

            for batch in get_batches(x_train, cfg.batch_size):
                gradient = 2 * contracted - psi_grad(batch, contracted)
                contracted = optimizer.update(contracted, gradient, cidx, pidx)

            tree.update_nodes(
                cidx,
                pidx,
                *decomp(
                    contracted,
                    child.kind,
                    child.side,
                    parent.kind,
                    direction,
                    cfg.max_bond,
                    cfg.sigma_thresh,
                ),
            )

        logging.info("Sweep %d - right sweep", sweep + 1)

        for iteration in tree.right_sweep():
            cidx, pidx, child, parent, direction = iteration

            contracted = contract(child, parent)
            psi_grad = PsiGradFunctor(tree, contracted.shape, cidx, pidx)

            for batch in get_batches(x_train, cfg.batch_size):
                gradient = 2 * contracted - psi_grad(batch, contracted)
                contracted = optimizer.update(contracted, gradient, cidx, pidx)

            tree.update_nodes(
                cidx,
                pidx,
                *decomp(
                    contracted,
                    child.kind,
                    child.side,
                    parent.kind,
                    direction,
                    cfg.max_bond,
                    cfg.sigma_thresh,
                ),
            )

        metrics = evaluate(tree, x_train, x_val)
        eval_metrics.append(metrics)
        log_metrics(sweep=sweep + 1, metrics=metrics)
        early_stop.update(metrics["vl"])

        if early_stop.should_stop:
            logging.info("Stopping early at sweep: %d", sweep + 1)
            break

    logging.info("Finished TTN training.")

    return eval_metrics

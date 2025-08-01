import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from ml_collections import config_dict
from sklearn.model_selection import train_test_split

from tn.data.batch import get_batches
from tn.mps import MPS
from tn.mps.contraction import build_psic_contraction_expr
from tn.mps.metrics import NLLFunctor
from tn.mps.ops import trunc_svd_decomp_normalize_left, trunc_svd_decomp_normalize_right
from tn.optimizer import MPSOptimizer


class PsiGradFunctor:
    """Functor to compute the data dependent part of the gradient during training.

    On initialization, the functor pre-computes and optimizes the contraction path

         cs
    o-o-####-o-...-o-o
    | | |  | |     | |
    o o o  o o ... o o

    As long as the shapes of all tensors remain unchanged (e.g., while iterating over batches), the
    functor can be reused.
    """

    def __init__(
        self,
        mps: MPS,
        contracted_shape: tuple[int, ...],
        contracted_site: int,
    ) -> None:
        """The constructor:

        Parameters
        ----------
        mps: MPS
            The Matrix Product State to contract.
        contracted_shape: tuple[int, int, int, int]
            Shape of the contracted order-4 tensor.
        contracted_site: int
            Index of the contracted order-4 tensor in the MPS.
        """
        self.mps = mps
        self.contracted_site = contracted_site

        self.expr = build_psic_contraction_expr(
            mps.shapes, contracted_shape, contracted_site, mps.d
        )

    def __call__(self, states: jax.Array, contracted: jax.Array) -> jax.Array:
        """Compute the data dependent part of the NLL gradient.

        2/|D| * Sum psi'(v)/psi(v)

        Parameters
        ----------
        states: jax.Array
            The states to compute the gradient on, shape (n_samples, n_sites, d)
        contracted: jax.Array
            The contracted order-4 tensor.

        Returns
        -------
        The data dependent part of the NLL loss with respect to the contracted order-4 tensor.

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
            The contracted order-4 tensor.
        state: jax.Array
            The state to compute the gradient on, shape (n_sites, d)

        Returns
        -------
        psi(v)
        """
        return jnp.trace(
            self.expr(
                *self.mps[: self.contracted_site],
                contracted,
                *self.mps[self.contracted_site + 2 :],
                *state,
                backend="jax",
            )
        )


def evaluate(mps: MPS, x_train: jax.Array, x_val: jax.Array) -> dict:
    """Evaluate the MPS

    Parameters
    ----------
    mps: MPS
        The MPS to evaluate
    x_train: jax.Array
        Training data
    x_val: jax.Array
        validation data

    Returns
    -------
    A dictonary containing training and validation loss, as well as the average
    bond dimension of the MPS.
    """
    loss_fn = NLLFunctor(mps.shapes, mps.d)

    return {
        "tl": loss_fn(x_train, mps).mean(),
        "vl": loss_fn(x_val, mps).mean(),
        "ab": mps.average_bond_dim,
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
        "Sweep: %d, tl: %.4f, , vl: %.4f, ab: %.4f.",
        sweep,
        metrics["tl"],
        metrics["vl"],
        metrics["ab"],
    )


def train_mps(
    mps: MPS,
    x: jax.Array,
    *,
    cfg: config_dict.FrozenConfigDict,
    validation_split: float = 0.2,
    patience: int = 2,
) -> list[dict]:
    """MPS Training

    [1] Han et al., "Unsupervised Generative Modeling Using Matrix Product States" (2018).

    Parameters
    ----------
    mps: MPS,
        The initial Matrix Product State to train on
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
    """

    eval_metrics = []

    mps.left_canonicalize()

    adam = optax.adam(cfg.lrate)
    optimizer = MPSOptimizer(adam, n_bonds=mps.n_sites - 1)

    x_train, x_val = train_test_split(x, test_size=validation_split)

    logging.info("Starting MPS training...")
    logging.info("Computing initial metrics...")

    metrics = evaluate(mps, x_train, x_val)
    eval_metrics.append(metrics)
    log_metrics(sweep=0, metrics=metrics)

    early_stop = EarlyStopping(min_delta=0.01, patience=patience)
    early_stop.update(metrics["vl"])

    for sweep in range(cfg.n_sweeps):
        logging.info("Sweep %d - left sweep", sweep + 1)

        for lidx, ridx, lhs, rhs in mps.two_site_reverse():
            contracted = jnp.einsum("abc,cde->abde", lhs, rhs)
            psi_grad = PsiGradFunctor(mps, contracted.shape, lidx)

            for batch in get_batches(x_train, cfg.batch_size):
                gradient = 2 * contracted - psi_grad(batch, contracted)
                contracted = optimizer.update(contracted, gradient, lidx)

            mps.update_ln(
                lidx,
                ridx,
                *trunc_svd_decomp_normalize_right(
                    contracted, cfg.max_bond, cfg.sigma_thresh
                ),
            )

        logging.info("Sweep %d - right sweep", sweep + 1)

        for lidx, ridx, lhs, rhs in mps.two_site_forward():
            contracted = jnp.einsum("abc,cde->abde", lhs, rhs)
            psi_grad = PsiGradFunctor(mps, contracted.shape, lidx)

            for batch in get_batches(x_train, cfg.batch_size):
                gradient = 2 * contracted - psi_grad(batch, contracted)
                contracted = optimizer.update(contracted, gradient, lidx)

            mps.update_rn(
                lidx,
                ridx,
                *trunc_svd_decomp_normalize_left(
                    contracted, cfg.max_bond, cfg.sigma_thresh
                ),
            )

        metrics = evaluate(mps, x_train, x_val)
        eval_metrics.append(metrics)
        log_metrics(sweep=sweep + 1, metrics=metrics)
        early_stop.update(metrics["vl"])

        if early_stop.should_stop:
            logging.info("Stopping early at sweep: %d", sweep + 1)
            break

    logging.info("Finished MPS training.")

    return eval_metrics

from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax import linen as nn
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping
from ml_collections import config_dict
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

from tn.ml.ae.model import ae


@jax.vmap
def squared_error(x_hat: jax.Array, x: jax.Array) -> jax.Array:
    """Squared error.

    Parameters
    ----------
    x_hat: jax.Array
        Reconstruction
    x: jax.Array
        Original

    Returns
    -------
    The squared error ½ * Σ(xh - x).
    """
    return jnp.inner(x_hat - x, x_hat - x) / 2.0


@jax.jit
def compute_metrics(x_hat: jax.Array, x: jax.Array) -> dict:
    """Compute all evaluation metrics (Currently only MSE)

    Parameters
    ----------
    x_hat: jax.Array
        Reconstruction
    x: jax.Array
        Original

    Returns
    -------
    Dictonary containing the metrics
    """
    mse = squared_error(x_hat, x).mean()
    return {"mse": mse}


@partial(jax.jit, static_argnames=["cfg"])
def train_step(
    state: train_state.TrainState, batch: jax.Array, cfg: config_dict.FrozenConfigDict
) -> train_state.TrainState:
    """Single training step for the autoencoder.

    Parameters
    ----------
    state: train_state.TrainState
        State of the autoencoder
    batch: jax.Array
        mini-batch to perform the training step on
    cfg: config_dict.FrozenConfigDict
        model/training configuration

    Returns
    -------
    The updated state after the gradient update.
    """

    def loss_fn(params):
        x_hat = ae(cfg).apply({"params": params}, batch)

        loss = squared_error(x_hat, batch).mean()
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@partial(jax.jit, static_argnames=["cfg"])
def evaluate(params: Any, x: jax.Array, cfg: config_dict.FrozenConfigDict) -> dict:
    """Evaluate the autoencoder on data.

    Parameters
    ----------
    params: Any
        Autoencoder parameters
    x: jax.Array
        The data to evaluate on
    cfg: config_dict.FrozenConfigDict
        Model/training configuration
    """

    def eval_fn(auto_encoder):
        x_hat = auto_encoder(x)
        metrics = compute_metrics(x_hat, x)
        return metrics

    return nn.apply(eval_fn, ae(cfg))({"params": params})


@partial(jax.jit, static_argnames=["cfg"])
def score_ae(params: Any, x: jax.Array, cfg: config_dict.FrozenConfigDict) -> jax.Array:
    """Score samples.

    Parameters
    ----------
    params: Any
        Autoencoder parameters
    x: jax.Array
        The data to score
    cfg: config_dict.FrozenConfigDict
        Model/training configuration
    """

    def score_fn(auto_encoder):
        x_hat = auto_encoder(x)
        return squared_error(x_hat, x)

    return nn.apply(score_fn, ae(cfg))({"params": params})


def log_metrics(epoch: int, metrics: dict) -> None:
    """Log AE metrics.

    Parameters
    ----------
    epoch: int
        The current training epoch
    metrics: dict
        dictonary containing the evaluation metrics
    """
    logging.info("Epoch: %d, mse loss: %.4f", epoch, metrics["mse"])


def train_ae(
    x: jax.typing.ArrayLike,
    *,
    cfg: config_dict.FrozenConfigDict,
    random_key: jax.Array,
    validation_split: float = 0.2,
    patience: int = 2,
    state: Optional[train_state.TrainState] = None,
) -> tuple[list[dict], train_state.TrainState]:
    """Train Autoencoder.

    Parameters
    ----------
    x: jax.typing.ArrayLike
        Training data, will be split according to validation_split
    cfg: config_dict.FrozenConfigDict
        Training/Model configuration
    random_key: jax.Array
        Random key for sampling
    validation_split: float = 0.2
        Ration of data to use for validation/evaluation during training
    patience: int = 2
        Number of sweeps to be patience for no further improvement, before
        stopping early.
    state: Optional[train_state.TrainState] = None
        Optional training state, if training is continued after a a previous run.

    Returns
    -------
    Evaluation metrics over all training epochs and the training state
    """
    n_features = x.shape[1]
    rng = np.random.default_rng()

    key, init_key = jax.random.split(random_key)

    if not state:
        init_data = jnp.ones((cfg.batch_size, n_features), dtype=float)
        params = ae(cfg).init(init_key, init_data)["params"]

        state = train_state.TrainState.create(
            apply_fn=ae(cfg).apply, params=params, tx=optax.adam(cfg.lrate)
        )

    x_train, x_test = train_test_split(x, test_size=validation_split)
    eval_metrics = []

    logging.info("Starting AE training...")
    logging.info("Computing initial metrics...")

    metrics = evaluate(state.params, x_test, cfg)
    eval_metrics.append(metrics)
    log_metrics(epoch=0, metrics=metrics)

    early_stop = EarlyStopping(min_delta=0.01, patience=patience)
    early_stop.update(metrics["mse"])

    for epoch in range(cfg.n_epochs):
        rng.shuffle(x_train, axis=0)

        for batch_slice in gen_batches(x_train.shape[0], cfg.batch_size):
            batch = x_train[batch_slice]
            state = train_step(state, batch, cfg)

        metrics = evaluate(state.params, x_test, cfg)
        eval_metrics.append(metrics)
        log_metrics(epoch + 1, metrics)
        early_stop.update(metrics["mse"])

        if early_stop.should_stop:
            logging.info("Stopping early at epoch: %d", epoch + 1)
            break

    return eval_metrics, state

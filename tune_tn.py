# ruff: noqa: E402

import random
import sys
from datetime import datetime
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import optuna
from absl import app, flags, logging

from tn.tune.mps import Objective as MPSObjective
from tn.tune.ttn import Objective as TTNObjective

FLAGS = flags.FLAGS
flags.DEFINE_enum("tn", None, ["mps", "ttn"], help="Tensor network")
flags.DEFINE_enum(
    "dataset",
    None,
    ["satellite", "ecg5000", "spambase"],
    short_name="d",
    help="Dataset to tune on",
)
flags.DEFINE_integer("trials", 30, short_name="t", help="Number of trials")


STORAGE = "sqlite:///tune.db"
N_SPLITS = 5


def get_objective(tn: str, dataset: str) -> Callable:
    """Get the tuning objective.

    Parameters
    ----------
    tn: str
        The tensor network type to tune
    dataset: str
        The dataset to tune on

    Returns
    -------
    The tuning objective.
    """
    seed = random.randint(0, sys.maxsize)

    match tn:
        case "mps":
            return MPSObjective(dataset=dataset, seed=seed, n_splits=N_SPLITS)
        case "ttn":
            return TTNObjective(dataset=dataset, seed=seed, n_splits=N_SPLITS)
        case _:
            raise ValueError(f"No tensor network type {tn}")


def main(_) -> None:
    """Main hyper-parameter tuning run."""
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    study_name = f"{FLAGS.tn}-{FLAGS.dataset}-{time}"

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5),
    )
    objective = get_objective(FLAGS.tn, FLAGS.dataset)

    logging.info("Starting study %s...", study_name)
    study.optimize(objective, n_trials=FLAGS.trials)


if __name__ == "__main__":
    app.run(main)

import random
import sys
from datetime import datetime

import optuna
from absl import app, flags, logging

from tn.tune.ae import get_objective

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "dataset",
    None,
    ["satellite", "ecg5000", "spambase"],
    short_name="d",
    help="Dataset to tune on",
)
flags.DEFINE_integer("trials", 100, short_name="t", help="Number of trials")


STORAGE = "sqlite:///tune.db"
N_SPLITS = 5


def main(_) -> None:
    """Main hyper-parameter tuning run."""
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    study_name = f"{FLAGS.ae}-{FLAGS.dataset}-{time}"
    seed = random.randint(0, sys.maxsize)

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5),
    )
    objective = get_objective(FLAGS.dataset, seed, N_SPLITS)

    logging.info("Starting study %s...", study_name)
    study.optimize(objective, n_trials=FLAGS.trials)


if __name__ == "__main__":
    app.run(main)

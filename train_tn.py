# ruff: noqa: E402

import os
from datetime import datetime
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

from absl import app, flags, logging
from ml_collections import config_dict, config_flags

from tn.data.fold import TrainingKFold
from tn.encoding import get_encoder
from tn.mps.mps import random_mps, save_mps
from tn.mps.train import train_mps
from tn.ttn.train import train_tree
from tn.ttn.tree import random_tree, save_ttn

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string("outdir", "./", short_name="o", help="Output directory")
flags.DEFINE_enum(
    "dataset", None, ["satellite", "ecg5000", "spambase"], short_name="d", help="Dataset"
)
flags.DEFINE_integer("seed", None, short_name="s", help="Random seed")
flags.DEFINE_integer("folds", 10, short_name="f", help="Number fo training folds")
flags.DEFINE_float("contamination", 0.05, short_name="c", help="Outlier contamination")
flags.DEFINE_enum(
    "model",
    default="mps",
    enum_values=["mps", "ttn"],
    short_name="m",
    help="model type",
)


def get_save_path(full: bool = False) -> str:
    """Build the path to save trained models to.

    Parameters
    ----------
    full: bool = False
        If set to true, return absolute path
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    path = f"{FLAGS.outdir}/{FLAGS.model}_{FLAGS.dataset}_{time}/"

    os.makedirs(path, exist_ok=True)

    if full:
        path = f"{os.getcwd()}/{path}"

    return path


def _get_encoder(cfg: config_dict.ConfigDict) -> Any:
    encoder_kwargs = cfg.get("encoder_kwargs", {})
    return get_encoder(cfg.encoder, **encoder_kwargs)


def _train_mps(fold: TrainingKFold, cfg: config_dict.ConfigDict) -> None:
    """Train MPS.

    Parameters
    ----------
    fold: TrainingKFold
        The training fold
    cfg: config_dict.ConfigDict
        Model configuration
    """
    cfg.n_sweeps = 5

    encoder = _get_encoder(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    save_path = get_save_path()

    key = jax.random.key(FLAGS.seed)

    for idx, (x, _) in enumerate(fold.split()):
        logging.info("Training MPS %d", idx + 1)

        key, subkey = jax.random.split(key)

        x_enc = encoder(x)

        mps = random_mps(n_sites=x.shape[1], bond_dim=2, random_key=subkey, d=cfg.d)
        train_mps(mps, x_enc, cfg=cfg)

        filename = f"mps_{FLAGS.dataset}_{idx}.pickle"
        save_mps(mps, save_path + filename)

        logging.info("Saved to %s%s", save_path, filename)


def _train_ttn(fold: TrainingKFold, cfg: config_dict.ConfigDict) -> None:
    """Train TTN.

    Parameters
    ----------
    fold: TrainingKFold
        The training fold
    cfg: config_dict.ConfigDict
        Model configuration
    """
    cfg.n_sweeps = 5

    encoder = _get_encoder(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    save_path = get_save_path()

    key = jax.random.key(FLAGS.seed)

    for idx, (x, _) in enumerate(fold.split()):
        logging.info("Training TTN %d", idx + 1)

        key, subkey = jax.random.split(key)

        x_enc = encoder(x)

        tree = random_tree(data_dim=x.shape[1], bond_dim=2, d=cfg.d, random_key=subkey)
        train_tree(tree, x_enc, cfg=cfg)

        filename = f"ttn_{FLAGS.dataset}_{idx}.pickle"
        save_ttn(tree, save_path + filename)

        logging.info("Saved to %s%s", save_path, filename)


def main(_):
    cfg = CONFIG.value
    cfg.unlock()

    fold = TrainingKFold(
        dataset=FLAGS.dataset,
        contamination=FLAGS.contamination,
        n_splits=FLAGS.folds,
        shuffle=True,
        scaler="minmax",
        seed=FLAGS.seed,
        include_test=False,
    )

    match FLAGS.model:
        case "mps":
            _train_mps(fold, cfg)
        case "ttn":
            _train_ttn(fold, cfg)
        case _:
            logging.fatal(
                "Model type %s not defined. This should not happen", FLAGS.model
            )


if __name__ == "__main__":
    app.run(main)

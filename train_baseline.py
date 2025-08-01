import os
import pickle
from datetime import datetime

import jax
import orbax.checkpoint
from absl import app, flags, logging
from flax.training import orbax_utils
from ml_collections import config_dict, config_flags

from tn.data.fold import TrainingKFold
from tn.ml.ae.train import train_ae
from tn.ml.simple import SVM, IsolationForest

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string("outdir", "./", short_name="o", help="Output directory")
flags.DEFINE_enum(
    "dataset",
    None,
    ["satellite", "ecg5000", "spambase"],
    short_name="d",
    help="Dataset",
)
flags.DEFINE_integer("seed", None, short_name="s", help="Random seed")
flags.DEFINE_integer("folds", 10, short_name="f", help="Number fo training folds")
flags.DEFINE_float("contamination", 0.05, short_name="c", help="Outlier contamination")
flags.DEFINE_enum(
    "model",
    default="svm",
    enum_values=["svm", "ifo", "ae"],
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


def _train_svm(fold: TrainingKFold, cfg: config_dict.ConfigDict) -> None:
    """Train SVM.

    Parameters
    ----------
    fold: TrainingKFold
        The training fold
    cfg: config_dict.ConfigDict
        Model configuration
    """
    cfg.nu = FLAGS.contamination
    cfg = config_dict.FrozenConfigDict(cfg)

    save_path = get_save_path()

    for idx, (x, _) in enumerate(fold.split()):
        logging.info("Training SVM %d", idx + 1)
        svm = SVM(cfg=cfg, n_jobs=-1)
        svm.fit(x)

        filename = f"svm_{FLAGS.dataset}_{idx}.pickle"

        with open(save_path + filename, "wb") as outfile:
            pickle.dump(svm, outfile)

            logging.info("Saved to %s%s", save_path, filename)


def _train_if(fold: TrainingKFold, cfg: config_dict.ConfigDict) -> None:
    """Train Isolation Forest.

    Parameters
    ----------
    fold: TrainingKFold
        The training fold
    cfg: config_dict.ConfigDict
        Model configuration
    """
    cfg = config_dict.FrozenConfigDict(cfg)

    save_path = get_save_path()

    for idx, (x, _) in enumerate(fold.split()):
        logging.info("Training IsolationForest %d", idx + 1)

        isof = IsolationForest(cfg=cfg, n_jobs=-1)
        isof.fit(x)

        filename = f"isof_{FLAGS.dataset}_{idx}.pickle"

        with open(save_path + filename, "wb") as outfile:
            pickle.dump(isof, outfile)

            logging.info("Saved to %s%s", save_path, filename)


def _train_ae(fold: TrainingKFold, cfg: config_dict.ConfigDict) -> None:
    """Train Autoencoder.

    Parameters
    ----------
    fold: TrainingKFold
        The training fold
    cfg: config_dict.ConfigDict
        Model configuration
    """
    cfg = config_dict.FrozenConfigDict(cfg)

    save_path = get_save_path(full=True)

    key = jax.random.key(FLAGS.seed)

    for idx, (x, _) in enumerate(fold.split()):
        logging.info("Training AE %d", idx + 1)
        key, subkey = jax.random.split(key)

        eval_metrics, state = train_ae(x, cfg=cfg, random_key=subkey)
        ckpt = {"model": state, "eval": eval_metrics}

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)

        ckpt_path = f"ae_{FLAGS.dataset}_{idx}"
        orbax_checkpointer.save(save_path + ckpt_path, ckpt, save_args=save_args)
        logging.info("Saved to %s%s", save_path, ckpt_path)


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
        case "svm":
            _train_svm(fold, cfg)
        case "ifo":
            _train_if(fold, cfg)
        case "ae":
            _train_ae(fold, cfg)
        case _:
            logging.fatal(
                "Model type %s not defined. This should not happen", FLAGS.model
            )


if __name__ == "__main__":
    app.run(main)

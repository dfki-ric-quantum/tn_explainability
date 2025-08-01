# ruff: noqa: E402

import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import orbax.checkpoint
from absl import app, flags, logging
from ml_collections import config_dict, config_flags
from sklearn.metrics import roc_auc_score

from config.ae import get_config as get_ae_config
from config.mps import get_config as get_mps_config
from config.ttn import get_config as get_ttn_config
from tn import stats as tnstats
from tn.data.fold import TrainingKFold
from tn.encoding import get_encoder
from tn.explain import SingleMarginalPDF, von_neumann_entropy
from tn.ml.ae.train import score_ae
from tn.mps.metrics import NLLFunctor as MPSLoss
from tn.mps.mps import load_mps
from tn.mps.rdm import conditional_reduced_density_matrix as mps_crdm
from tn.mps.rdm import reduced_density_matrix as mps_rdm
from tn.ttn.metrics import NLLFunctor as TTNLoss
from tn.ttn.rdm import reduced_density_matrix as ttn_rdm
from tn.ttn.tree import load_ttn
from tn.util import AUCROC



def get_fold(dataset: str, analysis_cfg: config_dict.FrozenConfigDict) -> TrainingKFold:
    """Get Training k-fold.

    Parameters
    ----------
    dataset: str
        The data set to load
    analysis_cfg: config_dict.FrozenConfigDict
        Analysis configuration

    Returns
    -------
    A TrainingKFold object for the data set
    """
    return TrainingKFold(
        dataset=dataset,
        contamination=analysis_cfg.cont,
        n_splits=analysis_cfg.n_folds,
        shuffle=True,
        scaler="minmax",
        seed=analysis_cfg.seed,
        include_test=True,
    )


def create_aucroc(
    auc_scores_train: np.ndarray,
    auc_scores_test: np.ndarray,
) -> AUCROC:
    """Create AUC ROC score statistics wrapper object.

    Parameters
    ----------
    auc_scores_train: np.ndarray
        Scores across all runs for the training set
    auc_scores_test: np.ndarray
        Scores across all runs for the hold-out set

    Returns
    -------
    An AUCROC wrapper object that contains mean and standard deviation for
    the training set and test set across all runs.
    """
    return AUCROC(
        train=(
            auc_scores_train.mean(),
            auc_scores_train.std(),
        ),
        test=(
            auc_scores_test.mean(),
            auc_scores_test.std(),
        ),
    )


def simple_auc_roc(fold: TrainingKFold, model_type: str, files: dict) -> AUCROC:
    """Compute AUC ROC scores for the one-class SVM and Isolation Forest models.

    Parameters
    ----------
    fold: TrainingKFold
        The data set split into k folds as used during training/evaluation
    model_type: str
        Either "svm" or "ifo", depending on the model type
    files: dict
        Paths to the model files

    Returns
    -------
    An AUCROC wrapper object holding the statistics
    """
    model_path = files[fold.dataset][model_type]

    auc_scores_train = np.empty(fold.n_splits, dtype=np.float32)
    auc_scores_test = np.empty(fold.n_splits, dtype=np.float32)

    for idx, (x_train, x_test, y_train, y_test) in enumerate(fold.split()):
        with open(model_path.format(str(idx)), "rb") as f:
            model = pickle.load(f)

        y_train_sc_labels = np.where(y_train == 0.0, 0.0, 1.0)
        y_test_sc_labels = np.where(y_test == 0.0, 0.0, 1.0)

        train_scores = model.score(x_train)
        test_scores = model.score(x_test)

        auc_scores_train[idx] = roc_auc_score(y_train_sc_labels, train_scores)
        auc_scores_test[idx] = roc_auc_score(y_test_sc_labels, test_scores)

    return create_aucroc(
        auc_scores_train,
        auc_scores_test,
    )


def ae_auc_roc(
    fold: TrainingKFold, cfg: config_dict.FrozenConfigDict, files: dict
) -> AUCROC:
    """Compute AUC ROC scores for the auto encoder

    Parameters
    ----------
    fold: TrainingKFold
        The data set split into k folds as used during training/evaluation
    cfg: config_dict.FrozenConfigDict
        The model/training configuration
    files: dict
        Paths to the model files

    Returns
    -------
    An AUCROC wrapper object holding the statistics
    """
    ae_path = files[fold.dataset]["ae"]

    auc_scores_train = np.empty(fold.n_splits, dtype=np.float32)
    auc_scores_test = np.empty(fold.n_splits, dtype=np.float32)

    for idx, (x_train, x_test, y_train, y_test) in enumerate(fold.split()):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(os.path.abspath(ae_path.format(str(idx))))
        params = restored["model"]["params"]

        y_train_sc_labels = np.where(y_train == 0.0, 0.0, 1.0)
        y_test_sc_labels = np.where(y_test == 0.0, 0.0, 1.0)

        train_scores = score_ae(params, x_train, cfg)
        test_scores = score_ae(params, x_test, cfg)

        auc_scores_train[idx] = roc_auc_score(y_train_sc_labels, train_scores)
        auc_scores_test[idx] = roc_auc_score(y_test_sc_labels, test_scores)

    return create_aucroc(
        auc_scores_train,
        auc_scores_test,
    )


def mps_auc_roc(
    fold: TrainingKFold, cfg: config_dict.FrozenConfigDict, files: dict
) -> AUCROC:
    """Compute AUC ROC scores for the matrix product state

    Parameters
    ----------
    fold: TrainingKFold
        The data set split into k folds as used during training/evaluation
    cfg: config_dict.FrozenConfigDict
        The model/training configuration
    files: dict
        Paths to the model files

    Returns
    -------
    An AUCROC wrapper object holding the statistics
    """
    mps_path = files[fold.dataset]["mps"]

    auc_scores_train = np.empty(fold.n_splits, dtype=np.float32)
    auc_scores_test = np.empty(fold.n_splits, dtype=np.float32)

    encoder = get_encoder(cfg.encoder, **cfg.encoder_kwargs)

    for idx, (x_train, x_test, y_train, y_test) in enumerate(fold.split()):
        x_train = encoder(x_train)
        x_test = encoder(x_test)

        mps = load_mps(mps_path.format(str(idx)))
        loss_fn = MPSLoss(mps.shapes, mps.d)

        y_train_sc_labels = np.where(y_train == 0.0, 0.0, 1.0)
        y_test_sc_labels = np.where(y_test == 0.0, 0.0, 1.0)

        train_scores = loss_fn(x_train, mps)
        test_scores = loss_fn(x_test, mps)

        auc_scores_train[idx] = roc_auc_score(y_train_sc_labels, train_scores)
        auc_scores_test[idx] = roc_auc_score(y_test_sc_labels, test_scores)

    return create_aucroc(
        auc_scores_train,
        auc_scores_test,
    )


def ttn_auc_roc(
    fold: TrainingKFold, cfg: config_dict.FrozenConfigDict, files: dict
) -> AUCROC:
    """Compute AUC ROC scores for the tree tensor network

    Parameters
    ----------
    fold: TrainingKFold
        The data set split into k folds as used during training/evaluation
    cfg: config_dict.FrozenConfigDict
        The model/training configuration
    files: dict
        Paths to the model files

    Returns
    -------
    An AUCROC wrapper object holding the statistics
    """
    ttn_path = files[fold.dataset]["ttn"]

    auc_scores_train = np.empty(fold.n_splits, dtype=np.float32)
    auc_scores_test = np.empty(fold.n_splits, dtype=np.float32)

    encoder = get_encoder(cfg.encoder, **cfg.encoder_kwargs)

    for idx, (x_train, x_test, y_train, y_test) in enumerate(fold.split()):
        x_train = encoder(x_train)
        x_test = encoder(x_test)

        ttn = load_ttn(ttn_path.format(str(idx)))
        loss_fn = TTNLoss(ttn.shapes, ttn.leaf_mask, ttn.d)

        y_train_sc_labels = np.where(y_train == 0.0, 0.0, 1.0)
        y_test_sc_labels = np.where(y_test == 0.0, 0.0, 1.0)

        train_scores = loss_fn(x_train, ttn)
        test_scores = loss_fn(x_test, ttn)

        auc_scores_train[idx] = roc_auc_score(y_train_sc_labels, train_scores)
        auc_scores_test[idx] = roc_auc_score(y_test_sc_labels, test_scores)

    return create_aucroc(
        auc_scores_train,
        auc_scores_test,
    )


def collect_roc(
    analysis_cfg: config_dict.FrozenConfigDict,
) -> dict[str, dict[str, AUCROC]]:
    """Collect AUC ROC score statistics over all trained models and data sets.

    Parameters
    ----------
    analysis_cfg: config_dict.FrozenConfigDict
        Analysis configuration

    Returns
    -------
    A dictionary of the struture:
    {
        "dataset1": {
            "model1": AUCROC,
            ...
        },
        ...
    }
    where AUCROC is a wrapper holding AUC ROC score statistics
    """
    datasets = ["ecg5000", "satellite", "spambase"]

    out: dict[str, dict[str, AUCROC]] = dict()

    for ds in datasets:
        fold = get_fold(ds, analysis_cfg)

        mps_cfg = config_dict.FrozenConfigDict(get_mps_config(ds))
        ttn_cfg = config_dict.FrozenConfigDict(get_ttn_config(ds))
        ae_cfg = config_dict.FrozenConfigDict(get_ae_config(ds))

        out[ds] = {
            "MPS": mps_auc_roc(fold, mps_cfg, analysis_cfg.files),
            "TTN": ttn_auc_roc(fold, ttn_cfg, analysis_cfg.files),
            "SVM": simple_auc_roc(fold, "svm", analysis_cfg.files),
            "IFO": simple_auc_roc(fold, "ifo", analysis_cfg.files),
            "AE": ae_auc_roc(fold, ae_cfg, analysis_cfg.files),
        }

    return out


def collect_ecg(
    analysis_cfg: config_dict.FrozenConfigDict,
) -> dict:
    """Collect data of tensor network model distributions for the ECG 5000 data set

    Parameters
    ----------
    analysis_cfg: config_dict.FrozenConfigDict
        Analysis configuration

    Returns
    -------
    A dictonary containing the expected value and expected standard deviation for
    the MPS and TTN model.
    """
    N_FEATURES = 140

    mps = load_mps(analysis_cfg.files["mps"])
    ttn = load_ttn(analysis_cfg.files["ttn"])

    mps_cfg = config_dict.FrozenConfigDict(get_mps_config("ecg5000"))
    ttn_cfg = config_dict.FrozenConfigDict(get_ttn_config("ecg5000"))

    mps_encoder = get_encoder(mps_cfg.encoder, **mps_cfg.encoder_kwargs)
    ttn_encoder = get_encoder(ttn_cfg.encoder, **ttn_cfg.encoder_kwargs)

    mps_rdms = [None] * mps.n_sites

    for site in reversed(range(mps.n_sites)):
        mps_rdms[site] = mps_rdm(mps, [site])

    ttn_rdms = [None] * N_FEATURES

    for fidx in reversed(range(N_FEATURES)):
        ttn_rdms[fidx] = ttn_rdm(ttn, [fidx])

    mps_marginals = [
        SingleMarginalPDF(rdm, encoder=mps_encoder, low=0.0, high=1.0)
        for rdm in mps_rdms
    ]
    mps_mu = np.array([mps_marginals[idx].expected_value for idx in range(mps.n_sites)])
    mps_std = np.sqrt(
        np.array([mps_marginals[idx].variance for idx in range(mps.n_sites)])
    )

    ttn_marginals = [
        SingleMarginalPDF(rdm, encoder=ttn_encoder, low=0.0, high=1.0)
        for rdm in ttn_rdms
    ]
    ttn_mu = np.array([ttn_marginals[idx].expected_value for idx in range(N_FEATURES)])
    ttn_std = np.sqrt(
        np.array([ttn_marginals[idx].variance for idx in range(N_FEATURES)])
    )

    return {
        "mps_mu": mps_mu,
        "mps_std": mps_std,
        "ttn_mu": ttn_mu,
        "ttn_std": ttn_std,
    }


def collect_sat(analysis_cfg: config_dict.FrozenConfigDict) -> dict:
    """Collect all-to-all feature correlation, estimated mutual information and mutual
    information assigned by the tensor network models for the satellite data set.

    Parameters
    ----------
    analysis_cfg: config_dict.FrozenConfigDict
        Analysis configuration

    Returns
    -------
    A dictonary containing the measures above.
    """
    fold = get_fold("satellite", analysis_cfg)

    x_train, _, y_train, _ = next(fold.split())
    n_features = x_train.shape[1]

    inlier_idxs = np.where(y_train == 0)
    outlier_idxs = np.where(y_train != 0)

    data_corr = tnstats.all_to_all_correlation(x_train)
    inlier_corr = tnstats.all_to_all_correlation(x_train[inlier_idxs])
    outlier_corr = tnstats.all_to_all_correlation(x_train[outlier_idxs])

    data_mi = tnstats.mutual_information(x_train, data_corr)[0]
    inlier_mi = tnstats.mutual_information(x_train[inlier_idxs], inlier_corr)[0]
    outlier_mi = tnstats.mutual_information(x_train[outlier_idxs], outlier_corr)[0]

    mps = load_mps(analysis_cfg.files["mps"])
    ttn = load_ttn(analysis_cfg.files["ttn"])

    mps_rdms = [None] * n_features
    ttn_rdms = [None] * n_features

    for fidx in reversed(range(n_features)):
        mps_rdms[fidx] = mps_rdm(mps, [fidx])
        ttn_rdms[fidx] = ttn_rdm(ttn, [fidx])

    mps_vnes = [von_neumann_entropy(rdm) for rdm in mps_rdms]
    ttn_vnes = [von_neumann_entropy(rdm) for rdm in ttn_rdms]

    mps_miq = np.empty((n_features, n_features), dtype=float)
    ttn_miq = np.empty((n_features, n_features), dtype=float)

    for i in range(n_features):
        mps_miq[i, i] = 0
        for j in range(i + 1, n_features):
            rdm_ij = mps_rdm(mps, [i, j])
            mps_miq[i, j] = mps_vnes[i] + mps_vnes[j] - von_neumann_entropy(rdm_ij)
            mps_miq[j, i] = mps_miq[i, j]

    for i in range(n_features):
        ttn_miq[i, i] = 0
        for j in range(i + 1, n_features):
            rdm_ij = ttn_rdm(ttn, [i, j])
            ttn_miq[i, j] = ttn_vnes[i] + ttn_vnes[j] - von_neumann_entropy(rdm_ij)
            ttn_miq[j, i] = ttn_miq[i, j]

    return {
        "all": {
            "corr": data_corr,
            "mi": data_mi,
        },
        "inlier": {
            "corr": inlier_corr,
            "mi": inlier_mi,
        },
        "outlier": {
            "corr": outlier_corr,
            "mi": outlier_mi,
        },
        "mps": {
            "vnes": mps_vnes,
            "mi": mps_miq,
        },
        "ttn": {
            "vnes": ttn_vnes,
            "mi": ttn_miq,
        },
    }


def collect_spam(analysis_cfg: config_dict.FrozenConfigDict) -> dict:
    """Collect data for partial reconstruction on the spambase data set.

    Parameters
    ----------
    analysis_cfg: config_dict.FrozenConfigDict
        Analysis configuration

    Returns
    -------
    A dictonary with the analysis results
    """
    fold = get_fold("spambase", analysis_cfg)

    _, x_test, _, y_test = next(fold.split())
    n_features = x_test.shape[1]

    mps_cfg = config_dict.FrozenConfigDict(get_mps_config("spambase"))
    mps_encoder = get_encoder(mps_cfg.encoder, **mps_cfg.encoder_kwargs)

    mps = load_mps(analysis_cfg.files["mps"])

    mps_rdms = [None] * n_features

    for fidx in reversed(range(n_features)):
        mps_rdms[fidx] = mps_rdm(mps, [fidx])

    mps_mds = [
        SingleMarginalPDF(rdm, encoder=mps_encoder, low=0.0, high=1.0)
        for rdm in mps_rdms
    ]

    mps_mu = np.array([mps_mds[idx].expected_value for idx in range(n_features)])
    mps_std = np.sqrt(np.array([mps_mds[idx].variance for idx in range(n_features)]))

    outlier = x_test[np.where(y_test != 0)][analysis_cfg.outlier_idx]
    corrected = outlier.copy()

    oenc = np.array(mps_encoder(np.array([outlier])))

    for idx in analysis_cfg.correction_idxs:
        crdm = mps_crdm(
            mps,
            sub_system=[idx],
            cond_vals=list(range(idx)),
            x=np.expand_dims(oenc[0, :idx, :], axis=0),
        )
        mds = SingleMarginalPDF(crdm, encoder=mps_encoder, low=0.0, high=1.0)
        val = mds.expected_value

        corrected[idx] = val
        oenc[0, idx, :] = mps_encoder.encode_single(val)

    return {
        "out_idx": analysis_cfg.outlier_idx,
        "cidxs": analysis_cfg.correction_idxs,
        "outlier": outlier,
        "corrected": corrected,
        "mps_mu": mps_mu,
        "mps_std": mps_std,
    }


FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string("outfile", None, short_name="o", help="Output file")


def main(_):
    tasks = {
        "roc": collect_roc,
        "ecg": collect_ecg,
        "sat": collect_sat,
        "spam": collect_spam,
    }

    analysis_cfg = CONFIG.value

    if not analysis_cfg:
        logging.error("Provide a configuration..., see --help for details")
        exit

    logging.info("Peforming analysis: %s", analysis_cfg.analysis)

    out = tasks[analysis_cfg.analysis](analysis_cfg)

    with open(FLAGS.outfile, "wb") as outfile:
        pickle.dump(out, outfile)
        logging.info("Results written to %s", FLAGS.outfile)


if __name__ == "__main__":
    app.run(main)

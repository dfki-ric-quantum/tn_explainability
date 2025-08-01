from ml_collections import config_dict


def get_config(analysis: str) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.analysis = analysis
    cfg.seed = 42
    cfg.n_folds = 10
    cfg.cont = 0.05

    match analysis:
        case "roc":
            cfg.files = {
                "ecg5000": {
                    "svm": "results/svm_ecg5000/svm_ecg5000_{}.pickle",
                    "ifo": "results/ifo_ecg5000/isof_ecg5000_{}.pickle",
                    "ae": "results/ae_ecg5000/ae_ecg5000_{}/",
                    "mps": "results/mps_ecg5000/mps_ecg5000_{}.pickle",
                    "ttn": "results/ttn_ecg5000/ttn_ecg5000_{}.pickle",
                },
                "satellite": {
                    "svm": "results/svm_satellite/svm_satellite_{}.pickle",
                    "ifo": "results/ifo_satellite/isof_satellite_{}.pickle",
                    "ae": "results/ae_satellite/ae_satellite_{}/",
                    "mps": "results/mps_satellite/mps_satellite_{}.pickle",
                    "ttn": "results/ttn_satellite/ttn_satellite_{}.pickle",
                },
                "spambase": {
                    "svm": "results/svm_spambase/svm_spambase_{}.pickle",
                    "ifo": "results/ifo_spambase/isof_spambase_{}.pickle",
                    "ae": "results/ae_spambase/ae_spambase_{}/",
                    "mps": "results/mps_spambase/mps_spambase_{}.pickle",
                    "ttn": "results/ttn_spambase/ttn_spambase_{}.pickle",
                },
            }

        case "ecg":
            cfg.files = {
                "mps": "results/mps_ecg5000/mps_ecg5000_0.pickle",
                "ttn": "results/ttn_ecg5000/ttn_ecg5000_0.pickle",
            }

        case "sat":
            cfg.files = {
                "mps": "results/mps_satellite/mps_satellite_0.pickle",
                "ttn": "results/ttn_satellite/ttn_satellite_0.pickle",
            }

        case "spam":
            cfg.files = {
                "mps": "results/mps_spambase/mps_spambase_0.pickle",
                "ttn": "results/ttn_spambase/ttn_spambase_0.pickle",
            }
            cfg.outlier_idx = 2 # Outlier sample to use
            cfg.correction_idxs = [3, 6, 22, 52, 53, 54, 55, 56] # Feature indices to correct

        case _:
            raise ValueError(f"No config for {analysis}")

    return cfg

from ml_collections import config_dict


def get_config(dataset: str) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    match dataset:
        case "ecg5000":
            cfg.n_estimators = 369
            cfg.max_samples = 0.1
        case "satellite":
            cfg.n_estimators = 157
            cfg.max_samples = 0.8
        case "spambase":
            cfg.n_estimators = 292
            cfg.max_samples = 0.1
    return cfg

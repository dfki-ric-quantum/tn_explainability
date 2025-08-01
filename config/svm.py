from ml_collections import config_dict


def get_config(dataset: str) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    match dataset:
        case "ecg5000":
            cfg.kernel = "rbf"
            cfg.gamma = 0.5
            cfg.n_components = 582
        case "satellite":
            cfg.kernel = "rbf"
            cfg.gamma = 4.8
            cfg.n_components = 120
        case "spambase":
            cfg.kernel = "rbf"
            cfg.gamma = 4.9
            cfg.n_components = 142

    return cfg

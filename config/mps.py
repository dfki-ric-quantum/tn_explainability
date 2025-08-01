from ml_collections import config_dict


def get_config(dataset: str) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    match dataset:
        case "ecg5000":
            cfg.bond_dim = 2
            cfg.max_bond = 30
            cfg.sigma_thresh = 0.05
            cfg.d = 4
            cfg.encoder = "legendre"
            cfg.encoder_kwargs = {"shifted": True, "p": [0, 1, 2, 3]}
            cfg.lrate = 0.001
            cfg.batch_size = 128
        case "satellite":
            cfg.bond_dim = 2
            cfg.max_bond = 20
            cfg.sigma_thresh = 0.05
            cfg.d = 5
            cfg.encoder = "legendre"
            cfg.encoder_kwargs = {"shifted": True, "p": [0, 1, 2, 3, 4]}
            cfg.lrate = 0.01
            cfg.batch_size = 128
        case "spambase":
            cfg.bond_dim = 2
            cfg.max_bond = 20
            cfg.sigma_thresh = 0.01
            cfg.d = 6
            cfg.encoder = "legendre"
            cfg.encoder_kwargs = {"shifted": True, "p": [0, 1, 2, 3, 4, 5]}
            cfg.lrate = 0.01
            cfg.batch_size = 64

    return cfg

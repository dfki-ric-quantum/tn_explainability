from ml_collections import config_dict


def get_config(dataset: str) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    match dataset:
        case "ecg5000":
            cfg.activation = "relu"
            cfg.batch_size = 64
            cfg.lrate = 0.002
            cfg.layers = [32, 16]
            cfg.n_latent = 42
            cfg.out_dim = 140
            cfg.n_epochs = 130
        case "satellite":
            cfg.activation = "relu"
            cfg.batch_size = 32
            cfg.lrate = 0.002
            cfg.layers = [256, 128]
            cfg.n_latent = 5
            cfg.out_dim = 36
            cfg.n_epochs = 20
        case "spambase":
            cfg.activation = "elu"
            cfg.batch_size = 128
            cfg.lrate = 0.002
            cfg.layers = [256, 128]
            cfg.n_latent = 22
            cfg.out_dim = 57
            cfg.n_epochs = 20
    return cfg

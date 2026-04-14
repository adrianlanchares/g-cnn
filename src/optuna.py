import optuna


def sample_basecnn_params(trial: optuna.Trial) -> dict:
    """Sample hyperparameters for the BaseCNN model."""
    params = {
        "num_filters": trial.suggest_categorical("num_filters", [16, 32, 64]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "num_conv_layers": trial.suggest_int("num_conv_layers", 2, 4),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }
    return params

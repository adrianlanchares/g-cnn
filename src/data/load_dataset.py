from omegaconf import DictConfig

from src.data.celeba import load_celeba_datasets
from src.data.chess import load_chess_tensor_dataset


def load_dataset(cfg: DictConfig):
    dataset_name = str(cfg.dataset_name).lower()

    if dataset_name in {"lichess/chess-position-evaluations", "chess"}:
        return load_chess_tensor_dataset(cfg)

    if dataset_name in {"celeba"}:
        return load_celeba_datasets(cfg)

    raise ValueError(
        "Unsupported dataset_name: "
        f"{cfg.dataset_name}. Supported values are chess and CelebA."
    )

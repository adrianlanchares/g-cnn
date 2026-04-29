from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.celeba import load_celeba_datasets
from src.data.chess import load_chess_tensor_dataset
from src.data.crc import load_crc_datasets


def load_dataset(
    cfg: DictConfig,
) -> tuple[Dataset, Dataset] | tuple[Dataset, Dataset, Dataset]:
    dataset_name: str = str(cfg.dataset_name).lower()

    if dataset_name in {"lichess/chess-position-evaluations", "chess"}:
        return load_chess_tensor_dataset(cfg)

    if dataset_name in {"celeba"}:
        return load_celeba_datasets(cfg)

    if dataset_name in {"crc", "nct-crc", "nct_crc"}:
        return load_crc_datasets(cfg)

    raise ValueError(
        "Unsupported dataset_name: "
        f"{cfg.dataset_name}. Supported values are chess, CelebA, and CRC."
    )

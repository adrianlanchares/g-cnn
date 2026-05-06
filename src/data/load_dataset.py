from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset
import torch

from src.data.celeba import load_celeba_datasets
from src.data.chess import load_chess_tensor_dataset
from src.data.crc import load_crc_datasets


LOADERS = {
    "chess": load_chess_tensor_dataset,
    "celeba": load_celeba_datasets,
    "crc": load_crc_datasets,
}


def load_dataset(
    cfg: DictConfig,
) -> tuple[Dataset, Dataset] | tuple[Dataset, Dataset, Dataset]:
    dataset_name: str = str(cfg.dataset_name).lower()

    if dataset_name not in LOADERS:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Supported datasets are: {list(LOADERS.keys())}."
        )
    
    loader_fn = LOADERS[dataset_name]
    datasets = loader_fn(cfg)

    data_fraction = float(getattr(cfg, "data_fraction", 1.0))
    if data_fraction >= 1.0:
        return datasets

    if data_fraction <= 0.0:
        raise ValueError("data_fraction must be in (0, 1].")

    seed = int(getattr(cfg, "data_fraction_seed", 21))
    generator = torch.Generator().manual_seed(seed)

    train_dataset = datasets[0]
    subset_len = max(1, int(len(train_dataset) * data_fraction))
    indices = torch.randperm(len(train_dataset), generator=generator)[:subset_len].tolist()
    train_subset = Subset(train_dataset, indices)

    return (train_subset, *datasets[1:])

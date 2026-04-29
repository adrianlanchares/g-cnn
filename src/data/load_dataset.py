from omegaconf import DictConfig
from torch.utils.data import Dataset

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
    return loader_fn(cfg)

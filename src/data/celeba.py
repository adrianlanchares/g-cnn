from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.data.dataset import CelebADataset


def _build_celeba_transform(cfg: DictConfig) -> transforms.Compose:
    transform_steps: list[object] = []

    if cfg.resize:
        transform_steps.append(transforms.Resize((cfg.image_size, cfg.image_size)))

    transform_steps.append(transforms.ToTensor())

    if cfg.normalize:
        transform_steps.append(
            transforms.Normalize(
                mean=list(cfg.normalize_mean),
                std=list(cfg.normalize_std),
            )
        )

    return transforms.Compose(transform_steps)


def _resolve_target_attr_index(
    base_dataset: datasets.CelebA, target_type: str, target_attribute: str | None
) -> int | None:
    if target_type != "attr" or target_attribute is None or target_attribute == "all":
        return None

    if target_attribute not in base_dataset.attr_names:
        raise ValueError(
            f"Unknown CelebA attribute '{target_attribute}'. "
            f"Available attributes: {base_dataset.attr_names}"
        )

    return base_dataset.attr_names.index(target_attribute)


def _build_raw_celeba_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = _build_celeba_transform(cfg)

    train_dataset = datasets.CelebA(
        root=str(data_dir),
        split="train",
        target_type=cfg.target_type,
        transform=transform,
        download=cfg.download,
    )
    valid_dataset = datasets.CelebA(
        root=str(data_dir),
        split="valid",
        target_type=cfg.target_type,
        transform=transform,
        download=cfg.download,
    )
    test_dataset = datasets.CelebA(
        root=str(data_dir),
        split="test",
        target_type=cfg.target_type,
        transform=transform,
        download=cfg.download,
    )

    return train_dataset, valid_dataset, test_dataset


def prepare_celeba_dataset(cfg: DictConfig) -> None:
    """Download CelebA (if needed) and validate config/attribute selection."""
    train_dataset, _, _ = _build_raw_celeba_datasets(cfg)
    _resolve_target_attr_index(train_dataset, cfg.target_type, cfg.target_attribute)


def load_celeba_datasets(
    cfg: DictConfig,
) -> (
    tuple[CelebADataset, CelebADataset]
    | tuple[CelebADataset, CelebADataset, CelebADataset]
):
    """Return train/test or train/valid/test torch datasets for CelebA."""
    train_base, valid_base, test_base = _build_raw_celeba_datasets(cfg)

    target_attr_idx = _resolve_target_attr_index(
        train_base, cfg.target_type, cfg.target_attribute
    )

    train_dataset = CelebADataset(train_base, target_attr_idx)
    test_dataset = CelebADataset(test_base, target_attr_idx)

    do_validation = bool(getattr(cfg, "do_validation", False))
    if do_validation:
        valid_dataset = CelebADataset(valid_base, target_attr_idx)
        print(
            "Loaded CelebA datasets with sizes "
            f"train={len(train_dataset):,}, valid={len(valid_dataset):,}, test={len(test_dataset):,}."
        )
        return train_dataset, valid_dataset, test_dataset

    print(
        "Loaded CelebA datasets with sizes "
        f"train={len(train_dataset):,}, test={len(test_dataset):,}."
    )
    return train_dataset, test_dataset

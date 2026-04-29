from __future__ import annotations

import zipfile
from pathlib import Path

import requests
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

FILES: dict[str, str] = {
    "NCT-CRC-HE-100K.zip": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1",
    "CRC-VAL-HE-7K.zip": "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1",
}

TRAIN_FOLDER = "NCT-CRC-HE-100K"
VAL_FOLDER = "CRC-VAL-HE-7K"


def _download_archive(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already exists, skipping: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(dest, "wb") as file_handle,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 20):
                file_handle.write(chunk)
                bar.update(len(chunk))


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(dest_dir)


def _build_crc_transform(cfg: DictConfig) -> transforms.Compose:
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


def prepare_crc_dataset(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = Path(getattr(cfg, "archive_dir", data_dir / "archives"))

    train_dir = data_dir / TRAIN_FOLDER
    val_dir = data_dir / VAL_FOLDER

    if train_dir.is_dir() and val_dir.is_dir():
        return

    if not cfg.download:
        missing = []
        if not train_dir.is_dir():
            missing.append(str(train_dir))
        if not val_dir.is_dir():
            missing.append(str(val_dir))
        raise FileNotFoundError(
            "CRC dataset not found and download disabled. Missing: "
            + ", ".join(missing)
        )

    for filename, url in FILES.items():
        archive_path = archive_dir / filename
        _download_archive(url, archive_path)
        _extract_archive(archive_path, data_dir)

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "CRC dataset extracted but expected folders not found: "
            f"{train_dir} and {val_dir}"
        )


def _build_crc_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    prepare_crc_dataset(cfg)

    data_dir = Path(cfg.data_dir)
    transform = _build_crc_transform(cfg)

    train_dataset = datasets.ImageFolder(
        root=str(data_dir / TRAIN_FOLDER), transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_dir / VAL_FOLDER), transform=transform
    )

    return train_dataset, val_dataset


def _split_validation_dataset(
    dataset: Dataset, validation_split: float, seed: int
) -> tuple[Dataset, Dataset]:
    if validation_split <= 0.0 or validation_split >= 1.0:
        raise ValueError(
            "validation_split must be in (0, 1) when do_validation is true."
        )

    total_len = len(dataset)
    valid_len = int(total_len * validation_split)
    test_len = total_len - valid_len

    generator = torch.Generator().manual_seed(seed)
    valid_subset, test_subset = random_split(
        dataset, [valid_len, test_len], generator=generator
    )
    return valid_subset, test_subset


def load_crc_datasets(
    cfg: DictConfig,
) -> tuple[Dataset, Dataset] | tuple[Dataset, Dataset, Dataset]:
    train_dataset, val_dataset = _build_crc_datasets(cfg)

    do_validation = bool(getattr(cfg, "do_validation", False))
    if do_validation:
        validation_split = float(getattr(cfg, "validation_split", 0.5))
        seed = int(getattr(cfg, "split_seed", 42))
        valid_dataset, test_dataset = _split_validation_dataset(
            val_dataset, validation_split, seed
        )
        print(
            "Loaded CRC datasets with sizes "
            f"train={len(train_dataset):,}, valid={len(valid_dataset):,}, test={len(test_dataset):,}."
        )
        return train_dataset, valid_dataset, test_dataset

    print(
        "Loaded CRC datasets with sizes "
        f"train={len(train_dataset):,}, test={len(val_dataset):,}."
    )
    return train_dataset, val_dataset

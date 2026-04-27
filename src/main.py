import hydra
from omegaconf import DictConfig

from src.data.load_dataset import load_dataset


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_splits = load_dataset(cfg.data)

    split_names = ["train", "valid", "test"] if len(dataset_splits) == 3 else ["train", "test"]
    for split_name, split_dataset in zip(split_names, dataset_splits):
        print(f"{split_name} dataset size: {len(split_dataset):,}")


if __name__ == "__main__":
    main()

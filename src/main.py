import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.load_dataset import load_dataset


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train_dataset, val_dataset, test_dataset = load_dataset(cfg.data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
        print(targets[0:100])

        break


if __name__ == "__main__":
    main()

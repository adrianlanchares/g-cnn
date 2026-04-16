import hydra
from omegaconf import DictConfig

from src.training.train_base_cnn import train_base_cnn

_TRAINERS = {
    "base_cnn": train_base_cnn,
}


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    mode: str = cfg.mode
    if mode not in _TRAINERS:
        raise ValueError(f"Unsupported mode: {mode}")

    trainer = _TRAINERS[mode]
    trainer(cfg)


if __name__ == "__main__":
    main()

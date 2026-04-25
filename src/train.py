import hydra
from omegaconf import DictConfig

from src.training import train_base_cnn, train_gecnn

_TRAINERS = {
    "base_cnn": train_base_cnn,
    "gecnn": train_gecnn,
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

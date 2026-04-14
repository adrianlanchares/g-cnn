import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.training.train_base_cnn import train_base_cnn

_TRAINERS = {
    "base_cnn": train_base_cnn,
}


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    model: str = HydraConfig.get().runtime.choices["mode"]
    if model not in _TRAINERS:
        raise ValueError(f"Unsupported model: {model}")

    trainer = _TRAINERS[model]
    trainer(cfg)


if __name__ == "__main__":
    main()

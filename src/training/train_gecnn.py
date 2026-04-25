from omegaconf import DictConfig

from src.training.train_base_cnn import train_base_cnn


def train_gecnn(cfg: DictConfig) -> None:
    """Currently has same functionality as base CNN"""

    train_base_cnn(cfg)

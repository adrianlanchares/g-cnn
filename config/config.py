from dataclasses import dataclass

from config.data import ChessDatasetConfig
from config.models import (
    ChessBaseCNNConfig,
    ChessGECNNConfig,
    MNISTBaseCNNConfig,
    MNISTGECNNConfig,
)
from config.paths import PathConfig


@dataclass
class Config:
    path_config: PathConfig = PathConfig()

    chess_base_cnn_config: ChessBaseCNNConfig = ChessBaseCNNConfig()
    mnist_base_cnn_config: MNISTBaseCNNConfig = MNISTBaseCNNConfig()

    chess_gecnn_config: ChessGECNNConfig = ChessGECNNConfig()
    mnist_gecnn_config: MNISTGECNNConfig = MNISTGECNNConfig()

    data_config: ChessDatasetConfig = ChessDatasetConfig()


cfg = Config()

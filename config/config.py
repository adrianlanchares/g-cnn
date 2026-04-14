from dataclasses import dataclass, field

from config.data import ChessDatasetConfig
from config.models import (
    ChessBaseCNNConfig,
    ChessGECNNConfig,
    MNISTBaseCNNConfig,
    MNISTGECNNConfig,
)


@dataclass
class Config:
    chess_base_cnn_config: ChessBaseCNNConfig = field(
        default_factory=ChessBaseCNNConfig
    )
    mnist_base_cnn_config: MNISTBaseCNNConfig = field(
        default_factory=MNISTBaseCNNConfig
    )

    chess_gecnn_config: ChessGECNNConfig = field(default_factory=ChessGECNNConfig)
    mnist_gecnn_config: MNISTGECNNConfig = field(default_factory=MNISTGECNNConfig)

    chess_data_config: ChessDatasetConfig = field(default_factory=ChessDatasetConfig)

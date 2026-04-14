from dataclasses import dataclass, field

from config.data import ChessDataConfig
from config.models import (
    BaseCNNConfig,
    ChessBaseCNNConfig,
    ChessGECNNConfig,
    GECNNConfig,
    MNISTBaseCNNConfig,
    MNISTGECNNConfig,
)
from config.train import ChessTrainConfig, MNISTTrainConfig, TrainConfig


@dataclass
class Config:
    base_cnn_config: BaseCNNConfig = field(default_factory=BaseCNNConfig)
    gecnn_config: GECNNConfig = field(default_factory=GECNNConfig)

    data_config: ChessDataConfig = field(default_factory=ChessDataConfig)

    train_config: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class ChessConfig:
    base_cnn_config: ChessBaseCNNConfig = field(default_factory=ChessBaseCNNConfig)
    gecnn_config: ChessGECNNConfig = field(default_factory=ChessGECNNConfig)

    data_config: ChessDataConfig = field(default_factory=ChessDataConfig)

    train_config: ChessTrainConfig = field(default_factory=ChessTrainConfig)


@dataclass
class MNISTConfig:
    base_cnn_config: MNISTBaseCNNConfig = field(default_factory=MNISTBaseCNNConfig)
    gecnn_config: MNISTGECNNConfig = field(default_factory=MNISTGECNNConfig)

    data_config: None = None  # TODO: add MNIST data config

    train_config: MNISTTrainConfig = field(default_factory=MNISTTrainConfig)

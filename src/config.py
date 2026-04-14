from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class BaseCNNConfig:
    in_channels: int = 12
    out_channels: int = 64

    hidden_channels: tuple[int] = (64, 128, 256)
    kernel_sizes: tuple[int] = (3, 3, 3)

    activation: torch.nn.Module = field(default_factory=torch.nn.ReLU)

    use_pooling: bool = True
    pool_kernel_size: int = 2
    pool_stride: int = 2

    linear_out_features: int = 1


@dataclass
class ChessBaseCNNConfig(BaseCNNConfig):
    # Override specific fields for the chess dataset

    use_pooling: bool = False  # Disable pooling for chess dataset


@dataclass
class MNISTBaseCNNConfig(BaseCNNConfig):
    # Override specific fields for the MNIST dataset

    in_channels: int = 1  # MNIST images are grayscale


@dataclass
class Config:
    base_cnn_config: BaseCNNConfig = BaseCNNConfig()
    chess_base_cnn_config: ChessBaseCNNConfig = ChessBaseCNNConfig()
    mnist_base_cnn_config: MNISTBaseCNNConfig = MNISTBaseCNNConfig()

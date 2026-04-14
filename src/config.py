from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"


############ BASE CNN ##############################


@dataclass
class BaseCNNConfig:
    in_channels: int = 12
    out_channels: int = 64

    hidden_channels: tuple[int] = (64, 128, 256)
    kernel_sizes: tuple[int] = (3, 3, 3)

    use_pooling: bool = True
    pool_kernel_size: int = 2
    pool_stride: int = 2

    linear_out_features: int = 1


@dataclass
class ChessBaseCNNConfig(BaseCNNConfig):
    # Override specific fields for the chess dataset

    use_pooling: bool = False


@dataclass
class MNISTBaseCNNConfig(BaseCNNConfig):
    # Override specific fields for the MNIST dataset

    in_channels: int = 1
    linear_out_features: int = 10


############# GECNN ##############################


@dataclass
class GECNNConfig:
    in_channels: int = 12
    out_channels: int = 64

    hidden_channels: tuple = (32, 64, 128)  # group size multiplies these internally
    kernel_sizes: tuple = (3, 3, 3)

    use_pooling: bool = True
    pool_kernel_size: int = 2
    pool_stride: int = 2

    linear_out_features: int = 1

    group: str = "Z2"  # "Z2" or "p4m"


@dataclass
class ChessGECNNConfig(GECNNConfig):
    in_channels: int = 12
    use_pooling: bool = False
    linear_out_features: int = 1
    group: str = "Z2"


@dataclass
class MNISTGECNNConfig(GECNNConfig):
    in_channels: int = 1
    use_pooling: bool = True
    linear_out_features: int = 10
    group: str = "p4m"


############### MAIN CONFIG ##############################


@dataclass
class Config:
    path_config: PathConfig = PathConfig()

    chess_base_cnn_config: ChessBaseCNNConfig = ChessBaseCNNConfig()
    mnist_base_cnn_config: MNISTBaseCNNConfig = MNISTBaseCNNConfig()

    chess_gecnn_config: ChessGECNNConfig = ChessGECNNConfig()
    mnist_gecnn_config: MNISTGECNNConfig = MNISTGECNNConfig()

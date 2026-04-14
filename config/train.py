from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TrainConfig:
    device: str = "mps"
    num_epochs: int = 100
    batch_size: int = 64

    lr: float = 1e-3
    loss_fn: str = "MSELoss"

    save_checkpoint_every: int = 5
    checkpoint_path: Path = PROJECT_ROOT / "models" / "chess"

    tensorboard_log_dir: Path = PROJECT_ROOT / "logs" / "chess"
    logs_per_epoch: int = 1000


@dataclass
class ChessTrainConfig(TrainConfig):
    checkpoint_path: Path = PROJECT_ROOT / "models" / "chess"
    tensorboard_log_dir: Path = PROJECT_ROOT / "logs" / "chess"

    loss_fn: str = "MSELoss"


@dataclass
class MNISTTrainConfig(TrainConfig):
    checkpoint_path: Path = PROJECT_ROOT / "models" / "mnist"
    tensorboard_log_dir: Path = PROJECT_ROOT / "logs" / "mnist"

    loss_fn: str = "CrossEntropyLoss"

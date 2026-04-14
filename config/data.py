from pathlib import Path

from dataclassess import dataclass

from config.paths import path_cfg


@dataclass
class ChessDatasetConfig:
    dataset_name: str = "Lichess/chess-position-evaluations"

    own_data_dir = path_cfg.data_dir / "chess"
    output_csv: Path = own_data_dir / "lichess_eval_subset.csv"

    hash_mod: int = 10_000
    hash_keep_below: int = 15  # keeps about 0.05% of rows
    max_rows: int = 1_000_000  # stop once enough rows are written

    clip_cp: int = 1000  # clip non-mate centipawn scores to [-1000, 1000]
    mate_value: int = 2000  # map certain mate to +/-2000

    dedup_fen: bool = True


@dataclass
class DataConfig:
    chess: ChessDatasetConfig = ChessDatasetConfig()


data_cfg = DataConfig()

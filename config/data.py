from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ChessDataConfig:
    dataset_name: str = "Lichess/chess-position-evaluations"

    own_data_dir = PROJECT_ROOT / "data" / "chess"
    output_csv: Path = own_data_dir / "lichess_eval_subset.csv"
    output_tensor_dataset: Path = own_data_dir / "lichess_eval_subset.pt"

    keep_all_fens: bool = (
        True  # if False, keep only a deterministic subset based on hashing
        # if true, keep all rows until max_rows is reached (faster download, but keep the first max_rows rows which may be less diverse)
    )

    hash_mod: int = 10_000_000_000
    hash_keep_below: int = 1_000_000_000
    max_rows: int = 1_000_000  # stop once enough rows are written

    clip_cp: int = 1000  # clip non-mate centipawn scores to [-1000, 1000]
    mate_value: int = 2000  # map certain mate to +/-2000

    dedup_fen: bool = True

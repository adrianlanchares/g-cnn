import csv
import hashlib
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import DictConfig

from src.data.dataset import ChessPositionEvaluationDataset


def stable_hash_int(text: str) -> int:
    """Stable integer hash from a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def keep_fen(fen: str, hash_mod: int, hash_keep_below: int) -> bool:
    """Deterministic subsampling rule."""
    return stable_hash_int(fen) % hash_mod < hash_keep_below


def cp_from_row(row: dict, clip_cp: int, mate_value: int) -> int:
    """
    Convert dataset row to a single numeric evaluation target.
    - If cp exists: clip it.
    - If mate exists: map to +/-MATE_VALUE.
    """
    cp = row["cp"]
    mate = row["mate"]

    if cp is not None:
        cp = int(cp)
        if cp > clip_cp:
            cp = clip_cp
        elif cp < -clip_cp:
            cp = -clip_cp
        return cp

    if mate is not None:
        mate = int(mate)
        if mate > 0:
            return mate_value
        elif mate < 0:
            return -mate_value
        return 0  # defensive fallback, should be rare/unexpected

    raise ValueError("Row has both cp=None and mate=None")


def get_fen_eval_csv(cfg: DictConfig):
    output_csv = Path(cfg.output_csv)

    if output_csv.is_file():
        print(f"Output already exists at {output_csv}. Skipping download.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Streaming mode avoids downloading the whole dataset first.
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)

    seen = set()
    written = 0
    processed = 0

    fieldnames = ["fen", "eval_cp"]

    errors = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in ds:
            processed += 1
            fen = row["fen"]

            if not cfg.keep_all_fens and not keep_fen(
                fen, cfg.hash_mod, cfg.hash_keep_below
            ):
                continue

            # Optional deduplication by FEN
            if cfg.dedup_fen:
                if fen in seen:
                    continue
                seen.add(fen)

            try:
                eval_cp = cp_from_row(row, cfg.clip_cp, cfg.mate_value)
            except ValueError:
                errors += 1
                continue

            out = {
                "fen": fen,
                "eval_cp": eval_cp,
            }

            writer.writerow(out)
            written += 1

            print(f"written={written:,} processed={processed:,}\t\t\t\t", end="\r")

            if written >= cfg.max_rows:
                break

    print(f"Done. Wrote {written:,} rows to {output_csv}, with {errors} errors.")
    return


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN board into a one-hot tensor with shape [12, 8, 8]."""
    piece_to_channel = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    board = fen.split(" ", 1)[0]
    rows = board.split("/")
    if len(rows) != 8:
        raise ValueError(f"Invalid FEN rows: {fen}")

    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    for rank, row in enumerate(rows):
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char)
                continue

            channel = piece_to_channel.get(char)
            if channel is None:
                raise ValueError(f"Invalid piece '{char}' in FEN: {fen}")

            if file_idx > 7:
                raise ValueError(f"Invalid file index in FEN: {fen}")

            tensor[channel, rank, file_idx] = 1.0
            file_idx += 1

        if file_idx != 8:
            raise ValueError(f"Invalid rank width in FEN: {fen}")

    return tensor


def csv_to_position_eval_tensors(
    csv_path: str, max_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read FEN/eval CSV and return tensors: positions [N,12,8,8], evals [N]."""
    print(f"Loading CSV from {csv_path}...")
    dataset = pd.read_csv(csv_path, usecols=["fen", "eval_cp"])
    print(f"Loaded {len(dataset):,} rows from CSV.")
    if dataset.empty:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    num_rows = len(dataset)
    limit = min(num_rows, max_rows)

    positions = torch.empty((limit, 12, 8, 8), dtype=torch.float32)
    fens = dataset["fen"].to_numpy()

    for i, fen in enumerate(fens):
        if i >= limit:
            break
        positions[i].copy_(fen_to_tensor(fen))
        print(f"Processed {i + 1:,}/{limit:,} rows\t\t\t\t", end="\r")

    evals = torch.from_numpy(dataset["eval_cp"].to_numpy(dtype="float32", copy=False))[
        :limit
    ]

    # Squash evals to [-1, 1] range for better training stability
    evals = torch.sigmoid(evals / 400) * 2 - 1

    print(f"Done. Processed {limit:,} rows.")
    return positions, evals


def build_and_save_tensor_dataset(
    cfg: DictConfig,
):
    """Build tensors from the exported CSV and save them to disk."""
    output_path = Path(cfg.output_tensor_dataset)
    if output_path.is_file():
        print(f"Output already exists at {output_path}. Skipping processing.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions, evaluations = csv_to_position_eval_tensors(
        str(Path(cfg.output_csv)), cfg.max_rows
    )

    dataset = {
        "positions": positions,
        "evaluations": evaluations,
    }
    torch.save(dataset, output_path)
    print(
        f"Saved tensor dataset to {output_path} with positions shape "
        f"{tuple(positions.shape)} and evaluations shape {tuple(evaluations.shape)}."
    )
    return


def prepare_chess_dataset(cfg: DictConfig):
    """Full pipeline: export CSV from raw dataset, then build tensor dataset."""
    get_fen_eval_csv(cfg)
    build_and_save_tensor_dataset(cfg)

    return


def load_chess_tensor_dataset(cfg: DictConfig) -> ChessPositionEvaluationDataset:
    """Load the processed tensor dataset from disk and return a PyTorch Dataset."""
    dataset_path = Path(cfg.output_tensor_dataset)
    if not dataset_path.is_file():
        prepare_chess_dataset(cfg)

    data = torch.load(dataset_path)
    positions = data["positions"]
    evaluations = data["evaluations"]

    if len(positions) != len(evaluations):
        min_len = min(len(positions), len(evaluations))
        print(
            f"Warning: tensor dataset length mismatch detected at {dataset_path}. "
            f"Truncating to {min_len:,} samples."
        )
        positions = positions[:min_len]
        evaluations = evaluations[:min_len]

    train_split = cfg.train_split
    train_end = int(len(positions) * train_split)

    train_positions = positions[:train_end]
    train_evals = evaluations[:train_end]

    test_positions = positions[train_end:]
    test_evals = evaluations[train_end:]

    train_dataset = ChessPositionEvaluationDataset(train_positions, train_evals)
    test_dataset = ChessPositionEvaluationDataset(test_positions, test_evals)

    print(
        f"Loaded tensor dataset from {dataset_path} with positions shape "
        f"{tuple(positions.shape)} and evaluations shape {tuple(evaluations.shape)}."
    )
    return train_dataset, test_dataset

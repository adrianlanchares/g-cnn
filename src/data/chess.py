import csv
import hashlib
import os

import torch
from datasets import load_dataset

from config.data import ChessDataConfig
from src.data.dataset import ChessPositionEvaluationDataset


def stable_hash_int(text: str) -> int:
    """Stable integer hash from a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def keep_fen(cfg: ChessDataConfig, fen: str) -> bool:
    """Deterministic subsampling rule."""
    return stable_hash_int(fen) % cfg.hash_mod < cfg.hash_keep_below


def cp_from_row(row: dict, cfg: ChessDataConfig) -> int:
    """
    Convert dataset row to a single numeric evaluation target.
    - If cp exists: clip it.
    - If mate exists: map to +/-MATE_VALUE.
    """
    cp = row["cp"]
    mate = row["mate"]

    if cp is not None:
        cp = int(cp)
        if cp > cfg.clip_cp:
            cp = cfg.clip_cp
        elif cp < -cfg.clip_cp:
            cp = -cfg.clip_cp
        return cp

    if mate is not None:
        mate = int(mate)
        if mate > 0:
            return cfg.mate_value
        elif mate < 0:
            return -cfg.mate_value
        return 0  # defensive fallback, should be rare/unexpected

    raise ValueError("Row has both cp=None and mate=None")


def get_fen_eval_csv(cfg: ChessDataConfig = ChessDataConfig()):
    if os.path.isfile(cfg.output_csv):
        print(f"Output already exists at {cfg.output_csv}. Skipping download.")
        return

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Streaming mode avoids downloading the whole dataset first.
    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)

    seen = set()
    written = 0
    processed = 0

    fieldnames = ["fen", "eval_cp"]

    errors = 0

    with open(cfg.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in ds:
            processed += 1
            fen = row["fen"]

            if not cfg.keep_all_fens and not keep_fen(cfg, fen):
                continue

            # Optional deduplication by FEN
            if cfg.dedup_fen:
                if fen in seen:
                    continue
                seen.add(fen)

            try:
                eval_cp = cp_from_row(row, cfg)
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

    print(f"Done. Wrote {written:,} rows to {cfg.output_csv}, with {errors} errors.")
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


def csv_to_position_eval_tensors(csv_path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read FEN/eval CSV and return tensors: positions [N,12,8,8], evals [N]."""
    position_tensors = []
    evaluations = []
    processed = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row["fen"]
            eval_cp = float(row["eval_cp"])

            position_tensors.append(fen_to_tensor(fen))
            evaluations.append(eval_cp)
            processed += 1

            print(f"processed={processed:,}\t\t\t", end="\r")

    if not position_tensors:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    positions = torch.stack(position_tensors, dim=0)
    evals = torch.tensor(evaluations, dtype=torch.float32)
    print(f"Done. Processed {processed:,} rows.")
    return positions, evals


def build_and_save_tensor_dataset(
    cfg: ChessDataConfig = ChessDataConfig(),
):
    """Build tensors from the exported CSV and save them to disk."""
    output_path = cfg.output_tensor_dataset
    if os.path.isfile(output_path):
        print(f"Output already exists at {output_path}. Skipping processing.")
        return

    cfg.output_tensor_dataset.parent.mkdir(parents=True, exist_ok=True)

    positions, evaluations = csv_to_position_eval_tensors(cfg.output_csv)

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


def prepare_chess_dataset(cfg: ChessDataConfig = ChessDataConfig()):
    """Full pipeline: export CSV from raw dataset, then build tensor dataset."""
    get_fen_eval_csv(cfg)
    build_and_save_tensor_dataset(cfg)

    return


def load_chess_tensor_dataset(
    cfg: ChessDataConfig = ChessDataConfig(),
) -> ChessPositionEvaluationDataset:
    """Load the processed tensor dataset from disk and return a PyTorch Dataset."""
    dataset_path = cfg.output_tensor_dataset
    if not os.path.isfile(dataset_path):
        prepare_chess_dataset(cfg)

    data = torch.load(dataset_path)
    positions = data["positions"]
    evaluations = data["evaluations"]

    print(
        f"Loaded tensor dataset from {dataset_path} with positions shape "
        f"{tuple(positions.shape)} and evaluations shape {tuple(evaluations.shape)}."
    )
    return ChessPositionEvaluationDataset(positions, evaluations)

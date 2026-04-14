import csv
import hashlib

from datasets import load_dataset

from config.data import data_cfg


def stable_hash_int(text: str) -> int:
    """Stable integer hash from a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def keep_fen(fen: str) -> bool:
    """Deterministic subsampling rule."""
    return (
        stable_hash_int(fen) % data_cfg.chess.hash_mod < data_cfg.chess.hash_keep_below
    )


def cp_from_row(row: dict) -> int:
    """
    Convert dataset row to a single numeric evaluation target.
    - If cp exists: clip it.
    - If mate exists: map to +/-MATE_VALUE.
    """
    cp = row["cp"]
    mate = row["mate"]

    if cp is not None:
        cp = int(cp)
        if cp > data_cfg.chess.clip_cp:
            cp = data_cfg.chess.clip_cp
        elif cp < -data_cfg.chess.clip_cp:
            cp = -data_cfg.chess.clip_cp
        return cp

    if mate is not None:
        mate = int(mate)
        if mate > 0:
            return data_cfg.chess.mate_value
        elif mate < 0:
            return -data_cfg.chess.mate_value
        return 0  # defensive fallback, should be rare/unexpected

    raise ValueError("Row has both cp=None and mate=None")


def get_fen_eval_csv():
    # Streaming mode avoids downloading the whole dataset first.
    ds = load_dataset(data_cfg.chess.dataset_name, split="train", streaming=True)

    seen = set()
    written = 0
    processed = 0

    fieldnames = ["fen", "eval_cp"]

    with open(data_cfg.chess.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in ds:
            processed += 1
            fen = row["fen"]

            # Deterministic subset
            if not keep_fen(fen):
                continue

            # Optional deduplication by FEN
            if data_cfg.chess.dedup_fen:
                if fen in seen:
                    continue
                seen.add(fen)

            try:
                eval_cp = cp_from_row(row)
            except ValueError:
                continue

            out = {
                "fen": fen,
                "eval_cp": eval_cp,
            }

            writer.writerow(out)
            written += 1

            if written % 10000 == 0:
                print(f"written={written:,} processed={processed:,}")

            if written >= data_cfg.chess.max_rows:
                break

    print(f"Done. Wrote {written:,} rows to {data_cfg.chess.output_csv}")


if __name__ == "__main__":
    get_fen_eval_csv()

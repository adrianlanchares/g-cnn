from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_run_name(name: str) -> tuple[str, str, float] | None:
    parts = name.split("_")

    if parts[0] == "gecnn":
        mode = parts[0]

        aug_label = parts[1]
        if parts[2] != "frac":
            return None

        fraction_str = parts[3].replace("p", ".")
        try:
            fraction = float(fraction_str)
        except ValueError:
            return None

        return mode, aug_label, fraction

    else:
        mode = f"{parts[0]}_{parts[1]}"

        aug_label = parts[2]
        if parts[3] != "frac":
            return None

        fraction_str = parts[4].replace("p", ".")
        try:
            fraction = float(fraction_str)
        except ValueError:
            return None

        return mode, aug_label, fraction


def _label_for_run(mode: str, aug_label: str) -> str | None:
    if mode == "base_cnn" and aug_label == "noaug":
        return "CNN"
    if mode == "base_cnn" and aug_label == "aug":
        return "CNN-Aug"
    if mode == "gecnn" and aug_label == "noaug":
        return "G-CNN"
    return None


def _load_metrics(metrics_path: Path) -> tuple[float | None, float | None]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    base_metrics = metrics.get("base", {})
    base_prf = metrics.get("prf", {})
    accuracy = base_metrics.get("accuracy")
    f1_macro = base_prf.get("f1_macro")
    return accuracy, f1_macro


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accuracy and macro-F1 vs training fraction."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/final"),
        help="Path containing experiment run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/final/plots.png"),
        help="Output plot image path.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Missing root directory: {root}")

    series: dict[str, list[tuple[float, float | None, float | None]]] = {
        "CNN": [],
        "CNN-Aug": [],
        "G-CNN": [],
    }

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        parsed = _parse_run_name(run_dir.name)
        if parsed is None:
            continue

        mode, aug_label, fraction = parsed
        label = _label_for_run(mode, aug_label)
        if label is None:
            continue

        metrics_path = run_dir / "metrics" / "metrics.json"
        if not metrics_path.is_file():
            continue

        accuracy, f1_macro = _load_metrics(metrics_path)
        series[label].append((fraction, accuracy, f1_macro))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    metric_info = [
        ("Accuracy", 1),
        ("Macro-F1", 2),
    ]

    for label, points in series.items():
        if not points:
            continue
        points_sorted = sorted(points, key=lambda item: item[0])
        fractions = [item[0] for item in points_sorted]
        accuracies = [item[1] for item in points_sorted]
        f1s = [item[2] for item in points_sorted]

        axes[0].plot(fractions, accuracies, marker="o", label=label)
        axes[1].plot(fractions, f1s, marker="o", label=label)

    axes[0].set_title("Accuracy vs Training Fraction")
    axes[1].set_title("Macro-F1 vs Training Fraction")
    for ax in axes:
        ax.set_xlabel("Training Fraction")
        ax.set_ylabel("Metric")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

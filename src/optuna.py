import argparse
import math
import shutil
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import optuna

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "optuna"


def _sample_basecnn_params(trial: optuna.Trial) -> dict:
    """Sample hyperparameters for the BaseCNN model."""

    # Model architecture parameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 5)
    hidden_channels = []
    kernel_sizes = []
    strides = []
    for i in range(num_conv_layers):
        hidden_channels.append(
            trial.suggest_int(f"hidden_channels_{i}", 16, 256, log=True)
        )
        kernel_sizes.append(trial.suggest_int(f"kernel_size_{i}", 3, 7, step=2))
        strides.append(trial.suggest_int(f"stride_{i}", 1, 2))

    return {
        "model.hidden_channels": hidden_channels,
        "model.kernel_sizes": kernel_sizes,
        "model.strides": strides,
        "train.lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "train.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def sample_params(mode: str, trial: optuna.Trial) -> dict:
    """Sample hyperparameters based on the specified mode."""
    samplers = {
        "base_cnn": _sample_basecnn_params,
        # Future modes can be added here
    }

    if mode not in samplers:
        raise ValueError(
            f"Unsupported mode '{mode}'. Available modes: {list(samplers.keys())}"
        )
    return samplers[mode](trial)


def _read_best_metric(log_dir: Path) -> float:
    if not log_dir.exists():
        return float("nan")

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])

    if "eval/avg_loss" in tags:
        values = [event.value for event in accumulator.Scalars("eval/avg_loss")]
        if values:
            return float(min(values))
    return float("nan")


def _build_trial_command(
    args: argparse.Namespace, trial: optuna.Trial, trial_dir: Path
) -> str:
    """Build a command string to run a training trial with the given parameters."""

    sampled = sample_params(args.mode, trial)

    cmd: list[str] = [
        "python",
        "-m",
        "src.train",
        f"problem={args.problem}",
        f"mode={args.mode}",
        f"seed={args.seed + trial.number}",
        f"hydra.run.dir={trial_dir.as_posix()}",
    ]

    for key, value in sampled.items():
        if value is None:
            continue

        if isinstance(value, float):
            if math.isfinite(value):
                cmd.append(f"{key}={value:.12g}")
            else:
                cmd.append(f"{key}={value}")
        else:
            cmd.append(f"{key}={value}")

    return cmd


def _objective(args: argparse.Namespace, output_root: Path):
    def objective(trial: optuna.Trial) -> float:
        study_prefix = args.study_name if args.study_name else args.algorithm
        trial_dir = output_root / study_prefix / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_trial_command(args, trial, trial_dir)
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

        (trial_dir / "stdout.log").write_text(process.stdout or "", encoding="utf-8")
        (trial_dir / "stderr.log").write_text(process.stderr or "", encoding="utf-8")

        if process.returncode != 0:
            raise optuna.exceptions.TrialPruned(
                f"Training failed with exit code {process.returncode}."
            )

        score = _read_best_metric(trial_dir / "logs")
        if not math.isfinite(score):
            raise optuna.exceptions.TrialPruned(
                "No valid score found in TensorBoard logs."
            )

        trial.report(score, step=0)
        return score

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="tpe",
        help="Optuna sampling algorithm to use (e.g., 'tpe')",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional name for the Optuna study (used in output directory naming)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="chess",
        help="Problem to optimize (e.g., 'chess')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="base_cnn",
        help="Mode for sampling hyperparameters (e.g., 'base_cnn')",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///study.db",
        help="Database storage URL for Optuna study",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=OUTPUT_ROOT,
        help="Root directory for Optuna trial outputs",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials to run (overrides hardcoded value in main())",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="Number of startup trials for Optuna optimization (overrides hardcoded value in main())",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.n_startup_trials
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials)

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )

    objective = _objective(args, output_root)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
        )
    except KeyboardInterrupt:
        pass

    if study.best_trial is not None:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best score: {study.best_trial.value:.4f}")
        print("Best params:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

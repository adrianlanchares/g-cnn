from __future__ import annotations

from pathlib import Path
from typing import Callable

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from src.training import train_base_cnn, train_gecnn

_TRAINERS: dict[str, Callable[[DictConfig], None]] = {
    "base_cnn": train_base_cnn,
    "gecnn": train_gecnn,
}


def _run_training(cfg: DictConfig) -> None:
    mode: str = cfg.mode
    if mode not in _TRAINERS:
        raise ValueError(f"Unsupported mode: {mode}")
    trainer = _TRAINERS[mode]
    trainer(cfg)


def main() -> None:
    seeds = [21, 42, 87]
    data_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    use_augmentations = [False, True]
    modes = ["base_cnn", "gecnn"]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_root = Path(__file__).resolve().parents[1] / "config"
    hydra.initialize(version_base=None, config_path=str(config_root))

    for mode in modes:
        for use_augmentation in use_augmentations:
            for data_fraction in data_fractions:
                for seed in seeds:
                    aug_label = "aug" if use_augmentation else "noaug"
                    fraction_label = f"frac_{data_fraction:.2f}".replace(".", "p")
                    run_name = f"{mode}_{aug_label}_{fraction_label}_seed_{seed}"

                    overrides = [
                        f"mode={mode}",
                        f"seed={seed}",
                        f"data.use_augmentation={use_augmentation}",
                        f"data.data_fraction={data_fraction}",
                        f"data.data_fraction_seed={seed}",
                        f"data.split_seed={seed}",
                        "hydra.run.dir=${hydra:runtime.cwd}/outputs/${problem}/"
                        + run_name,
                        f"train.checkpoint_path=${{hydra:runtime.output_dir}}/checkpoints_{run_name}",
                        f"train.tensorboard_log_dir=${{hydra:runtime.output_dir}}/logs_{run_name}",
                    ]

                    cfg = hydra.compose(config_name="config", overrides=overrides)

                    print(
                        "Running experiment:",
                        f"mode={mode}",
                        f"augmentation={use_augmentation}",
                        f"data_fraction={data_fraction}",
                        f"seed={seed}",
                    )
                    _run_training(cfg)


if __name__ == "__main__":
    main()

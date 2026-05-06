from __future__ import annotations

from typing import Callable

import hydra
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    seeds = [21, 42, 87]
    data_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    use_augmentations = [False, True]
    modes = ["base_cnn", "gecnn"]

    for mode in modes:
        for use_augmentation in use_augmentations:
            for data_fraction in data_fractions:
                for seed in seeds:
                    cfg.mode = mode
                    cfg.seed = seed
                    cfg.data.use_augmentation = use_augmentation
                    cfg.data.data_fraction = data_fraction
                    cfg.data.data_fraction_seed = seed
                    cfg.data.split_seed = seed

                    aug_label = "aug" if use_augmentation else "noaug"
                    fraction_label = f"frac_{data_fraction:.2f}".replace(".", "p")
                    run_name = f"{mode}_{aug_label}_{fraction_label}_seed_{seed}"

                    cfg.hydra.run.dir = (
                        f"${{hydra:runtime.cwd}}/outputs/${{problem}}/{run_name}"
                    )
                    cfg.train.checkpoint_path = (
                        f"${{hydra:runtime.output_dir}}/checkpoints_{run_name}"
                    )
                    cfg.train.tensorboard_log_dir = (
                        f"${{hydra:runtime.output_dir}}/logs_{run_name}"
                    )

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

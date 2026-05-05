import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.data.crc import with_crc_augmentation
from src.data.load_dataset import load_dataset
from src.training.train_functions import evaluate


def _build_loss_fn(
    loss_name: str, dataset: Dataset, device: torch.device
) -> torch.nn.Module:
    if loss_name == "BCEWithLogitsLoss":
        from src.training.train_functions import compute_pos_weight

        pos_weight = compute_pos_weight(dataset).to(device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "MSELoss":
        return torch.nn.MSELoss()
    if loss_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


def _resolve_model_path(cfg: DictConfig) -> Path:
    model_path = getattr(cfg, "model_path", None) or getattr(
        cfg.eval, "model_path", None
    )
    if model_path in (None, "", "null"):
        raise ValueError(
            "Missing model_path. Pass eval.model_path=... or model_path=..."
        )
    return Path(model_path)


def _get_metrics_output_path(model_path: Path) -> Path:
    run_dir = model_path.parent.parent
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / f"{model_path.stem}.json"


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    data_config = cfg.data
    train_config = cfg.train
    model_config = cfg.model
    device = torch.device(train_config.device)

    dataset_splits = load_dataset(data_config)
    if len(dataset_splits) == 3:
        _, _, test_dataset = dataset_splits
    else:
        _, test_dataset = dataset_splits

    model_path = _resolve_model_path(cfg)
    model: torch.nn.Module = instantiate(model_config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    loss_fn = _build_loss_fn(train_config.loss_fn, test_dataset, device)

    base_dataloader = DataLoader(
        test_dataset, batch_size=train_config.batch_size, shuffle=False
    )
    base_metrics = evaluate(
        model=model,
        loss_fn=loss_fn,
        dataloader=base_dataloader,
        device=device,
        accuracy_config=getattr(train_config, "accuracy", None),
    )

    augmented_dataset = with_crc_augmentation(test_dataset)

    augmented_dataloader = DataLoader(
        augmented_dataset, batch_size=train_config.batch_size, shuffle=False
    )
    augmented_metrics = evaluate(
        model=model,
        loss_fn=loss_fn,
        dataloader=augmented_dataloader,
        device=device,
        accuracy_config=getattr(train_config, "accuracy", None),
    )

    metrics_output_path = _get_metrics_output_path(model_path)
    payload = {
        "model_path": str(model_path),
        "metrics": {
            "no_augmentation": base_metrics,
            "with_augmentation": augmented_metrics,
        },
    }
    metrics_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Evaluation metrics for {model_path}:")
    print("No augmentation:")
    for name, value in base_metrics.items():
        print(f"  {name}: {value:.6f}")
    print("With augmentation:")
    for name, value in augmented_metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    main()

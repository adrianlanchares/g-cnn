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


def _get_consistency_config(accuracy_config: object | None) -> float:
    if accuracy_config is None:
        return 0.5
    if isinstance(accuracy_config, dict):
        return float(accuracy_config.get("threshold", 0.5))
    return float(getattr(accuracy_config, "threshold", 0.5))


def _build_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for positions, _ in dataloader:
            positions = positions.to(device)
            batch_outputs = model(positions).detach()
            outputs.append(batch_outputs)

    preds = torch.cat(outputs, dim=0)
    if preds.ndim == 1:
        preds = preds.unsqueeze(1)
    if preds.shape[1] > 1:
        return preds.argmax(dim=1)

    probs = torch.sigmoid(preds)
    return (probs > threshold).int().squeeze(1)


def _collect_base_images(dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    images: list[torch.Tensor] = []
    with torch.no_grad():
        for positions, _ in dataloader:
            images.append(positions.to(device).detach())
    return torch.cat(images, dim=0)


def _compute_consistency(
    base_preds: torch.Tensor,
    base_images: torch.Tensor,
    aug_preds: torch.Tensor,
    aug_images: torch.Tensor,
) -> float:
    if base_preds.shape != aug_preds.shape:
        raise ValueError(
            "Base and augmented predictions have different shapes: "
            f"{base_preds.shape} vs {aug_preds.shape}."
        )

    unchanged_mask = (base_images == aug_images).view(base_images.shape[0], -1).all(
        dim=1
    )
    changed_mask = ~unchanged_mask
    if not changed_mask.any():
        return 1.0

    if base_preds.ndim == 1:
        matches = base_preds[changed_mask] == aug_preds[changed_mask]
        return matches.float().mean().item()

    matches = (base_preds[changed_mask] == aug_preds[changed_mask]).all(dim=1)
    return matches.float().mean().item()


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

    num_aug_repeats = int(getattr(cfg.eval, "num_aug_repeats", 1))
    num_aug_repeats = max(num_aug_repeats, 1)
    accuracy_config = getattr(train_config, "accuracy", None)
    consistency_threshold = _get_consistency_config(accuracy_config)

    base_images = _collect_base_images(base_dataloader, device)
    base_preds = _build_predictions(
        model=model,
        dataloader=base_dataloader,
        device=device,
        threshold=consistency_threshold,
    )

    augmented_metrics_runs: list[dict[str, float]] = []
    consistency_runs: list[float] = []

    for _ in range(num_aug_repeats):
        augmented_dataset = with_crc_augmentation(test_dataset)
        augmented_dataloader = DataLoader(
            augmented_dataset, batch_size=train_config.batch_size, shuffle=False
        )
        augmented_metrics_runs.append(
            evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=augmented_dataloader,
                device=device,
                accuracy_config=accuracy_config,
            )
        )

        aug_images = _collect_base_images(augmented_dataloader, device)
        aug_preds = _build_predictions(
            model=model,
            dataloader=augmented_dataloader,
            device=device,
            threshold=consistency_threshold,
        )
        consistency_runs.append(
            _compute_consistency(base_preds, base_images, aug_preds, aug_images)
        )

    augmented_metrics: dict[str, float] = {}
    if augmented_metrics_runs:
        keys = augmented_metrics_runs[0].keys()
        for key in keys:
            augmented_metrics[key] = sum(
                metrics[key] for metrics in augmented_metrics_runs
            ) / len(augmented_metrics_runs)

    prediction_consistency = sum(consistency_runs) / max(len(consistency_runs), 1)

    metrics_output_path = _get_metrics_output_path(model_path)
    payload = {
        "model_path": str(model_path),
        "metrics": {
            "no_augmentation": base_metrics,
            "with_augmentation_avg": augmented_metrics,
            "prediction_consistency": prediction_consistency,
        },
    }
    metrics_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Evaluation metrics for {model_path}:")
    print("No augmentation:")
    for name, value in base_metrics.items():
        print(f"  {name}: {value:.6f}")
    print("With augmentation (avg):")
    for name, value in augmented_metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Prediction consistency: {prediction_consistency:.6f}")
    print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    main()

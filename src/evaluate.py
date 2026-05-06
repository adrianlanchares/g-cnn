import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


def _resolve_model_path(model_path: str) -> Path:
    if model_path in (None, "", "null"):
        raise ValueError(
            "Missing model_path. Pass eval.model_path=... or model_path=..."
        )
    return Path(model_path)


def _get_metrics_output_path(model_path: Path) -> Path:
    run_dir = model_path.parent.parent
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / "metrics.json"


def _compute_macro_prf(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tp: torch.Tensor | None = None
    fp: torch.Tensor | None = None
    fn: torch.Tensor | None = None

    with torch.no_grad():
        for positions, targets in dataloader:
            positions = positions.to(device)
            targets = targets.to(device).long().view(-1)

            outputs = model(positions)
            preds = outputs.argmax(dim=1).view(-1)

            num_classes = outputs.shape[1]
            if tp is None:
                tp = torch.zeros(num_classes, device=device)
                fp = torch.zeros(num_classes, device=device)
                fn = torch.zeros(num_classes, device=device)

            for cls in range(num_classes):
                pred_is_cls = preds == cls
                target_is_cls = targets == cls
                tp[cls] += (pred_is_cls & target_is_cls).sum()
                fp[cls] += (pred_is_cls & ~target_is_cls).sum()
                fn[cls] += (~pred_is_cls & target_is_cls).sum()

    if tp is None or fp is None or fn is None:
        return {"precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision_macro": precision.mean().item(),
        "recall_macro": recall.mean().item(),
        "f1_macro": f1.mean().item(),
        **{f"f1_class_{idx}": value.item() for idx, value in enumerate(f1)},
    }


MODEL_PATHS = []


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

    for model_path in tqdm(MODEL_PATHS, desc="Evaluating models"):
        print(f"\nEvaluating model: {model_path}")
        model_path = _resolve_model_path(model_path)
        model: torch.nn.Module = instantiate(model_config).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        loss_fn = _build_loss_fn(train_config.loss_fn, test_dataset, device)

        dataloader = DataLoader(
            test_dataset, batch_size=train_config.batch_size, shuffle=False
        )
        metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            dataloader=dataloader,
            device=device,
            accuracy_config=getattr(train_config, "accuracy", None),
        )
        base_prf = _compute_macro_prf(model=model, dataloader=dataloader, device=device)

        metrics_output_path = _get_metrics_output_path(model_path)
        payload = {
            "model_path": str(model_path),
            "metrics": {
                "base": metrics,
                "prf": base_prf,
            },
        }
        metrics_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"Evaluation metrics for {model_path}:")
        print("No augmentation:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.6f}")
        print("No augmentation (macro PRF):")
        for name, value in base_prf.items():
            print(f"  {name}: {value:.6f}")
        print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    main()

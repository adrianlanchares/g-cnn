from pathlib import Path
from typing import cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.data.load_dataset import load_dataset
from src.models.modules import get_group_spec
from src.training.train_functions import (
    BatchMetricsFn,
    compute_pos_weight,
    evaluate,
    train_one_epoch,
    validate,
)


def _transform_positions(
    positions: torch.Tensor,
    rotation_k: int,
    mirrored: bool,
) -> torch.Tensor:
    transformed = positions
    if mirrored:
        transformed = torch.flip(transformed, dims=(-1,))
    if rotation_k:
        transformed = torch.rot90(transformed, k=rotation_k, dims=(-2, -1))
    return transformed


def _build_invariance_metrics_fn(group: str) -> BatchMetricsFn:
    group_spec = get_group_spec(group)
    non_identity_elements: list[tuple[int, int]] = [
        element for element in group_spec.elements if not (element[0] == 0 and element[1] == 0)
    ]

    def metrics_fn(
        model: torch.nn.Module,
        positions: torch.Tensor,
        outputs: torch.Tensor,
    ) -> dict[str, float]:
        if not non_identity_elements:
            return {
                "invariance_abs_error": 0.0,
                "invariance_rel_error": 0.0,
            }

        baseline = outputs.reshape(outputs.shape[0], -1)
        baseline_abs_mean = baseline.abs().mean(dim=1)

        abs_errors: list[torch.Tensor] = []
        rel_errors: list[torch.Tensor] = []

        for rotation_k, mirror in non_identity_elements:
            transformed_positions = _transform_positions(
                positions,
                rotation_k=rotation_k,
                mirrored=bool(mirror),
            )
            transformed_outputs = model(transformed_positions).reshape(outputs.shape[0], -1)

            abs_error_per_sample = (baseline - transformed_outputs).abs().mean(dim=1)
            rel_error_per_sample = abs_error_per_sample / (baseline_abs_mean + 1e-8)

            abs_errors.append(abs_error_per_sample)
            rel_errors.append(rel_error_per_sample)

        abs_error = torch.stack(abs_errors, dim=1).mean().item()
        rel_error = torch.stack(rel_errors, dim=1).mean().item()

        return {
            "invariance_abs_error": abs_error,
            "invariance_rel_error": rel_error,
        }

    return metrics_fn


def train_gecnn(cfg: DictConfig) -> None:
    """Train the GECNN model and track symmetry diagnostics."""

    data_config = cfg.data
    train_config = cfg.train
    model_config = cfg.model

    print(f"Using device: {train_config.device}")
    device = torch.device(train_config.device)

    dataset_splits: tuple[Dataset, ...] = cast(tuple[Dataset, ...], load_dataset(data_config))
    if len(dataset_splits) == 3:
        train_dataset, valid_dataset, test_dataset = dataset_splits
    else:
        train_dataset, test_dataset = dataset_splits
        valid_dataset = None

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True
    )
    valid_dataloader = (
        DataLoader(valid_dataset, batch_size=train_config.batch_size, shuffle=False)
        if valid_dataset is not None
        else None
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=train_config.batch_size, shuffle=False
    )

    model: torch.nn.Module = instantiate(model_config).to(device)

    if train_config.loss_fn == "BCEWithLogitsLoss":
        pos_weight = compute_pos_weight(train_dataset).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif train_config.loss_fn == "MSELoss":
        loss_fn = torch.nn.MSELoss()
    elif train_config.loss_fn == "CrossEntropyLoss":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {train_config.loss_fn}")

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr
    )

    tensorboard_log_dir = Path(train_config.tensorboard_log_dir)
    checkpoint_dir = Path(train_config.checkpoint_path)

    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    invariance_metrics_fn: BatchMetricsFn = _build_invariance_metrics_fn(
        model_config.group
    )

    for epoch in range(train_config.num_epochs):
        avg_train_loss, epoch_time_sec = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch,
            writer=writer,
        )

        if valid_dataloader is not None:
            valid_metrics = validate(
                model=model,
                loss_fn=loss_fn,
                dataloader=valid_dataloader,
                device=device,
                batch_metrics_fn=invariance_metrics_fn,
            )
            test_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=test_dataloader,
                device=device,
                batch_metrics_fn=invariance_metrics_fn,
            )
            avg_eval_loss: float = valid_metrics["loss"]
            invariance_abs_error: float = valid_metrics["invariance_abs_error"]
            invariance_rel_error: float = valid_metrics["invariance_rel_error"]
            avg_test_loss: float | None = test_metrics["loss"]
            test_invariance_abs_error: float | None = test_metrics[
                "invariance_abs_error"
            ]
            test_invariance_rel_error: float | None = test_metrics[
                "invariance_rel_error"
            ]
        else:
            eval_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=test_dataloader,
                device=device,
                batch_metrics_fn=invariance_metrics_fn,
            )

            avg_eval_loss = eval_metrics["loss"]
            invariance_abs_error = eval_metrics["invariance_abs_error"]
            invariance_rel_error = eval_metrics["invariance_rel_error"]
            avg_test_loss = None
            test_invariance_abs_error = None
            test_invariance_rel_error = None

        writer.add_scalar("train/avg_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("eval/avg_loss", avg_eval_loss, epoch + 1)
        writer.add_scalar("eval/invariance_abs_error", invariance_abs_error, epoch + 1)
        writer.add_scalar("eval/invariance_rel_error", invariance_rel_error, epoch + 1)
        if avg_test_loss is not None:
            writer.add_scalar("test/avg_loss", avg_test_loss, epoch + 1)
            writer.add_scalar(
                "test/invariance_abs_error", test_invariance_abs_error, epoch + 1
            )
            writer.add_scalar(
                "test/invariance_rel_error", test_invariance_rel_error, epoch + 1
            )
        writer.add_scalar("train/epoch_time_sec", epoch_time_sec, epoch + 1)

        if avg_test_loss is not None:
            print(
                f"Epoch {epoch + 1}/{train_config.num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Valid Loss: {avg_eval_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}, "
                f"Valid Invariance Abs Error: {invariance_abs_error:.6f}, "
                f"Valid Invariance Rel Error: {invariance_rel_error:.6f}, "
                f"Test Invariance Abs Error: {test_invariance_abs_error:.6f}, "
                f"Test Invariance Rel Error: {test_invariance_rel_error:.6f}, "
                f"Epoch Time: {epoch_time_sec:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{train_config.num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Eval Loss: {avg_eval_loss:.4f}, "
                f"Invariance Abs Error: {invariance_abs_error:.6f}, "
                f"Invariance Rel Error: {invariance_rel_error:.6f}, "
                f"Epoch Time: {epoch_time_sec:.2f}s"
            )

        if (epoch + 1) % train_config.save_checkpoint_every == 0:
            checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Saved checkpoint to {checkpoint_file}")

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    writer.flush()
    writer.close()

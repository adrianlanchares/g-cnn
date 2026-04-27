import time
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


BatchMetricsFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], dict[str, float]]


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss: float = 0.0
    total_samples: int = 0
    epoch_start_time: float = time.perf_counter()

    for step, (positions, evaluations) in enumerate(dataloader):
        global_step = epoch * len(dataloader) + step

        positions = positions.to(device)
        evaluations = evaluations.to(device)

        outputs = model(positions)
        loss = loss_fn(outputs.squeeze(), evaluations)

        batch_size = positions.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer is not None:
            writer.add_scalar("train/step_loss", loss.item(), global_step)

    epoch_time_sec = time.perf_counter() - epoch_start_time
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, epoch_time_sec


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    batch_metrics_fn: BatchMetricsFn | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss: float = 0.0
    total_samples: int = 0
    metric_sums: dict[str, float] = {}

    with torch.no_grad():
        for positions, evaluations in dataloader:
            positions = positions.to(device)
            evaluations = evaluations.to(device)

            outputs = model(positions)
            loss = loss_fn(outputs.squeeze(), evaluations)

            batch_size = positions.shape[0]
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if batch_metrics_fn is not None:
                batch_metrics = batch_metrics_fn(model, positions, outputs)
                for metric_name, metric_value in batch_metrics.items():
                    metric_sums[metric_name] = (
                        metric_sums.get(metric_name, 0.0) + metric_value * batch_size
                    )

    metrics: dict[str, float] = {"loss": total_loss / max(total_samples, 1)}
    for metric_name, metric_sum in metric_sums.items():
        metrics[metric_name] = metric_sum / max(total_samples, 1)

    return metrics


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    batch_metrics_fn: BatchMetricsFn | None = None,
) -> dict[str, float]:
    return validate(
        model=model,
        loss_fn=loss_fn,
        dataloader=dataloader,
        device=device,
        batch_metrics_fn=batch_metrics_fn,
    )

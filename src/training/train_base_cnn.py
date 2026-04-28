from pathlib import Path
from typing import cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.data.load_dataset import load_dataset
from src.training.train_functions import (
    compute_pos_weight,
    evaluate,
    train_one_epoch,
    validate,
)


def train_base_cnn(
    cfg: DictConfig,
) -> None:
    """Train the BaseCNN model on the chess dataset."""

    data_config = cfg.data
    train_config = cfg.train
    model_config = cfg.model

    print(f"Using device: {train_config.device}")
    device = torch.device(train_config.device)

    dataset_splits: tuple[Dataset, ...] = cast(
        tuple[Dataset, ...], load_dataset(data_config)
    )
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

    # Initialize model, loss function, and optimizer
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

    # Train for a few epochs
    for epoch in range(train_config.num_epochs):
        avg_train_loss, epoch_time_sec = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch,
            writer=writer,
            write_every=train_config.train_write_every,
        )
        if valid_dataloader is not None:
            valid_metrics = validate(
                model=model,
                loss_fn=loss_fn,
                dataloader=valid_dataloader,
                device=device,
            )
            test_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=test_dataloader,
                device=device,
            )
            avg_eval_loss: float = valid_metrics["loss"]
            avg_test_loss: float | None = test_metrics["loss"]
        else:
            eval_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=test_dataloader,
                device=device,
            )
            avg_eval_loss = eval_metrics["loss"]
            avg_test_loss = None

        writer.add_scalar("train/avg_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("eval/avg_loss", avg_eval_loss, epoch + 1)
        if avg_test_loss is not None:
            writer.add_scalar("test/avg_loss", avg_test_loss, epoch + 1)
        writer.add_scalar("train/epoch_time_sec", epoch_time_sec, epoch + 1)

        if avg_test_loss is not None:
            print(
                f"Epoch {epoch + 1}/{train_config.num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Valid Loss: {avg_eval_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}, "
                f"Epoch Time: {epoch_time_sec:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{train_config.num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Eval Loss: {avg_eval_loss:.4f}, "
                f"Epoch Time: {epoch_time_sec:.2f}s"
            )

        if (epoch + 1) % train_config.save_checkpoint_every == 0:
            checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Saved checkpoint to {checkpoint_file}")

    # save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    writer.flush()
    writer.close()

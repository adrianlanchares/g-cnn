import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from src.data.chess import load_chess_tensor_dataset
from src.models.base_cnn import BaseCNN


def _train_one_epoch(
    model: BaseCNN,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    writer: SummaryWriter,
):
    """Train the model for one epoch on the given dataset."""
    model.train()
    total_loss = 0.0

    for step, (positions, evaluations) in enumerate(dataloader):
        print(f"Step {step + 1}/{len(dataloader)}\t\t\t", end="\r")
        positions = positions.to(device)
        evaluations = evaluations.to(device)

        outputs = model(positions)
        loss = loss_fn(outputs.squeeze(), evaluations)
        total_loss += loss.item()

        writer.add_scalar(
            "train/step_loss", loss.item(), epoch * len(dataloader) + step
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n\n")

    return total_loss / len(dataloader)


def evaluate(
    model: BaseCNN, loss_fn: torch.nn.Module, dataloader: DataLoader, device: str
):
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for positions, evaluations in dataloader:
            positions = positions.to(device)
            evaluations = evaluations.to(device)

            outputs = model(positions)
            loss = loss_fn(outputs.squeeze(), evaluations)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(
    cfg: Config,
):
    """Train the BaseCNN model on the chess dataset."""

    data_cfg = cfg.data_config
    train_cfg = cfg.train_config

    print(f"Using device: {train_cfg.device}")

    # Load dataset
    dataset = load_chess_tensor_dataset(data_cfg)
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = BaseCNN(cfg.base_cnn_config).to(train_cfg.device)

    if train_cfg.loss_fn == "MSELoss":
        loss_fn = torch.nn.MSELoss()
    elif train_cfg.loss_fn == "CrossEntropyLoss":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {train_cfg.loss_fn}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    train_cfg.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    train_cfg.checkpoint_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(train_cfg.tensorboard_log_dir))

    # Train for a few epochs
    for epoch in range(train_cfg.num_epochs):
        avg_train_loss = _train_one_epoch(
            model, loss_fn, optimizer, dataloader, train_cfg.device, epoch, writer
        )
        avg_eval_loss = evaluate(model, loss_fn, dataloader, train_cfg.device)

        writer.add_scalar("eval/avg_loss", avg_eval_loss, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{train_cfg.num_epochs}, Average Loss: {avg_train_loss:.4f}"
        )

        if (epoch + 1) % train_cfg.save_checkpoint_every == 0:
            checkpoint_path = (
                train_cfg.checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # save final model
    final_model_path = train_cfg.checkpoint_path / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    writer.flush()
    writer.close()

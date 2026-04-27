import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, positions: torch.Tensor, evaluations: torch.Tensor) -> None:
        """
        Dataset for chess position evaluations.

        Args:
            positions: Tensor of shape [N, 12, 8, 8] containing the board states.
            evaluations: Tensor of shape [N] containing the evaluation scores.
        """
        self.positions = positions
        self.evaluations = evaluations

    def __len__(self) -> int:
        return len(self.evaluations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.evaluations[idx]


class CelebADataset(Dataset):
    def __init__(self, base_dataset: Dataset, target_attr_idx: int | None = None) -> None:
        """
        Lightweight wrapper around torchvision CelebA datasets.

        Args:
            base_dataset: Any dataset that returns (image, target).
            target_attr_idx: Optional attribute index when target is an attribute vector.
        """
        self.base_dataset = base_dataset
        self.target_attr_idx = target_attr_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.base_dataset[idx]

        if self.target_attr_idx is not None and torch.is_tensor(target):
            target = target[self.target_attr_idx]

        if not torch.is_tensor(target):
            target = torch.as_tensor(target)

        return image, target.float()

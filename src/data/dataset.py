import torch
from torch.utils.data import Dataset


class ChessPositionEvaluationDataset(Dataset):
    def __init__(self, positions: torch.Tensor, evaluations: torch.Tensor):
        """
        Dataset for chess position evaluations.

        Args:
            positions: Tensor of shape [N, 12, 8, 8] containing the board states.
            evaluations: Tensor of shape [N] containing the evaluation scores.
        """
        self.positions = positions
        self.evaluations = evaluations

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]

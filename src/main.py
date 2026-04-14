import torch

from config.data import ChessDatasetConfig


def main():
    cfg = ChessDatasetConfig()

    dataset_path = cfg.output_tensor_dataset

    data = torch.load(dataset_path, map_location="cpu")

    positions = data["positions"]
    evaluations = data["evaluations"]

    print(f"Positions shape: {positions.shape}")
    print(f"Evaluations shape: {evaluations.shape}")

    print(positions[-1])


if __name__ == "__main__":
    main()

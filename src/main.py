import hydra
from omegaconf import DictConfig

from src.data.chess import load_chess_tensor_dataset

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = load_chess_tensor_dataset(cfg.data)

    positions = dataset.positions
    evaluations = dataset.evaluations

    print(f"Positions shape: {positions.shape}")
    print(f"Evaluations shape: {evaluations.shape}")

    print(positions[-1])


if __name__ == "__main__":
    main()

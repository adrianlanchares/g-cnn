from config import Config
from src.training.train_base_cnn import train

if __name__ == "__main__":
    cfg = Config()
    train(cfg)

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    config_root = Path(__file__).resolve().parents[1] / "config"

    base_model_cfg = OmegaConf.load(config_root / "model" / "base_cnn.yaml")
    ge_model_cfg = OmegaConf.load(config_root / "model" / "gecnn.yaml")

    base_cfg = OmegaConf.create({"data": cfg.data, "model": base_model_cfg})
    ge_cfg = OmegaConf.create({"data": cfg.data, "model": ge_model_cfg})
    OmegaConf.resolve(base_cfg)
    OmegaConf.resolve(ge_cfg)

    base_model = instantiate(base_cfg.model)
    ge_model = instantiate(ge_cfg.model)

    print(f"BaseCNN params: {base_model._get_param_count()}")
    print(f"GECNN params: {ge_model._get_param_count()}")


if __name__ == "__main__":
    main()

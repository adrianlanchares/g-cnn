from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = project_root / "data"


path_cfg = PathConfig()

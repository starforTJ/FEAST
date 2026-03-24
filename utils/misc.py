from pathlib import Path
from typing import Dict
import numpy as np
import torch
import yaml

def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
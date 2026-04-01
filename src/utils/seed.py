import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    print(f">>> 设置随机种子: {seed}", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
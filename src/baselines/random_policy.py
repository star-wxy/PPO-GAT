import numpy as np


class RandomPolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def predict(self, obs: np.ndarray) -> int:
        return int(np.random.randint(0, self.action_dim))
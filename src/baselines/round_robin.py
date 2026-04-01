class RoundRobinPolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.current = 0

    def predict(self, obs) -> int:
        action = self.current
        self.current = (self.current + 1) % self.action_dim
        return action
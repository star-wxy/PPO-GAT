import numpy as np


class GreedyCPUPolicy:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def predict(self, obs: np.ndarray) -> int:
        # obs = [task_size, task_deadline, task_priority,
        #        node_free_cpu * N,
        #        node_latency * N,
        #        node_load_ratio * N,
        #        robot_energy * R]
        start = 3
        end = 3 + self.num_nodes
        node_free = obs[start:end]
        return int(np.argmax(node_free))
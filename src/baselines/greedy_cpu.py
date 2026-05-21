import numpy as np


class GreedyCPUPolicy:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def predict(self, obs: np.ndarray) -> int:
        # Current obs = [task fields * 8, node_free_cpu * N,
        #        node_latency * N,
        #        node_load_ratio * N,
        #        robot_energy * R, robot_local_cpu * R, robot_queue * R]
        # Legacy obs used 3 task fields.
        start = 8 if len(obs) >= 8 + self.num_nodes * 3 else 3
        end = 3 + self.num_nodes
        if start == 8:
            end = start + self.num_nodes
        node_free = obs[start:end]
        return int(np.argmax(node_free))

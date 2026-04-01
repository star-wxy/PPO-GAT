from dataclasses import dataclass


@dataclass
class ComputeNode:
    node_id: int
    node_type: str
    cpu_capacity: float
    latency: float
    energy_factor: float
    cpu_used: float = 0.0

    @property
    def cpu_free(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_used)

    @property
    def load_ratio(self) -> float:
        if self.cpu_capacity <= 0:
            return 0.0
        return self.cpu_used / self.cpu_capacity
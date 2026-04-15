from dataclasses import dataclass, field


@dataclass
class Robot:
    robot_id: int
    energy: float
    home_node_id: int = 0
    local_cpu: float = 2.0
    task_rate: float = 1.0
    task_size_bias: float = 1.0
    deadline_bias: float = 1.0
    task_queue: list = field(default_factory=list)

    @property
    def queue_length(self) -> int:
        return len(self.task_queue)

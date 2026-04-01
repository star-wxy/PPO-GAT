from dataclasses import dataclass, field


@dataclass
class Robot:
    robot_id: int
    energy: float
    local_cpu: float = 2.0
    task_queue: list = field(default_factory=list)
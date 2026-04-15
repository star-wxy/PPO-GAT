from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    source_robot_id: int
    size: float
    deadline: int
    priority: int
    task_type: int = 0
    local_compute_demand: float = 0.0
    transmission_demand: float = 0.0
    assigned_node: int | None = None
    finished: bool = False

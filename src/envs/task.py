from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    size: float
    deadline: int
    priority: int
    assigned_node: int | None = None
    finished: bool = False
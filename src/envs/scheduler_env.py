import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.node import ComputeNode
from src.envs.robot import Robot
from src.envs.task import Task


class SchedulerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict):
        super().__init__()

        self.num_robots = env_config["num_robots"]
        self.num_nodes = env_config["num_nodes"]
        self.max_steps = env_config["max_steps"]

        self.robot_init_energy = env_config["robot"]["init_energy"]
        self.node_configs = env_config["nodes"]

        self.task_size_min = env_config["task"]["size_min"]
        self.task_size_max = env_config["task"]["size_max"]
        self.deadline_min = env_config["task"]["deadline_min"]
        self.deadline_max = env_config["task"]["deadline_max"]
        self.priority_levels = env_config["task"]["priority_levels"]

        self.action_space = spaces.Discrete(self.num_nodes)

        # 观测设计：
        # [task_size, task_deadline, task_priority,
        #  node_free_cpu * N,
        #  node_latency * N,
        #  node_load_ratio * N,
        #  robot_energy * R]
        obs_dim = 3 + self.num_nodes + self.num_nodes + self.num_nodes + self.num_robots

        self.observation_space = spaces.Box(
            low=0.0,
            high=1e6,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.robots: list[Robot] = []
        self.nodes: list[ComputeNode] = []
        self.current_task: Task | None = None
        self.current_robot_id: int = 0
        self.task_id_counter: int = 0
        self.step_count: int = 0

    def _build_entities(self) -> None:
        self.robots = [
            Robot(robot_id=i, energy=self.robot_init_energy)
            for i in range(self.num_robots)
        ]

        self.nodes = []
        for i, cfg in enumerate(self.node_configs):
            self.nodes.append(
                ComputeNode(
                    node_id=i,
                    node_type=cfg["type"],
                    cpu_capacity=float(cfg["cpu_capacity"]),
                    latency=float(cfg["latency"]),
                    energy_factor=float(cfg["energy_factor"]),
                )
            )

    def _sample_task(self) -> Task:
        self.task_id_counter += 1
        size = random.uniform(self.task_size_min, self.task_size_max)
        deadline = random.randint(self.deadline_min, self.deadline_max)
        priority = random.randint(1, self.priority_levels)
        return Task(
            task_id=self.task_id_counter,
            size=size,
            deadline=deadline,
            priority=priority,
        )

    def _get_obs(self) -> np.ndarray:
        assert self.current_task is not None

        node_free = [node.cpu_free for node in self.nodes]
        node_latency = [node.latency for node in self.nodes]
        node_load = [node.load_ratio for node in self.nodes]
        robot_energy = [robot.energy for robot in self.robots]

        obs = np.array(
            [
                self.current_task.size,
                self.current_task.deadline,
                self.current_task.priority,
            ]
            + node_free
            + node_latency
            + node_load
            + robot_energy,
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "current_robot_id": self.current_robot_id,
            "task_id": None if self.current_task is None else self.current_task.task_id,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.task_id_counter = 0
        self.current_robot_id = 0

        self._build_entities()
        self.current_task = self._sample_task()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.current_task is not None

        self.step_count += 1

        chosen_node = self.nodes[action]
        current_robot = self.robots[self.current_robot_id]

        task_size = float(self.current_task.size)
        task_deadline = float(self.current_task.deadline)
        task_priority = int(self.current_task.priority)

        # 1. 计算执行时间
        compute_time = task_size / max(chosen_node.cpu_capacity, 1e-6)

        # 2. 排队时延：当前负载越高，排队越严重
        queue_delay = chosen_node.load_ratio * 1.5

        # 3. 网络时延
        network_latency = chosen_node.latency

        # 4. 总时延
        total_time = compute_time + queue_delay + network_latency

        # 5. 过载惩罚
        overload_penalty = 0.0
        if chosen_node.cpu_free >= task_size:
            chosen_node.cpu_used += task_size
        else:
            overload_penalty = 1.0 + 2.0 * (task_size - chosen_node.cpu_free) / max(
                chosen_node.cpu_capacity, 1e-6
            )
            chosen_node.cpu_used = min(
                chosen_node.cpu_capacity,
                chosen_node.cpu_used + task_size * 0.5,
            )

        # 6. 能耗：任务越大、节点能耗因子越高，消耗越高
        energy_cost = (0.4 + 0.25 * task_size) * chosen_node.energy_factor
        current_robot.energy = max(0.0, current_robot.energy - energy_cost)

        # 7. deadline 惩罚
        deadline_penalty = max(0.0, total_time - task_deadline / 2.0)

        # 8. 优先级权重：优先级越高，超时越不能接受
        priority_weight = 1.0 + 0.5 * (task_priority - 1)

        # 9. 奖励函数（v2）
        reward = (
            6.0
            - 2.5 * total_time
            - 1.2 * energy_cost
            - priority_weight * 2.0 * deadline_penalty
            - 2.0 * overload_penalty
        )

        # 10. 每步释放少量资源，让拥塞能积累
        for node in self.nodes:
            release_amount = 0.4 * node.cpu_capacity
            node.cpu_used = max(0.0, node.cpu_used - release_amount)

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info["reward"] = float(reward)
        info["chosen_node"] = int(action)
        info["compute_time"] = float(compute_time)
        info["queue_delay"] = float(queue_delay)
        info["network_latency"] = float(network_latency)
        info["total_time"] = float(total_time)
        info["energy_cost"] = float(energy_cost)
        info["deadline_penalty"] = float(deadline_penalty)
        info["overload_penalty"] = float(overload_penalty)
        info["task_size"] = float(task_size)
        info["task_deadline"] = float(task_deadline)
        info["task_priority"] = int(task_priority)
        info["node_type"] = chosen_node.node_type
        info["node_load_ratio"] = float(chosen_node.load_ratio)

        self.current_robot_id = (self.current_robot_id + 1) % self.num_robots
        self.current_task = self._sample_task()

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
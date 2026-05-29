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

        self.robot_cfg = env_config["robot"]
        self.robot_profiles = env_config.get("robots", [])
        self.node_configs = env_config["nodes"]
        self.task_type_configs = env_config.get("task_types", [])
        self.network_cfg = env_config.get("network", {})
        self.reward_cfg = env_config.get("reward", {})
        self.congestion_propagation_enabled = bool(
            self.network_cfg.get("congestion_propagation_enabled", True)
        )

        self.task_size_min = env_config["task"]["size_min"]
        self.task_size_max = env_config["task"]["size_max"]
        self.deadline_min = env_config["task"]["deadline_min"]
        self.deadline_max = env_config["task"]["deadline_max"]
        self.priority_levels = env_config["task"]["priority_levels"]

        self.action_space = spaces.Discrete(self.num_nodes)

        obs_dim = 8 + self.num_nodes * 3 + self.num_robots * 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1e6,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.robots: list[Robot] = []
        self.nodes: list[ComputeNode] = []
        self.pending_tasks: list[Task] = []
        self.current_task: Task | None = None
        self.current_robot_id: int = 0
        self.task_id_counter: int = 0
        self.step_count: int = 0
        self.generated_tasks_total: int = 0
        self.completed_tasks_total: int = 0

        self.max_robot_energy = float(self.robot_cfg["init_energy"])
        self.max_robot_local_cpu = max(
            [profile.get("local_cpu", 0.0) for profile in self.robot_profiles]
            + [self.robot_cfg.get("local_cpu_max", 3.5)]
        )
        self.max_robot_queue = max(3, self.num_robots * 2)
        self.task_arrival_base = float(self.robot_cfg.get("task_arrival_base", 0.35))
        self.task_arrival_scale = float(self.robot_cfg.get("task_arrival_scale", 0.35))
        self.task_arrival_cap = float(self.robot_cfg.get("task_arrival_cap", 0.95))
        self.max_new_tasks_per_step = self.robot_cfg.get("max_new_tasks_per_step")
        self.max_new_tasks_per_step = (
            None
            if self.max_new_tasks_per_step is None
            else int(self.max_new_tasks_per_step)
        )
        self.charging_enabled = bool(self.robot_cfg.get("charging_enabled", False))
        self.charge_rate = float(self.robot_cfg.get("charge_rate", 0.0))
        self.critical_energy_ratio = float(self.robot_cfg.get("critical_energy_ratio", 0.08))
        self.recover_energy_ratio = float(self.robot_cfg.get("recover_energy_ratio", 0.22))
        self.charging_task_rate_scale = float(self.robot_cfg.get("charging_task_rate_scale", 0.0))
        self.idle_charge_rate_scale = float(self.robot_cfg.get("idle_charge_rate_scale", 0.15))
        self.charging_robot_ids: set[int] = set()
        self.backlog_penalty_mode = str(self.reward_cfg.get("backlog_penalty_mode", "linear"))
        self.backlog_penalty_coef = float(self.reward_cfg.get("backlog_penalty_coef", 0.09))
        self.backlog_soft_limit = int(self.reward_cfg.get("backlog_soft_limit", self.num_robots))
        self.backlog_delta_penalty_coef = float(
            self.reward_cfg.get("backlog_delta_penalty_coef", 0.0)
        )
        self.total_time_penalty_coef = float(self.reward_cfg.get("total_time_penalty_coef", 2.7))
        self.energy_penalty_coef = float(self.reward_cfg.get("energy_penalty_coef", 1.05))
        self.deadline_penalty_coef = float(self.reward_cfg.get("deadline_penalty_coef", 2.4))
        self.overload_penalty_coef = float(self.reward_cfg.get("overload_penalty_coef", 2.6))
        self.topology_penalty_coef = float(self.reward_cfg.get("topology_penalty_coef", 0.45))
        self.robot_queue_penalty_coef = float(self.reward_cfg.get("robot_queue_penalty_coef", 0.32))
        self.node_pressure_penalty_coef = float(
            self.reward_cfg.get("node_pressure_penalty_coef", 0.85)
        )
        self.reward_mode = str(self.reward_cfg.get("mode", "static"))
        self.queue_penalty_coef = float(self.reward_cfg.get("queue_penalty_coef", 0.0))
        self.latency_penalty_coef = float(self.reward_cfg.get("latency_penalty_coef", 0.0))
        self.balance_bonus_coef = float(self.reward_cfg.get("balance_bonus_coef", 0.0))
        self.slack_bonus_coef = float(self.reward_cfg.get("slack_bonus_coef", 0.18))
        self.dynamic_min_multiplier = float(self.reward_cfg.get("dynamic_min_multiplier", 0.5))
        self.dynamic_max_multiplier = float(self.reward_cfg.get("dynamic_max_multiplier", 3.0))
        self.bandwidth_scale = float(self.network_cfg.get("bandwidth_scale", 1.0))
        self.transfer_base_latency = float(self.network_cfg.get("transfer_base_latency", 0.03))
        self.transfer_hop_latency = float(self.network_cfg.get("transfer_hop_latency", 0.10))
        self.transfer_demand_latency = float(self.network_cfg.get("transfer_demand_latency", 0.05))
        self.transfer_base_energy = float(self.network_cfg.get("transfer_base_energy", 0.10))
        self.transfer_hop_energy = float(self.network_cfg.get("transfer_hop_energy", 0.16))
        self.transfer_demand_energy = float(self.network_cfg.get("transfer_demand_energy", 0.08))
        self.max_network_latency = (
            max(float(cfg["latency"]) for cfg in self.node_configs)
            + self.transfer_base_latency
            + self.transfer_hop_latency * max(self.num_nodes - 1, 1)
            + self.transfer_demand_latency * 1.5 / max(self.bandwidth_scale, 1e-6)
        )
        self.cloud_node_ids = [
            idx for idx, cfg in enumerate(self.node_configs) if cfg.get("type") == "cloud"
        ]

    def _clipped_dynamic_coef(self, base_coef: float, multiplier: float) -> float:
        min_coef = base_coef * self.dynamic_min_multiplier
        max_coef = base_coef * self.dynamic_max_multiplier
        return float(np.clip(base_coef * multiplier, min_coef, max_coef))

    def _context_factors(
        self,
        *,
        current_robot: Robot,
        current_task: Task,
        network_latency: float,
        robot_energy_before: float | None = None,
    ) -> dict[str, float]:
        node_loads = [float(node.load_ratio) for node in self.nodes]
        load_level = float(np.clip(np.mean(node_loads), 0.0, 1.0)) if node_loads else 0.0
        hotspot_level = float(np.clip(max(node_loads), 0.0, 1.0)) if node_loads else 0.0
        robot_energy = current_robot.energy if robot_energy_before is None else robot_energy_before
        energy_risk = float(
            np.clip(1.0 - robot_energy / max(self.max_robot_energy, 1e-6), 0.0, 1.0)
        )
        priority_norm = float(
            np.clip((current_task.priority - 1) / max(self.priority_levels - 1, 1), 0.0, 1.0)
        )
        deadline_norm = float(
            np.clip(current_task.deadline / max(self.deadline_max, 1), 0.0, 1.0)
        )
        urgency_level = float(np.clip(priority_norm * (1.0 - deadline_norm), 0.0, 1.0))
        comm_risk = float(
            np.clip(network_latency / max(self.max_network_latency, 1e-6), 0.0, 1.0)
        )
        task_scale = float(np.clip(current_task.size / max(self.task_size_max, 1e-6), 0.0, 1.5))
        compute_pressure = float(np.clip(task_scale / 1.25, 0.0, 1.0))

        load_context = float(np.clip(0.6 * load_level + 0.4 * hotspot_level, 0.0, 1.0))
        normal_context = float(
            np.clip(
                1.0
                - max(load_context, energy_risk, urgency_level, comm_risk, compute_pressure),
                0.0,
                1.0,
            )
        )

        return {
            "load_level": load_level,
            "hotspot_level": hotspot_level,
            "energy_risk": energy_risk,
            "urgency_level": urgency_level,
            "comm_risk": comm_risk,
            "task_scale": task_scale,
            "compute_pressure": compute_pressure,
            "load_context": load_context,
            "normal_context": normal_context,
        }

    def _dynamic_reward_weights(
        self,
        context: dict[str, float],
        *,
        priority_weight: float,
    ) -> dict[str, float]:
        load_context = context["load_context"]
        energy_risk = context["energy_risk"]
        urgency_level = context["urgency_level"]
        comm_risk = context["comm_risk"]
        compute_pressure = context["compute_pressure"]

        energy_mult = 1.0 + 1.5 * energy_risk + 0.8 * comm_risk - 0.4 * urgency_level
        deadline_mult = 1.0 + 2.0 * urgency_level + 0.5 * load_context
        overload_mult = 1.0 + 1.5 * load_context + 0.8 * compute_pressure
        queue_mult = 1.0 + 1.2 * load_context + 1.0 * urgency_level + 0.6 * compute_pressure
        latency_mult = 1.0 + 1.2 * comm_risk + 0.8 * urgency_level + 0.5 * energy_risk
        balance_mult = 1.0 + 1.5 * load_context + 0.8 * compute_pressure
        slack_mult = 1.0 + 0.8 * urgency_level

        deadline_coef = self._clipped_dynamic_coef(
            self.deadline_penalty_coef,
            deadline_mult,
        )

        return {
            "energy": self._clipped_dynamic_coef(self.energy_penalty_coef, energy_mult),
            "deadline": priority_weight * deadline_coef,
            "deadline_base": deadline_coef,
            "overload": self._clipped_dynamic_coef(self.overload_penalty_coef, overload_mult),
            "queue": self._clipped_dynamic_coef(self.queue_penalty_coef, queue_mult)
            if self.queue_penalty_coef > 0
            else 0.0,
            "latency": self._clipped_dynamic_coef(self.latency_penalty_coef, latency_mult)
            if self.latency_penalty_coef > 0
            else 0.0,
            "balance": self._clipped_dynamic_coef(self.balance_bonus_coef, balance_mult)
            if self.balance_bonus_coef > 0
            else 0.0,
            "slack": self._clipped_dynamic_coef(self.slack_bonus_coef, slack_mult)
            if self.slack_bonus_coef > 0
            else 0.0,
        }

    def _task_type_name(self, task_type_id: int) -> str:
        if 0 <= task_type_id < len(self.task_type_configs):
            return str(self.task_type_configs[task_type_id].get("name", task_type_id))
        return str(task_type_id)

    def _infer_scenario_context(
        self,
        *,
        current_robot: Robot,
        current_task: Task,
        context: dict[str, float],
    ) -> tuple[str, dict[str, float]]:
        robot_queue_pressure = float(
            np.clip(current_robot.queue_length / max(self.max_robot_queue, 1), 0.0, 1.0)
        )
        priority_norm = float(
            np.clip((current_task.priority - 1) / max(self.priority_levels - 1, 1), 0.0, 1.0)
        )
        deadline_pressure = float(
            np.clip(1.0 - current_task.deadline / max(self.deadline_max, 1), 0.0, 1.0)
        )
        transmission_pressure = float(
            np.clip(current_task.transmission_demand / 1.5, 0.0, 1.0)
        )
        type_name = self._task_type_name(current_task.task_type).lower()

        emergency_type_bonus = 0.15 if type_name in {"perception", "manipulation", "inspection"} else 0.0
        high_latency_type_bonus = 0.12 if type_name in {"mapping", "perception"} else 0.0
        high_load_type_bonus = 0.10 if type_name in {"mapping", "perception"} else 0.0

        scores = {
            "low_energy": float(
                np.clip(
                    0.72 * context["energy_risk"]
                    + 0.18 * transmission_pressure
                    + 0.10 * context["comm_risk"],
                    0.0,
                    1.0,
                )
            ),
            "high_load": float(
                np.clip(
                    0.66 * context["load_context"]
                    + 0.18 * context["compute_pressure"]
                    + 0.16 * robot_queue_pressure
                    + high_load_type_bonus,
                    0.0,
                    1.0,
                )
            ),
            "emergency": float(
                np.clip(
                    0.58 * context["urgency_level"]
                    + 0.22 * priority_norm
                    + 0.20 * deadline_pressure
                    + emergency_type_bonus,
                    0.0,
                    1.0,
                )
            ),
            "high_latency": float(
                np.clip(
                    0.62 * context["comm_risk"]
                    + 0.26 * transmission_pressure
                    + 0.12 * context["energy_risk"]
                    + high_latency_type_bonus,
                    0.0,
                    1.0,
                )
            ),
        }

        inferred_scenario = max(scores, key=scores.get)
        if scores[inferred_scenario] < 0.35:
            inferred_scenario = "normal"

        scores["normal"] = float(context["normal_context"])
        return inferred_scenario, scores

    def _robot_anchor_node(self, robot_id: int) -> int:
        return self.robots[robot_id].home_node_id if self.robots else min(robot_id, self.num_nodes - 1)

    def _topology_distance(self, src_node: int, dst_node: int) -> int:
        return abs(int(src_node) - int(dst_node))

    def _robot_to_node_transfer_latency(self, robot_id: int, node_id: int, task: Task) -> float:
        anchor = self._robot_anchor_node(robot_id)
        distance = self._topology_distance(anchor, node_id)
        bandwidth = max(self.bandwidth_scale, 1e-6)
        return (
            self.transfer_base_latency
            + self.transfer_hop_latency * distance
            + self.transfer_demand_latency * task.transmission_demand / bandwidth
        )

    def _robot_to_node_transfer_energy(self, robot_id: int, node_id: int, task: Task) -> float:
        anchor = self._robot_anchor_node(robot_id)
        distance = self._topology_distance(anchor, node_id)
        bandwidth = max(self.bandwidth_scale, 1e-6)
        return (
            self.transfer_base_energy
            + self.transfer_hop_energy * distance
            + self.transfer_demand_energy * task.transmission_demand / bandwidth
        )

    def _propagate_congestion(self, chosen_node_id: int, task_size: float) -> None:
        spill_strength = min(0.35, 0.04 * task_size)
        for node in self.nodes:
            if node.node_id == chosen_node_id:
                continue

            distance = self._topology_distance(chosen_node_id, node.node_id)
            if distance > 1:
                continue

            spill_ratio = spill_strength / (distance + 1.0)
            node.cpu_used = min(
                node.cpu_capacity,
                node.cpu_used + spill_ratio * node.cpu_capacity,
            )

    def _is_robot_charging(self, robot: Robot) -> bool:
        return robot.robot_id in self.charging_robot_ids

    def _update_robot_charging_states(self) -> tuple[int, int]:
        if not self.charging_enabled:
            return 0, 0

        started_count = 0
        recovered_count = 0
        critical_energy = self.max_robot_energy * self.critical_energy_ratio
        recover_energy = self.max_robot_energy * self.recover_energy_ratio

        for robot in self.robots:
            was_charging = robot.robot_id in self.charging_robot_ids
            if robot.energy <= critical_energy:
                if not was_charging:
                    started_count += 1
                self.charging_robot_ids.add(robot.robot_id)
                was_charging = True

            if was_charging:
                robot.energy = min(self.max_robot_energy, robot.energy + self.charge_rate)
                if robot.energy >= recover_energy:
                    self.charging_robot_ids.discard(robot.robot_id)
                    recovered_count += 1
            elif self.charge_rate > 0 and self.idle_charge_rate_scale > 0:
                robot.energy = min(
                    self.max_robot_energy,
                    robot.energy + self.charge_rate * self.idle_charge_rate_scale,
                )

        return started_count, recovered_count

    def _build_entities(self) -> None:
        self.robots = []
        for robot_id in range(self.num_robots):
            profile = self.robot_profiles[robot_id] if robot_id < len(self.robot_profiles) else {}
            home_node_id = int(profile.get("home_node_id", min(robot_id, self.num_nodes - 1)))
            local_cpu = float(
                profile.get(
                    "local_cpu",
                    random.uniform(
                        self.robot_cfg.get("local_cpu_min", 1.5),
                        self.robot_cfg.get("local_cpu_max", 3.5),
                    ),
                )
            )
            task_rate = float(
                profile.get(
                    "task_rate",
                    random.uniform(
                        self.robot_cfg.get("task_rate_min", 0.8),
                        self.robot_cfg.get("task_rate_max", 1.2),
                    ),
                )
            )
            task_size_bias = float(
                profile.get(
                    "task_size_bias",
                    random.uniform(
                        self.robot_cfg.get("task_size_bias_min", 0.85),
                        self.robot_cfg.get("task_size_bias_max", 1.2),
                    ),
                )
            )
            deadline_bias = float(
                profile.get(
                    "deadline_bias",
                    random.uniform(
                        self.robot_cfg.get("deadline_bias_min", 0.85),
                        self.robot_cfg.get("deadline_bias_max", 1.15),
                    ),
                )
            )
            self.robots.append(
                Robot(
                    robot_id=robot_id,
                    energy=float(self.robot_cfg["init_energy"]),
                    home_node_id=home_node_id,
                    local_cpu=local_cpu,
                    task_rate=task_rate,
                    task_size_bias=task_size_bias,
                    deadline_bias=deadline_bias,
                )
            )

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

    def _generate_task_for_robot(self, robot_id: int) -> Task:
        robot = self.robots[robot_id]
        self.task_id_counter += 1
        self.generated_tasks_total += 1

        task_type_id = random.randrange(len(self.task_type_configs)) if self.task_type_configs else 0
        task_type_cfg = self.task_type_configs[task_type_id] if self.task_type_configs else {}

        size_scale = float(task_type_cfg.get("size_scale", 1.0))
        deadline_scale = float(task_type_cfg.get("deadline_scale", 1.0))
        priority_bias = int(task_type_cfg.get("priority_bias", 0))
        local_compute_scale = float(task_type_cfg.get("local_compute_scale", 1.0))
        transmission_scale = float(task_type_cfg.get("transmission_scale", 1.0))

        size = random.uniform(self.task_size_min, self.task_size_max)
        size *= robot.task_size_bias * size_scale
        size = float(np.clip(size, self.task_size_min, self.task_size_max * 1.5))

        deadline = random.randint(self.deadline_min, self.deadline_max)
        deadline = int(round(deadline * robot.deadline_bias * deadline_scale))
        deadline = max(1, deadline)

        priority = random.randint(1, self.priority_levels)
        priority = int(np.clip(priority + priority_bias, 1, self.priority_levels))

        local_compute_demand = size * local_compute_scale / max(robot.local_cpu, 1e-3)
        transmission_demand = size * transmission_scale / max(self.task_size_max, 1e-3)

        return Task(
            task_id=self.task_id_counter,
            source_robot_id=robot_id,
            size=size,
            deadline=deadline,
            priority=priority,
            task_type=task_type_id,
            local_compute_demand=float(local_compute_demand),
            transmission_demand=float(transmission_demand),
        )

    def _enqueue_new_tasks(self, ensure_at_least_one: bool = False) -> int:
        created_count = 0
        for robot in self.robots:
            if (
                self.max_new_tasks_per_step is not None
                and created_count >= self.max_new_tasks_per_step
            ):
                break

            task_rate_scale = (
                self.charging_task_rate_scale
                if self._is_robot_charging(robot)
                else 1.0
            )
            task_probability = min(
                self.task_arrival_cap,
                self.task_arrival_base + self.task_arrival_scale * robot.task_rate * task_rate_scale,
            )
            if random.random() <= task_probability:
                task = self._generate_task_for_robot(robot.robot_id)
                robot.task_queue.append(task)
                self.pending_tasks.append(task)
                created_count += 1

        if ensure_at_least_one and created_count == 0 and self.robots:
            fallback_robot = max(self.robots, key=lambda robot: (robot.task_rate, -robot.robot_id))
            task = self._generate_task_for_robot(fallback_robot.robot_id)
            fallback_robot.task_queue.append(task)
            self.pending_tasks.append(task)
            created_count += 1

        return created_count

    def _backlog_penalty(self, backlog_size: int) -> float:
        backlog_excess = max(backlog_size - self.backlog_soft_limit, 0)
        if backlog_excess <= 0:
            return 0.0

        if self.backlog_penalty_mode == "log":
            return self.backlog_penalty_coef * float(np.log1p(backlog_excess))
        if self.backlog_penalty_mode == "sqrt":
            return self.backlog_penalty_coef * float(np.sqrt(backlog_excess))
        return self.backlog_penalty_coef * float(backlog_excess)

    def _current_backlog_size(self) -> int:
        return len(self.pending_tasks) + (1 if self.current_task is not None else 0)

    def _select_next_task(self) -> None:
        if not self.pending_tasks:
            self._enqueue_new_tasks(ensure_at_least_one=True)

        if not self.pending_tasks:
            self.current_task = None
            return

        self.pending_tasks.sort(
            key=lambda task: (-task.priority, task.deadline, task.source_robot_id, task.task_id)
        )
        self.current_task = self.pending_tasks.pop(0)
        self.current_robot_id = self.current_task.source_robot_id

        robot_queue = self.robots[self.current_robot_id].task_queue
        self.robots[self.current_robot_id].task_queue = [
            task for task in robot_queue if task.task_id != self.current_task.task_id
        ]

    def _get_obs(self) -> np.ndarray:
        assert self.current_task is not None

        node_free = [node.cpu_free for node in self.nodes]
        node_latency = [node.latency for node in self.nodes]
        node_load = [node.load_ratio for node in self.nodes]
        robot_energy = [robot.energy for robot in self.robots]
        robot_local_cpu = [robot.local_cpu for robot in self.robots]
        robot_queue_norm = [robot.queue_length / self.max_robot_queue for robot in self.robots]

        source_robot = self.robots[self.current_task.source_robot_id]
        source_robot_id_norm = (
            self.current_task.source_robot_id / max(self.num_robots - 1, 1)
            if self.num_robots > 1
            else 0.0
        )
        task_type_norm = (
            self.current_task.task_type / max(len(self.task_type_configs) - 1, 1)
            if len(self.task_type_configs) > 1
            else 0.0
        )

        obs = np.array(
            [
                self.current_task.size,
                self.current_task.deadline,
                self.current_task.priority,
                task_type_norm,
                source_robot_id_norm,
                source_robot.energy,
                source_robot.local_cpu,
                source_robot.queue_length / self.max_robot_queue,
            ]
            + node_free
            + node_latency
            + node_load
            + robot_energy
            + robot_local_cpu
            + robot_queue_norm,
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "current_robot_id": self.current_robot_id,
            "task_id": None if self.current_task is None else self.current_task.task_id,
            "pending_tasks": len(self.pending_tasks),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.step_count = 0
        self.task_id_counter = 0
        self.current_robot_id = 0
        self.pending_tasks = []
        self.generated_tasks_total = 0
        self.completed_tasks_total = 0
        self.charging_robot_ids = set()

        self._build_entities()
        self._enqueue_new_tasks(ensure_at_least_one=True)
        self._select_next_task()

        if self.current_task is None:
            raise RuntimeError("No task could be generated during reset.")

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.current_task is not None

        self.step_count += 1
        backlog_before_step = self._current_backlog_size()
        charging_started_count, charging_recovered_count = self._update_robot_charging_states()

        chosen_node = self.nodes[action]
        current_task = self.current_task
        current_robot = self.robots[current_task.source_robot_id]
        robot_energy_before = float(current_robot.energy)

        task_size = float(current_task.size)
        task_deadline = float(current_task.deadline)
        task_priority = int(current_task.priority)
        anchor_node = self._robot_anchor_node(current_task.source_robot_id)
        topology_distance = self._topology_distance(anchor_node, action)
        edge_nodes = [node for node in self.nodes if node.node_type != "cloud"]
        edge_load_mean = (
            sum(node.load_ratio for node in edge_nodes) / max(len(edge_nodes), 1)
            if edge_nodes
            else chosen_node.load_ratio
        )
        edge_free_mean = (
            sum(node.cpu_free for node in edge_nodes) / max(len(edge_nodes), 1)
            if edge_nodes
            else chosen_node.cpu_free
        )
        cloud_candidate = chosen_node.node_type == "cloud"

        compute_time = task_size / max(chosen_node.cpu_capacity, 1e-6)
        queue_delay = chosen_node.load_ratio * (1.4 + 0.2 * current_task.transmission_demand)
        transfer_latency = self._robot_to_node_transfer_latency(current_task.source_robot_id, action, current_task)
        network_latency = chosen_node.latency + transfer_latency
        total_time = compute_time + queue_delay + network_latency

        overload_penalty = 0.0
        if chosen_node.cpu_free >= task_size:
            chosen_node.cpu_used += task_size
        else:
            overload_penalty = 1.0 + 2.2 * (task_size - chosen_node.cpu_free) / max(
                chosen_node.cpu_capacity, 1e-6
            )
            chosen_node.cpu_used = min(
                chosen_node.cpu_capacity,
                chosen_node.cpu_used + task_size * 0.55,
            )

        if self.congestion_propagation_enabled:
            self._propagate_congestion(action, task_size)

        local_compute_cost = current_task.local_compute_demand / max(current_robot.local_cpu, 1e-6)
        transfer_energy = self._robot_to_node_transfer_energy(current_task.source_robot_id, action, current_task)
        energy_cost = (
            (0.28 + 0.18 * task_size) * chosen_node.energy_factor
            + transfer_energy * (0.22 + 0.06 * task_size)
            + 0.08 * local_compute_cost
        )
        current_robot.energy = max(0.0, current_robot.energy - energy_cost)

        deadline_penalty = max(0.0, total_time - task_deadline / 2.0)
        priority_weight = 1.0 + 0.5 * (task_priority - 1)

        context_factors: dict[str, float] = {}
        dynamic_weights: dict[str, float] = {}
        scenario_scores: dict[str, float] = {}
        inferred_scenario = "static"
        dynamic_queue_penalty = 0.0
        dynamic_latency_penalty = 0.0
        dynamic_balance_bonus = 0.0

        if self.reward_mode == "dynamic_context":
            context_factors = self._context_factors(
                current_robot=current_robot,
                current_task=current_task,
                network_latency=network_latency,
                robot_energy_before=robot_energy_before,
            )
            dynamic_weights = self._dynamic_reward_weights(
                context_factors,
                priority_weight=priority_weight,
            )
            inferred_scenario, scenario_scores = self._infer_scenario_context(
                current_robot=current_robot,
                current_task=current_task,
                context=context_factors,
            )

            if inferred_scenario == "low_energy":
                dynamic_weights["energy"] *= 1.10
                dynamic_weights["latency"] *= 1.05
            elif inferred_scenario == "high_load":
                dynamic_weights["overload"] *= 1.10
                dynamic_weights["queue"] *= 1.08
                dynamic_weights["balance"] *= 1.08
            elif inferred_scenario == "emergency":
                dynamic_weights["deadline"] *= 1.12
                dynamic_weights["queue"] *= 1.06
                dynamic_weights["slack"] *= 1.06
            elif inferred_scenario == "high_latency":
                dynamic_weights["latency"] *= 1.12
                dynamic_weights["energy"] *= 1.04

            balance_bonus = max(1.0 - chosen_node.load_ratio, 0.0)
            dynamic_queue_penalty = dynamic_weights["queue"] * queue_delay
            dynamic_latency_penalty = dynamic_weights["latency"] * network_latency
            dynamic_balance_bonus = dynamic_weights["balance"] * balance_bonus

            reward = (
                8.0
                - self.total_time_penalty_coef * total_time
                - dynamic_weights["energy"] * energy_cost
                - dynamic_weights["deadline"] * deadline_penalty
                - dynamic_weights["overload"] * overload_penalty
                - dynamic_queue_penalty
                - dynamic_latency_penalty
                - self.topology_penalty_coef * topology_distance
                - self.robot_queue_penalty_coef * current_robot.queue_length
                + dynamic_balance_bonus
            )
        else:
            reward = (
                8.0
                - self.total_time_penalty_coef * total_time
                - self.energy_penalty_coef * energy_cost
                - priority_weight * self.deadline_penalty_coef * deadline_penalty
                - self.overload_penalty_coef * overload_penalty
                - self.topology_penalty_coef * topology_distance
                - self.robot_queue_penalty_coef * current_robot.queue_length
            )

        system_backlog_penalty = self._backlog_penalty(backlog_before_step)
        node_pressure_penalty = (
            self.node_pressure_penalty_coef
            * chosen_node.load_ratio
            * (0.8 + 0.5 * current_task.transmission_demand)
        )
        remote_cloud_penalty = 0.0
        if chosen_node.node_type == "cloud":
            remote_cloud_penalty = 0.35 + 0.18 * topology_distance + 0.12 * current_task.transmission_demand

        locality_bonus = 0.0
        if topology_distance == 0:
            locality_bonus = 0.22 + 0.08 * min(current_robot.local_cpu / max(task_size, 1.0), 1.5)
        elif topology_distance == 1 and chosen_node.node_type != "cloud":
            locality_bonus = 0.10

        slack_ratio = max((task_deadline - total_time) / max(task_deadline, 1.0), 0.0)
        if self.reward_mode == "dynamic_context":
            slack_bonus = dynamic_weights.get("slack", self.slack_bonus_coef) * slack_ratio
        else:
            slack_bonus = self.slack_bonus_coef * slack_ratio
        congestion_relief_bonus = 0.0
        cloud_misuse_penalty = 0.0
        if cloud_candidate:
            edge_congestion = max(edge_load_mean - 0.52, 0.0)
            edge_shortage = max(task_size - edge_free_mean, 0.0) / max(task_size, 1.0)
            queue_urgency = max(current_robot.queue_length - 1, 0) / max(self.max_robot_queue, 1)
            big_task_factor = min(task_size / max(self.task_size_max * 0.78, 1.0), 1.25)
            relaxed_deadline_factor = min(task_deadline / max(self.deadline_max, 1), 1.2)
            transmission_friendly = max(1.0 - current_task.transmission_demand, 0.0)

            congestion_relief_bonus = (
                1.25 * edge_congestion
                + 0.85 * edge_shortage
                + 0.55 * queue_urgency
                + 0.35 * big_task_factor * relaxed_deadline_factor * transmission_friendly
            )

            insufficient_reason = max(0.45 - edge_congestion, 0.0) + max(0.20 - edge_shortage, 0.0)
            cloud_misuse_penalty = (
                0.95 * insufficient_reason
                + 0.35 * max(current_task.transmission_demand - 0.95, 0.0)
                + 0.22 * max(0.55 - relaxed_deadline_factor, 0.0)
            )

        reward = (
            reward
            - system_backlog_penalty
            - node_pressure_penalty
            - remote_cloud_penalty
            - cloud_misuse_penalty
            + locality_bonus
            + slack_bonus
            + congestion_relief_bonus
        )

        for node in self.nodes:
            release_amount = 0.35 * node.cpu_capacity
            node.cpu_used = max(0.0, node.cpu_used - release_amount)

        self.completed_tasks_total += 1
        new_tasks_generated = self._enqueue_new_tasks()
        backlog_after_enqueue = len(self.pending_tasks)
        self._select_next_task()

        terminated = False
        truncated = self.step_count >= self.max_steps
        if self.current_task is None:
            truncated = True

        backlog_after_step = self._current_backlog_size()
        backlog_delta = backlog_after_step - backlog_before_step
        backlog_delta_penalty = self.backlog_delta_penalty_coef * max(backlog_delta, 0)
        reward -= backlog_delta_penalty

        info = self._get_info()
        info["reward"] = float(reward)
        info["chosen_node"] = int(action)
        info["compute_time"] = float(compute_time)
        info["queue_delay"] = float(queue_delay)
        info["transfer_latency"] = float(transfer_latency)
        info["network_latency"] = float(network_latency)
        info["bandwidth_scale"] = float(self.bandwidth_scale)
        info["total_time"] = float(total_time)
        info["energy_cost"] = float(energy_cost)
        info["deadline_penalty"] = float(deadline_penalty)
        info["overload_penalty"] = float(overload_penalty)
        info["task_size"] = float(task_size)
        info["task_deadline"] = float(task_deadline)
        info["task_priority"] = int(task_priority)
        info["task_type"] = int(current_task.task_type)
        info["source_robot_id"] = int(current_task.source_robot_id)
        info["source_robot_local_cpu"] = float(current_robot.local_cpu)
        info["node_type"] = chosen_node.node_type
        info["node_load_ratio"] = float(chosen_node.load_ratio)
        info["congestion_propagation_enabled"] = bool(self.congestion_propagation_enabled)
        info["current_robot_energy_before"] = float(robot_energy_before)
        info["current_robot_energy"] = float(current_robot.energy)
        info["robot_anchor_node"] = int(anchor_node)
        info["topology_distance"] = int(topology_distance)
        info["backlog_before_step"] = int(backlog_before_step)
        info["backlog_after_enqueue"] = int(backlog_after_enqueue)
        info["backlog_after_step"] = int(backlog_after_step)
        info["backlog_delta"] = int(backlog_delta)
        info["new_tasks_generated"] = int(new_tasks_generated)
        info["generated_tasks_total"] = int(self.generated_tasks_total)
        info["completed_tasks_total"] = int(self.completed_tasks_total)
        info["charging_robots"] = int(len(self.charging_robot_ids))
        info["charging_started_count"] = int(charging_started_count)
        info["charging_recovered_count"] = int(charging_recovered_count)
        info["system_backlog_penalty"] = float(system_backlog_penalty)
        info["backlog_delta_penalty"] = float(backlog_delta_penalty)
        info["node_pressure_penalty"] = float(node_pressure_penalty)
        info["remote_cloud_penalty"] = float(remote_cloud_penalty)
        info["congestion_relief_bonus"] = float(congestion_relief_bonus)
        info["cloud_misuse_penalty"] = float(cloud_misuse_penalty)
        info["locality_bonus"] = float(locality_bonus)
        info["slack_bonus"] = float(slack_bonus)
        info["reward_mode"] = self.reward_mode
        info["inferred_scenario"] = inferred_scenario
        info["dynamic_queue_penalty"] = float(dynamic_queue_penalty)
        info["dynamic_latency_penalty"] = float(dynamic_latency_penalty)
        info["dynamic_balance_bonus"] = float(dynamic_balance_bonus)
        if self.reward_mode == "dynamic_context":
            for name, value in context_factors.items():
                info[f"context_{name}"] = float(value)
            for name, value in dynamic_weights.items():
                info[f"dynamic_{name}_coef"] = float(value)
            for name, value in scenario_scores.items():
                info[f"scenario_{name}_score"] = float(value)

        obs = (
            self._get_obs()
            if self.current_task is not None
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

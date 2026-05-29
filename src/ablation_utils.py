from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np

from src.envs.multi_robot_scheduler_env import SchedulerEnv


def deep_update(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return a recursively updated copy of a config dictionary."""
    result = deepcopy(base)
    if not overrides:
        return result

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


class RobotStateAblationWrapper(gym.ObservationWrapper):
    """Mask robot-state fields while keeping the original observation shape."""

    def __init__(self, env: SchedulerEnv):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.array(observation, dtype=np.float32, copy=True)
        env = self.unwrapped
        num_nodes = int(env.num_nodes)
        num_robots = int(env.num_robots)

        # Current robot id, energy, local CPU, and queue pressure.
        obs[4] = 0.0
        obs[5] = float(env.max_robot_energy)
        obs[6] = float(env.max_robot_local_cpu)
        obs[7] = 0.0

        robot_start = 8 + num_nodes * 3
        obs[robot_start : robot_start + num_robots] = float(env.max_robot_energy)
        obs[robot_start + num_robots : robot_start + 2 * num_robots] = float(
            env.max_robot_local_cpu
        )
        obs[robot_start + 2 * num_robots : robot_start + 3 * num_robots] = 0.0
        return obs


def make_scheduler_env(
    env_cfg: dict[str, Any],
    *,
    observation_ablation: str | None = None,
) -> gym.Env:
    env: gym.Env = SchedulerEnv(env_cfg)
    if observation_ablation in {None, "", "none"}:
        return env
    if observation_ablation == "robot_state":
        return RobotStateAblationWrapper(env)
    raise ValueError(f"Unknown observation ablation: {observation_ablation}")

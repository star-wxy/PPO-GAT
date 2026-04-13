from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.baselines.greedy_cpu import GreedyCPUPolicy
from src.baselines.random_policy import RandomPolicy
from src.baselines.round_robin import RoundRobinPolicy
from src.envs.scheduler_env import SchedulerEnv
from src.utils.config import load_yaml
from src.utils.display_names import get_display_name


def evaluate_policy(policy_name: str, policy, env: SchedulerEnv, max_steps: int) -> dict:
    print(f">>> evaluating policy: {policy_name}", flush=True)

    obs, _ = env.reset()

    total_reward = 0.0
    total_compute_time = 0.0
    total_queue_delay = 0.0
    total_network_latency = 0.0
    total_time = 0.0
    total_energy_cost = 0.0
    total_deadline_penalty = 0.0
    total_overload_penalty = 0.0
    step_count = 0

    for _ in range(max_steps):
        if policy_name == "ppo":
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = int(policy.predict(obs))

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_compute_time += info.get("compute_time", 0.0)
        total_queue_delay += info.get("queue_delay", 0.0)
        total_network_latency += info.get("network_latency", 0.0)
        total_time += info.get("total_time", 0.0)
        total_energy_cost += info.get("energy_cost", 0.0)
        total_deadline_penalty += info.get("deadline_penalty", 0.0)
        total_overload_penalty += info.get("overload_penalty", 0.0)

        step_count += 1
        if terminated or truncated:
            break

    result = {
        "policy": get_display_name(policy_name),
        "steps": step_count,
        "avg_reward": total_reward / max(step_count, 1),
        "total_reward": total_reward,
        "avg_compute_time": total_compute_time / max(step_count, 1),
        "avg_queue_delay": total_queue_delay / max(step_count, 1),
        "avg_network_latency": total_network_latency / max(step_count, 1),
        "avg_total_time": total_time / max(step_count, 1),
        "avg_energy_cost": total_energy_cost / max(step_count, 1),
        "avg_deadline_penalty": total_deadline_penalty / max(step_count, 1),
        "avg_overload_penalty": total_overload_penalty / max(step_count, 1),
    }

    print(f">>> evaluation finished: {policy_name} -> {result}", flush=True)
    return result


def main():
    print(">>> compare_baselines.py started", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train_plain_ppo.yaml")
    print(">>> configs loaded", flush=True)

    max_steps = env_cfg["max_steps"]
    results = []

    model_path = Path(train_cfg["checkpoint_dir"]) / train_cfg["model_name"]
    print(f">>> PPO model path: {model_path}", flush=True)
    print(f">>> model file exists: {model_path.with_suffix('.zip').exists()}", flush=True)

    ppo_env = SchedulerEnv(env_cfg)
    ppo_model = PPO.load(model_path)
    results.append(evaluate_policy("ppo", ppo_model, ppo_env, max_steps))

    random_env = SchedulerEnv(env_cfg)
    random_policy = RandomPolicy(action_dim=env_cfg["num_nodes"])
    results.append(evaluate_policy("random", random_policy, random_env, max_steps))

    rr_env = SchedulerEnv(env_cfg)
    rr_policy = RoundRobinPolicy(action_dim=env_cfg["num_nodes"])
    results.append(evaluate_policy("round_robin", rr_policy, rr_env, max_steps))

    greedy_env = SchedulerEnv(env_cfg)
    greedy_policy = GreedyCPUPolicy(num_nodes=env_cfg["num_nodes"])
    results.append(evaluate_policy("greedy_cpu", greedy_policy, greedy_env, max_steps))

    df = pd.DataFrame(results)
    print("\n=== baseline comparison results ===", flush=True)
    print(df.to_string(index=False), flush=True)

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "policy_baseline_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n>>> results saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

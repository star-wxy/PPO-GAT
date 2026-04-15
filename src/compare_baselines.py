from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.baselines.greedy_cpu import GreedyCPUPolicy
from src.baselines.random_policy import RandomPolicy
from src.baselines.round_robin import RoundRobinPolicy
from src.envs.multi_robot_scheduler_env import SchedulerEnv
from src.utils.config import load_yaml
from src.utils.display_names import get_display_name


EVAL_SEEDS = [11, 23, 37, 41, 53]
METRIC_KEYS = [
    "steps",
    "avg_reward",
    "total_reward",
    "avg_compute_time",
    "avg_queue_delay",
    "avg_network_latency",
    "avg_total_time",
    "avg_energy_cost",
    "avg_deadline_penalty",
    "avg_overload_penalty",
]


def evaluate_single_seed(policy_name: str, policy, env_cfg: dict, max_steps: int, eval_seed: int) -> dict:
    env = SchedulerEnv(env_cfg)
    obs, _ = env.reset(seed=eval_seed)

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

    return {
        "policy": policy_name,
        "seed": eval_seed,
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


def aggregate_seed_results(policy_name: str, seed_results: list[dict]) -> dict:
    aggregated = {"policy": get_display_name(policy_name), "eval_runs": len(seed_results)}

    for metric in METRIC_KEYS:
        values = np.array([result[metric] for result in seed_results], dtype=np.float64)
        aggregated[metric] = float(values.mean())
        aggregated[f"{metric}_std"] = float(values.std(ddof=0))

    return aggregated


def evaluate_policy(policy_name: str, policy, env_cfg: dict, max_steps: int) -> tuple[dict, pd.DataFrame]:
    print(f">>> evaluating policy: {policy_name}", flush=True)

    seed_results = []
    for eval_seed in EVAL_SEEDS:
        print(f">>> seed run: {policy_name} / seed={eval_seed}", flush=True)
        seed_results.append(evaluate_single_seed(policy_name, policy, env_cfg, max_steps, eval_seed))

    summary = aggregate_seed_results(policy_name, seed_results)
    print(f">>> evaluation summary: {summary}", flush=True)
    print(flush=True)
    return summary, pd.DataFrame(seed_results)


def main():
    print(">>> compare_baselines.py started", flush=True)
    print(f">>> evaluation seeds: {EVAL_SEEDS}", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train_plain_ppo.yaml")
    print(">>> configs loaded", flush=True)

    max_steps = env_cfg["max_steps"]
    summary_results = []
    per_seed_frames = []

    model_path = Path(train_cfg["checkpoint_dir"]) / train_cfg["model_name"]
    print(f">>> PPO model path: {model_path}", flush=True)
    print(f">>> model file exists: {model_path.with_suffix('.zip').exists()}", flush=True)

    ppo_model = PPO.load(model_path)
    ppo_summary, ppo_seed_df = evaluate_policy("ppo", ppo_model, env_cfg, max_steps)
    summary_results.append(ppo_summary)
    per_seed_frames.append(ppo_seed_df)

    random_policy = RandomPolicy(action_dim=env_cfg["num_nodes"])
    random_summary, random_seed_df = evaluate_policy("random", random_policy, env_cfg, max_steps)
    summary_results.append(random_summary)
    per_seed_frames.append(random_seed_df)

    rr_policy = RoundRobinPolicy(action_dim=env_cfg["num_nodes"])
    rr_summary, rr_seed_df = evaluate_policy("round_robin", rr_policy, env_cfg, max_steps)
    summary_results.append(rr_summary)
    per_seed_frames.append(rr_seed_df)

    greedy_policy = GreedyCPUPolicy(num_nodes=env_cfg["num_nodes"])
    greedy_summary, greedy_seed_df = evaluate_policy("greedy_cpu", greedy_policy, env_cfg, max_steps)
    summary_results.append(greedy_summary)
    per_seed_frames.append(greedy_seed_df)

    summary_df = pd.DataFrame(summary_results).sort_values(by="avg_reward", ascending=False)
    per_seed_df = pd.concat(per_seed_frames, ignore_index=True)
    per_seed_df["policy"] = per_seed_df["policy"].map(get_display_name)

    print("\n=== baseline comparison results (mean/std) ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "policy_baseline_comparison.csv"
    per_seed_csv_path = output_dir / "policy_baseline_comparison_per_seed.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    per_seed_df.to_csv(per_seed_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n>>> summary results saved to: {summary_csv_path}", flush=True)
    print(f">>> per-seed results saved to: {per_seed_csv_path}", flush=True)


if __name__ == "__main__":
    main()

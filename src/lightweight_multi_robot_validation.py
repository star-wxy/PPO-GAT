from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.baselines.greedy_cpu import GreedyCPUPolicy
from src.envs.multi_robot_scheduler_env import SchedulerEnv
from src.utils.config import load_yaml


def load_policy(checkpoint_path: Path):
    if checkpoint_path.exists():
        try:
            print(f">>> using trained model: {checkpoint_path}", flush=True)
            return "ppo_gat_scoring", PPO.load(checkpoint_path)
        except Exception as exc:
            print(f">>> failed to load scoring checkpoint, fallback to GreedyCPU: {exc}", flush=True)
            return "greedy_cpu_fallback", None

    print(f">>> scoring checkpoint not found, fallback to GreedyCPU: {checkpoint_path}", flush=True)
    return "greedy_cpu_fallback", None


def main():
    print(">>> lightweight_multi_robot_validation.py started", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train_scoring_gat.yaml")

    checkpoint_path = Path(train_cfg["checkpoint_dir"]) / f"{train_cfg['model_name']}.zip"
    policy_name, model = load_policy(checkpoint_path)

    env = SchedulerEnv(env_cfg)
    fallback_policy = GreedyCPUPolicy(num_nodes=env_cfg["num_nodes"])

    obs, info = env.reset(seed=20260415)
    events = []
    total_reward = 0.0
    initial_backlog = len(env.pending_tasks) + (1 if env.current_task is not None else 0)

    for step_idx in range(env_cfg["max_steps"]):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = int(fallback_policy.predict(obs))

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        events.append(
            {
                "step": step_idx + 1,
                "policy": policy_name,
                "task_id": step_info.get("task_id"),
                "source_robot_id": step_info.get("source_robot_id"),
                "task_type": step_info.get("task_type"),
                "task_size": step_info.get("task_size"),
                "task_deadline": step_info.get("task_deadline"),
                "task_priority": step_info.get("task_priority"),
                "chosen_node": step_info.get("chosen_node"),
                "node_type": step_info.get("node_type"),
                "robot_anchor_node": step_info.get("robot_anchor_node"),
                "topology_distance": step_info.get("topology_distance"),
                "pending_tasks": step_info.get("pending_tasks"),
                "backlog_before_step": step_info.get("backlog_before_step"),
                "backlog_after_enqueue": step_info.get("backlog_after_enqueue"),
                "backlog_after_step": step_info.get("backlog_after_step"),
                "backlog_delta": step_info.get("backlog_delta"),
                "new_tasks_generated": step_info.get("new_tasks_generated"),
                "generated_tasks_total": step_info.get("generated_tasks_total"),
                "completed_tasks_total": step_info.get("completed_tasks_total"),
                "compute_time": step_info.get("compute_time"),
                "queue_delay": step_info.get("queue_delay"),
                "transfer_latency": step_info.get("transfer_latency"),
                "network_latency": step_info.get("network_latency"),
                "total_time": step_info.get("total_time"),
                "energy_cost": step_info.get("energy_cost"),
                "deadline_penalty": step_info.get("deadline_penalty"),
                "overload_penalty": step_info.get("overload_penalty"),
                "system_backlog_penalty": step_info.get("system_backlog_penalty"),
                "node_pressure_penalty": step_info.get("node_pressure_penalty"),
                "remote_cloud_penalty": step_info.get("remote_cloud_penalty"),
                "locality_bonus": step_info.get("locality_bonus"),
                "slack_bonus": step_info.get("slack_bonus"),
                "current_robot_energy": step_info.get("current_robot_energy"),
                "reward": reward,
            }
        )

        if terminated or truncated:
            break

    events_df = pd.DataFrame(events)
    generated_tasks_total = int(events_df["generated_tasks_total"].iloc[-1]) if not events_df.empty else 0
    completed_tasks_total = int(events_df["completed_tasks_total"].iloc[-1]) if not events_df.empty else 0
    final_backlog = int(events_df["backlog_after_step"].iloc[-1]) if not events_df.empty else initial_backlog
    backlog_growth = final_backlog - initial_backlog
    completion_ratio = (
        completed_tasks_total / generated_tasks_total if generated_tasks_total > 0 else 0.0
    )
    positive_backlog_ratio = (
        float((events_df["backlog_delta"] > 0).mean()) if not events_df.empty else 0.0
    )
    cloud_offload_ratio = (
        float((events_df["node_type"] == "cloud").mean()) if not events_df.empty else 0.0
    )

    summary = {
        "policy": policy_name,
        "steps": len(events_df),
        "total_reward": float(total_reward),
        "avg_reward": float(events_df["reward"].mean()) if not events_df.empty else 0.0,
        "avg_total_time": float(events_df["total_time"].mean()) if not events_df.empty else 0.0,
        "avg_queue_delay": float(events_df["queue_delay"].mean()) if not events_df.empty else 0.0,
        "avg_energy_cost": float(events_df["energy_cost"].mean()) if not events_df.empty else 0.0,
        "avg_overload_penalty": float(events_df["overload_penalty"].mean()) if not events_df.empty else 0.0,
        "avg_pending_tasks": float(events_df["pending_tasks"].mean()) if not events_df.empty else 0.0,
        "initial_backlog": int(initial_backlog),
        "final_backlog": int(final_backlog),
        "backlog_growth": int(backlog_growth),
        "avg_backlog_delta": float(events_df["backlog_delta"].mean()) if not events_df.empty else 0.0,
        "positive_backlog_ratio": float(positive_backlog_ratio),
        "generated_tasks_total": int(generated_tasks_total),
        "completed_tasks_total": int(completed_tasks_total),
        "completion_ratio": float(completion_ratio),
        "cloud_offload_ratio": float(cloud_offload_ratio),
        "avg_system_backlog_penalty": float(events_df["system_backlog_penalty"].mean())
        if not events_df.empty
        else 0.0,
        "avg_node_pressure_penalty": float(events_df["node_pressure_penalty"].mean())
        if not events_df.empty
        else 0.0,
        "avg_remote_cloud_penalty": float(events_df["remote_cloud_penalty"].mean())
        if not events_df.empty
        else 0.0,
        "avg_locality_bonus": float(events_df["locality_bonus"].mean()) if not events_df.empty else 0.0,
        "avg_slack_bonus": float(events_df["slack_bonus"].mean()) if not events_df.empty else 0.0,
    }
    summary_df = pd.DataFrame([summary])

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "lightweight_multi_robot_validation_events.csv"
    summary_path = output_dir / "lightweight_multi_robot_validation_summary.csv"
    events_df.to_csv(events_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=== lightweight multi-robot validation summary ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\n>>> event log saved to: {events_path}", flush=True)
    print(f">>> summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

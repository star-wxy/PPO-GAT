from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.envs.scheduler_env import SchedulerEnv
from src.utils.config import load_yaml
from src.utils.display_names import get_display_name


def adapt_observation_for_model(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    current_dim = int(obs.shape[0])
    if current_dim == expected_dim:
        return obs

    if current_dim == expected_dim + 2:
        legacy_obs = np.concatenate([obs[:3], obs[5:]]).astype(np.float32, copy=False)
        if legacy_obs.shape[0] == expected_dim:
            return legacy_obs

    if current_dim + 2 == expected_dim:
        raise ValueError(
            f"Model expects observation dim {expected_dim}, but env returned legacy dim {current_dim}."
        )

    raise ValueError(
        f"Unsupported observation shape mismatch: env={current_dim}, model={expected_dim}."
    )


def evaluate_model(model_name: str, model_path: Path, env_cfg: dict) -> dict:
    print(f">>> evaluating model: {model_name}", flush=True)
    print(f">>> model path: {model_path}", flush=True)

    env = SchedulerEnv(env_cfg)
    model = PPO.load(model_path)
    expected_obs_dim = int(model.observation_space.shape[0])

    obs, _ = env.reset()

    total_reward = 0.0
    total_compute_time = 0.0
    total_queue_delay = 0.0
    total_network_latency = 0.0
    total_total_time = 0.0
    total_energy_cost = 0.0
    total_deadline_penalty = 0.0
    total_overload_penalty = 0.0
    step_count = 0

    for _ in range(env_cfg["max_steps"]):
        model_obs = adapt_observation_for_model(obs, expected_obs_dim)
        action, _ = model.predict(model_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward
        total_compute_time += info.get("compute_time", 0.0)
        total_queue_delay += info.get("queue_delay", 0.0)
        total_network_latency += info.get("network_latency", 0.0)
        total_total_time += info.get("total_time", 0.0)
        total_energy_cost += info.get("energy_cost", 0.0)
        total_deadline_penalty += info.get("deadline_penalty", 0.0)
        total_overload_penalty += info.get("overload_penalty", 0.0)

        step_count += 1
        if terminated or truncated:
            break

    result = {
        "model": model_name,
        "steps": step_count,
        "avg_reward": total_reward / max(step_count, 1),
        "total_reward": total_reward,
        "avg_compute_time": total_compute_time / max(step_count, 1),
        "avg_queue_delay": total_queue_delay / max(step_count, 1),
        "avg_network_latency": total_network_latency / max(step_count, 1),
        "avg_total_time": total_total_time / max(step_count, 1),
        "avg_energy_cost": total_energy_cost / max(step_count, 1),
        "avg_deadline_penalty": total_deadline_penalty / max(step_count, 1),
        "avg_overload_penalty": total_overload_penalty / max(step_count, 1),
    }

    print(f">>> evaluation finished: {model_name}", flush=True)
    print(result, flush=True)
    print(flush=True)
    return result


def build_model_candidates(scoring_cfg: dict, naive_cfg: dict | None) -> dict[str, Path]:
    candidates = {
        "ppo_baseline": Path(scoring_cfg["checkpoint_dir"]) / "ppo_baseline.zip",
        "ppo_gat_scoring": Path(scoring_cfg["checkpoint_dir"]) / f"{scoring_cfg['model_name']}.zip",
    }

    if naive_cfg is not None:
        candidates["ppo_gat_naive"] = (
            Path(naive_cfg["checkpoint_dir"]) / f"{naive_cfg['model_name']}.zip"
        )

    return candidates


def main():
    print(">>> compare_ppo_models.py started", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train_scoring_gat.yaml")
    naive_cfg_path = Path("configs/train_naive_gat.yaml")
    naive_cfg = load_yaml(str(naive_cfg_path)) if naive_cfg_path.exists() else None
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    models_to_compare = build_model_candidates(train_cfg, naive_cfg)

    for model_name, model_path in models_to_compare.items():
        if not model_path.exists():
            print(f">>> skipping {model_name}, file not found: {model_path}", flush=True)
            continue

        try:
            result = evaluate_model(model_name, model_path, env_cfg)
            result["model"] = get_display_name(model_name)
            results.append(result)
        except Exception as exc:
            print(f">>> skipping {model_name}, evaluation failed: {exc}", flush=True)

    if not results:
        raise FileNotFoundError(
            "No comparable model files were found. Train the models first and verify the paths."
        )

    df = pd.DataFrame(results).sort_values(by="avg_reward", ascending=False)

    print("=== PPO comparison results ===", flush=True)
    print(df.to_string(index=False), flush=True)

    csv_path = output_dir / "ppo_gat_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n>>> results saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

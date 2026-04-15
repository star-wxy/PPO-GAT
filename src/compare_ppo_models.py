from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

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


def evaluate_single_seed(
    model_name: str,
    model_path: Path,
    env_cfg: dict,
    eval_seed: int,
) -> dict:
    env = SchedulerEnv(env_cfg)
    model = PPO.load(model_path)
    expected_obs_dim = int(model.observation_space.shape[0])

    obs, _ = env.reset(seed=eval_seed)

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

    return {
        "model": model_name,
        "seed": eval_seed,
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


def aggregate_seed_results(model_name: str, seed_results: list[dict]) -> dict:
    aggregated = {"model": get_display_name(model_name), "eval_runs": len(seed_results)}

    for metric in METRIC_KEYS:
        values = np.array([result[metric] for result in seed_results], dtype=np.float64)
        aggregated[metric] = float(values.mean())
        aggregated[f"{metric}_std"] = float(values.std(ddof=0))

    return aggregated


def evaluate_model(model_name: str, model_path: Path, env_cfg: dict) -> tuple[dict, pd.DataFrame]:
    print(f">>> evaluating model: {model_name}", flush=True)
    print(f">>> model path: {model_path}", flush=True)

    seed_results = []
    for eval_seed in EVAL_SEEDS:
        print(f">>> seed run: {model_name} / seed={eval_seed}", flush=True)
        seed_results.append(evaluate_single_seed(model_name, model_path, env_cfg, eval_seed))

    summary = aggregate_seed_results(model_name, seed_results)
    print(f">>> evaluation summary: {summary}", flush=True)
    print(flush=True)
    return summary, pd.DataFrame(seed_results)


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
    print(f">>> evaluation seeds: {EVAL_SEEDS}", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train_scoring_gat.yaml")
    naive_cfg_path = Path("configs/train_naive_gat.yaml")
    naive_cfg = load_yaml(str(naive_cfg_path)) if naive_cfg_path.exists() else None
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_results = []
    per_seed_frames = []
    models_to_compare = build_model_candidates(train_cfg, naive_cfg)

    for model_name, model_path in models_to_compare.items():
        if not model_path.exists():
            print(f">>> skipping {model_name}, file not found: {model_path}", flush=True)
            continue

        try:
            summary, seed_df = evaluate_model(model_name, model_path, env_cfg)
            summary_results.append(summary)
            per_seed_frames.append(seed_df)
        except Exception as exc:
            print(f">>> skipping {model_name}, evaluation failed: {exc}", flush=True)

    if not summary_results:
        raise FileNotFoundError(
            "No comparable model files were found. Train the models first and verify the paths."
        )

    summary_df = pd.DataFrame(summary_results).sort_values(by="avg_reward", ascending=False)
    per_seed_df = pd.concat(per_seed_frames, ignore_index=True)
    per_seed_df["model"] = per_seed_df["model"].map(get_display_name)

    print("=== PPO comparison results (mean/std) ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    summary_csv_path = output_dir / "ppo_gat_comparison.csv"
    per_seed_csv_path = output_dir / "ppo_gat_comparison_per_seed.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    per_seed_df.to_csv(per_seed_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n>>> summary results saved to: {summary_csv_path}", flush=True)
    print(f">>> per-seed results saved to: {per_seed_csv_path}", flush=True)


if __name__ == "__main__":
    main()

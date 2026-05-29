import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.ablation_utils import deep_update, make_scheduler_env
from src.utils.config import load_yaml
from src.utils.display_names import get_display_name


DEFAULT_EVAL_SEEDS = [11, 23, 37, 41, 53]
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
    "avg_context_load",
    "avg_context_energy_risk",
    "avg_context_urgency",
    "avg_context_comm_risk",
    "avg_charging_robots",
    "total_charging_starts",
    "total_charging_recoveries",
    "local_node_ratio",
    "edge_node_ratio",
    "regional_node_ratio",
    "cloud_node_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/ablations/mixed_context_ablation_manifest.yaml")
    parser.add_argument("--use-best-model", action="store_true")
    return parser.parse_args()


def adapt_observation_for_model(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    current_dim = int(obs.shape[0])
    if current_dim == expected_dim:
        return obs
    if current_dim == expected_dim + 2:
        legacy_obs = np.concatenate([obs[:3], obs[5:]]).astype(np.float32, copy=False)
        if legacy_obs.shape[0] == expected_dim:
            return legacy_obs
    raise ValueError(f"Unsupported observation shape mismatch: env={current_dim}, model={expected_dim}.")


def train_config_model_path(train_cfg: dict, use_best_model: bool) -> Path:
    if use_best_model:
        return Path(train_cfg["best_model_dir"]) / "best_model.zip"
    return Path(train_cfg["checkpoint_dir"]) / f"{train_cfg['model_name']}.zip"


def evaluate_single_seed(
    model_name: str,
    model_path: Path,
    env_cfg: dict,
    eval_seed: int,
    *,
    observation_ablation: str | None = None,
) -> dict:
    env = make_scheduler_env(env_cfg, observation_ablation=observation_ablation)
    model = PPO.load(model_path)
    expected_obs_dim = int(model.observation_space.shape[0])
    obs, _ = env.reset(seed=eval_seed)

    totals = {
        "reward": 0.0,
        "compute_time": 0.0,
        "queue_delay": 0.0,
        "network_latency": 0.0,
        "total_time": 0.0,
        "energy_cost": 0.0,
        "deadline_penalty": 0.0,
        "overload_penalty": 0.0,
        "context_load": 0.0,
        "context_energy_risk": 0.0,
        "context_urgency": 0.0,
        "context_comm_risk": 0.0,
        "charging_robots": 0.0,
        "charging_starts": 0.0,
        "charging_recoveries": 0.0,
    }
    node_type_counts = {"local": 0, "edge": 0, "regional": 0, "cloud": 0}
    step_count = 0

    for _ in range(env_cfg["max_steps"]):
        model_obs = adapt_observation_for_model(obs, expected_obs_dim)
        action, _ = model.predict(model_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        totals["reward"] += float(reward)
        totals["compute_time"] += info.get("compute_time", 0.0)
        totals["queue_delay"] += info.get("queue_delay", 0.0)
        totals["network_latency"] += info.get("network_latency", 0.0)
        totals["total_time"] += info.get("total_time", 0.0)
        totals["energy_cost"] += info.get("energy_cost", 0.0)
        totals["deadline_penalty"] += info.get("deadline_penalty", 0.0)
        totals["overload_penalty"] += info.get("overload_penalty", 0.0)
        totals["context_load"] += info.get("context_load_context", 0.0)
        totals["context_energy_risk"] += info.get("context_energy_risk", 0.0)
        totals["context_urgency"] += info.get("context_urgency_level", 0.0)
        totals["context_comm_risk"] += info.get("context_comm_risk", 0.0)
        totals["charging_robots"] += info.get("charging_robots", 0.0)
        totals["charging_starts"] += info.get("charging_started_count", 0.0)
        totals["charging_recoveries"] += info.get("charging_recovered_count", 0.0)
        node_type = str(info.get("node_type", ""))
        if node_type in node_type_counts:
            node_type_counts[node_type] += 1

        step_count += 1
        if terminated or truncated:
            break

    denom = max(step_count, 1)
    return {
        "model": model_name,
        "seed": eval_seed,
        "steps": step_count,
        "avg_reward": totals["reward"] / denom,
        "total_reward": totals["reward"],
        "avg_compute_time": totals["compute_time"] / denom,
        "avg_queue_delay": totals["queue_delay"] / denom,
        "avg_network_latency": totals["network_latency"] / denom,
        "avg_total_time": totals["total_time"] / denom,
        "avg_energy_cost": totals["energy_cost"] / denom,
        "avg_deadline_penalty": totals["deadline_penalty"] / denom,
        "avg_overload_penalty": totals["overload_penalty"] / denom,
        "avg_context_load": totals["context_load"] / denom,
        "avg_context_energy_risk": totals["context_energy_risk"] / denom,
        "avg_context_urgency": totals["context_urgency"] / denom,
        "avg_context_comm_risk": totals["context_comm_risk"] / denom,
        "avg_charging_robots": totals["charging_robots"] / denom,
        "total_charging_starts": totals["charging_starts"],
        "total_charging_recoveries": totals["charging_recoveries"],
        "local_node_ratio": node_type_counts["local"] / denom,
        "edge_node_ratio": node_type_counts["edge"] / denom,
        "regional_node_ratio": node_type_counts["regional"] / denom,
        "cloud_node_ratio": node_type_counts["cloud"] / denom,
    }


def aggregate_seed_results(model_name: str, seed_results: list[dict]) -> dict:
    aggregated = {"model": get_display_name(model_name), "eval_runs": len(seed_results)}
    for metric in METRIC_KEYS:
        values = np.array([result[metric] for result in seed_results], dtype=np.float64)
        aggregated[metric] = float(values.mean())
        aggregated[f"{metric}_std"] = float(values.std(ddof=0))
    return aggregated


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    eval_seeds = manifest.get("eval_seeds", DEFAULT_EVAL_SEEDS)
    output_prefix = manifest.get("output_prefix", "ablation_comparison")
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    default_env_cfg = load_yaml(manifest["eval_env_config"])
    summary_results = []
    per_seed_frames = []

    print(">>> compare_ablation_models.py started", flush=True)
    print(f">>> manifest: {args.manifest}", flush=True)
    print(f">>> eval seeds: {eval_seeds}", flush=True)

    for entry in manifest["models"]:
        model_name = entry["name"]
        train_cfg = load_yaml(entry["train_config"])
        model_path = train_config_model_path(train_cfg, args.use_best_model)
        if not model_path.exists():
            print(f">>> skipping {model_name}, file not found: {model_path}", flush=True)
            continue

        eval_env_config = entry.get("eval_env_config", manifest["eval_env_config"])
        env_cfg = load_yaml(eval_env_config) if eval_env_config != manifest["eval_env_config"] else default_env_cfg
        env_cfg = deep_update(env_cfg, entry.get("eval_env_overrides"))
        observation_ablation = entry.get(
            "eval_observation_ablation",
            train_cfg.get("observation_ablation", "none"),
        )

        print(f">>> evaluating {model_name}: {model_path}", flush=True)
        seed_results = [
            evaluate_single_seed(
                model_name,
                model_path,
                env_cfg,
                int(seed),
                observation_ablation=observation_ablation,
            )
            for seed in eval_seeds
        ]
        summary_results.append(aggregate_seed_results(model_name, seed_results))
        per_seed_frames.append(pd.DataFrame(seed_results))

    if not summary_results:
        raise FileNotFoundError("No ablation model checkpoints were found.")

    summary_df = pd.DataFrame(summary_results).sort_values(by="avg_reward", ascending=False)
    per_seed_df = pd.concat(per_seed_frames, ignore_index=True)
    per_seed_df["model"] = per_seed_df["model"].map(get_display_name)

    summary_path = output_dir / f"{output_prefix}.csv"
    per_seed_path = output_dir / f"{output_prefix}_per_seed.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    per_seed_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    print("=== Ablation comparison results (mean/std) ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f">>> summary results saved to: {summary_path}", flush=True)
    print(f">>> per-seed results saved to: {per_seed_path}", flush=True)


if __name__ == "__main__":
    main()

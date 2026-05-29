import argparse
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.ablation_utils import deep_update, make_scheduler_env
from src.models.feature_extractors import GATFeatureExtractor
from src.models.node_scoring_gat_extractor import NodeScoringGATFeatureExtractor
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/scenarios/mixed_context_20r_10n.yaml")
    parser.add_argument("--train-config", required=True)
    return parser.parse_args()


def build_policy_kwargs(env_cfg: dict, train_cfg: dict) -> dict:
    model_variant = train_cfg.get("model_variant", "scoring_gat")
    net_arch = train_cfg.get("net_arch")

    if model_variant == "plain_ppo":
        return dict(
            net_arch=net_arch or dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.ReLU,
        )

    if model_variant == "naive_gat":
        return dict(
            features_extractor_class=GATFeatureExtractor,
            features_extractor_kwargs=dict(
                num_nodes=env_cfg["num_nodes"],
                num_robots=env_cfg["num_robots"],
                features_dim=int(train_cfg.get("features_dim", 96)),
                hidden_dim=int(train_cfg.get("hidden_dim", 32)),
                gat_heads=int(train_cfg.get("gat_heads", 2)),
            ),
            net_arch=net_arch or dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.ReLU,
        )

    if model_variant in {"scoring_gat", "scoring_no_heuristic"}:
        use_heuristic_score = bool(
            train_cfg.get("use_heuristic_score", model_variant != "scoring_no_heuristic")
        )
        return dict(
            features_extractor_class=NodeScoringGATFeatureExtractor,
            features_extractor_kwargs=dict(
                num_nodes=env_cfg["num_nodes"],
                num_robots=env_cfg["num_robots"],
                node_cpu_capacities=[node["cpu_capacity"] for node in env_cfg["nodes"]],
                node_latencies=[node["latency"] for node in env_cfg["nodes"]],
                node_energy_factors=[node["energy_factor"] for node in env_cfg["nodes"]],
                max_task_size=env_cfg["task"]["size_max"],
                max_task_deadline=env_cfg["task"]["deadline_max"],
                max_task_priority=env_cfg["task"]["priority_levels"],
                max_robot_energy=env_cfg["robot"]["init_energy"],
                max_robot_local_cpu=env_cfg["robot"]["local_cpu_max"],
                features_dim=int(train_cfg.get("features_dim", 128)),
                hidden_dim=int(train_cfg.get("hidden_dim", 48)),
                gat_heads=int(train_cfg.get("gat_heads", 2)),
                dropout=float(train_cfg.get("dropout", 0.03)),
                use_heuristic_score=use_heuristic_score,
                heuristic_gate_init=float(train_cfg.get("heuristic_gate_init", 0.85)),
            ),
            net_arch=net_arch or dict(pi=[64, 64], vf=[128, 64]),
            activation_fn=nn.SiLU,
        )

    raise ValueError(f"Unknown model_variant: {model_variant}")


def main():
    print(">>> train_ablation_model.py started", flush=True)

    args = parse_args()
    raw_env_cfg = load_yaml(args.env_config)
    train_cfg = load_yaml(args.train_config)
    env_cfg = deep_update(raw_env_cfg, train_cfg.get("env_overrides"))
    observation_ablation = train_cfg.get("observation_ablation", "none")
    print(f">>> env config: {args.env_config}", flush=True)
    print(f">>> train config: {args.train_config}", flush=True)
    print(f">>> model variant: {train_cfg.get('model_variant', 'scoring_gat')}", flush=True)
    print(f">>> observation ablation: {observation_ablation}", flush=True)

    set_seed(train_cfg["seed"])

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    tensorboard_log = Path(train_cfg["tensorboard_log"])
    best_model_dir = Path(train_cfg["best_model_dir"])
    eval_log_path = Path(train_cfg["eval_log_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_path.mkdir(parents=True, exist_ok=True)

    env = Monitor(make_scheduler_env(env_cfg, observation_ablation=observation_ablation))
    eval_env = Monitor(make_scheduler_env(env_cfg, observation_ablation=observation_ablation))
    policy_kwargs = build_policy_kwargs(env_cfg, train_cfg)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg["ent_coef"],
        vf_coef=train_cfg["vf_coef"],
        max_grad_norm=train_cfg["max_grad_norm"],
        n_epochs=train_cfg["n_epochs"],
        verbose=1,
        tensorboard_log=str(tensorboard_log),
        device=train_cfg["device"],
        policy_kwargs=policy_kwargs,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_path),
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=eval_callback)

    model_path = checkpoint_dir / train_cfg["model_name"]
    model.save(model_path)
    print(f">>> model saved to: {model_path}.zip", flush=True)
    print(f">>> best eval model saved under: {best_model_dir}", flush=True)


if __name__ == "__main__":
    main()

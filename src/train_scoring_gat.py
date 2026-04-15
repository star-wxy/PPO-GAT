from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.multi_robot_scheduler_env import SchedulerEnv
from src.models.node_scoring_gat_extractor import NodeScoringGATFeatureExtractor
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def main():
    print(">>> train_scoring_gat.py started", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train_scoring_gat.yaml")
    print(">>> configs loaded", flush=True)

    set_seed(train_cfg["seed"])
    print(f">>> seed set to {train_cfg['seed']}", flush=True)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    tensorboard_log = Path(train_cfg["tensorboard_log"])
    best_model_dir = Path(train_cfg["best_model_dir"])
    eval_log_path = Path(train_cfg["eval_log_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_path.mkdir(parents=True, exist_ok=True)
    print(">>> output directories ready", flush=True)

    env = Monitor(SchedulerEnv(env_cfg))
    eval_env = Monitor(SchedulerEnv(env_cfg))
    print(">>> environment created", flush=True)

    policy_kwargs = dict(
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
            features_dim=128,
            hidden_dim=48,
            gat_heads=2,
            dropout=0.03,
        ),
        net_arch=dict(
            pi=[64, 64],
            vf=[128, 64],
        ),
        activation_fn=nn.SiLU,
    )

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
    print(">>> node-scoring PPO+GAT model created, training starts", flush=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_path),
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=eval_callback,
    )

    model_path = checkpoint_dir / train_cfg["model_name"]
    model.save(model_path)

    print(f">>> model saved to: {model_path}.zip", flush=True)
    print(f">>> best eval model saved under: {best_model_dir}", flush=True)


if __name__ == "__main__":
    main()

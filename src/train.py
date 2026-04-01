from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.scheduler_env import SchedulerEnv
from src.models.feature_extractors import GATFeatureExtractor
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def main():
    print(">>> train.py 已启动", flush=True)

    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train.yaml")
    print(">>> 配置文件加载完成", flush=True)

    set_seed(train_cfg["seed"])
    print(f">>> 随机种子已设置: {train_cfg['seed']}", flush=True)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    tensorboard_log = Path(train_cfg["tensorboard_log"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    print(">>> 输出目录已创建", flush=True)

    env = SchedulerEnv(env_cfg)
    env = Monitor(env)
    print(">>> 环境创建完成", flush=True)

    policy_kwargs = dict(
        features_extractor_class=GATFeatureExtractor,
        features_extractor_kwargs=dict(
            num_nodes=env_cfg["num_nodes"],
            num_robots=env_cfg["num_robots"],
            features_dim=64,
            hidden_dim=32,
            gat_heads=4,
        ),
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        gamma=train_cfg["gamma"],
        verbose=1,
        tensorboard_log=str(tensorboard_log),
        device=train_cfg["device"],
        policy_kwargs=policy_kwargs,
    )
    print(">>> PPO+GAT 模型创建完成，开始训练", flush=True)

    model.learn(total_timesteps=train_cfg["total_timesteps"])

    model_path = checkpoint_dir / train_cfg["model_name"]
    model.save(model_path)

    print(f">>> 模型已保存到: {model_path}.zip", flush=True)


if __name__ == "__main__":
    main()
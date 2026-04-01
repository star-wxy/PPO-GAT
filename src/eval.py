from pathlib import Path

from stable_baselines3 import PPO

from src.envs.scheduler_env import SchedulerEnv
from src.utils.config import load_yaml


def main():
    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train.yaml")

    env = SchedulerEnv(env_cfg)

    model_path = Path(train_cfg["checkpoint_dir"]) / train_cfg["model_name"]
    print(f">>> 正在加载模型: {model_path}.zip", flush=True)

    model = PPO.load(model_path)

    obs, info = env.reset()
    total_reward = 0.0

    for step in range(env_cfg["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward

        print(
            f"step={step + 1}, action={action}, reward={reward:.4f}, "
            f"total_time={info.get('total_time', 0):.4f}, "
            f"energy={info.get('energy_cost', 0):.4f}"
        )

        if terminated or truncated:
            break

    print(f"\nEvaluation total reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()
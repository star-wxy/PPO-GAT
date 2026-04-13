DISPLAY_NAME_MAP = {
    "ppo": "PPO-Baseline",
    "ppo_baseline": "PPO-Baseline",
    "random": "Random-Policy",
    "round_robin": "RoundRobin-Policy",
    "greedy_cpu": "GreedyCPU-Policy",
    "ppo_gat_naive": "PPO-GAT-Naive",
    "ppo_gat_naive_best_eval": "PPO-GAT-Naive-BestEval",
    "ppo_gat_scoring": "PPO-GAT-Scoring",
    "ppo_gat_scoring_best_eval": "PPO-GAT-Scoring-BestEval",
}


def get_display_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name)

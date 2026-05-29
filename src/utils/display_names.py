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
    "ablation_full_scoring": "Full-Scoring",
    "ablation_no_gat": "No-GAT",
    "ablation_no_node_scoring": "No-NodeScoring",
    "ablation_no_heuristic_gate": "No-HeuristicGate",
    "ablation_fixed_reward": "Fixed-Reward",
    "ablation_no_robot_state": "No-RobotState",
    "ablation_no_congestion": "No-Congestion",
    "ablation_no_charging": "No-Charging",
}


def get_display_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name)

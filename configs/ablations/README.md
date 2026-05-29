# Mixed-context ablation experiments

These configs implement the ablation set used to justify the main method:

- `train_full_scoring_mixed_context.yaml`: full PPO-GAT-Scoring with dynamic context reward.
- `train_no_gat_mixed_context.yaml`: removes GAT and node scoring, using plain PPO.
- `train_no_node_scoring_mixed_context.yaml`: keeps naive GAT but removes explicit node scoring.
- `train_no_heuristic_gate_mixed_context.yaml`: keeps node scoring but removes heuristic score fusion.
- `train_fixed_reward_mixed_context.yaml`: keeps PPO-GAT-Scoring but trains with static reward weights.
- `train_no_robot_state_mixed_context.yaml`: masks robot-state observation fields during training/evaluation.
- `train_no_congestion_mixed_context.yaml`: trains without topology congestion propagation.
- `train_no_charging_mixed_context.yaml`: trains without low-energy charging/recovery dynamics.

The default configs are thesis-grade settings: `100000` training steps per ablation,
evaluation every `10000` steps, and `3` eval episodes per checkpoint.

Run all training and comparison:

```powershell
scripts/run_ablation_experiment.ps1
```

Only compare existing best checkpoints:

```powershell
scripts/run_ablation_experiment.ps1 -CompareOnly
```

Outputs:

- `outputs/results/ablation_mixed_context.csv`
- `outputs/results/ablation_mixed_context_per_seed.csv`

The comparison evaluates all policies on the full mixed-context environment by default.
The `No-RobotState` policy also receives masked robot-state observations during evaluation,
matching its training condition.

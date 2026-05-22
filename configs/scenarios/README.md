# Scenario configs for dynamic reward validation

These configs are isolated from the main 20-robot/10-node setup and do not
overwrite existing result files. They are intended for controlled validation of
the context-aware dynamic reward mode across these three PPO variants:

- PPO-Baseline
- PPO-GAT-Naive
- PPO-GAT-Scoring

## Files

- `high_load_24r_10n.yaml`
  - Stress target: high load and queue pressure.
  - Changes: 24 robots, higher task arrival rate, lower edge/regional CPU, lower bandwidth.
  - Expected dynamic response: higher `dynamic_overload_coef`, `dynamic_queue_coef`,
    and `dynamic_balance_coef`.

- `low_energy_20r_10n.yaml`
  - Stress target: low battery and energy-sensitive scheduling.
  - Changes: lower initial robot energy, higher node energy factors, lower bandwidth.
  - Expected dynamic response: higher `dynamic_energy_coef` and reduced
    `avg_energy_cost` under comparable policies.

- `emergency_20r_10n.yaml`
  - Stress target: urgent high-priority tasks.
  - Changes: tighter deadline range, higher task arrival rate, higher priority bias.
  - Expected dynamic response: higher `dynamic_deadline_coef`,
    `dynamic_queue_coef`, and `dynamic_slack_coef`.

- `high_latency_20r_12n.yaml`
  - Stress target: low bandwidth and high communication latency.
  - Changes: 12 nodes, lower bandwidth, higher per-hop and transmission latency,
    higher task transmission demand.
  - Expected dynamic response: higher `dynamic_latency_coef` and lower selected
    `network_latency` after scenario-specific training.

## Notes

Configs that keep `20r_10n` preserve the current observation/action dimensions.
Configs with a different robot or node count require scenario-specific training;
existing 20r/10n checkpoints are not dimension-compatible with those environments.

Each scenario has three dedicated train configs:

```text
train_plain_ppo_<scenario>.yaml
train_naive_gat_<scenario>.yaml
train_scoring_gat_<scenario>.yaml
```

Their output paths use `outputs/scenario_*`, so they do not overwrite the main
20r/10n checkpoints, best models, tensorboard logs, or evaluation logs.

For static-vs-dynamic ablation, copy the target scenario config and change:

```yaml
reward:
  mode: static
```

Use a different output prefix for every run so existing static results remain
untouched.

## Three-model comparison

After training the three scenario-specific models, compare them with:

```powershell
.\scripts\compare_ppo_models.ps1 `
  --env-config configs/scenarios/low_energy_20r_10n.yaml `
  --baseline-train-config configs/scenarios/train_plain_ppo_low_energy_20r_10n.yaml `
  --naive-train-config configs/scenarios/train_naive_gat_low_energy_20r_10n.yaml `
  --scoring-train-config configs/scenarios/train_scoring_gat_low_energy_20r_10n.yaml `
  --use-best-model `
  --output-prefix scenario_low_energy_ppo_models_dynamic
```

The output CSV includes model metrics and dynamic reward diagnostics such as
`avg_context_energy_risk`, `avg_dynamic_energy_coef`,
`avg_dynamic_deadline_coef`, and `avg_dynamic_latency_coef`.

Or train and compare the three PPO variants in one command:

```powershell
.\scripts\run_ppo_model_scenario.ps1 `
  -EnvConfig configs/scenarios/low_energy_20r_10n.yaml `
  -BaselineTrainConfig configs/scenarios/train_plain_ppo_low_energy_20r_10n.yaml `
  -NaiveTrainConfig configs/scenarios/train_naive_gat_low_energy_20r_10n.yaml `
  -ScoringTrainConfig configs/scenarios/train_scoring_gat_low_energy_20r_10n.yaml `
  -OutputPrefix scenario_low_energy_ppo_models_dynamic
```

Use `-CompareOnly` when the three scenario-specific checkpoints already exist.

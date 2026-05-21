$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$envConfig = "configs/env_20r_10n.yaml"
$baselineTrainConfig = "configs/train_plain_ppo_20r_10n.yaml"
$naiveTrainConfig = "configs/train_naive_gat_20r_10n.yaml"
$scoringTrainConfig = "configs/train_scoring_gat_20r_10n.yaml"

Write-Host ">>> 20r_10n experiment started"
Write-Host ">>> env config: $envConfig"

Write-Host ">>> [1/5] training PPO baseline"
& (Join-Path $scriptRoot "train_ppo_baseline.ps1") `
    --env-config $envConfig `
    --train-config $baselineTrainConfig

Write-Host ">>> [2/5] training naive PPO+GAT"
& (Join-Path $scriptRoot "train_ppo_naive.ps1") `
    --env-config $envConfig `
    --train-config $naiveTrainConfig

Write-Host ">>> [3/5] training node-scoring PPO+GAT"
& (Join-Path $scriptRoot "train_ppo_scoring.ps1") `
    --env-config $envConfig `
    --train-config $scoringTrainConfig

Write-Host ">>> [4/5] comparing final checkpoints"
& (Join-Path $scriptRoot "compare_ppo_models.ps1") `
    --env-config $envConfig `
    --baseline-train-config $baselineTrainConfig `
    --naive-train-config $naiveTrainConfig `
    --scoring-train-config $scoringTrainConfig `
    --output-prefix "ppo_gat_comparison_20r_10n"

Write-Host ">>> [5/5] comparing best models and running lightweight validation"
& (Join-Path $scriptRoot "compare_ppo_models.ps1") `
    --env-config $envConfig `
    --baseline-train-config $baselineTrainConfig `
    --naive-train-config $naiveTrainConfig `
    --scoring-train-config $scoringTrainConfig `
    --output-prefix "ppo_gat_comparison_20r_10n_best" `
    --use-best-model

& (Join-Path $scriptRoot "run_lightweight_multi_robot_validation.ps1") `
    --env-config $envConfig `
    --train-config $scoringTrainConfig `
    --output-prefix "lightweight_multi_robot_validation_20r_10n"

Write-Host ">>> 20r_10n experiment completed"
Write-Host ">>> results:"
Write-Host ">>> outputs/results/ppo_gat_comparison_20r_10n.csv"
Write-Host ">>> outputs/results/ppo_gat_comparison_20r_10n_best.csv"
Write-Host ">>> outputs/results/lightweight_multi_robot_validation_20r_10n_summary.csv"

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ">>> running PPO baseline training"
& (Join-Path $scriptRoot "train_ppo_baseline.ps1")

Write-Host ">>> running naive PPO+GAT training"
& (Join-Path $scriptRoot "train_ppo_naive.ps1")

Write-Host ">>> running node-scoring PPO+GAT training"
& (Join-Path $scriptRoot "train_ppo_scoring.ps1")

Write-Host ">>> running PPO and GAT strategy comparison"
& (Join-Path $scriptRoot "compare_ppo_models.ps1")

Write-Host ">>> generating charts"
& (Join-Path $scriptRoot "plot_results.ps1")

Write-Host ">>> GAT experiment pipeline completed"

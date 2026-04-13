$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ">>> running PPO baseline training"
& (Join-Path $scriptRoot "train_ppo_baseline.ps1")

Write-Host ">>> running baseline policy comparison"
& (Join-Path $scriptRoot "compare_baselines.ps1")

Write-Host ">>> generating charts"
& (Join-Path $scriptRoot "plot_results.ps1")

Write-Host ">>> baseline experiment pipeline completed"

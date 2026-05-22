param(
    [Parameter(Mandatory = $true)]
    [string]$EnvConfig,

    [Parameter(Mandatory = $true)]
    [string]$BaselineTrainConfig,

    [Parameter(Mandatory = $true)]
    [string]$NaiveTrainConfig,

    [Parameter(Mandatory = $true)]
    [string]$ScoringTrainConfig,

    [Parameter(Mandatory = $true)]
    [string]$OutputPrefix,

    [switch]$CompareOnly
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$powerShellExe = (Get-Process -Id $PID).Path

function Invoke-ScenarioChildScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $powerShellExe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Child script failed with exit code $LASTEXITCODE`: $ScriptPath"
    }
}

Write-Host ">>> PPO model scenario experiment started"
Write-Host ">>> env config: $EnvConfig"
Write-Host ">>> baseline train config: $BaselineTrainConfig"
Write-Host ">>> naive train config: $NaiveTrainConfig"
Write-Host ">>> scoring train config: $ScoringTrainConfig"

if (-not $CompareOnly) {
    Write-Host ">>> [1/4] training PPO baseline"
    Invoke-ScenarioChildScript `
        -ScriptPath (Join-Path $scriptRoot "train_ppo_baseline.ps1") `
        -Arguments @("--env-config", $EnvConfig, "--train-config", $BaselineTrainConfig)

    Write-Host ">>> [2/4] training naive PPO+GAT"
    Invoke-ScenarioChildScript `
        -ScriptPath (Join-Path $scriptRoot "train_ppo_naive.ps1") `
        -Arguments @("--env-config", $EnvConfig, "--train-config", $NaiveTrainConfig)

    Write-Host ">>> [3/4] training node-scoring PPO+GAT"
    Invoke-ScenarioChildScript `
        -ScriptPath (Join-Path $scriptRoot "train_ppo_scoring.ps1") `
        -Arguments @("--env-config", $EnvConfig, "--train-config", $ScoringTrainConfig)
} else {
    Write-Host ">>> CompareOnly enabled; skipping training"
}

Write-Host ">>> [4/4] comparing best PPO models"
Invoke-ScenarioChildScript `
    -ScriptPath (Join-Path $scriptRoot "compare_ppo_models.ps1") `
    -Arguments @(
        "--env-config", $EnvConfig,
        "--baseline-train-config", $BaselineTrainConfig,
        "--naive-train-config", $NaiveTrainConfig,
        "--scoring-train-config", $ScoringTrainConfig,
        "--use-best-model",
        "--output-prefix", $OutputPrefix
    )

Write-Host ">>> scenario experiment completed"
Write-Host ">>> outputs/results/$OutputPrefix.csv"
Write-Host ">>> outputs/results/${OutputPrefix}_per_seed.csv"

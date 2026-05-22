param(
    [switch]$CompareOnly
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$scenarios = @(
    @{
        Name = "low_energy_20r_10n"
        EnvConfig = "configs/scenarios/low_energy_20r_10n.yaml"
        BaselineTrainConfig = "configs/scenarios/train_plain_ppo_low_energy_20r_10n.yaml"
        NaiveTrainConfig = "configs/scenarios/train_naive_gat_low_energy_20r_10n.yaml"
        ScoringTrainConfig = "configs/scenarios/train_scoring_gat_low_energy_20r_10n.yaml"
        OutputPrefix = "scenario_low_energy_ppo_models_dynamic"
    },
    @{
        Name = "high_load_24r_10n"
        EnvConfig = "configs/scenarios/high_load_24r_10n.yaml"
        BaselineTrainConfig = "configs/scenarios/train_plain_ppo_high_load_24r_10n.yaml"
        NaiveTrainConfig = "configs/scenarios/train_naive_gat_high_load_24r_10n.yaml"
        ScoringTrainConfig = "configs/scenarios/train_scoring_gat_high_load_24r_10n.yaml"
        OutputPrefix = "scenario_high_load_ppo_models_dynamic"
    },
    @{
        Name = "emergency_20r_10n"
        EnvConfig = "configs/scenarios/emergency_20r_10n.yaml"
        BaselineTrainConfig = "configs/scenarios/train_plain_ppo_emergency_20r_10n.yaml"
        NaiveTrainConfig = "configs/scenarios/train_naive_gat_emergency_20r_10n.yaml"
        ScoringTrainConfig = "configs/scenarios/train_scoring_gat_emergency_20r_10n.yaml"
        OutputPrefix = "scenario_emergency_ppo_models_dynamic"
    },
    @{
        Name = "high_latency_20r_12n"
        EnvConfig = "configs/scenarios/high_latency_20r_12n.yaml"
        BaselineTrainConfig = "configs/scenarios/train_plain_ppo_high_latency_20r_12n.yaml"
        NaiveTrainConfig = "configs/scenarios/train_naive_gat_high_latency_20r_12n.yaml"
        ScoringTrainConfig = "configs/scenarios/train_scoring_gat_high_latency_20r_12n.yaml"
        OutputPrefix = "scenario_high_latency_ppo_models_dynamic"
    }
)

$startedAt = Get-Date
Write-Host ">>> all PPO model scenario experiments started at $startedAt"
Write-Host ">>> scenario count: $($scenarios.Count)"

if ($CompareOnly) {
    Write-Host ">>> CompareOnly enabled; every scenario will skip training and only compare best models"
}

for ($i = 0; $i -lt $scenarios.Count; $i++) {
    $scenario = $scenarios[$i]
    $index = $i + 1

    Write-Host ""
    Write-Host "============================================================"
    Write-Host ">>> [$index/$($scenarios.Count)] scenario started: $($scenario.Name)"
    Write-Host "============================================================"

    $argsForScenario = @{
        EnvConfig = $scenario.EnvConfig
        BaselineTrainConfig = $scenario.BaselineTrainConfig
        NaiveTrainConfig = $scenario.NaiveTrainConfig
        ScoringTrainConfig = $scenario.ScoringTrainConfig
        OutputPrefix = $scenario.OutputPrefix
    }

    if ($CompareOnly) {
        $argsForScenario.CompareOnly = $true
    }

    & (Join-Path $scriptRoot "run_ppo_model_scenario.ps1") @argsForScenario

    Write-Host ">>> [$index/$($scenarios.Count)] scenario completed: $($scenario.Name)"
    Write-Host ">>> summary: outputs/results/$($scenario.OutputPrefix).csv"
    Write-Host ">>> per-seed: outputs/results/$($scenario.OutputPrefix)_per_seed.csv"
}

$finishedAt = Get-Date
$elapsed = $finishedAt - $startedAt
Write-Host ""
Write-Host "============================================================"
Write-Host ">>> all PPO model scenario experiments completed at $finishedAt"
Write-Host ">>> elapsed: $elapsed"
Write-Host "============================================================"

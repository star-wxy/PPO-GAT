param(
    [string]$EnvConfig = "configs/scenarios/mixed_context_20r_10n.yaml",
    [string]$Manifest = "configs/ablations/mixed_context_ablation_manifest.yaml",
    [switch]$CompareOnly
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$powerShellExe = (Get-Process -Id $PID).Path
$trainConfigs = @(
    "configs/ablations/train_full_scoring_mixed_context.yaml",
    "configs/ablations/train_no_gat_mixed_context.yaml",
    "configs/ablations/train_no_node_scoring_mixed_context.yaml",
    "configs/ablations/train_no_heuristic_gate_mixed_context.yaml",
    "configs/ablations/train_fixed_reward_mixed_context.yaml",
    "configs/ablations/train_no_robot_state_mixed_context.yaml",
    "configs/ablations/train_no_congestion_mixed_context.yaml",
    "configs/ablations/train_no_charging_mixed_context.yaml"
)

function Invoke-ChildScript {
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

Write-Host ">>> ablation experiment started"
Write-Host ">>> env config: $EnvConfig"
Write-Host ">>> manifest: $Manifest"

if (-not $CompareOnly) {
    $idx = 1
    foreach ($trainConfig in $trainConfigs) {
        Write-Host ">>> [$idx/$($trainConfigs.Count)] training ablation: $trainConfig"
        Invoke-ChildScript `
            -ScriptPath (Join-Path $scriptRoot "train_ablation_model.ps1") `
            -Arguments @("-EnvConfig", $EnvConfig, "-TrainConfig", $trainConfig)
        $idx += 1
    }
} else {
    Write-Host ">>> CompareOnly enabled; skipping training"
}

Write-Host ">>> comparing ablation best models"
Invoke-ChildScript `
    -ScriptPath (Join-Path $scriptRoot "compare_ablation_models.ps1") `
    -Arguments @("-Manifest", $Manifest, "-UseBestModel")

Write-Host ">>> ablation experiment completed"
Write-Host ">>> outputs/results/ablation_mixed_context.csv"
Write-Host ">>> outputs/results/ablation_mixed_context_per_seed.csv"

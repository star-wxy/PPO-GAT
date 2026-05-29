param(
    [string]$EnvConfig = "configs/scenarios/mixed_context_20r_10n.yaml",
    [Parameter(Mandatory = $true)]
    [string]$TrainConfig
)

$env:PYTHONPATH = (Get-Location).Path

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m src.train_ablation_model --env-config $EnvConfig --train-config $TrainConfig
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m src.train_ablation_model --env-config $EnvConfig --train-config $TrainConfig
    exit $LASTEXITCODE
}

$condaPython = "D:\MiniConda\envs\mytest\python.exe"
if (Test-Path -LiteralPath $condaPython) {
    & $condaPython -m src.train_ablation_model --env-config $EnvConfig --train-config $TrainConfig
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first."

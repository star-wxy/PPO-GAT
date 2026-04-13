$env:PYTHONPATH = (Get-Location).Path

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m src.compare_ppo_models
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m src.compare_ppo_models
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first, then rerun scripts/compare_ppo_models.ps1."

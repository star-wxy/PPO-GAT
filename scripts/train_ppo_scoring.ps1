$env:PYTHONPATH = (Get-Location).Path

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m src.train_scoring_gat
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m src.train_scoring_gat
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first, then rerun scripts/train_ppo_scoring.ps1."

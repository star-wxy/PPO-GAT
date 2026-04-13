$env:PYTHONPATH = (Get-Location).Path

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m src.compare_baselines
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m src.compare_baselines
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first, then rerun scripts/compare_baselines.ps1."

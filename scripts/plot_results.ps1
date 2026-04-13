$env:PYTHONPATH = (Get-Location).Path

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m src.plot_results
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m src.plot_results
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first, then rerun scripts/plot_results.ps1."

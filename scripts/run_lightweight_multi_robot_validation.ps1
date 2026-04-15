$env:PYTHONPATH = (Get-Location).Path

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m src.lightweight_multi_robot_validation
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m src.lightweight_multi_robot_validation
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first, then rerun scripts/run_lightweight_multi_robot_validation.ps1."

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

$baselineLogDir = Join-Path $projectRoot "outputs/tensorboard_baseline"
$naiveLogDir = Join-Path $projectRoot "outputs/tensorboard_naive"
$scoringLogDir = Join-Path $projectRoot "outputs/tensorboard_scoring"

$logdirs = @(
    "baseline=$baselineLogDir",
    "naive=$naiveLogDir",
    "scoring=$scoringLogDir"
)

$logdirSpec = $logdirs -join ","

$hasAnyLog = (Test-Path $baselineLogDir) -or (Test-Path $naiveLogDir) -or (Test-Path $scoringLogDir)
if (-not $hasAnyLog) {
    throw "No TensorBoard log directories were found under $projectRoot\\outputs. Run training first, then rerun scripts/run_tensorboard.ps1."
}

$tensorboardCmd = Get-Command tensorboard -ErrorAction SilentlyContinue
if ($tensorboardCmd) {
    & $tensorboardCmd.Source --logdir_spec $logdirSpec --port 6006
    exit $LASTEXITCODE
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source -m tensorboard.main --logdir_spec $logdirSpec --port 6006
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -m tensorboard.main --logdir_spec $logdirSpec --port 6006
    exit $LASTEXITCODE
}

throw "Neither tensorboard nor python/py is available in PATH. Please activate the Python environment used for training, then rerun scripts/run_tensorboard.ps1."

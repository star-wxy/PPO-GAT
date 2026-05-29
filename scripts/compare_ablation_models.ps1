param(
    [string]$Manifest = "configs/ablations/mixed_context_ablation_manifest.yaml",
    [switch]$UseBestModel
)

$env:PYTHONPATH = (Get-Location).Path
$arguments = @("-m", "src.compare_ablation_models", "--manifest", $Manifest)
if ($UseBestModel) {
    $arguments += "--use-best-model"
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source @arguments
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source @arguments
    exit $LASTEXITCODE
}

$condaPython = "D:\MiniConda\envs\mytest\python.exe"
if (Test-Path -LiteralPath $condaPython) {
    & $condaPython @arguments
    exit $LASTEXITCODE
}

throw "Python interpreter not found in PATH. Please activate your environment first."

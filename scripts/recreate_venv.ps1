# Use when repair_venv.ps1 cannot delete files (still "access denied"):
# 1. Close ALL Jupyter kernels for this project in Cursor/VS Code.
# 2. Optional: Task Manager -> end any "python.exe" you do not need.
# 3. Close Cursor completely, reopen, run this script from PowerShell.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"

if (Test-Path $venv) {
    Remove-Item -LiteralPath $venv -Recurse -Force
    Write-Host "Removed old .venv"
}

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {
    Write-Host "python not on PATH; use full path to py -3.10 launcher"
    exit 1
}

& $py -m venv $venv
$pip = Join-Path $venv "Scripts\python.exe"
& $pip -m pip install --upgrade pip
& $pip -m pip install -r (Join-Path $root "requirements.txt")

& $pip -c "import matplotlib; import matplotlib.backends.registry; import seaborn; import ydata_profiling; print('OK')"

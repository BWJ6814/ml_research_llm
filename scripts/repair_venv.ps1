# Run AFTER closing: Jupyter kernels, Python REPL, and any process using .venv
# If Remove-Item says "access denied", use recreate_venv.ps1 after closing Cursor.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$site = Join-Path $root ".venv\Lib\site-packages"
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Missing .venv at $root"
    exit 1
}

Write-Host "Removing broken matplotlib / pillow temp dirs under site-packages..."
$toRemove = @(
    "matplotlib",
    "matplotlib-3.10.0.dist-info",
    "matplotlib-3.10.8.dist-info",
    "~atplotlib",
    "~atplotlib-3.10.8.dist-info",
    "~il",
    "~illow-12.2.0.dist-info"
)
foreach ($name in $toRemove) {
    $p = Join-Path $site $name
    if (Test-Path $p) {
        try {
            Remove-Item -LiteralPath $p -Recurse -Force
            Write-Host "  removed $name"
        } catch {
            Write-Host "FAILED to remove $name — close Jupyter kernels & Python using this venv, then retry or run scripts\recreate_venv.ps1"
            throw
        }
    }
}

Write-Host "Reinstalling from requirements.txt..."
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $root "requirements.txt") --force-reinstall --no-cache-dir

Write-Host "Smoke test..."
& $py -c "import matplotlib; import matplotlib.backends.registry; import seaborn; import ydata_profiling; import statsmodels; import openpyxl; print('OK', matplotlib.__version__)"

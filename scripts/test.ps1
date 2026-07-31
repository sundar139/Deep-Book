<#
.SYNOPSIS
    Test runner for DeepBook. Run the tier that matches what you just changed.

.EXAMPLE
    .\scripts\test.ps1 fast     # every commit  (~2 s)
    .\scripts\test.ps1 slow     # before pushing (~1 min)
    .\scripts\test.ps1 gpu      # after touching the model
    .\scripts\test.ps1 all      # before opening a PR
    .\scripts\test.ps1 cov      # coverage report -> htmlcov\index.html
#>
param(
    [ValidateSet("fast", "slow", "gpu", "data", "all", "cov")]
    [string]$Tier = "fast"
)

$ErrorActionPreference = "Stop"
if (-not $env:VIRTUAL_ENV -and -not $env:CONDA_PREFIX) {
    Write-Warning "No virtual environment active. Run .\.venv\Scripts\Activate.ps1 first."
}

switch ($Tier) {
    "fast" { pytest -m "fast and not slow and not gpu and not data" --maxfail=1 }
    "slow" { pytest -m "slow and not gpu and not data" --durations=20 }
    "gpu"  { pytest -m "gpu" }
    "data" { pytest -m "data" }
    "all"  { pytest }
    "cov"  { pytest --cov=src --cov-report=html --cov-report=term-missing }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`n$Tier tier passed." -ForegroundColor Green

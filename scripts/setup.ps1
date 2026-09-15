# Set up the environment: virtualenv + dependencies + tests.
# Usage: .\scripts\setup.ps1 [-SkipTests]
#
# NOTE: this file is intentionally ASCII-only. Windows PowerShell 5.1 decodes
# BOM-less .ps1 files as ANSI, so non-ASCII text here can corrupt parsing.
[CmdletBinding()]
param([switch]$SkipTests)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo '.venv'

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtualenv: $venv"
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'python -m venv failed. Python 3.11+ is required.' }
}

$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Interpreter not found: $py" }

Write-Host 'Upgrading pip'
& $py -m pip install --upgrade pip --quiet

Write-Host 'Installing requirements'
& $py -m pip install -r (Join-Path $repo 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed.' }

if (-not $SkipTests) {
    Write-Host 'Running tests'
    Push-Location $repo
    try {
        & $py -m pytest tests -q
        if ($LASTEXITCODE -ne 0) { throw 'Tests are not green; fix them before continuing.' }
    } finally { Pop-Location }
}

Write-Host ''
Write-Host 'Done. Next steps:'
Write-Host '  Copy-Item .env.example .env   # then edit it and set SKINTONE_API_KEY'
Write-Host '  .\scripts\run.ps1'

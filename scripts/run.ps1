# Run the API service. Binds to loopback only; expose it via tunnel.ps1.
# Usage: .\scripts\run.ps1 [-Reload] [-Port 8000]
#
# NOTE: this file is intentionally ASCII-only (see setup.ps1 for the reason).
[CmdletBinding()]
param([switch]$Reload, [int]$Port = 0)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

function Get-Interpreter {
    param([string]$Repo)
    # Support both layouts: venv inside the repo (fresh clone) or one level up
    # (this machine's development layout).
    foreach ($cand in @((Join-Path $Repo '.venv\Scripts\python.exe'),
                        (Join-Path (Split-Path -Parent $Repo) '.venv\Scripts\python.exe'))) {
        if (Test-Path $cand) { return $cand }
    }
    return 'python'
}

function Read-DotEnv {
    param([string]$Path)
    $h = @{}
    if (-not (Test-Path $Path)) { return $h }
    foreach ($line in Get-Content $Path) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -lt 1) { continue }
        $h[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim()
    }
    return $h
}

$envMap = Read-DotEnv (Join-Path $repo '.env')
if ($envMap.Count -eq 0) {
    Write-Warning 'No .env found (copy .env.example and edit it). Falling back to defaults.'
}

$py = Get-Interpreter -Repo $repo
$bindHost = if ($envMap['SKINTONE_HOST']) { $envMap['SKINTONE_HOST'] } else { '127.0.0.1' }
if ($Port -gt 0) { $bindPort = $Port }
elseif ($envMap['SKINTONE_PORT']) { $bindPort = [int]$envMap['SKINTONE_PORT'] }
else { $bindPort = 8000 }

if (-not $envMap['SKINTONE_API_KEY']) {
    Write-Warning 'SKINTONE_API_KEY is empty - the service does no authentication. Fine for local use; set it before opening a tunnel.'
}

Write-Host "Interpreter: $py"
Write-Host "Listening:   http://${bindHost}:${bindPort}"
Write-Host "Health:      curl.exe http://${bindHost}:${bindPort}/v1/health"
Write-Host ''

# --factory is required: main.py is an application factory and deliberately does
# not build the app at import time (otherwise importing it would create the data
# directory). Using skintone.main:app fails with "Attribute 'app' not found".
$uvArgs = @('-m', 'uvicorn', '--factory', 'skintone.main:create_app',
            '--host', $bindHost, '--port', "$bindPort")
if ($Reload) { $uvArgs += '--reload' }

Push-Location $repo
try { & $py @uvArgs } finally { Pop-Location }

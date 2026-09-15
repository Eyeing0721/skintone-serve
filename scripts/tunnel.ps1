# Cloudflared named tunnel: run / status / url / local
#
# The tunnel was created through the Cloudflare API and its DNS record is bound.
# Its configuration is REMOTELY managed (config_src=cloudflare), so locally we
# need neither config.yml nor cert.pem - only the connector token. This script
# therefore never calls `cloudflared tunnel create` or `route dns`.
#
# Usage:
#   .\scripts\tunnel.ps1 run      start the connector (foreground)
#   .\scripts\tunnel.ps1 status   process count + probe through the tunnel
#   .\scripts\tunnel.ps1 url      print the public URL
#   .\scripts\tunnel.ps1 local    print the loopback URL
#
# NOTE: this file is intentionally ASCII-only (see setup.ps1 for the reason).
[CmdletBinding()]
param([Parameter(Position = 0)][ValidateSet('run', 'status', 'url', 'local')][string]$Command = 'run')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

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
$tunnelName = if ($envMap['SKINTONE_TUNNEL_NAME']) { $envMap['SKINTONE_TUNNEL_NAME'] } else { 'skintone-api' }
$hostname = if ($envMap['SKINTONE_TUNNEL_HOSTNAME']) { $envMap['SKINTONE_TUNNEL_HOSTNAME'] } else { 'skintest.0721.luxe' }
$port = if ($envMap['SKINTONE_PORT']) { $envMap['SKINTONE_PORT'] } else { '8000' }
$tokenFile = if ($envMap['SKINTONE_TUNNEL_TOKEN_FILE']) { $envMap['SKINTONE_TUNNEL_TOKEN_FILE'] } else { '..\.cloudflared-token' }

# Resolve relative paths against the repo root.
if (-not [System.IO.Path]::IsPathRooted($tokenFile)) { $tokenFile = Join-Path $repo $tokenFile }
$tokenFile = [System.IO.Path]::GetFullPath($tokenFile)

$cf = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
if (-not (Test-Path $cf)) {
    $found = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($found) { $cf = $found.Source }
    else { throw 'cloudflared not found. Install it, or fix the $cf path in this script.' }
}

$publicUrl = "https://$hostname"
$localUrl = "http://127.0.0.1:$port"

switch ($Command) {
    'url' { Write-Output $publicUrl; break }
    'local' { Write-Output $localUrl; break }

    'status' {
        $proc = @(Get-Process cloudflared -ErrorAction SilentlyContinue)
        Write-Host "cloudflared processes: $($proc.Count)"
        Write-Host "Probing origin  $localUrl/v1/health"
        & curl.exe -sS -m 10 -o NUL -w "  local  http_code=%{http_code}`n" "$localUrl/v1/health"
        Write-Host "Probing tunnel  $publicUrl/v1/health"
        & curl.exe -sS -m 25 -o NUL -w "  tunnel http_code=%{http_code}  dns=%{time_namelookup}s total=%{time_total}s`n" "$publicUrl/v1/health"
        Write-Host ''
        Write-Host 'Hint: 200 = full chain works; 502 = tunnel up but the backend is down; 000 = DNS/tunnel not live.'
        break
    }

    'run' {
        if (-not (Test-Path $tokenFile)) {
            throw "Connector token file not found: $tokenFile`nThe tunnel config lives on Cloudflare's side; this token is all the connector needs."
        }
        # Pass the token via environment variable so it never shows up in the
        # command line or the process list.
        $env:TUNNEL_TOKEN = (Get-Content $tokenFile -Raw).Trim()
        if (-not $env:TUNNEL_TOKEN) { throw "Token file is empty: $tokenFile" }
        Write-Host "Tunnel: $tunnelName -> $publicUrl -> $localUrl"
        Write-Host 'Start the backend in another window: .\scripts\run.ps1'
        Write-Host ''
        & $cf tunnel --no-autoupdate run
        break
    }
}

$ErrorActionPreference = "Stop"

Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

function Get-UvCommand {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        return $uv.Source
    }

    $candidate = Join-Path $env:APPDATA "Python\Python314\Scripts\uv.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    return $null
}

function Get-PythonCommand {
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return $python.Source
    }

    throw "Python was not found. Run uv sync first or install Python."
}

function Wait-ForHealth {
    param(
        [string]$BaseUrl,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Timed out waiting for server health at $BaseUrl"
}

$uv = Get-UvCommand
if ($null -ne $uv) {
    Write-Step "Syncing local environment"
    Invoke-CheckedCommand $uv "sync" "--frozen" "--extra" "dev"
}

$python = Get-PythonCommand
$serverProcess = $null
$baseUrl = "http://127.0.0.1:8000"

try {
    Write-Step "OpenEnv validation"
    Invoke-CheckedCommand $python "-m" "openenv.cli" "validate" "."

    Write-Step "Unit tests"
    Invoke-CheckedCommand $python "-m" "pytest" "tests\test_env.py"

    Write-Step "Task graders"
    Invoke-CheckedCommand $python "scripts\run_graders.py"

    Write-Step "Starting local server"
    $serverProcess = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $RepoRoot `
        -PassThru

    Wait-ForHealth -BaseUrl $baseUrl

    Write-Step "Ping + reset smoke test"
    Invoke-CheckedCommand $python "scripts\ping_env.py" $baseUrl

    Write-Step "Baseline inference"
    Invoke-CheckedCommand $python "inference.py"

    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -eq $docker) {
        Write-Host ""
        Write-Host "Docker not found, skipping docker build." -ForegroundColor Yellow
    } else {
        $dockerAvailable = $false
        $stdoutPath = [System.IO.Path]::GetTempFileName()
        $stderrPath = [System.IO.Path]::GetTempFileName()
        try {
            $probe = Start-Process `
                -FilePath $docker.Source `
                -ArgumentList @("info") `
                -NoNewWindow `
                -Wait `
                -PassThru `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath
            $dockerAvailable = ($probe.ExitCode -eq 0)
        } finally {
            Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
        }

        if ($dockerAvailable) {
            Write-Step "Docker build"
            Invoke-CheckedCommand $docker.Source "build" "-t" "atc-openenv-local" "."
        } else {
            Write-Host ""
            Write-Host "Docker engine is not available, skipping docker build." -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Local checklist completed successfully." -ForegroundColor Green
} finally {
    if ($null -ne $serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
    }
}

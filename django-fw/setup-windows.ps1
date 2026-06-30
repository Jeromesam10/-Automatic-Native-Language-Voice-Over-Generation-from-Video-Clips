#Requires -Version 5.1
<#
.SYNOPSIS
    Voice Over Gen - one-shot Windows setup.
.DESCRIPTION
    Installs Ollama + DeepSeek, creates a Python virtual environment,
    installs dependencies, configures .env, runs migrations, and verifies
    the DeepSeek connection.
.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -DeepSeekModel "deepseek-r1:7b" -DjangoPort 8000
.NOTES
    Run from PowerShell. If script execution is blocked, start with:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>

[CmdletBinding()]
param(
    [string]$DeepSeekModel = "",
    [int]$DjangoPort = 8000,
    [string]$PythonBin = "python"
)

$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $AppDir ".venv"
$EnvFile = Join-Path $AppDir ".env"
$OllamaBaseUrl = "http://127.0.0.1:11434"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-TotalRamGb {
    try {
        $bytes = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory
        return [math]::Round($bytes / 1GB)
    }
    catch {
        return 8
    }
}

function Resolve-Model {
    if ($DeepSeekModel -ne "") {
        return $DeepSeekModel
    }

    $ramGb = Get-TotalRamGb
    if ($ramGb -lt 12) {
        $model = "deepseek-r1:1.5b"
    }
    elseif ($ramGb -lt 24) {
        $model = "deepseek-r1:7b"
    }
    else {
        $model = "deepseek-r1:8b"
    }

    Write-Step "Detected ~${ramGb}GB RAM, using model: $model"
    return $model
}

function Install-Ollama {
    if (Test-Command "ollama") {
        Write-Step "Ollama already installed"
        return
    }

    Write-Step "Installing Ollama"
    if (Test-Command "winget") {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    }
    else {
        $installer = Join-Path $env:TEMP "OllamaSetup.exe"
        Write-Step "Downloading Ollama installer"
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
        Write-Step "Running Ollama installer"
        Start-Process -FilePath $installer -ArgumentList "/silent" -Wait
    }

    $ollamaPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if (Test-Path $ollamaPath) {
        $env:Path = "$env:Path;$ollamaPath"
    }

    if (-not (Test-Command "ollama")) {
        throw "Ollama installation finished but 'ollama' is not on PATH. Restart the terminal and re-run this script."
    }
}

function Start-OllamaService {
    Write-Step "Starting Ollama"
    if (-not (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    }

    for ($i = 0; $i -lt 30; $i++) {
        try {
            Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -TimeoutSec 3 | Out-Null
            return
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    throw "Ollama did not become ready on $OllamaBaseUrl"
}

function Get-DeepSeekModel {
    param([string]$Model)
    Write-Step "Pulling DeepSeek model: $Model"
    ollama pull $Model
}

function Initialize-PythonEnv {
    if (-not (Test-Command $PythonBin)) {
        throw "Python not found. Install Python 3.10+ from https://www.python.org/downloads/ and ensure it is on PATH."
    }

    Write-Step "Creating Python virtual environment"
    if (-not (Test-Path $VenvDir)) {
        & $PythonBin -m venv $VenvDir
    }

    $venvPython = Join-Path $VenvDir "Scripts\python.exe"

    Write-Step "Installing Python dependencies"
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $AppDir "requirements.txt")

    return $venvPython
}

function New-SecretKey {
    param([string]$VenvPython)
    return (& $VenvPython -c "import secrets; print(secrets.token_urlsafe(50))").Trim()
}

function New-EnvFile {
    param([string]$VenvPython, [string]$Model)

    if (Test-Path $EnvFile) {
        Write-Step ".env already exists, keeping current file"
        return
    }

    Write-Step "Creating .env"
    $secretKey = New-SecretKey -VenvPython $VenvPython

    $content = @"
SECRET_KEY=$secretKey
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1

DJANGO_PORT=$DjangoPort
GUNICORN_WORKERS=2

OLLAMA_BASE_URL=$OllamaBaseUrl
DEEPSEEK_MODEL=$Model
OLLAMA_TIMEOUT=120
"@

    Set-Content -Path $EnvFile -Value $content -Encoding UTF8
}

function Invoke-DjangoSetup {
    param([string]$VenvPython)
    Write-Step "Running Django migrations"
    Push-Location $AppDir
    try {
        & $VenvPython manage.py migrate --noinput
    }
    finally {
        Pop-Location
    }
}

function Test-Setup {
    param([string]$VenvPython)
    Write-Step "Verifying setup"
    Push-Location $AppDir
    try {
        & $VenvPython manage.py check
        & $VenvPython manage.py test_deepseek "Say hello in one short sentence."
    }
    finally {
        Pop-Location
    }
}

function Write-Summary {
    param([string]$Model, [string]$VenvPython)

    Write-Host @"

Setup complete.

Project directory: $AppDir
Environment file:  $EnvFile
DeepSeek model:    $Model

Run the development server:
  $VenvPython manage.py runserver 0.0.0.0:$DjangoPort

Test DeepSeek from the CLI:
  $VenvPython manage.py test_deepseek "Your prompt"

Health endpoint (after the server starts):
  http://127.0.0.1:$DjangoPort/api/v1/voiceover/health

If you expose this machine on your network, add its IP to ALLOWED_HOSTS in .env.

"@ -ForegroundColor Green
}

function Main {
    Write-Step "Voice Over Gen setup starting (Windows)"

    Install-Ollama
    Start-OllamaService

    $model = Resolve-Model
    Get-DeepSeekModel -Model $model

    $venvPython = Initialize-PythonEnv
    New-EnvFile -VenvPython $venvPython -Model $model
    Invoke-DjangoSetup -VenvPython $venvPython
    Test-Setup -VenvPython $venvPython
    Write-Summary -Model $model -VenvPython $venvPython
}

Main

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt

if (Test-Path .\build) { Remove-Item .\build -Recurse -Force }
if (Test-Path .\dist) { Remove-Item .\dist -Recurse -Force }

python -m PyInstaller `
  --noconfirm `
  --onedir `
  --name AIWorkstationHub `
  --collect-all gradio `
  --collect-all gradio_client `
  .\ai_workstation_launcher.py

Write-Host "Build complete: $Root\dist\AIWorkstationHub\AIWorkstationHub.exe"

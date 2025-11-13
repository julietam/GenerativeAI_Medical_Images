param(
  [string]$OutDir = "data/openneuro/ds005533",
  [string]$Version = "1.0.0"
)

$ErrorActionPreference = 'Stop'

# Check for openneuro CLI
$cli = Get-Command openneuro -ErrorAction SilentlyContinue
if (-not $cli) {
  Write-Host "OpenNeuro CLI not found. Install with:" -ForegroundColor Yellow
  Write-Host "  npm install -g openneuro-cli" -ForegroundColor Yellow
  Write-Host "Then run this script again." -ForegroundColor Yellow
  exit 1
}

# Download only anat T1w/T2w files to keep it light
# openneuro supports include patterns
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

Write-Host "Downloading dataset ds005533:$Version (anat T1w/T2w only) ..."
openneuro download ds005533:$Version $OutDir --include "sub-*/**/anat/*_T1w.nii*" --include "sub-*/**/anat/*_T2w.nii*"

Write-Host "Download complete: $OutDir"

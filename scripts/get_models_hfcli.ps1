#!/usr/bin/env pwsh
# ============================================================================
# get_models_hfcli.ps1 - Download Models via HuggingFace CLI
# ============================================================================
#
# Alternative to get_models.ps1 that uses HuggingFace CLI instead of git.
# This method is often faster, more reliable, and doesn't require Git LFS.
#
# Downloads:
#   - TerraMind 1.0 Large (IBM/ESA) → weights/hf/TerraMind-1.0-large
#   - Prithvi EO 2.0 600M (IBM/NASA) → weights/hf/Prithvi-EO-2.0-600M
#
# Usage:
#   .\scripts\get_models_hfcli.ps1                 # Download both models
#   .\scripts\get_models_hfcli.ps1 -TerraMindOnly  # Only TerraMind
#   .\scripts\get_models_hfcli.ps1 -PrithviOnly    # Only Prithvi
#   .\scripts\get_models_hfcli.ps1 -Force          # Re-download existing
#
# Exit Codes:
#   0 - Success
#   1 - Error occurred
#
# Requirements:
#   - Python 3.11+
#   - pip
#   - Internet connection
#   - Disk space: ~13 GB
#
# Advantages over git clone:
#   - Faster downloads (multi-threaded)
#   - Automatic resume on interruption
#   - No Git LFS required
#   - Smaller storage footprint (no .git directory)
#
# ============================================================================

param(
    [Parameter(Mandatory = $false)]
    [switch]$TerraMindOnly,
    
    [Parameter(Mandatory = $false)]
    [switch]$PrithviOnly,
    
    [Parameter(Mandatory = $false)]
    [switch]$Force,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory = $false)]
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================

# Project root
$ProjectRoot = Split-Path $PSScriptRoot -Parent

# Model download directory
$WeightsDir = Join-Path $ProjectRoot "weights"
$HFDir = Join-Path $WeightsDir "hf"

# Model repositories
$TERRAMIND_REPO = "ibm-esa-geospatial/TerraMind-1.0-large"
$PRITHVI_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"

# Local directories
$TERRAMIND_DIR = Join-Path $HFDir "TerraMind-1.0-large"
$PRITHVI_DIR = Join-Path $HFDir "Prithvi-EO-2.0-600M"

# ANSI color codes
$ColorReset = "`e[0m"
$ColorBold = "`e[1m"
$ColorGreen = "`e[32m"
$ColorYellow = "`e[33m"
$ColorRed = "`e[31m"
$ColorCyan = "`e[36m"
$ColorBlue = "`e[34m"

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Header {
    param([string]$Message)
    
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}═══════════════════════════════════════════════════════════════════════════════${ColorReset}"
    Write-Host "${ColorBold}${ColorGreen}$Message${ColorReset}"
    Write-Host "${ColorBold}${ColorCyan}═══════════════════════════════════════════════════════════════════════════════${ColorReset}"
    Write-Host ""
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "${ColorBold}${ColorBlue}▸ $Message${ColorReset}"
    Write-Host "${ColorCyan}$("-" * 79)${ColorReset}"
}

function Write-Success {
    param([string]$Message)
    Write-Host "${ColorGreen}✓ $Message${ColorReset}"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "${ColorYellow}⚠ $Message${ColorReset}"
}

function Write-Error {
    param([string]$Message)
    Write-Host "${ColorBold}${ColorRed}✗ $Message${ColorReset}"
}

function Write-Info {
    param([string]$Message)
    Write-Host "${ColorCyan}  $Message${ColorReset}"
}

function Test-CommandExists {
    param([string]$Command)
    
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Get-DirectorySize {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return 0
    }
    
    try {
        $size = (Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue | 
            Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        return $size ?? 0
    }
    catch {
        return 0
    }
}

function Format-ByteSize {
    param([long]$Bytes)
    
    if ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    }
    elseif ($Bytes -ge 1MB) {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    }
    elseif ($Bytes -ge 1KB) {
        return "{0:N2} KB" -f ($Bytes / 1KB)
    }
    else {
        return "$Bytes bytes"
    }
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

function Test-Prerequisites {
    Write-Section "Checking Prerequisites"
    
    $allGood = $true
    
    # Check Python
    if (Test-CommandExists "python") {
        $pythonVersion = python --version 2>&1
        Write-Success "Python: $pythonVersion"
    }
    else {
        Write-Error "Python not found"
        Write-Info "Install from: https://www.python.org/downloads/"
        $allGood = $false
    }
    
    # Check pip
    if (Test-CommandExists "pip") {
        $pipVersion = (pip --version) -replace "pip ", "" -replace " from.*", ""
        Write-Success "pip: $pipVersion"
    }
    else {
        Write-Error "pip not found"
        Write-Info "Install with: python -m ensurepip"
        $allGood = $false
    }
    
    # Check disk space
    $drive = (Get-Location).Drive.Name
    $driveInfo = Get-PSDrive -Name $drive -PSProvider FileSystem
    $freeGB = [math]::Round($driveInfo.Free / 1GB, 2)
    $requiredGB = 15
    
    if ($freeGB -ge $requiredGB) {
        Write-Success "Disk space: $freeGB GB available (need ~$requiredGB GB)"
    }
    else {
        Write-Warning "Low disk space: $freeGB GB available (recommended: $requiredGB GB)"
    }
    
    Write-Host ""
    
    return $allGood
}

# ============================================================================
# HuggingFace CLI Installation
# ============================================================================

function Install-HuggingFaceCLI {
    Write-Section "Installing/Updating HuggingFace CLI"
    
    try {
        Write-Info "Running: pip install -U \"huggingface_hub[cli]\""
        Write-Host ""
        
        if ($Verbose) {
            python -m pip install -U "huggingface_hub[cli]"
        }
        else {
            python -m pip install -U "huggingface_hub[cli]" 2>&1 | Out-Null
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed with exit code $LASTEXITCODE"
        }
        
        Write-Host ""
        Write-Success "HuggingFace CLI installed/updated"
        
        # Verify installation
        $hfVersion = python -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Version: $hfVersion"
        }
        
        return $true
    }
    catch {
        Write-Error "Failed to install HuggingFace CLI"
        Write-Info "Error: $_"
        return $false
    }
}

# ============================================================================
# Model Download Functions
# ============================================================================

function Download-Model {
    param(
        [string]$RepoId,
        [string]$LocalDir,
        [string]$ModelName
    )
    
    Write-Section "Downloading $ModelName"
    
    # Check if already exists
    if (Test-Path $LocalDir) {
        if (-not $Force) {
            Write-Warning "Directory already exists: $LocalDir"
            Write-Info "Use -Force to re-download"
            return $true
        }
        else {
            Write-Warning "Removing existing directory (forced)"
            Remove-Item $LocalDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    # Create parent directory
    $parentDir = Split-Path $LocalDir -Parent
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    
    Write-Info "Repository: $RepoId"
    Write-Info "Destination: $LocalDir"
    Write-Info "Method: HuggingFace CLI (multi-threaded download)"
    Write-Host ""
    
    # Download using HuggingFace CLI
    try {
        Write-Info "Downloading files..."
        $startTime = Get-Date
        
        # Build command
        $hfCmd = "hf"
        $hfArgs = @(
            "download",
            $RepoId,
            "--local-dir", $LocalDir
        )
        
        # Add resume flag (automatic in newer versions)
        # $hfArgs += "--resume-download"
        
        Write-Host ""
        
        if ($Verbose) {
            & $hfCmd @hfArgs
        }
        else {
            & $hfCmd @hfArgs 2>&1 | ForEach-Object {
                if ($_ -match "Downloading|Fetching|100%|Downloaded") {
                    Write-Host $_
                }
            }
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "HuggingFace CLI download failed with exit code $LASTEXITCODE"
        }
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        # Get directory size
        $size = Get-DirectorySize -Path $LocalDir
        $sizeStr = Format-ByteSize -Bytes $size
        
        Write-Host ""
        Write-Success "Downloaded successfully"
        Write-Info "Time: $([math]::Round($duration, 1))s"
        Write-Info "Size: $sizeStr"
        
        return $true
    }
    catch {
        Write-Error "Failed to download model"
        Write-Info "Error: $_"
        return $false
    }
}

function Show-ModelInfo {
    param(
        [string]$ModelDir,
        [string]$ModelName
    )
    
    if (-not (Test-Path $ModelDir)) {
        return
    }
    
    Write-Host ""
    Write-Host "${ColorBold}$ModelName Information:${ColorReset}"
    
    # Check for key files
    $files = @{
        "config.json"       = "Configuration"
        "model.safetensors" = "Model weights (safetensors)"
        "pytorch_model.bin" = "Model weights (PyTorch)"
        "README.md"         = "Documentation"
        ".gitattributes"    = "Git attributes"
    }
    
    foreach ($file in $files.GetEnumerator()) {
        $filePath = Join-Path $ModelDir $file.Key
        if (Test-Path $filePath) {
            $fileSize = (Get-Item $filePath).Length
            $fileSizeStr = Format-ByteSize -Bytes $fileSize
            Write-Info "✓ $($file.Value): $fileSizeStr"
        }
    }
    
    # Total size
    $totalSize = Get-DirectorySize -Path $ModelDir
    $totalSizeStr = Format-ByteSize -Bytes $totalSize
    Write-Info "Total size: $totalSizeStr"
}

# ============================================================================
# Main Logic
# ============================================================================

function Main {
    # Banner
    Write-Header "Model Download Script - HuggingFace CLI Method"
    
    Write-Info "Target directory: $HFDir"
    Write-Info "Method: HuggingFace CLI (faster, auto-resume)"
    
    if ($TerraMindOnly) {
        Write-Info "Models: TerraMind only"
    }
    elseif ($PrithviOnly) {
        Write-Info "Models: Prithvi only"
    }
    else {
        Write-Info "Models: TerraMind + Prithvi"
    }
    
    Write-Host ""
    
    # Prerequisites
    if (-not (Test-Prerequisites)) {
        Write-Error "Prerequisites check failed"
        return 1
    }
    
    # Install HuggingFace CLI
    if (-not $SkipInstall) {
        if (-not (Install-HuggingFaceCLI)) {
            Write-Error "Failed to install HuggingFace CLI"
            return 1
        }
    }
    else {
        Write-Warning "Skipping HuggingFace CLI installation (--SkipInstall)"
        Write-Host ""
    }
    
    # Check if hf command exists
    if (-not (Test-CommandExists "hf")) {
        Write-Error "HuggingFace CLI 'hf' command not found"
        Write-Info "Try running without --SkipInstall flag"
        return 1
    }
    
    # Download models
    $success = $true
    
    if (-not $PrithviOnly) {
        Write-Host ""
        if (-not (Download-Model -RepoId $TERRAMIND_REPO -LocalDir $TERRAMIND_DIR -ModelName "TerraMind 1.0 Large")) {
            $success = $false
        }
    }
    
    if (-not $TerraMindOnly) {
        Write-Host ""
        if (-not (Download-Model -RepoId $PRITHVI_REPO -LocalDir $PRITHVI_DIR -ModelName "Prithvi EO 2.0 600M")) {
            $success = $false
        }
    }
    
    # Summary
    Write-Header "Download Summary"
    
    if (-not $PrithviOnly -and (Test-Path $TERRAMIND_DIR)) {
        Show-ModelInfo -ModelDir $TERRAMIND_DIR -ModelName "TerraMind 1.0 Large"
    }
    
    if (-not $TerraMindOnly -and (Test-Path $PRITHVI_DIR)) {
        Show-ModelInfo -ModelDir $PRITHVI_DIR -ModelName "Prithvi EO 2.0 600M"
    }
    
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}$("-" * 79)${ColorReset}"
    
    if ($success) {
        Write-Success "All models downloaded successfully!"
        
        Write-Host ""
        Write-Info "Next steps:"
        Write-Info "  1. Verify models: ls $HFDir"
        Write-Info "  2. Update config: configs/hardware.lowvr.yaml"
        Write-Info "  3. Test import: python -c 'from transformers import AutoModel'"
    }
    else {
        Write-Error "Some downloads failed"
        Write-Info "Check errors above and try again"
        return 1
    }
    
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}$("-" * 79)${ColorReset}"
    
    # Advantages of HF CLI method
    Write-Host ""
    Write-Host "${ColorBold}Advantages of HuggingFace CLI:${ColorReset}"
    Write-Info "✓ Faster downloads (multi-threaded)"
    Write-Info "✓ Automatic resume on interruption"
    Write-Info "✓ No Git LFS required"
    Write-Info "✓ Smaller storage footprint (no .git directory)"
    Write-Info "✓ Better progress indicators"
    Write-Host ""
    
    # License reminder
    Write-Host "${ColorBold}License Information:${ColorReset}"
    Write-Info "Both models are licensed under Apache License 2.0"
    Write-Info "TerraMind: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large"
    Write-Info "Prithvi:   https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M"
    Write-Host ""
    
    return 0
}

# ============================================================================
# Entry Point
# ============================================================================

try {
    $exitCode = Main
    exit $exitCode
}
catch {
    Write-Host ""
    Write-Error "Unexpected error occurred"
    Write-Host "${ColorRed}$($_.Exception.Message)${ColorReset}"
    Write-Host "${ColorRed}$($_.ScriptStackTrace)${ColorReset}"
    Write-Host ""
    exit 1
}

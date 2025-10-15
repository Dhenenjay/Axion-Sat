#!/usr/bin/env pwsh
# ============================================================================
# get_models.ps1 - Download Foundation Models from HuggingFace
# ============================================================================
#
# This script downloads the required foundation models for Axion-Sat:
#   - TerraMind 1.0 Large (IBM/ESA)
#   - Prithvi EO 2.0 600M (IBM/NASA)
#
# Usage:
#   .\scripts\get_models.ps1                    # Full download
#   .\scripts\get_models.ps1 -PointersOnly      # Metadata only (no weights)
#   .\scripts\get_models.ps1 -TerraMindOnly     # Only TerraMind
#   .\scripts\get_models.ps1 -PrithviOnly       # Only Prithvi
#
# Exit Codes:
#   0 - Success
#   1 - Error occurred
#
# Requirements:
#   - git (with Git LFS support)
#   - Internet connection
#   - Disk space: ~13 GB for full download, ~100 MB for pointers only
#
# ============================================================================

param(
    [Parameter(Mandatory = $false)]
    [switch]$PointersOnly,
    
    [Parameter(Mandatory = $false)]
    [switch]$TerraMindOnly,
    
    [Parameter(Mandatory = $false)]
    [switch]$PrithviOnly,
    
    [Parameter(Mandatory = $false)]
    [switch]$Force,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
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
$TERRAMIND_REPO = "https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large"
$PRITHVI_REPO = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M"

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
    
    # Check git
    if (Test-CommandExists "git") {
        $gitVersion = (git --version) -replace "git version ", ""
        Write-Success "git: $gitVersion"
    }
    else {
        Write-Error "git not found"
        Write-Info "Install from: https://git-scm.com/downloads"
        $allGood = $false
    }
    
    # Check git-lfs
    if (Test-CommandExists "git-lfs") {
        $lfsVersion = (git lfs version) -replace "git-lfs/", "" -replace " \(.*", ""
        Write-Success "git-lfs: $lfsVersion"
    }
    else {
        Write-Warning "git-lfs not installed (will install now)"
    }
    
    # Check disk space
    $drive = (Get-Location).Drive.Name
    $driveInfo = Get-PSDrive -Name $drive -PSProvider FileSystem
    $freeGB = [math]::Round($driveInfo.Free / 1GB, 2)
    
    if ($PointersOnly) {
        $requiredGB = 1
    }
    else {
        $requiredGB = 15
    }
    
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
# Model Download Functions
# ============================================================================

function Install-GitLFS {
    Write-Section "Installing Git LFS"
    
    try {
        $output = git lfs install 2>&1
        
        if ($LASTEXITCODE -eq 0 -or $output -match "already initialized") {
            Write-Success "Git LFS installed/configured"
            return $true
        }
        else {
            Write-Error "Failed to install Git LFS"
            Write-Info "Error: $output"
            return $false
        }
    }
    catch {
        Write-Error "Failed to install Git LFS: $_"
        return $false
    }
}

function Clone-Model {
    param(
        [string]$RepoUrl,
        [string]$LocalDir,
        [string]$ModelName,
        [bool]$SkipLFS
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
    
    # Set environment variable if pointers only
    if ($SkipLFS) {
        $env:GIT_LFS_SKIP_SMUDGE = "1"
        Write-Info "Mode: Pointers only (metadata, no weights)"
    }
    else {
        Write-Info "Mode: Full download (including model weights)"
    }
    
    Write-Info "Repository: $RepoUrl"
    Write-Info "Destination: $LocalDir"
    Write-Host ""
    
    # Clone repository
    try {
        Write-Info "Cloning repository..."
        $startTime = Get-Date
        
        if ($Verbose) {
            git clone --progress $RepoUrl $LocalDir
        }
        else {
            git clone --progress $RepoUrl $LocalDir 2>&1 | Out-Null
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Git clone failed with exit code $LASTEXITCODE"
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
        
        if ($SkipLFS) {
            Write-Host ""
            Write-Warning "Model weights not downloaded (pointers only mode)"
            Write-Info "To download weights later, run:"
            Write-Info "  cd $LocalDir"
            Write-Info "  git lfs pull"
        }
        
        return $true
    }
    catch {
        Write-Error "Failed to clone repository"
        Write-Info "Error: $_"
        return $false
    }
    finally {
        # Clean up environment variable
        if ($SkipLFS) {
            Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
        }
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
    Write-Header "Model Download Script - Axion-Sat Foundation Models"
    
    Write-Info "Target directory: $HFDir"
    Write-Info "Download mode: $(if ($PointersOnly) { "Pointers only (metadata)" } else { "Full download (with weights)" })"
    
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
    
    # Install Git LFS
    if (-not (Install-GitLFS)) {
        Write-Error "Failed to install Git LFS"
        return 1
    }
    
    # Download models
    $success = $true
    
    if (-not $PrithviOnly) {
        Write-Host ""
        if (-not (Clone-Model -RepoUrl $TERRAMIND_REPO -LocalDir $TERRAMIND_DIR -ModelName "TerraMind 1.0 Large" -SkipLFS $PointersOnly)) {
            $success = $false
        }
    }
    
    if (-not $TerraMindOnly) {
        Write-Host ""
        if (-not (Clone-Model -RepoUrl $PRITHVI_REPO -LocalDir $PRITHVI_DIR -ModelName "Prithvi EO 2.0 600M" -SkipLFS $PointersOnly)) {
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
        
        if (-not $PointersOnly) {
            Write-Host ""
            Write-Info "Next steps:"
            Write-Info "  1. Verify models: ls $HFDir"
            Write-Info "  2. Update config: configs/hardware.lowvr.yaml"
            Write-Info "  3. Test import: python -c 'from transformers import AutoModel'"
        }
        else {
            Write-Host ""
            Write-Info "To download model weights later:"
            if (-not $PrithviOnly) {
                Write-Info "  cd $TERRAMIND_DIR && git lfs pull"
            }
            if (-not $TerraMindOnly) {
                Write-Info "  cd $PRITHVI_DIR && git lfs pull"
            }
        }
    }
    else {
        Write-Error "Some downloads failed"
        Write-Info "Check errors above and try again"
        return 1
    }
    
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}$("-" * 79)${ColorReset}"
    
    # License reminder
    Write-Host ""
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

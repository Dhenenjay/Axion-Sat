#!/usr/bin/env pwsh
# ============================================================================
# 04_disk_budget.ps1 - Disk Space Budget Checker
# ============================================================================
#
# This script verifies that sufficient disk space is available before running
# data-intensive operations. It checks the DATA_DIR (or current drive) and
# aborts execution if free space falls below the threshold.
#
# Usage:
#   .\scripts\04_disk_budget.ps1                    # Check default DATA_DIR
#   .\scripts\04_disk_budget.ps1 -MinGB 50          # Require 50 GB free
#   .\scripts\04_disk_budget.ps1 -Path "D:\Data"    # Check specific path
#   .\scripts\04_disk_budget.ps1 -Verbose           # Show detailed info
#
# Exit Codes:
#   0 - Sufficient disk space available
#   1 - Insufficient disk space (aborted)
#   2 - Error occurred
#
# Why This Matters:
#   - Satellite tile downloads can consume 10-100+ GB
#   - Model training generates large checkpoint files
#   - Out-of-space errors during processing cause data corruption
#   - Prevention is better than cleanup after failure
#
# ============================================================================

param(
    [Parameter(Mandatory = $false)]
    [string]$Path = $null,
    
    [Parameter(Mandatory = $false)]
    [int]$MinGB = 20,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory = $false)]
    [switch]$Json,
    
    [Parameter(Mandatory = $false)]
    [switch]$WarningOnly
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================

# Default minimum free space (in GB)
$DEFAULT_MIN_GB = 20

# Warning threshold (in GB) - warn if below this, even if above minimum
$WARNING_THRESHOLD_GB = 50

# ANSI color codes (if terminal supports them)
$ColorReset = "`e[0m"
$ColorBold = "`e[1m"
$ColorGreen = "`e[32m"
$ColorYellow = "`e[33m"
$ColorRed = "`e[31m"
$ColorCyan = "`e[36m"

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

function Write-Success {
    param([string]$Message)
    Write-Host "${ColorGreen}✓ $Message${ColorReset}"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "${ColorYellow}⚠ WARNING: $Message${ColorReset}"
}

function Write-Error {
    param([string]$Message)
    Write-Host "${ColorBold}${ColorRed}✗ ERROR: $Message${ColorReset}"
}

function Write-Info {
    param([string]$Message)
    Write-Host "${ColorCyan}$Message${ColorReset}"
}

function Get-DataDirFromEnv {
    <#
    .SYNOPSIS
    Get DATA_DIR from environment variable or .env file
    #>
    
    # Try environment variable first
    $dataDir = $env:DATA_DIR
    if ($dataDir) {
        return $dataDir
    }
    
    # Try loading from .env file
    $envFile = Join-Path $PSScriptRoot ".." ".env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match '^\s*DATA_DIR\s*=\s*(.+)\s*$') {
                $dataDir = $Matches[1].Trim('"').Trim("'")
                return $dataDir
            }
        }
    }
    
    # Default to ./data relative to project root
    $projectRoot = Split-Path $PSScriptRoot -Parent
    return Join-Path $projectRoot "data"
}

function Get-DiskSpaceInfo {
    <#
    .SYNOPSIS
    Get disk space information for a given path
    
    .OUTPUTS
    Hashtable with TotalGB, UsedGB, FreeGB, FreePercent, DriveLetter
    #>
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    
    try {
        # Resolve to absolute path
        $absolutePath = (Resolve-Path -Path $Path -ErrorAction SilentlyContinue).Path
        if (-not $absolutePath) {
            # Path doesn't exist yet, use parent or drive root
            $absolutePath = Split-Path $Path -Parent
            if (-not $absolutePath) {
                $absolutePath = (Get-Location).Drive.Root
            }
        }
        
        # Get the drive letter
        $drive = Split-Path $absolutePath -Qualifier
        if (-not $drive) {
            # Fallback: use current drive
            $drive = (Get-Location).Drive.Name + ":"
        }
        
        # Get drive info using Get-PSDrive
        $driveInfo = Get-PSDrive -Name $drive.TrimEnd(':') -PSProvider FileSystem -ErrorAction Stop
        
        # Calculate sizes in GB
        $totalBytes = $driveInfo.Used + $driveInfo.Free
        $totalGB = [math]::Round($totalBytes / 1GB, 2)
        $freeGB = [math]::Round($driveInfo.Free / 1GB, 2)
        $usedGB = [math]::Round($driveInfo.Used / 1GB, 2)
        $freePercent = [math]::Round(($driveInfo.Free / $totalBytes) * 100, 1)
        
        return @{
            DriveLetter = $drive
            TotalGB     = $totalGB
            UsedGB      = $usedGB
            FreeGB      = $freeGB
            FreePercent = $freePercent
            Path        = $absolutePath
        }
    }
    catch {
        Write-Error "Failed to get disk space info for path: $Path"
        Write-Error "Error: $_"
        return $null
    }
}

function Format-Bytes {
    <#
    .SYNOPSIS
    Format bytes into human-readable string
    #>
    param([long]$Bytes)
    
    if ($Bytes -ge 1TB) {
        return "{0:N2} TB" -f ($Bytes / 1TB)
    }
    elseif ($Bytes -ge 1GB) {
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

function Show-DiskUsageBar {
    <#
    .SYNOPSIS
    Display a visual progress bar of disk usage
    #>
    param(
        [Parameter(Mandatory = $true)]
        [double]$UsedPercent
    )
    
    $barLength = 50
    $filledLength = [math]::Floor(($UsedPercent / 100) * $barLength)
    $emptyLength = $barLength - $filledLength
    
    # Use simple ASCII characters for progress bar
    $bar = ("#" * $filledLength) + ("-" * $emptyLength)
    
    # Color based on usage
    if ($UsedPercent -ge 90) {
        $color = $ColorRed
    }
    elseif ($UsedPercent -ge 75) {
        $color = $ColorYellow
    }
    else {
        $color = $ColorGreen
    }
    
    $percentStr = [string]::Format("{0:0.0}", $UsedPercent)
    Write-Host "  Usage: ${color}[${bar}]${ColorReset} ${percentStr}%"
}

function Get-DirectorySize {
    <#
    .SYNOPSIS
    Get the size of a directory in bytes
    #>
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

# ============================================================================
# Main Logic
# ============================================================================

function Check-DiskSpace {
    param(
        [string]$CheckPath,
        [int]$MinimumGB,
        [bool]$VerboseOutput,
        [bool]$JsonOutput,
        [bool]$WarnOnly
    )
    
    Write-Header "Disk Space Budget Check"
    
    # Determine path to check
    if (-not $CheckPath) {
        $CheckPath = Get-DataDirFromEnv
        Write-Info "▸ Using DATA_DIR: $CheckPath"
    }
    else {
        Write-Info "▸ Checking path: $CheckPath"
    }
    
    Write-Info "▸ Minimum required: $MinimumGB GB"
    Write-Host ""
    
    # Get disk space info
    $diskInfo = Get-DiskSpaceInfo -Path $CheckPath
    
    if (-not $diskInfo) {
        Write-Error "Failed to retrieve disk space information"
        return 2
    }
    
    # Display results
    Write-Host "${ColorBold}Disk Space Information:${ColorReset}"
    Write-Host "  Drive:        $($diskInfo.DriveLetter)"
    Write-Host "  Total:        $($diskInfo.TotalGB) GB"
    Write-Host "  Used:         $($diskInfo.UsedGB) GB"
    Write-Host "  Free:         ${ColorBold}$($diskInfo.FreeGB) GB${ColorReset}"
    Write-Host "  Free:         $($diskInfo.FreePercent)%"
    Write-Host ""
    
    # Show usage bar
    $usedPercent = 100 - $diskInfo.FreePercent
    Show-DiskUsageBar -UsedPercent $usedPercent
    Write-Host ""
    
    # Show verbose info
    if ($VerboseOutput) {
        Write-Host "${ColorBold}Additional Information:${ColorReset}"
        Write-Host "  Checked path: $($diskInfo.Path)"
        
        # Show size of common directories if they exist
        $projectRoot = Split-Path $PSScriptRoot -Parent
        $commonDirs = @{
            "cache"   = Join-Path $projectRoot "cache"
            "outputs" = Join-Path $projectRoot "outputs"
            "weights" = Join-Path $projectRoot "weights"
            "data"    = Join-Path $projectRoot "data"
        }
        
        Write-Host "  Directory sizes:"
        foreach ($dir in $commonDirs.GetEnumerator()) {
            if (Test-Path $dir.Value) {
                $size = Get-DirectorySize -Path $dir.Value
                $sizeStr = Format-Bytes -Bytes $size
                Write-Host "    • $($dir.Key): $sizeStr"
            }
        }
        Write-Host ""
    }
    
    # Check against minimum requirement
    $hasEnoughSpace = $diskInfo.FreeGB -ge $MinimumGB
    
    # Output JSON if requested
    if ($JsonOutput) {
        $result = @{
            drive          = $diskInfo.DriveLetter
            total_gb       = $diskInfo.TotalGB
            used_gb        = $diskInfo.UsedGB
            free_gb        = $diskInfo.FreeGB
            free_percent   = $diskInfo.FreePercent
            minimum_gb     = $MinimumGB
            has_enough     = $hasEnoughSpace
            path           = $diskInfo.Path
            timestamp      = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        } | ConvertTo-Json -Depth 2
        
        Write-Host $result
        return $(if ($hasEnoughSpace) { 0 } else { 1 })
    }
    
    # Status message
    Write-Host "${ColorBold}Status:${ColorReset}"
    
    if ($hasEnoughSpace) {
        $available = $diskInfo.FreeGB - $MinimumGB
        Write-Success "Sufficient disk space available"
        Write-Host "  Required:  $MinimumGB GB"
        Write-Host "  Available: $($diskInfo.FreeGB) GB"
        $availableStr = [string]::Format("{0:0.00}", $available)
        Write-Host "  Margin:    $availableStr GB"
        Write-Host ""
        
        # Warning if approaching threshold
        if ($diskInfo.FreeGB -lt $WARNING_THRESHOLD_GB) {
            Write-Warning "Free space is below recommended threshold of $WARNING_THRESHOLD_GB GB"
            Write-Host "  Consider cleaning up old files or expanding storage"
            Write-Host ""
        }
        
        return 0
    }
    else {
        $shortfall = $MinimumGB - $diskInfo.FreeGB
        
        if ($WarnOnly) {
            Write-Warning "Insufficient disk space (warning only)"
            Write-Host "  Required:  $MinimumGB GB"
            Write-Host "  Available: $($diskInfo.FreeGB) GB"
            $shortfallStr2 = [string]::Format("{0:0.00}", $shortfall)
            Write-Host "  Shortfall: $shortfallStr2 GB"
            Write-Host ""
            return 0
        }
        else {
            Write-Error "Insufficient disk space!"
            Write-Host "  Required:  $MinimumGB GB"
            Write-Host "  Available: $($diskInfo.FreeGB) GB"
            $shortfallStr = [string]::Format("{0:0.00}", $shortfall)
            Write-Host "  Shortfall: ${ColorRed}${shortfallStr} GB${ColorReset}"
            Write-Host ""
            
            Write-Warning "Suggestions to free up space:"
            Write-Host "  1. Clean cache:         .\scripts\cleanup_cache.ps1"
            Write-Host "  2. Remove old outputs:  Remove-Item .\outputs\* -Recurse -Force"
            Write-Host "  3. Clear temp files:    Remove-Item .\temp\* -Recurse -Force"
            Write-Host "  4. Run disk cleanup:    cleanmgr.exe"
            Write-Host "  5. Move data to external drive"
            Write-Host ""
            
            return 1
        }
    }
}

# ============================================================================
# Entry Point
# ============================================================================

try {
    $exitCode = Check-DiskSpace `
        -CheckPath $Path `
        -MinimumGB $MinGB `
        -VerboseOutput:$Verbose `
        -JsonOutput:$Json `
        -WarnOnly:$WarningOnly
    
    exit $exitCode
}
catch {
    Write-Host ""
    Write-Host "ERROR: Unexpected error occurred" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    Write-Host ""
    exit 2
}

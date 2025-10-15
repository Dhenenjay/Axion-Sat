#!/usr/bin/env pwsh
# ==============================================================================
# 04_disk_budget.ps1 - Disk Space Budget Checker
# ==============================================================================
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
# ==============================================================================

param(
    [Parameter(Mandatory = $false)]
    [string]$Path = $null,
    
    [Parameter(Mandatory = $false)]
    [int]$MinGB = 20,
    
    [Parameter(Mandatory = $false)]
    [switch]$VerboseOutput,
    
    [Parameter(Mandatory = $false)]
    [switch]$Json,
    
    [Parameter(Mandatory = $false)]
    [switch]$WarningOnly
)

$ErrorActionPreference = "Stop"

# ==============================================================================
# Configuration
# ==============================================================================

$DEFAULT_MIN_GB = 20
$WARNING_THRESHOLD_GB = 50

# ==============================================================================
# Helper Functions
# ==============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Get-DataDirFromEnv {
    # Try environment variable first
    $dataDir = $env:DATA_DIR
    if ($dataDir) {
        return $dataDir
    }
    
    # Try loading from .env file
    $envFile = Join-Path (Join-Path $PSScriptRoot "..") ".env"
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
        Write-Host "Failed to get disk space info for path: $Path" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        return $null
    }
}

function Format-Bytes {
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
        $color = "Red"
    }
    elseif ($UsedPercent -ge 75) {
        $color = "Yellow"
    }
    else {
        $color = "Green"
    }
    
    $percentStr = "{0:0.0}" -f $UsedPercent
    Write-Host "  Usage: " -NoNewline
    Write-Host "[$bar]" -ForegroundColor $color -NoNewline
    Write-Host " $percentStr%"
}

function Get-DirectorySize {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return 0
    }
    
    try {
        $size = (Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue | 
            Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        if ($null -eq $size) {
            return 0
        }
        return $size
    }
    catch {
        return 0
    }
}

# ==============================================================================
# Main Logic
# ==============================================================================

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
        Write-Info "> Using DATA_DIR: $CheckPath"
    }
    else {
        Write-Info "> Checking path: $CheckPath"
    }
    
    Write-Info "> Minimum required: $MinimumGB GB"
    Write-Host ""
    
    # Get disk space info
    $diskInfo = Get-DiskSpaceInfo -Path $CheckPath
    
    if (-not $diskInfo) {
        Write-Error "Failed to retrieve disk space information"
        return 2
    }
    
    # Display results
    Write-Host "Disk Space Information:" -ForegroundColor White
    Write-Host "  Drive:        $($diskInfo.DriveLetter)"
    Write-Host "  Total:        $($diskInfo.TotalGB) GB"
    Write-Host "  Used:         $($diskInfo.UsedGB) GB"
    Write-Host "  Free:         $($diskInfo.FreeGB) GB" -ForegroundColor White
    Write-Host "  Free:         $($diskInfo.FreePercent)%"
    Write-Host ""
    
    # Show usage bar
    $usedPercent = 100 - $diskInfo.FreePercent
    Show-DiskUsageBar -UsedPercent $usedPercent
    Write-Host ""
    
    # Show verbose info
    if ($VerboseOutput) {
        Write-Host "Additional Information:" -ForegroundColor White
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
                Write-Host "    - $($dir.Key): $sizeStr"
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
    Write-Host "Status:" -ForegroundColor White
    
    if ($hasEnoughSpace) {
        $available = $diskInfo.FreeGB - $MinimumGB
        $availableStr = "{0:0.00}" -f $available
        
        Write-Success "Sufficient disk space available"
        Write-Host "  Required:  $MinimumGB GB"
        Write-Host "  Available: $($diskInfo.FreeGB) GB"
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
        $shortfallStr = "{0:0.00}" -f $shortfall
        
        if ($WarnOnly) {
            Write-Warning "Insufficient disk space (warning only)"
            Write-Host "  Required:  $MinimumGB GB"
            Write-Host "  Available: $($diskInfo.FreeGB) GB"
            Write-Host "  Shortfall: $shortfallStr GB"
            Write-Host ""
            return 0
        }
        else {
            Write-Error "Insufficient disk space!"
            Write-Host "  Required:  $MinimumGB GB"
            Write-Host "  Available: $($diskInfo.FreeGB) GB"
            Write-Host "  Shortfall: " -NoNewline
            Write-Host "$shortfallStr GB" -ForegroundColor Red
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

# ==============================================================================
# Entry Point
# ==============================================================================

try {
    $exitCode = Check-DiskSpace `
        -CheckPath $Path `
        -MinimumGB $MinGB `
        -VerboseOutput:$VerboseOutput `
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

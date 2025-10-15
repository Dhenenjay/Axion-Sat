#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Safely clean raw satellite data after tile generation is verified.

.DESCRIPTION
    This script deletes raw Sentinel SAFE archives and COG files only after
    confirming that:
    1. Tile manifest CSV files exist
    2. Manifests contain minimum number of rows (successful tile generation)
    3. Tile files referenced in manifests exist on disk
    
    This prevents accidental data loss by ensuring tiles are properly generated
    before removing source data.

.PARAMETER RawDir
    Directory containing raw SAFE/COG files to clean (default: data/raw)

.PARAMETER ManifestDir
    Directory containing tile manifest CSV files (default: data/index)

.PARAMETER MinTiles
    Minimum number of tiles required in manifest before deletion (default: 10)

.PARAMETER VerifyTiles
    Verify that tile files actually exist on disk (default: true)

.PARAMETER DryRun
    Show what would be deleted without actually deleting (default: false)

.PARAMETER Force
    Skip confirmation prompts (default: false)

.PARAMETER KeepManifests
    Comma-separated list of manifest files to check (default: all *.csv)

.EXAMPLE
    .\scripts\clean_raw.ps1
    # Interactive mode with defaults

.EXAMPLE
    .\scripts\clean_raw.ps1 -MinTiles 100 -DryRun
    # Dry run requiring at least 100 tiles

.EXAMPLE
    .\scripts\clean_raw.ps1 -RawDir data/raw/sentinel1 -ManifestDir data/index -Force
    # Clean Sentinel-1 data without prompts

.EXAMPLE
    .\scripts\clean_raw.ps1 -KeepManifests "train_tiles.csv,val_tiles.csv" -MinTiles 50
    # Only check specific manifests

.NOTES
    Author: Axion-Sat Project
    Version: 1.0.0
    
    Safety Features:
    - Validates manifests exist and have minimum rows
    - Optionally verifies tile files exist
    - Dry-run mode for testing
    - Confirmation prompts (unless -Force)
    - Detailed logging of actions
    - Rollback not possible - ensure backups exist!
#>

[CmdletBinding()]
param(
    [Parameter(HelpMessage="Directory containing raw SAFE/COG files")]
    [string]$RawDir = "data\raw",
    
    [Parameter(HelpMessage="Directory containing tile manifest CSV files")]
    [string]$ManifestDir = "data\index",
    
    [Parameter(HelpMessage="Minimum number of tiles required before deletion")]
    [int]$MinTiles = 10,
    
    [Parameter(HelpMessage="Verify tile files exist on disk")]
    [bool]$VerifyTiles = $true,
    
    [Parameter(HelpMessage="Show what would be deleted without deleting")]
    [switch]$DryRun,
    
    [Parameter(HelpMessage="Skip confirmation prompts")]
    [switch]$Force,
    
    [Parameter(HelpMessage="Comma-separated list of specific manifests to check")]
    [string]$KeepManifests = ""
)

# ============================================================================
# Configuration
# ============================================================================

$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

# SAFE file patterns (Sentinel-1 and Sentinel-2)
$SAFEPatterns = @(
    "S1*.SAFE",
    "S2*.SAFE"
)

# COG file patterns
$COGPatterns = @(
    "*.tif",
    "*.tiff",
    "S1*.cog",
    "S2*.cog"
)

# Manifest CSV patterns
$ManifestPatterns = @(
    "*_tiles.csv",
    "tile_index.csv",
    "dataset*.csv"
)

# ============================================================================
# Helper Functions
# ============================================================================

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("-" * 80) -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor DarkCyan
    Write-Host ("-" * 80) -ForegroundColor DarkCyan
}

function Format-FileSize {
    param([long]$Bytes)
    
    if ($Bytes -ge 1TB) {
        return "{0:N2} TB" -f ($Bytes / 1TB)
    } elseif ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    } elseif ($Bytes -ge 1MB) {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    } elseif ($Bytes -ge 1KB) {
        return "{0:N2} KB" -f ($Bytes / 1KB)
    } else {
        return "{0} bytes" -f $Bytes
    }
}

function Get-DirectorySize {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return 0
    }
    
    $size = (Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue | 
             Measure-Object -Property Length -Sum).Sum
    
    return $size
}

function Test-ManifestValid {
    param(
        [string]$ManifestPath,
        [int]$MinRows
    )
    
    if (-not (Test-Path $ManifestPath)) {
        return $false, "Manifest file not found"
    }
    
    try {
        # Read CSV
        $csv = Import-Csv -Path $ManifestPath
        $rowCount = ($csv | Measure-Object).Count
        
        if ($rowCount -lt $MinRows) {
            return $false, "Only $rowCount rows (minimum: $MinRows)"
        }
        
        return $true, "$rowCount rows"
    }
    catch {
        return $false, "Failed to read CSV: $($_.Exception.Message)"
    }
}

function Test-TilesExist {
    param(
        [string]$ManifestPath
    )
    
    try {
        $csv = Import-Csv -Path $ManifestPath
        
        # Check if tile_path column exists
        if (-not ($csv[0].PSObject.Properties.Name -contains "tile_path")) {
            return $false, 0, 0, "No 'tile_path' column found"
        }
        
        $totalTiles = 0
        $existingTiles = 0
        
        foreach ($row in $csv) {
            $totalTiles++
            $tilePath = $row.tile_path
            
            # Handle relative paths
            if (-not [System.IO.Path]::IsPathRooted($tilePath)) {
                $tilePath = Join-Path (Get-Location) $tilePath
            }
            
            if (Test-Path $tilePath) {
                $existingTiles++
            }
        }
        
        $allExist = ($existingTiles -eq $totalTiles)
        $message = "$existingTiles / $totalTiles tiles exist"
        
        return $allExist, $existingTiles, $totalTiles, $message
    }
    catch {
        return $false, 0, 0, "Failed to verify tiles: $($_.Exception.Message)"
    }
}

function Get-RawFiles {
    param([string]$Directory)
    
    $files = @()
    
    if (-not (Test-Path $Directory)) {
        Write-Warning "Raw directory not found: $Directory"
        return $files
    }
    
    # Find SAFE directories
    foreach ($pattern in $SAFEPatterns) {
        $found = Get-ChildItem -Path $Directory -Filter $pattern -Directory -Recurse -ErrorAction SilentlyContinue
        $files += $found
    }
    
    # Find COG files
    foreach ($pattern in $COGPatterns) {
        $found = Get-ChildItem -Path $Directory -Filter $pattern -File -Recurse -ErrorAction SilentlyContinue
        $files += $found
    }
    
    return $files
}

function Get-ManifestFiles {
    param(
        [string]$Directory,
        [string[]]$SpecificManifests
    )
    
    $manifests = @()
    
    if (-not (Test-Path $Directory)) {
        Write-Warning "Manifest directory not found: $Directory"
        return $manifests
    }
    
    if ($SpecificManifests.Count -gt 0) {
        # Check specific manifests
        foreach ($name in $SpecificManifests) {
            $path = Join-Path $Directory $name
            if (Test-Path $path) {
                $manifests += Get-Item $path
            } else {
                Write-Warning "Specified manifest not found: $name"
            }
        }
    } else {
        # Find all manifest files
        foreach ($pattern in $ManifestPatterns) {
            $found = Get-ChildItem -Path $Directory -Filter $pattern -File -Recurse -ErrorAction SilentlyContinue
            $manifests += $found
        }
    }
    
    return $manifests
}

# ============================================================================
# Main Script
# ============================================================================

function Main {
    Write-Header "Axion-Sat Raw Data Cleanup Utility"
    
    # Display configuration
    Write-ColorOutput "Configuration:" -Color Yellow
    Write-Host "  Raw Directory    : $RawDir"
    Write-Host "  Manifest Directory: $ManifestDir"
    Write-Host "  Minimum Tiles    : $MinTiles"
    Write-Host "  Verify Tiles     : $VerifyTiles"
    Write-Host "  Dry Run          : $DryRun"
    Write-Host "  Force Mode       : $Force"
    
    if ($KeepManifests) {
        Write-Host "  Specific Manifests: $KeepManifests"
    }
    
    Write-Host ""
    
    # Step 1: Find manifest files
    Write-Section "Step 1: Checking Tile Manifests"
    
    $specificManifests = @()
    if ($KeepManifests) {
        $specificManifests = $KeepManifests -split ',' | ForEach-Object { $_.Trim() }
    }
    
    $manifests = Get-ManifestFiles -Directory $ManifestDir -SpecificManifests $specificManifests
    
    if ($manifests.Count -eq 0) {
        Write-ColorOutput "ERROR: No manifest files found!" -Color Red
        Write-ColorOutput "Cannot proceed without tile manifests." -Color Red
        exit 1
    }
    
    Write-ColorOutput "Found $($manifests.Count) manifest file(s):" -Color Green
    
    $allManifestsValid = $true
    $totalValidatedTiles = 0
    
    foreach ($manifest in $manifests) {
        Write-Host ""
        Write-Host "  Manifest: $($manifest.Name)"
        
        # Check row count
        $isValid, $message = Test-ManifestValid -ManifestPath $manifest.FullName -MinRows $MinTiles
        
        if ($isValid) {
            Write-ColorOutput "    ✓ Valid: $message" -Color Green
            
            # Extract tile count from message
            if ($message -match '(\d+) rows') {
                $totalValidatedTiles += [int]$matches[1]
            }
            
            # Verify tiles exist if requested
            if ($VerifyTiles) {
                $tilesExist, $existing, $total, $tileMessage = Test-TilesExist -ManifestPath $manifest.FullName
                
                if ($tilesExist) {
                    Write-ColorOutput "    ✓ Tiles exist: $tileMessage" -Color Green
                } else {
                    Write-ColorOutput "    ⚠ Tiles missing: $tileMessage" -Color Yellow
                    
                    # Only fail if more than 10% are missing
                    $missingPercent = (($total - $existing) / $total) * 100
                    if ($missingPercent -gt 10) {
                        Write-ColorOutput "    ✗ Too many tiles missing (${missingPercent}%)" -Color Red
                        $allManifestsValid = $false
                    }
                }
            }
        } else {
            Write-ColorOutput "    ✗ Invalid: $message" -Color Red
            $allManifestsValid = $false
        }
    }
    
    if (-not $allManifestsValid) {
        Write-Host ""
        Write-ColorOutput "ERROR: One or more manifests are invalid!" -Color Red
        Write-ColorOutput "Cannot delete raw data - tile generation may be incomplete." -Color Red
        exit 1
    }
    
    Write-Host ""
    Write-ColorOutput "✓ All manifests valid ($totalValidatedTiles total tiles)" -Color Green
    
    # Step 2: Find raw files
    Write-Section "Step 2: Scanning Raw Data"
    
    $rawFiles = Get-RawFiles -Directory $RawDir
    
    if ($rawFiles.Count -eq 0) {
        Write-ColorOutput "No raw files found in $RawDir" -Color Yellow
        Write-ColorOutput "Nothing to clean." -Color Yellow
        exit 0
    }
    
    # Calculate sizes
    $totalSize = 0
    $safeCount = 0
    $cogCount = 0
    
    foreach ($file in $rawFiles) {
        if ($file.PSIsContainer) {
            # SAFE directory
            $safeCount++
            $size = Get-DirectorySize -Path $file.FullName
            $totalSize += $size
        } else {
            # COG file
            $cogCount++
            $totalSize += $file.Length
        }
    }
    
    Write-Host "Found raw data:"
    Write-Host "  SAFE archives: $safeCount"
    Write-Host "  COG files    : $cogCount"
    Write-Host "  Total size   : $(Format-FileSize $totalSize)"
    Write-Host ""
    
    # Step 3: Confirm deletion
    Write-Section "Step 3: Deletion Summary"
    
    if ($DryRun) {
        Write-ColorOutput "DRY RUN MODE - No files will be deleted" -Color Yellow
        Write-Host ""
    }
    
    Write-Host "The following will be deleted:"
    Write-Host ""
    
    # Group by type
    $safeFiles = $rawFiles | Where-Object { $_.PSIsContainer }
    $cogFiles = $rawFiles | Where-Object { -not $_.PSIsContainer }
    
    if ($safeFiles.Count -gt 0) {
        Write-ColorOutput "SAFE Archives ($($safeFiles.Count)):" -Color Cyan
        foreach ($file in $safeFiles | Select-Object -First 10) {
            $size = Get-DirectorySize -Path $file.FullName
            Write-Host "  - $($file.Name) [$(Format-FileSize $size)]"
        }
        if ($safeFiles.Count -gt 10) {
            Write-Host "  ... and $($safeFiles.Count - 10) more"
        }
        Write-Host ""
    }
    
    if ($cogFiles.Count -gt 0) {
        Write-ColorOutput "COG Files ($($cogFiles.Count)):" -Color Cyan
        foreach ($file in $cogFiles | Select-Object -First 10) {
            Write-Host "  - $($file.Name) [$(Format-FileSize $file.Length)]"
        }
        if ($cogFiles.Count -gt 10) {
            Write-Host "  ... and $($cogFiles.Count - 10) more"
        }
        Write-Host ""
    }
    
    Write-ColorOutput "Total space to reclaim: $(Format-FileSize $totalSize)" -Color Yellow
    Write-Host ""
    
    # Confirmation
    if (-not $Force -and -not $DryRun) {
        Write-ColorOutput "WARNING: This action cannot be undone!" -Color Red
        Write-Host ""
        $confirmation = Read-Host "Type 'DELETE' to confirm deletion"
        
        if ($confirmation -ne "DELETE") {
            Write-ColorOutput "Deletion cancelled." -Color Yellow
            exit 0
        }
    }
    
    # Step 4: Delete files
    if (-not $DryRun) {
        Write-Section "Step 4: Deleting Raw Data"
        
        $deletedCount = 0
        $failedCount = 0
        $reclaimedSpace = 0
        
        foreach ($file in $rawFiles) {
            try {
                $filePath = $file.FullName
                $fileName = $file.Name
                
                if ($file.PSIsContainer) {
                    # SAFE directory
                    $size = Get-DirectorySize -Path $filePath
                    Write-Verbose "Deleting SAFE: $fileName"
                    Remove-Item -Path $filePath -Recurse -Force
                    $reclaimedSpace += $size
                } else {
                    # COG file
                    Write-Verbose "Deleting COG: $fileName"
                    $size = $file.Length
                    Remove-Item -Path $filePath -Force
                    $reclaimedSpace += $size
                }
                
                $deletedCount++
                
                # Progress indicator
                if ($deletedCount % 10 -eq 0) {
                    $percent = [math]::Round(($deletedCount / $rawFiles.Count) * 100, 1)
                    Write-Host "  Progress: $deletedCount / $($rawFiles.Count) ($percent%)" -NoNewline
                    Write-Host "`r" -NoNewline
                }
            }
            catch {
                Write-Warning "Failed to delete $($file.Name): $($_.Exception.Message)"
                $failedCount++
            }
        }
        
        Write-Host ""
        Write-Host ""
        
        # Summary
        Write-Section "Deletion Complete"
        Write-ColorOutput "✓ Successfully deleted: $deletedCount files/directories" -Color Green
        Write-ColorOutput "✓ Reclaimed disk space: $(Format-FileSize $reclaimedSpace)" -Color Green
        
        if ($failedCount -gt 0) {
            Write-ColorOutput "⚠ Failed to delete: $failedCount items" -Color Yellow
        }
    } else {
        Write-Section "Dry Run Complete"
        Write-ColorOutput "No files were deleted (dry run mode)" -Color Yellow
        Write-ColorOutput "Would reclaim: $(Format-FileSize $totalSize)" -Color Cyan
    }
    
    Write-Host ""
    Write-ColorOutput "Done!" -Color Green
}

# ============================================================================
# Entry Point
# ============================================================================

try {
    Main
}
catch {
    Write-Host ""
    Write-ColorOutput "ERROR: $($_.Exception.Message)" -Color Red
    Write-Host $_.ScriptStackTrace
    exit 1
}

#!/usr/bin/env pwsh
# ============================================================================
# 05_dev_watch.ps1 - Development File Watcher
# ============================================================================
#
# This script watches for file changes in vc_lib/ and app/ directories and
# automatically runs linting (ruff), formatting checks (black), and tests
# (pytest) on change.
#
# Usage:
#   .\scripts\05_dev_watch.ps1                  # Watch vc_lib and app
#   .\scripts\05_dev_watch.ps1 -Fast            # Skip black, quick tests only
#   .\scripts\05_dev_watch.ps1 -Verbose         # Show detailed output
#   .\scripts\05_dev_watch.ps1 -NoTests         # Skip pytest
#
# Exit:
#   Ctrl+C to stop watching
#
# Requirements:
#   - ruff: pip install ruff
#   - black: pip install black
#   - pytest: pip install pytest
#
# ============================================================================

param(
    [Parameter(Mandatory = $false)]
    [switch]$Fast,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory = $false)]
    [switch]$NoTests,
    
    [Parameter(Mandatory = $false)]
    [switch]$NoLint,
    
    [Parameter(Mandatory = $false)]
    [switch]$NoFormat,
    
    [Parameter(Mandatory = $false)]
    [int]$Delay = 2
)

$ErrorActionPreference = "Continue"

# ============================================================================
# Configuration
# ============================================================================

# Project root
$ProjectRoot = Split-Path $PSScriptRoot -Parent

# Directories to watch
$WatchDirs = @(
    (Join-Path $ProjectRoot "vc_lib"),
    (Join-Path $ProjectRoot "app")
)

# File patterns to watch
$FilePatterns = @("*.py", "*.pyi")

# ANSI color codes
$ColorReset = "`e[0m"
$ColorBold = "`e[1m"
$ColorGreen = "`e[32m"
$ColorYellow = "`e[33m"
$ColorRed = "`e[31m"
$ColorCyan = "`e[36m"
$ColorBlue = "`e[34m"

# Global state
$script:LastRunTime = Get-Date
$script:RunCount = 0
$script:SuccessCount = 0
$script:FailCount = 0

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

function Get-Timestamp {
    return Get-Date -Format "HH:mm:ss"
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

function Get-ChangedFiles {
    param(
        [string[]]$Directories,
        [DateTime]$Since
    )
    
    $changedFiles = @()
    
    foreach ($dir in $Directories) {
        if (Test-Path $dir) {
            $files = Get-ChildItem -Path $dir -Recurse -Include $FilePatterns -File -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -gt $Since }
            
            $changedFiles += $files
        }
    }
    
    return $changedFiles
}

# ============================================================================
# Check Functions
# ============================================================================

function Invoke-RuffCheck {
    param([bool]$VerboseOutput)
    
    Write-Section "Running Ruff (Linter)"
    
    if (-not (Test-CommandExists "ruff")) {
        Write-Warning "ruff not installed. Skipping."
        return $true
    }
    
    try {
        $ruffArgs = @("check", "vc_lib", "app")
        if ($VerboseOutput) {
            $ruffArgs += "--verbose"
        }
        
        $output = & ruff @ruffArgs 2>&1
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Success "Ruff: No issues found"
            return $true
        }
        else {
            Write-Error "Ruff: Found issues"
            if ($VerboseOutput -or ($output -match "error" -or $output -match "warning")) {
                $output | ForEach-Object { Write-Info $_ }
            }
            return $false
        }
    }
    catch {
        Write-Error "Ruff: Error running command"
        Write-Info $_.Exception.Message
        return $false
    }
}

function Invoke-BlackCheck {
    param([bool]$VerboseOutput)
    
    Write-Section "Running Black (Formatter Check)"
    
    if (-not (Test-CommandExists "black")) {
        Write-Warning "black not installed. Skipping."
        return $true
    }
    
    try {
        $blackArgs = @("--check", "--diff", "vc_lib", "app")
        if (-not $VerboseOutput) {
            $blackArgs += "--quiet"
        }
        
        $output = & black @blackArgs 2>&1
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Success "Black: All files properly formatted"
            return $true
        }
        else {
            Write-Error "Black: Files need formatting"
            if ($VerboseOutput) {
                Write-Info "Run 'black vc_lib app' to fix"
                $output | ForEach-Object { Write-Info $_ }
            }
            else {
                Write-Info "Run 'black vc_lib app' to fix formatting issues"
            }
            return $false
        }
    }
    catch {
        Write-Error "Black: Error running command"
        Write-Info $_.Exception.Message
        return $false
    }
}

function Invoke-PytestQuick {
    param([bool]$VerboseOutput, [bool]$FastMode)
    
    Write-Section "Running Pytest (Quick Tests)"
    
    if (-not (Test-CommandExists "pytest")) {
        Write-Warning "pytest not installed. Skipping."
        return $true
    }
    
    try {
        # Fast mode: only run tests marked as 'fast' or with shorter timeout
        if ($FastMode) {
            $pytestArgs = @(
                "tests/",
                "-m", "not slow",
                "-x",  # Stop on first failure
                "--tb=short",
                "--quiet"
            )
        }
        else {
            $pytestArgs = @(
                "tests/",
                "-x",  # Stop on first failure
                "--tb=short"
            )
        }
        
        if ($VerboseOutput) {
            $pytestArgs += "-v"
        }
        else {
            $pytestArgs += "--quiet"
        }
        
        $output = & pytest @pytestArgs 2>&1
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Success "Pytest: All tests passed"
            return $true
        }
        else {
            Write-Error "Pytest: Tests failed"
            # Always show output for failures
            $output | ForEach-Object { Write-Info $_ }
            return $false
        }
    }
    catch {
        Write-Error "Pytest: Error running command"
        Write-Info $_.Exception.Message
        return $false
    }
}

# ============================================================================
# Main Check Function
# ============================================================================

function Invoke-AllChecks {
    param(
        [bool]$VerboseOutput,
        [bool]$SkipTests,
        [bool]$SkipLint,
        [bool]$SkipFormat,
        [bool]$FastMode
    )
    
    $script:RunCount++
    $timestamp = Get-Timestamp
    $allPassed = $true
    
    Write-Header "Check Run #$($script:RunCount) - $timestamp"
    
    # Run ruff
    if (-not $SkipLint) {
        $ruffPassed = Invoke-RuffCheck -VerboseOutput $VerboseOutput
        $allPassed = $allPassed -and $ruffPassed
    }
    
    # Run black
    if (-not $SkipFormat) {
        $blackPassed = Invoke-BlackCheck -VerboseOutput $VerboseOutput
        $allPassed = $allPassed -and $blackPassed
    }
    
    # Run pytest
    if (-not $SkipTests) {
        $pytestPassed = Invoke-PytestQuick -VerboseOutput $VerboseOutput -FastMode $FastMode
        $allPassed = $allPassed -and $pytestPassed
    }
    
    # Summary
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}$("-" * 79)${ColorReset}"
    
    if ($allPassed) {
        $script:SuccessCount++
        Write-Host "${ColorBold}${ColorGreen}✓ All checks passed!${ColorReset}"
    }
    else {
        $script:FailCount++
        Write-Host "${ColorBold}${ColorRed}✗ Some checks failed${ColorReset}"
    }
    
    $successRate = if ($script:RunCount -gt 0) { 
        [math]::Round(($script:SuccessCount / $script:RunCount) * 100, 1) 
    } else { 
        0 
    }
    
    Write-Host "${ColorCyan}  Runs: $($script:RunCount) | Passed: $($script:SuccessCount) | Failed: $($script:FailCount) | Success Rate: $successRate%${ColorReset}"
    Write-Host "${ColorBold}${ColorCyan}$("-" * 79)${ColorReset}"
    Write-Host ""
    
    $script:LastRunTime = Get-Date
}

# ============================================================================
# Main Watch Loop
# ============================================================================

function Start-DevWatch {
    param(
        [string[]]$Directories,
        [bool]$VerboseOutput,
        [bool]$SkipTests,
        [bool]$SkipLint,
        [bool]$SkipFormat,
        [bool]$FastMode,
        [int]$DelaySeconds
    )
    
    # Banner
    Clear-Host
    Write-Host "${ColorBold}${ColorCyan}"
    Write-Host "╔═══════════════════════════════════════════════════════════════════════════════╗"
    Write-Host "║                         AXION-SAT DEV WATCHER                                 ║"
    Write-Host "║                      File Change Detection & Validation                        ║"
    Write-Host "╚═══════════════════════════════════════════════════════════════════════════════╝"
    Write-Host "${ColorReset}"
    Write-Host ""
    Write-Info "Watching directories:"
    foreach ($dir in $Directories) {
        if (Test-Path $dir) {
            Write-Info "  • $dir"
        }
        else {
            Write-Warning "  • $dir (does not exist)"
        }
    }
    Write-Host ""
    Write-Info "File patterns: $($FilePatterns -join ', ')"
    Write-Info "Check delay: $DelaySeconds seconds"
    Write-Host ""
    Write-Info "Enabled checks:"
    Write-Info "  • Ruff (linter): $(if ($SkipLint) { "❌ Disabled" } else { "✓ Enabled" })"
    Write-Info "  • Black (formatter): $(if ($SkipFormat) { "❌ Disabled" } else { "✓ Enabled" })"
    Write-Info "  • Pytest (tests): $(if ($SkipTests) { "❌ Disabled" } else { "✓ Enabled" })"
    Write-Info "  • Fast mode: $(if ($FastMode) { "✓ Enabled" } else { "❌ Disabled" })"
    Write-Host ""
    Write-Info "Press ${ColorBold}Ctrl+C${ColorReset}${ColorCyan} to stop watching"
    Write-Host ""
    Write-Host "${ColorCyan}$("=" * 79)${ColorReset}"
    
    # Verify tools are installed
    Write-Host ""
    Write-Info "Checking required tools..."
    
    $missingTools = @()
    if (-not $SkipLint -and -not (Test-CommandExists "ruff")) {
        $missingTools += "ruff"
    }
    if (-not $SkipFormat -and -not (Test-CommandExists "black")) {
        $missingTools += "black"
    }
    if (-not $SkipTests -and -not (Test-CommandExists "pytest")) {
        $missingTools += "pytest"
    }
    
    if ($missingTools.Count -gt 0) {
        Write-Warning "Missing tools: $($missingTools -join ', ')"
        Write-Info "Install with: pip install $($missingTools -join ' ')"
        Write-Host ""
    }
    
    # Initial check
    Write-Info "Running initial checks..."
    Invoke-AllChecks `
        -VerboseOutput $VerboseOutput `
        -SkipTests $SkipTests `
        -SkipLint $SkipLint `
        -SkipFormat $SkipFormat `
        -FastMode $FastMode
    
    Write-Info "Watching for changes..."
    Write-Host ""
    
    # Watch loop
    try {
        while ($true) {
            Start-Sleep -Seconds $DelaySeconds
            
            # Check for changes
            $changedFiles = Get-ChangedFiles -Directories $Directories -Since $script:LastRunTime
            
            if ($changedFiles.Count -gt 0) {
                Write-Host ""
                Write-Host "${ColorYellow}⚡ Changes detected in $($changedFiles.Count) file(s):${ColorReset}"
                foreach ($file in $changedFiles | Select-Object -First 5) {
                    $relativePath = $file.FullName.Replace($ProjectRoot, ".")
                    Write-Info "  • $relativePath"
                }
                if ($changedFiles.Count -gt 5) {
                    Write-Info "  • ... and $($changedFiles.Count - 5) more"
                }
                Write-Host ""
                
                # Run checks
                Invoke-AllChecks `
                    -VerboseOutput $VerboseOutput `
                    -SkipTests $SkipTests `
                    -SkipLint $SkipLint `
                    -SkipFormat $SkipFormat `
                    -FastMode $FastMode
                
                Write-Info "Watching for changes..."
                Write-Host ""
            }
        }
    }
    catch [System.Management.Automation.PipelineStoppedException] {
        # User pressed Ctrl+C
        Write-Host ""
        Write-Host ""
        Write-Info "Watch stopped by user"
        Write-Host ""
    }
    finally {
        # Final stats
        Write-Host "${ColorBold}${ColorCyan}═══════════════════════════════════════════════════════════════════════════════${ColorReset}"
        Write-Host "${ColorBold}Session Summary${ColorReset}"
        Write-Host "${ColorCyan}$("-" * 79)${ColorReset}"
        Write-Host "  Total runs:      $($script:RunCount)"
        Write-Host "  Successful:      ${ColorGreen}$($script:SuccessCount)${ColorReset}"
        Write-Host "  Failed:          ${ColorRed}$($script:FailCount)${ColorReset}"
        
        if ($script:RunCount -gt 0) {
            $successRate = [math]::Round(($script:SuccessCount / $script:RunCount) * 100, 1)
            Write-Host "  Success rate:    $successRate%"
        }
        
        Write-Host "${ColorBold}${ColorCyan}═══════════════════════════════════════════════════════════════════════════════${ColorReset}"
        Write-Host ""
    }
}

# ============================================================================
# Entry Point
# ============================================================================

# Parse options
$skipTests = $NoTests.IsPresent
$skipLint = $NoLint.IsPresent
$skipFormat = $NoFormat.IsPresent -or $Fast.IsPresent
$fastMode = $Fast.IsPresent
$verboseOutput = $Verbose.IsPresent

# Validate directories exist
$validDirs = @()
foreach ($dir in $WatchDirs) {
    if (Test-Path $dir) {
        $validDirs += $dir
    }
    else {
        Write-Warning "Directory does not exist: $dir"
    }
}

if ($validDirs.Count -eq 0) {
    Write-Error "No valid directories to watch!"
    exit 1
}

# Start watching
Start-DevWatch `
    -Directories $validDirs `
    -VerboseOutput $verboseOutput `
    -SkipTests $skipTests `
    -SkipLint $skipLint `
    -SkipFormat $skipFormat `
    -FastMode $fastMode `
    -DelaySeconds $Delay

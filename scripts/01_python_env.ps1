<#
.SYNOPSIS
    Python environment setup script for Axion-Sat project.

.DESCRIPTION
    Creates a Python 3.11 virtual environment in .venv directory,
    activates it, upgrades pip, and displays version information.

.NOTES
    Requires: Python 3.11 installed and accessible via py launcher
    Author: Axion-Sat Team
#>

#Requires -Version 5.1

# Set strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ANSI color codes for output
$script:Colors = @{
    Red = "`e[91m"
    Green = "`e[92m"
    Yellow = "`e[93m"
    Blue = "`e[94m"
    Magenta = "`e[95m"
    Cyan = "`e[96m"
    Reset = "`e[0m"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "Reset"
    )
    $colorCode = $script:Colors[$Color]
    Write-Host "$colorCode$Message$($script:Colors.Reset)"
}

function Write-SectionHeader {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "===============================================================" "Cyan"
    Write-ColorOutput "  $Title" "Cyan"
    Write-ColorOutput "===============================================================" "Cyan"
}

function Test-Python311Available {
    <#
    .SYNOPSIS
        Checks if Python 3.11 is available via py launcher.
    #>
    try {
        $pythonVersion = py -3.11 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "[OK] Python 3.11 found: $pythonVersion" "Green"
            return $true
        } else {
            Write-ColorOutput "[ERROR] Python 3.11 not found" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Python 3.11 not available: $($_.Exception.Message)" "Red"
        return $false
    }
}

function New-PythonVirtualEnv {
    <#
    .SYNOPSIS
        Creates a Python 3.11 virtual environment in .venv directory.
    #>
    param(
        [string]$VenvPath = ".venv"
    )
    
    Write-SectionHeader "Creating Python Virtual Environment"
    
    # Check if .venv already exists
    if (Test-Path $VenvPath) {
        Write-ColorOutput "[!] Virtual environment already exists at: $VenvPath" "Yellow"
        
        # Check if it's a valid venv
        $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
        if (Test-Path $activateScript) {
            Write-ColorOutput "[OK] Using existing virtual environment" "Green"
            return $true
        } else {
            Write-ColorOutput "[!] Existing .venv directory is invalid, removing..." "Yellow"
            Remove-Item -Path $VenvPath -Recurse -Force
        }
    }
    
    try {
        Write-ColorOutput "[*] Creating virtual environment with Python 3.11..." "Blue"
        
        # Create the virtual environment
        py -3.11 -m venv $VenvPath
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "[OK] Virtual environment created successfully at: $VenvPath" "Green"
            return $true
        } else {
            Write-ColorOutput "[ERROR] Failed to create virtual environment" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error creating virtual environment: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Enable-VirtualEnv {
    <#
    .SYNOPSIS
        Activates the Python virtual environment.
    #>
    param(
        [string]$VenvPath = ".venv"
    )
    
    Write-SectionHeader "Activating Virtual Environment"
    
    $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    
    if (-not (Test-Path $activateScript)) {
        Write-ColorOutput "[ERROR] Activation script not found at: $activateScript" "Red"
        return $false
    }
    
    try {
        Write-ColorOutput "[*] Activating virtual environment..." "Blue"
        
        # Activate the virtual environment
        & $activateScript
        
        # Verify activation by checking if python is now from venv
        $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        
        if ($pythonPath -like "*$VenvPath*") {
            Write-ColorOutput "[OK] Virtual environment activated" "Green"
            Write-ColorOutput "    Python path: $pythonPath" "Green"
            return $true
        } else {
            Write-ColorOutput "[!] Warning: Virtual environment may not be properly activated" "Yellow"
            return $true
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error activating virtual environment: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Update-Pip {
    <#
    .SYNOPSIS
        Upgrades pip to the latest version.
    #>
    Write-SectionHeader "Upgrading pip"
    
    try {
        Write-ColorOutput "[*] Upgrading pip to latest version..." "Blue"
        
        # Upgrade pip
        python -m pip install --upgrade pip --quiet
        
        if ($LASTEXITCODE -eq 0) {
            # Get pip version
            $pipVersion = python -m pip --version
            Write-ColorOutput "[OK] pip upgraded successfully" "Green"
            Write-ColorOutput "    $pipVersion" "Green"
            return $true
        } else {
            Write-ColorOutput "[ERROR] Failed to upgrade pip" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error upgrading pip: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Show-PythonVersion {
    <#
    .SYNOPSIS
        Displays Python version information.
    #>
    Write-SectionHeader "Python Environment Information"
    
    try {
        # Python version
        $pythonVersion = python --version 2>&1
        Write-ColorOutput "[Python]  $pythonVersion" "Magenta"
        
        # Python executable path
        $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        Write-ColorOutput "[Path]    $pythonPath" "Magenta"
        
        # Pip version
        $pipVersion = python -m pip --version 2>&1
        Write-ColorOutput "[pip]     $pipVersion" "Magenta"
        
        # Virtual environment status
        if ($env:VIRTUAL_ENV) {
            Write-ColorOutput "[venv]    Active ($env:VIRTUAL_ENV)" "Magenta"
        } else {
            Write-ColorOutput "[venv]    Not detected (may need to check activation)" "Yellow"
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error retrieving version information: $($_.Exception.Message)" "Red"
    }
}

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

function Main {
    Write-ColorOutput @"

    ===============================================================
    
              AXION-SAT PYTHON ENVIRONMENT SETUP
    
    ===============================================================

"@ "Cyan"

    # Step 1: Check Python 3.11 availability
    Write-SectionHeader "Checking Python 3.11 Installation"
    
    if (-not (Test-Python311Available)) {
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Python 3.11 is required but not found!" "Red"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Please install Python 3.11 from:" "Yellow"
        Write-ColorOutput "  https://www.python.org/downloads/" "Yellow"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Or use winget:" "Yellow"
        Write-ColorOutput "  winget install Python.Python.3.11" "Yellow"
        Write-ColorOutput "" "Reset"
        exit 1
    }
    
    # Step 2: Create virtual environment
    $venvSuccess = New-PythonVirtualEnv -VenvPath ".venv"
    
    if (-not $venvSuccess) {
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "[ERROR] Failed to create virtual environment. Exiting." "Red"
        exit 1
    }
    
    # Step 3: Activate virtual environment
    $activateSuccess = Enable-VirtualEnv -VenvPath ".venv"
    
    if (-not $activateSuccess) {
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "[ERROR] Failed to activate virtual environment. Exiting." "Red"
        exit 1
    }
    
    # Step 4: Upgrade pip
    $pipSuccess = Update-Pip
    
    # Step 5: Display version information
    Show-PythonVersion
    
    # Final summary
    Write-SectionHeader "Setup Summary"
    
    $allSuccess = $venvSuccess -and $activateSuccess -and $pipSuccess
    
    if ($allSuccess) {
        Write-ColorOutput "[OK] Python environment setup completed successfully!" "Green"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "To activate the virtual environment in a new session, run:" "Green"
        Write-ColorOutput "  .\.venv\Scripts\Activate.ps1" "Cyan"
    } else {
        Write-ColorOutput "[!] Setup completed with warnings." "Yellow"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Please review the output above for any issues." "Yellow"
    }
    
    Write-Host ""
    Write-ColorOutput "===============================================================" "Cyan"
    Write-Host ""
}

# Run main function
Main

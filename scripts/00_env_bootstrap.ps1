<#
.SYNOPSIS
    Environment bootstrap script for Axion-Sat project setup.

.DESCRIPTION
    Checks for admin privileges, enables long paths, installs Git and Git LFS if missing,
    configures Git LFS, and prints version information.

.NOTES
    Requires: Windows 11, PowerShell 5.1+
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

function Test-Administrator {
    <#
    .SYNOPSIS
        Checks if the current PowerShell session has administrator privileges.
    #>
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Enable-LongPaths {
    <#
    .SYNOPSIS
        Enables Windows long path support and Git long paths.
    #>
    Write-SectionHeader "Enabling Long Path Support"
    
    try {
        # Enable Windows long path support (requires admin)
        Write-ColorOutput "[*] Checking Windows long path registry setting..." "Blue"
        $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
        $currentValue = Get-ItemProperty -Path $regPath -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
        
        if ($currentValue.LongPathsEnabled -ne 1) {
            Write-ColorOutput "[+] Enabling Windows long paths in registry..." "Yellow"
            Set-ItemProperty -Path $regPath -Name "LongPathsEnabled" -Value 1 -Type DWord
            Write-ColorOutput "[OK] Windows long paths enabled" "Green"
        } else {
            Write-ColorOutput "[OK] Windows long paths already enabled" "Green"
        }
        
        # Enable Git long paths at system level
        Write-ColorOutput "[*] Configuring Git long paths..." "Blue"
        $gitInstalled = Get-Command git -ErrorAction SilentlyContinue
        
        if ($gitInstalled) {
            git config --system core.longpaths true 2>$null
            $longPathsValue = git config --system --get core.longpaths 2>$null
            if ($longPathsValue -eq "true") {
                Write-ColorOutput "[OK] Git long paths enabled (system-level)" "Green"
            } else {
                Write-ColorOutput "[!] Warning: Could not verify Git long paths setting" "Yellow"
            }
        } else {
            Write-ColorOutput "[!] Git not installed yet, will configure after installation" "Yellow"
        }
        
        return $true
    }
    catch {
        Write-ColorOutput "[ERROR] Error enabling long paths: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Test-CommandExists {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Install-GitForWindows {
    <#
    .SYNOPSIS
        Installs Git for Windows using winget if not already installed.
    #>
    Write-SectionHeader "Checking Git Installation"
    
    if (Test-CommandExists "git") {
        $gitVersion = git --version
        Write-ColorOutput "[OK] Git is already installed: $gitVersion" "Green"
        return $true
    }
    
    Write-ColorOutput "[*] Git not found. Installing Git for Windows..." "Yellow"
    
    # Check if winget is available
    if (-not (Test-CommandExists "winget")) {
        Write-ColorOutput "[ERROR] Error: winget not found. Please install Git manually from https://git-scm.com/download/win" "Red"
        return $false
    }
    
    try {
        Write-ColorOutput "[+] Installing Git via winget..." "Blue"
        winget install --id Git.Git -e --source winget --silent --accept-package-agreements --accept-source-agreements
        
        # Refresh environment variables
        $machinePath = [System.Environment]::GetEnvironmentVariable("Path","Machine")
        $userPath = [System.Environment]::GetEnvironmentVariable("Path","User")
        $env:Path = $machinePath + ";" + $userPath
        
        # Verify installation
        if (Test-CommandExists "git") {
            $gitVersion = git --version
            Write-ColorOutput "[OK] Git installed successfully: $gitVersion" "Green"
            
            # Configure Git long paths after installation
            git config --system core.longpaths true 2>$null
            Write-ColorOutput "[OK] Git long paths configured" "Green"
            
            return $true
        } else {
            Write-ColorOutput "[ERROR] Git installation completed but git command not found. You may need to restart your terminal." "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error installing Git: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Install-GitLFS {
    <#
    .SYNOPSIS
        Installs Git LFS using winget if not already installed.
    #>
    Write-SectionHeader "Checking Git LFS Installation"
    
    if (Test-CommandExists "git-lfs") {
        $lfsVersion = git lfs version
        Write-ColorOutput "[OK] Git LFS is already installed: $lfsVersion" "Green"
        return $true
    }
    
    Write-ColorOutput "[*] Git LFS not found. Installing Git LFS..." "Yellow"
    
    # Check if winget is available
    if (-not (Test-CommandExists "winget")) {
        Write-ColorOutput "[ERROR] Error: winget not found. Please install Git LFS manually from https://git-lfs.github.com/" "Red"
        return $false
    }
    
    try {
        Write-ColorOutput "[+] Installing Git LFS via winget..." "Blue"
        winget install --id GitHub.GitLFS -e --source winget --silent --accept-package-agreements --accept-source-agreements
        
        # Refresh environment variables
        $machinePath = [System.Environment]::GetEnvironmentVariable("Path","Machine")
        $userPath = [System.Environment]::GetEnvironmentVariable("Path","User")
        $env:Path = $machinePath + ";" + $userPath
        
        # Verify installation
        if (Test-CommandExists "git-lfs") {
            $lfsVersion = git lfs version
            Write-ColorOutput "[OK] Git LFS installed successfully: $lfsVersion" "Green"
            return $true
        } else {
            Write-ColorOutput "[ERROR] Git LFS installation completed but git-lfs command not found. You may need to restart your terminal." "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error installing Git LFS: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Initialize-GitLFS {
    <#
    .SYNOPSIS
        Initializes Git LFS for the current user.
    #>
    Write-SectionHeader "Initializing Git LFS"
    
    try {
        Write-ColorOutput "[*] Running git lfs install..." "Blue"
        $output = git lfs install 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "[OK] Git LFS initialized successfully" "Green"
            Write-ColorOutput "    $output" "Green"
            return $true
        } else {
            Write-ColorOutput "[ERROR] Error initializing Git LFS: $output" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[ERROR] Error initializing Git LFS: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Show-VersionInfo {
    <#
    .SYNOPSIS
        Displays version information for installed tools.
    #>
    Write-SectionHeader "Version Information"
    
    # Git version
    if (Test-CommandExists "git") {
        $gitVersion = git --version
        Write-ColorOutput "[Git]        $gitVersion" "Magenta"
    } else {
        Write-ColorOutput "[Git]        Not installed" "Red"
    }
    
    # Git LFS version
    if (Test-CommandExists "git-lfs") {
        $lfsVersion = git lfs version
        Write-ColorOutput "[Git LFS]    $lfsVersion" "Magenta"
    } else {
        Write-ColorOutput "[Git LFS]    Not installed" "Red"
    }
    
    # PowerShell version
    Write-ColorOutput "[PowerShell] $($PSVersionTable.PSVersion.ToString())" "Magenta"
    
    # Windows version
    $osInfo = Get-CimInstance Win32_OperatingSystem
    Write-ColorOutput "[OS]         $($osInfo.Caption) (Build $($osInfo.BuildNumber))" "Magenta"
}

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

function Main {
    Write-ColorOutput @"

    ===============================================================
    
              AXION-SAT ENVIRONMENT BOOTSTRAP
    
    ===============================================================

"@ "Cyan"

    # Step 1: Check for administrator privileges
    Write-SectionHeader "Checking Administrator Privileges"
    
    if (Test-Administrator) {
        Write-ColorOutput "[OK] Running with administrator privileges" "Green"
    } else {
        Write-ColorOutput "[ERROR] NOT running as administrator!" "Red"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "This script requires administrator privileges to:" "Yellow"
        Write-ColorOutput "  * Enable Windows long path support" "Yellow"
        Write-ColorOutput "  * Configure system-level Git settings" "Yellow"
        Write-ColorOutput "  * Install software via winget" "Yellow"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Please run PowerShell as Administrator and try again." "Yellow"
        Write-ColorOutput "" "Reset"
        exit 1
    }
    
    # Step 2: Enable long paths
    $longPathsSuccess = Enable-LongPaths
    
    # Step 3: Install Git if missing
    $gitSuccess = Install-GitForWindows
    
    # Step 4: Install Git LFS if missing
    $lfsSuccess = Install-GitLFS
    
    # Step 5: Initialize Git LFS
    if ($lfsSuccess) {
        $lfsInitSuccess = Initialize-GitLFS
    }
    
    # Step 6: Display version information
    Show-VersionInfo
    
    # Final summary
    Write-SectionHeader "Bootstrap Summary"
    
    $allSuccess = $longPathsSuccess -and $gitSuccess -and $lfsSuccess
    
    if ($allSuccess) {
        Write-ColorOutput "[OK] All bootstrap tasks completed successfully!" "Green"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Your environment is now ready for Axion-Sat development." "Green"
    } else {
        Write-ColorOutput "[!] Some tasks completed with warnings or errors." "Yellow"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Please review the output above and address any issues." "Yellow"
    }
    
    Write-Host ""
    Write-ColorOutput "===============================================================" "Cyan"
    Write-Host ""
}

# Run main function
Main

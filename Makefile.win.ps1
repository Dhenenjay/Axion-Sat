# ============================================================================
# Makefile.win.ps1 - Windows PowerShell Build System for Axion-Sat
# ============================================================================
#
# A Make-like task runner for Windows PowerShell that orchestrates the
# Axion-Sat ML pipeline. Provides convenient aliases for common tasks.
#
# Usage:
#   .\Makefile.win.ps1 <task> [args...]
#
# Examples:
#   .\Makefile.win.ps1 env              # Bootstrap environment
#   .\Makefile.win.ps1 install          # Install Python dependencies
#   .\Makefile.win.ps1 gpu-check        # Check GPU availability
#   .\Makefile.win.ps1 train-s1         # Train stage 1 model
#   .\Makefile.win.ps1 help             # Show this help message
#
# ============================================================================

param(
    [Parameter(Position = 0, Mandatory = $false)]
    [string]$Task = "help",
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

# ============================================================================
# Configuration
# ============================================================================

$ErrorActionPreference = "Stop"
$ScriptsDir = Join-Path $PSScriptRoot "scripts"
$ProjectRoot = $PSScriptRoot

# ANSI color codes for pretty output
$ColorReset = "`e[0m"
$ColorBold = "`e[1m"
$ColorGreen = "`e[32m"
$ColorBlue = "`e[34m"
$ColorYellow = "`e[33m"
$ColorRed = "`e[31m"
$ColorCyan = "`e[36m"

# ============================================================================
# Helper Functions
# ============================================================================

function Write-TaskHeader {
    param([string]$TaskName, [string]$Description)
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}═══════════════════════════════════════════════════════════════════════════════${ColorReset}"
    Write-Host "${ColorBold}${ColorGreen}▶ Task: ${ColorBlue}$TaskName${ColorReset}"
    Write-Host "${ColorCyan}  $Description${ColorReset}"
    Write-Host "${ColorBold}${ColorCyan}═══════════════════════════════════════════════════════════════════════════════${ColorReset}"
    Write-Host ""
}

function Write-TaskSuccess {
    param([string]$TaskName)
    Write-Host ""
    Write-Host "${ColorBold}${ColorGreen}✓ Task '$TaskName' completed successfully!${ColorReset}"
    Write-Host ""
}

function Write-TaskError {
    param([string]$TaskName, [string]$ErrorMessage)
    Write-Host ""
    Write-Host "${ColorBold}${ColorRed}✗ Task '$TaskName' failed!${ColorReset}"
    Write-Host "${ColorRed}  Error: $ErrorMessage${ColorReset}"
    Write-Host ""
}

function Invoke-Script {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments = @()
    )
    
    if (-not (Test-Path $ScriptPath)) {
        throw "Script not found: $ScriptPath"
    }
    
    Write-Host "${ColorYellow}▸ Executing: $ScriptPath $($Arguments -join ' ')${ColorReset}"
    Write-Host ""
    
    & $ScriptPath @Arguments
    
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        throw "Script exited with code $LASTEXITCODE"
    }
}

function Test-Python {
    try {
        $pythonVersion = python --version 2>&1
        return $true
    } catch {
        return $false
    }
}

function Test-VirtualEnv {
    return $null -ne $env:VIRTUAL_ENV
}

# ============================================================================
# Task Definitions
# ============================================================================

function Task-Help {
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}Axion-Sat Build System - Available Tasks${ColorReset}"
    Write-Host "${ColorCyan}════════════════════════════════════════${ColorReset}"
    Write-Host ""
    Write-Host "${ColorBold}Setup & Installation:${ColorReset}"
    Write-Host "  ${ColorGreen}env${ColorReset}           Bootstrap environment (Conda/venv setup)"
    Write-Host "  ${ColorGreen}install${ColorReset}       Install Python dependencies from requirements.txt"
    Write-Host "  ${ColorGreen}gpu-check${ColorReset}     Verify GPU/CUDA availability"
    Write-Host ""
    Write-Host "${ColorBold}Development & Serving:${ColorReset}"
    Write-Host "  ${ColorGreen}run-server${ColorReset}    Start the inference/prediction server"
    Write-Host "  ${ColorGreen}warm-cache${ColorReset}    Pre-download and cache satellite tiles"
    Write-Host ""
    Write-Host "${ColorBold}Training Pipeline:${ColorReset}"
    Write-Host "  ${ColorGreen}train-s1${ColorReset}      Train stage 1 model (feature extraction)"
    Write-Host "  ${ColorGreen}train-s2${ColorReset}      Train stage 2 model (segmentation)"
    Write-Host "  ${ColorGreen}train-s3${ColorReset}      Train stage 3 model (refinement)"
    Write-Host ""
    Write-Host "${ColorBold}Evaluation & Demo:${ColorReset}"
    Write-Host "  ${ColorGreen}eval${ColorReset}          Run model evaluation on test set"
    Write-Host "  ${ColorGreen}demo${ColorReset}          Run interactive demo/visualization"
    Write-Host ""
    Write-Host "${ColorBold}Utilities:${ColorReset}"
    Write-Host "  ${ColorGreen}help${ColorReset}          Show this help message"
    Write-Host "  ${ColorGreen}list${ColorReset}          List all available tasks"
    Write-Host ""
    Write-Host "${ColorBold}Usage:${ColorReset}"
    Write-Host "  .\Makefile.win.ps1 ${ColorYellow}<task>${ColorReset} [args...]"
    Write-Host ""
    Write-Host "${ColorBold}Examples:${ColorReset}"
    Write-Host "  .\Makefile.win.ps1 env"
    Write-Host "  .\Makefile.win.ps1 install"
    Write-Host "  .\Makefile.win.ps1 gpu-check"
    Write-Host "  .\Makefile.win.ps1 train-s1"
    Write-Host "  .\Makefile.win.ps1 eval"
    Write-Host ""
}

function Task-List {
    Write-Host ""
    Write-Host "${ColorBold}${ColorCyan}Available Tasks:${ColorReset}"
    Write-Host ""
    $tasks = @(
        "env", "install", "gpu-check", "run-server", "warm-cache",
        "train-s1", "train-s2", "train-s3", "eval", "demo", "help", "list"
    )
    foreach ($t in $tasks) {
        Write-Host "  • $t"
    }
    Write-Host ""
}

# ============================================================================
# Task: env
# ============================================================================

function Task-Env {
    Write-TaskHeader "env" "Bootstrap Python environment (Conda/venv setup)"
    
    try {
        # Try bootstrap script first
        $bootstrapScript = Join-Path $ScriptsDir "00_env_bootstrap.ps1"
        if (Test-Path $bootstrapScript) {
            Invoke-Script $bootstrapScript
        } else {
            # Fallback to Python environment script
            $pythonEnvScript = Join-Path $ScriptsDir "01_python_env.ps1"
            if (Test-Path $pythonEnvScript) {
                Invoke-Script $pythonEnvScript
            } else {
                throw "Environment setup scripts not found"
            }
        }
        
        Write-TaskSuccess "env"
    } catch {
        Write-TaskError "env" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: install
# ============================================================================

function Task-Install {
    Write-TaskHeader "install" "Install Python dependencies from requirements.txt"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        $requirementsFile = Join-Path $ProjectRoot "requirements.txt"
        if (-not (Test-Path $requirementsFile)) {
            throw "requirements.txt not found in project root"
        }
        
        Write-Host "${ColorYellow}▸ Installing dependencies...${ColorReset}"
        Write-Host ""
        
        python -m pip install --upgrade pip
        python -m pip install -r $requirementsFile
        
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed with exit code $LASTEXITCODE"
        }
        
        Write-TaskSuccess "install"
    } catch {
        Write-TaskError "install" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: gpu-check
# ============================================================================

function Task-GpuCheck {
    Write-TaskHeader "gpu-check" "Verify GPU/CUDA availability"
    
    try {
        $gpuScript = Join-Path $ScriptsDir "02_gpu_smoke.ps1"
        if (-not (Test-Path $gpuScript)) {
            throw "GPU smoke test script not found: $gpuScript"
        }
        
        Invoke-Script $gpuScript
        
        Write-TaskSuccess "gpu-check"
    } catch {
        Write-TaskError "gpu-check" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: run-server
# ============================================================================

function Task-RunServer {
    Write-TaskHeader "run-server" "Start the inference/prediction server"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for server script in scripts/ or app/
        $serverScript = Join-Path $ScriptsDir "run_server.ps1"
        $fallbackScript = Join-Path $ProjectRoot "app\server.py"
        
        if (Test-Path $serverScript) {
            Invoke-Script $serverScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Starting server via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Server script not found. Expected: $serverScript or $fallbackScript"
        }
        
        Write-TaskSuccess "run-server"
    } catch {
        Write-TaskError "run-server" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: warm-cache
# ============================================================================

function Task-WarmCache {
    Write-TaskHeader "warm-cache" "Pre-download and cache satellite tiles"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for warm cache script
        $cacheScript = Join-Path $ScriptsDir "warm_cache.ps1"
        $fallbackScript = Join-Path $ProjectRoot "scripts\warm_cache.py"
        
        if (Test-Path $cacheScript) {
            Invoke-Script $cacheScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Warming cache via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Cache warming script not found. Expected: $cacheScript or $fallbackScript"
        }
        
        Write-TaskSuccess "warm-cache"
    } catch {
        Write-TaskError "warm-cache" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: train-s1
# ============================================================================

function Task-TrainS1 {
    Write-TaskHeader "train-s1" "Train stage 1 model (feature extraction)"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for training script
        $trainScript = Join-Path $ScriptsDir "train_s1.ps1"
        $fallbackScript = Join-Path $ProjectRoot "vc_lib\training\train_stage1.py"
        
        if (Test-Path $trainScript) {
            Invoke-Script $trainScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Training Stage 1 via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Stage 1 training script not found. Expected: $trainScript or $fallbackScript"
        }
        
        Write-TaskSuccess "train-s1"
    } catch {
        Write-TaskError "train-s1" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: train-s2
# ============================================================================

function Task-TrainS2 {
    Write-TaskHeader "train-s2" "Train stage 2 model (segmentation)"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for training script
        $trainScript = Join-Path $ScriptsDir "train_s2.ps1"
        $fallbackScript = Join-Path $ProjectRoot "vc_lib\training\train_stage2.py"
        
        if (Test-Path $trainScript) {
            Invoke-Script $trainScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Training Stage 2 via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Stage 2 training script not found. Expected: $trainScript or $fallbackScript"
        }
        
        Write-TaskSuccess "train-s2"
    } catch {
        Write-TaskError "train-s2" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: train-s3
# ============================================================================

function Task-TrainS3 {
    Write-TaskHeader "train-s3" "Train stage 3 model (refinement)"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for training script
        $trainScript = Join-Path $ScriptsDir "train_s3.ps1"
        $fallbackScript = Join-Path $ProjectRoot "vc_lib\training\train_stage3.py"
        
        if (Test-Path $trainScript) {
            Invoke-Script $trainScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Training Stage 3 via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Stage 3 training script not found. Expected: $trainScript or $fallbackScript"
        }
        
        Write-TaskSuccess "train-s3"
    } catch {
        Write-TaskError "train-s3" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: eval
# ============================================================================

function Task-Eval {
    Write-TaskHeader "eval" "Run model evaluation on test set"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for evaluation script
        $evalScript = Join-Path $ScriptsDir "eval.ps1"
        $fallbackScript = Join-Path $ProjectRoot "vc_lib\evaluation\evaluate.py"
        
        if (Test-Path $evalScript) {
            Invoke-Script $evalScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Running evaluation via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Evaluation script not found. Expected: $evalScript or $fallbackScript"
        }
        
        Write-TaskSuccess "eval"
    } catch {
        Write-TaskError "eval" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task: demo
# ============================================================================

function Task-Demo {
    Write-TaskHeader "demo" "Run interactive demo/visualization"
    
    try {
        if (-not (Test-Python)) {
            throw "Python not found. Run '.\Makefile.win.ps1 env' first."
        }
        
        # Check for demo script
        $demoScript = Join-Path $ScriptsDir "demo.ps1"
        $fallbackScript = Join-Path $ProjectRoot "app\demo.py"
        
        if (Test-Path $demoScript) {
            Invoke-Script $demoScript $Args
        } elseif (Test-Path $fallbackScript) {
            Write-Host "${ColorYellow}▸ Running demo via Python...${ColorReset}"
            Write-Host ""
            python $fallbackScript @Args
        } else {
            throw "Demo script not found. Expected: $demoScript or $fallbackScript"
        }
        
        Write-TaskSuccess "demo"
    } catch {
        Write-TaskError "demo" $_.Exception.Message
        exit 1
    }
}

# ============================================================================
# Task Router
# ============================================================================

function Invoke-Task {
    param([string]$TaskName)
    
    switch ($TaskName.ToLower()) {
        "help"        { Task-Help }
        "list"        { Task-List }
        "env"         { Task-Env }
        "install"     { Task-Install }
        "gpu-check"   { Task-GpuCheck }
        "run-server"  { Task-RunServer }
        "warm-cache"  { Task-WarmCache }
        "train-s1"    { Task-TrainS1 }
        "train-s2"    { Task-TrainS2 }
        "train-s3"    { Task-TrainS3 }
        "eval"        { Task-Eval }
        "demo"        { Task-Demo }
        default {
            Write-Host ""
            Write-Host "${ColorBold}${ColorRed}Error: Unknown task '$TaskName'${ColorReset}"
            Write-Host ""
            Write-Host "Run ${ColorYellow}.\Makefile.win.ps1 help${ColorReset} to see available tasks."
            Write-Host ""
            exit 1
        }
    }
}

# ============================================================================
# Main Entry Point
# ============================================================================

try {
    Invoke-Task $Task
} catch {
    Write-Host ""
    Write-Host "${ColorBold}${ColorRed}Fatal Error:${ColorReset}"
    Write-Host "${ColorRed}$($_.Exception.Message)${ColorReset}"
    Write-Host ""
    exit 1
}

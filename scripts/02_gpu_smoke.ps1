<#
.SYNOPSIS
    GPU smoke test script - Verifies CUDA availability in PyTorch.

.DESCRIPTION
    Checks if PyTorch can detect and use CUDA/GPU, displays GPU information
    including device name, total VRAM, and available VRAM. Exits with non-zero
    status if CUDA is not available.

.NOTES
    Requires: PyTorch with CUDA support, Python 3.11
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

function Test-GPUAvailability {
    <#
    .SYNOPSIS
        Tests if CUDA/GPU is available in PyTorch and retrieves GPU information.
    #>
    
    Write-SectionHeader "GPU SMOKE TEST"
    
    # Python script to check CUDA availability and get GPU info
    $pythonScript = @"
import sys
import torch
import json

try:
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        print(json.dumps({
            'cuda_available': False,
            'error': 'CUDA is not available in PyTorch'
        }))
        sys.exit(1)
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    
    # Get VRAM information
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / (1024**3)  # Convert to GB
    
    # Get currently allocated and free memory
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
    free_memory = total_memory - (reserved_memory)
    
    # Get PyTorch and CUDA versions
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda
    
    # Output JSON for PowerShell to parse
    result = {
        'cuda_available': True,
        'gpu_count': gpu_count,
        'gpu_name': gpu_name,
        'pytorch_version': pytorch_version,
        'cuda_version': cuda_version,
        'total_memory_gb': round(total_memory, 2),
        'allocated_memory_gb': round(allocated_memory, 2),
        'reserved_memory_gb': round(reserved_memory, 2),
        'free_memory_gb': round(free_memory, 2),
        'compute_capability': f'{props.major}.{props.minor}'
    }
    
    print(json.dumps(result))
    sys.exit(0)
    
except Exception as e:
    print(json.dumps({
        'cuda_available': False,
        'error': str(e)
    }))
    sys.exit(1)
"@
    
    try {
        Write-ColorOutput "[*] Checking PyTorch CUDA availability..." "Blue"
        
        # Execute Python script and capture output
        $result = python -c $pythonScript
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -ne 0) {
            # CUDA not available
            $errorInfo = $result | ConvertFrom-Json
            Write-ColorOutput "" "Reset"
            Write-ColorOutput "[ERROR] CUDA/GPU is NOT available!" "Red"
            Write-ColorOutput "" "Reset"
            Write-ColorOutput "Error: $($errorInfo.error)" "Red"
            Write-ColorOutput "" "Reset"
            Write-ColorOutput "Possible reasons:" "Yellow"
            Write-ColorOutput "  1. PyTorch CPU-only version is installed" "Yellow"
            Write-ColorOutput "  2. NVIDIA drivers are not properly installed" "Yellow"
            Write-ColorOutput "  3. CUDA toolkit is not compatible" "Yellow"
            Write-ColorOutput "  4. GPU is not detected by the system" "Yellow"
            Write-ColorOutput "" "Reset"
            Write-ColorOutput "Please ensure you have:" "Yellow"
            Write-ColorOutput "  - PyTorch with CUDA support installed" "Yellow"
            Write-ColorOutput "  - NVIDIA GPU drivers >= 535" "Yellow"
            Write-ColorOutput "  - Compatible CUDA toolkit (12.1 recommended)" "Yellow"
            Write-ColorOutput "" "Reset"
            return $false
        }
        
        # Parse successful result
        $gpuInfo = $result | ConvertFrom-Json
        
        # Display GPU information
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "[OK] CUDA is available!" "Green"
        Write-ColorOutput "" "Reset"
        
        Write-ColorOutput "GPU Information:" "Cyan"
        Write-ColorOutput "  Device Name:        $($gpuInfo.gpu_name)" "Magenta"
        Write-ColorOutput "  GPU Count:          $($gpuInfo.gpu_count)" "Magenta"
        Write-ColorOutput "  Compute Capability: $($gpuInfo.compute_capability)" "Magenta"
        Write-ColorOutput "" "Reset"
        
        Write-ColorOutput "VRAM (Video Memory):" "Cyan"
        Write-ColorOutput "  Total VRAM:         $($gpuInfo.total_memory_gb) GB" "Magenta"
        Write-ColorOutput "  Free VRAM:          $($gpuInfo.free_memory_gb) GB" "Green"
        Write-ColorOutput "  Allocated:          $($gpuInfo.allocated_memory_gb) GB" "Magenta"
        Write-ColorOutput "  Reserved:           $($gpuInfo.reserved_memory_gb) GB" "Magenta"
        Write-ColorOutput "" "Reset"
        
        Write-ColorOutput "Software Versions:" "Cyan"
        Write-ColorOutput "  PyTorch:            $($gpuInfo.pytorch_version)" "Magenta"
        Write-ColorOutput "  CUDA:               $($gpuInfo.cuda_version)" "Magenta"
        Write-ColorOutput "" "Reset"
        
        # Calculate memory utilization percentage
        $memoryUsed = $gpuInfo.reserved_memory_gb
        $memoryUtilization = [math]::Round(($memoryUsed / $gpuInfo.total_memory_gb) * 100, 1)
        
        Write-ColorOutput "Memory Utilization:" "Cyan"
        Write-ColorOutput "  Current Usage:      $memoryUtilization% ($memoryUsed GB / $($gpuInfo.total_memory_gb) GB)" "Magenta"
        
        if ($memoryUtilization -lt 20) {
            Write-ColorOutput "  Status:             Excellent - Plenty of VRAM available" "Green"
        } elseif ($memoryUtilization -lt 50) {
            Write-ColorOutput "  Status:             Good - Sufficient VRAM for training" "Green"
        } elseif ($memoryUtilization -lt 80) {
            Write-ColorOutput "  Status:             Moderate - Monitor memory usage" "Yellow"
        } else {
            Write-ColorOutput "  Status:             High - May need to reduce batch size" "Yellow"
        }
        
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "===============================================================" "Cyan"
        Write-ColorOutput "[SUCCESS] GPU smoke test passed!" "Green"
        Write-ColorOutput "===============================================================" "Cyan"
        Write-Host ""
        
        return $true
    }
    catch {
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "[ERROR] Failed to execute GPU test!" "Red"
        Write-ColorOutput "" "Reset"
        Write-ColorOutput "Error details: $($_.Exception.Message)" "Red"
        Write-ColorOutput "" "Reset"
        return $false
    }
}

function Main {
    Write-ColorOutput @"

    ===============================================================
    
                  AXION-SAT GPU SMOKE TEST
    
    ===============================================================

"@ "Cyan"

    # Run GPU availability test
    $testPassed = Test-GPUAvailability
    
    if ($testPassed) {
        # Exit with success code
        exit 0
    } else {
        # Exit with error code
        Write-ColorOutput "GPU smoke test FAILED - CUDA is not available" "Red"
        Write-Host ""
        exit 1
    }
}

# Run main function
Main

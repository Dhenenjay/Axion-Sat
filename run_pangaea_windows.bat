@echo off
REM Pangaea Benchmark on Windows - Single GPU Mode
REM Sets up distributed training environment variables for Windows

echo ================================================================================
echo 🚀 PANGAEA BENCHMARK - Windows Single GPU Mode
echo ================================================================================
echo.

set DATASET=%1
if "%DATASET%"=="" set DATASET=hlsburnscars

echo Dataset: %DATASET%
echo Target: 83.6%% (HLSBurnScars)
echo.

REM Set distributed training environment variables for single GPU
set MASTER_ADDR=localhost
set MASTER_PORT=29500
set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1

echo Setting up environment...
echo   RANK=%RANK%
echo   LOCAL_RANK=%LOCAL_RANK%
echo   WORLD_SIZE=%WORLD_SIZE%
echo.

cd pangaea-bench

echo ================================================================================
echo Starting Training...
echo ================================================================================
echo.

python pangaea/run.py ^
  encoder=terramind_optical ^
  decoder=seg_upernet ^
  dataset=%DATASET% ^
  task=segmentation ^
  criterion=cross_entropy ^
  preprocessing=seg_default ^
  lr_scheduler=multi_step_lr ^
  use_wandb=False ^
  finetune=False ^
  train=True ^
  work_dir=../results/pangaea_bench

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ✗ Training failed
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✓ Training complete!
echo ================================================================================
echo.
echo Results: %CD%\..\results\pangaea_bench
echo.
pause
exit /b 0

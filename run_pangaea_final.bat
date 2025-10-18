@echo off
REM Final Pangaea Benchmark - Windows Compatible
REM Uses patched runner to bypass distributed training issues

echo ================================================================================
echo 🚀 PANGAEA BENCHMARK - Production Stage 1
echo ================================================================================
echo.

set DATASET=%1
if "%DATASET%"=="" set DATASET=hlsburnscars

echo Configuration:
echo   Dataset: %DATASET%
echo   Model: TerraMind Optical (Production Stage 1)
echo   Target: 83.6%% (HLSBurnScars) - Frozen baseline
  echo   Goal: Beat baseline with fine-tuning
  echo   Mode: Single GPU (Windows compatible)
echo.

cd pangaea-bench

echo ================================================================================
echo Starting Training...
echo ================================================================================
echo.

python run_windows.py ^
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
  work_dir=../results/pangaea_final ^
  num_workers=0 ^
  test_num_workers=0 ^
  +use_tta=False ^
  +use_multiscale=False

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
echo Results: %CD%\..\results\pangaea_final
echo.

REM Try to show final results
for /f %%i in ('dir /b /od ..\results\pangaea_final\* 2^>nul') do set LATEST=%%i
if defined LATEST (
    echo Latest run: %LATEST%
    if exist "..\results\pangaea_final\%LATEST%\train.log" (
        findstr /C:"mIoU" "..\results\pangaea_final\%LATEST%\train.log" 2>nul | findstr /V "best_metric" | more +5
    )
)

echo.
pause
exit /b 0

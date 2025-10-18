@echo off
REM Evaluate with Patch Overlapping Enhancement
REM Uses 50% overlap with Gaussian weighting for smoother predictions

echo ================================================================================
echo 🚀 PANGAEA EVALUATION - Patch Overlapping Enhancement
echo ================================================================================
echo.

set CHECKPOINT_DIR=%1
if "%CHECKPOINT_DIR%"=="" (
    REM Use latest successful run
    set CHECKPOINT_DIR=20251017_131022_c3d2a2_terramind_optical_seg_upernet_hlsburnscars
)

echo Configuration:
echo   Checkpoint: %CHECKPOINT_DIR%
echo   Enhancement: 50%% Patch Overlap + Gaussian Weighting
echo   Baseline Score: 83.446%% mIoU
echo   Expected: +1-2%% mIoU = ~84.5-85.5%%
echo.

cd pangaea-bench

echo ================================================================================
echo Running evaluation...
echo ================================================================================
echo.

python run_windows.py ^
  ckpt_dir=../results/pangaea_final/%CHECKPOINT_DIR% ^
  train=False ^
  test_num_workers=0 ^
  task=segmentation ^
  criterion=cross_entropy ^
  preprocessing=seg_default ^
  encoder=terramind_optical ^
  decoder=seg_upernet ^
  dataset=hlsburnscars ^
  +use_tta=False ^
  +use_multiscale=False

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ✗ Evaluation failed
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✓ Evaluation complete!
echo ================================================================================
echo.

REM Show test results
if exist "..\results\pangaea_final\%CHECKPOINT_DIR%\test.log-0" (
    echo Final Test Scores with Patch Overlapping:
    echo.
    findstr /C:"Mean" "..\results\pangaea_final\%CHECKPOINT_DIR%\test.log-0" | findstr /C:"mIoU" /C:"mIoU"
)

echo.
pause
exit /b 0

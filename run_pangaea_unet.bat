@echo off
REM Train with UNet Decoder instead of UPerNet
REM UNet is simpler and may handle boundary artifacts better

echo ================================================================================
echo 🚀 PANGAEA BENCHMARK - UNet Decoder
echo ================================================================================
echo.

echo Configuration:
echo   Dataset: HLSBurnScars
echo   Encoder: TerraMind Optical
echo   Decoder: UNet (instead of UPerNet)
echo   Baseline (UPerNet): 83.446%% mIoU
echo   Expected: Potentially better for sharp boundaries
echo.

cd pangaea-bench

echo ================================================================================
echo Starting Training with UNet decoder...
echo ================================================================================
echo.

python run_windows.py ^
  encoder=terramind_optical ^
  decoder=seg_unet ^
  dataset=hlsburnscars ^
  task=segmentation ^
  criterion=cross_entropy ^
  preprocessing=seg_default ^
  lr_scheduler=multi_step_lr ^
  use_wandb=False ^
  finetune=False ^
  train=True ^
  work_dir=../results/pangaea_unet ^
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

REM Show final results
for /f %%i in ('dir /b /od ..\results\pangaea_unet\* 2^>nul') do set LATEST=%%i
if defined LATEST (
    echo Latest run: %LATEST%
    if exist "..\results\pangaea_unet\%LATEST%\train.log-0" (
        echo.
        echo Final Results:
        findstr /C:"[test]" "..\results\pangaea_unet\%LATEST%\train.log-0" | findstr /C:"Mean"
    )
)

echo.
pause
exit /b 0

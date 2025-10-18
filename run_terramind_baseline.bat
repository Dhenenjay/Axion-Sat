@echo off
REM Run TerraMind Baseline on MADOS dataset
REM MADOS will auto-download if not present

echo ================================================================================
echo   AXION-SAT TERRAMIND BASELINE BENCHMARK
echo ================================================================================
echo.
echo Dataset: MADOS (Marine pollution detection)
echo Model: TerraMind-1.0-large
echo Configuration: Baseline (no enhancements)
echo.
echo Expected time: 1-2 hours
echo.
pause

cd C:\Users\Dhenenjay\Axion-Sat\pangaea-bench

echo.
echo Step 1: Running baseline TerraMind on MADOS...
echo.

torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py ^
  --config-name=train ^
  dataset=mados ^
  encoder=terramind_large ^
  decoder=seg_upernet ^
  preprocessing=seg_default ^
  criterion=cross_entropy ^
  task=segmentation ^
  batch_size=4 ^
  use_wandb=False ^
  finetune=False ^
  train=True ^
  task.trainer.n_epochs=0 ^
  work_dir=C:\Users\Dhenenjay\Axion-Sat\pangaea_results

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: Baseline run failed!
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   BASELINE COMPLETE!
echo ================================================================================
echo.
echo Results saved to: C:\Users\Dhenenjay\Axion-Sat\pangaea_results
echo.
echo Next: Run with TTA enhancement
echo   Command: Same as above but add: use_tta=True
echo.
pause

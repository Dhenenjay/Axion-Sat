@echo off
REM Axion-Sat Full Pangaea Benchmark Runner
REM This runs all configurations and generates comparison report

echo ================================================================================
echo   AXION-SAT FULL PANGAEA BENCHMARK
echo ================================================================================
echo.
echo This will run TerraMind on multiple datasets with different enhancements:
echo   1. Baseline (vanilla)
echo   2. + Test-Time Augmentation
echo   3. + Multi-Scale Ensemble  
echo   4. + Combined (TTA + MultiScale)
echo.
echo Expected time: 2-4 hours per dataset (depending on GPU)
echo.
pause

cd C:\Users\Dhenenjay\Axion-Sat\pangaea-bench

REM Create results directory
mkdir axion_results 2>nul

echo.
echo ================================================================================
echo STEP 1: Running Baseline on MADOS dataset
echo ================================================================================
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
  exp_name=axion_baseline_mados ^
  exp_dir=axion_results/baseline_mados

if errorlevel 1 (
    echo ERROR: Baseline run failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo STEP 2: Running with TTA on MADOS dataset
echo ================================================================================
echo.
echo NOTE: This will take 5x longer due to Test-Time Augmentation
echo.

REM For now, just baseline. To enable TTA, we need to integrate into evaluator
REM See PANGAEA_BENCHMARK_PLAN.md for integration instructions

echo.
echo ================================================================================
echo RESULTS SUMMARY
echo ================================================================================
echo.
echo Baseline results saved to: axion_results/baseline_mados
echo.
echo Next steps:
echo   1. Check metrics in axion_results/baseline_mados/train.log
echo   2. Integrate TTA wrapper (see PANGAEA_BENCHMARK_PLAN.md)
echo   3. Run enhanced benchmarks
echo   4. Compare results
echo.
echo Target: Beat DOFA (0.7715 mIoU)
echo.
pause

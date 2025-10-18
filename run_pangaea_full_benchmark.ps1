# Full Pangaea Benchmark Runner
# Runs baseline and TTA on multiple datasets and tracks results

$datasets = @(
    'hlsburnscars',
    'sen1floods11',
    'potsdam',
    'opencanopy'
)

$resultsFile = "pangaea_benchmark_results.csv"

# Initialize results file
"Dataset,Config,Mean_IoU,Mean_F1,Mean_Accuracy,Status,Time" | Out-File -FilePath $resultsFile

foreach ($dataset in $datasets) {
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "BENCHMARKING: $dataset" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    
    # Run baseline
    Write-Host "`n[1/2] Running BASELINE..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    $cmd = "python run_benchmark_windows.py dataset=$dataset encoder=terramind_optical decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation batch_size=1 finetune=False train=True task.trainer.n_epochs=0 work_dir=./results num_workers=0 test_num_workers=0"
    
    $output = Invoke-Expression $cmd 2>&1
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMinutes
    
    if ($LASTEXITCODE -eq 0) {
        # Parse results
        $miou = ($output | Select-String "Mean.*IoU" | Select-Object -Last 1) -replace '.*:\s*(\d+\.?\d*)\s*.*', '$1'
        Write-Host "✓ Baseline completed - Mean IoU: $miou" -ForegroundColor Green
        "$dataset,Baseline,$miou,,,Success,$([math]::Round($duration,2))" | Out-File -FilePath $resultsFile -Append
    } else {
        Write-Host "✗ Baseline failed" -ForegroundColor Red
        "$dataset,Baseline,,,Failed,$([math]::Round($duration,2))" | Out-File -FilePath $resultsFile -Append
        continue
    }
    
    # Run TTA
    Write-Host "`n[2/2] Running TTA..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    $cmd = "python run_benchmark_windows.py dataset=$dataset encoder=terramind_optical decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation batch_size=1 finetune=False train=True task.trainer.n_epochs=0 work_dir=./results num_workers=0 test_num_workers=0 +use_tta=True"
    
    $output = Invoke-Expression $cmd 2>&1
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMinutes
    
    if ($LASTEXITCODE -eq 0) {
        $miou = ($output | Select-String "Mean.*IoU" | Select-Object -Last 1) -replace '.*:\s*(\d+\.?\d*)\s*.*', '$1'
        Write-Host "✓ TTA completed - Mean IoU: $miou" -ForegroundColor Green
        "$dataset,TTA,$miou,,,Success,$([math]::Round($duration,2))" | Out-File -FilePath $resultsFile -Append
    } else {
        Write-Host "✗ TTA failed" -ForegroundColor Red
        "$dataset,TTA,,,Failed,$([math]::Round($duration,2))" | Out-File -FilePath $resultsFile -Append
    }
    
    Write-Host "`n$dataset completed!" -ForegroundColor Green
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "BENCHMARK COMPLETE - Results saved to $resultsFile" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# Display results
Get-Content $resultsFile

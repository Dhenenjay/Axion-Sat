# Lambda Labs Setup Helper (Windows)
# This script helps you set up and transfer code to Lambda Labs

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lambda Labs Instance Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Get Lambda Labs instance IP
Write-Host "Step 1: Create Lambda Labs Instance" -ForegroundColor Yellow
Write-Host "  1. Go to https://cloud.lambdalabs.com/instances" -ForegroundColor Gray
Write-Host "  2. Launch instance (1x A100 or RTX 6000 Ada recommended)" -ForegroundColor Gray
Write-Host "  3. Copy the instance IP address" -ForegroundColor Gray
Write-Host ""

$LAMBDA_IP = Read-Host "Enter Lambda Labs instance IP"

if ([string]::IsNullOrWhiteSpace($LAMBDA_IP)) {
    Write-Host "Error: IP address required" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Testing SSH connection..." -ForegroundColor Yellow

# Test SSH connection
$sshTest = ssh -i "C:\Users\Dhenenjay\Downloads\Dhenenjay.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$LAMBDA_IP "echo 'Connection successful'"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Cannot connect to Lambda Labs instance" -ForegroundColor Red
    Write-Host "Make sure:" -ForegroundColor Yellow
    Write-Host "  - Instance is running" -ForegroundColor Gray
    Write-Host "  - IP address is correct" -ForegroundColor Gray
    Write-Host "  - SSH key path is correct" -ForegroundColor Gray
    exit 1
}

Write-Host "✓ SSH connection successful" -ForegroundColor Green
Write-Host ""

# Step 3: Transfer code
Write-Host "Step 3: Transferring code to Lambda Labs..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes depending on connection speed..." -ForegroundColor Gray
Write-Host ""

# Use rsync for faster transfer (if available), otherwise scp
Write-Host "Transferring Axion-Sat directory..." -ForegroundColor Gray

scp -i "C:\Users\Dhenenjay\Downloads\Dhenenjay.pem" `
    -o StrictHostKeyChecking=no `
    -r "C:\Users\Dhenenjay\Axion-Sat\pangaea-bench" `
    -r "C:\Users\Dhenenjay\Axion-Sat\weights" `
    ubuntu@${LAMBDA_IP}:~/

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during file transfer" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Code transferred successfully" -ForegroundColor Green
Write-Host ""

# Step 4: Transfer setup script
Write-Host "Step 4: Transferring setup script..." -ForegroundColor Yellow

scp -i "C:\Users\Dhenenjay\Downloads\Dhenenjay.pem" `
    -o StrictHostKeyChecking=no `
    "C:\Users\Dhenenjay\Axion-Sat\lambda_setup_and_run.sh" `
    ubuntu@${LAMBDA_IP}:~/

Write-Host "✓ Setup script transferred" -ForegroundColor Green
Write-Host ""

# Step 5: Run setup
Write-Host "Step 5: Running setup and starting training..." -ForegroundColor Yellow
Write-Host ""
Write-Host "You will now be connected to the Lambda Labs instance." -ForegroundColor Cyan
Write-Host "The training will start automatically in a tmux session." -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  - Detach from tmux: Ctrl+B then D" -ForegroundColor Gray
Write-Host "  - Reattach to tmux: tmux attach -t pyramid_training" -ForegroundColor Gray
Write-Host "  - Check GPU: nvidia-smi" -ForegroundColor Gray
Write-Host "  - Check progress: tail -f ~/pangaea-bench/training.log" -ForegroundColor Gray
Write-Host ""

Write-Host "Press Enter to connect and start training..." -ForegroundColor Green
Read-Host

# Connect and run setup
ssh -i "C:\Users\Dhenenjay\Downloads\Dhenenjay.pem" `
    -o StrictHostKeyChecking=no `
    ubuntu@$LAMBDA_IP `
    "chmod +x ~/lambda_setup_and_run.sh && ~/lambda_setup_and_run.sh"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To reconnect and check progress:" -ForegroundColor Yellow
Write-Host "  ssh -i C:\Users\Dhenenjay\Downloads\Dhenenjay.pem ubuntu@$LAMBDA_IP" -ForegroundColor Gray
Write-Host "  tmux attach -t pyramid_training" -ForegroundColor Gray
Write-Host ""
Write-Host "Results will be in: ~/Axion-Sat/results/pangaea_pyramid/" -ForegroundColor Cyan

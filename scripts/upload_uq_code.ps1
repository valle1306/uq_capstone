# PowerShell script to upload UQ experiment code to Amarel
# Run: powershell -ExecutionPolicy Bypass -File .\scripts\upload_uq_code.ps1

$remoteUser = "hpl14"
$remoteHost = "amarel.rutgers.edu"
$remotePath = "/scratch/hpl14/uq_capstone"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "UPLOADING UQ EXPERIMENT CODE TO AMAREL" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
ssh $remoteUser@$remoteHost "echo 'Connection successful'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot connect to Amarel" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Connection successful" -ForegroundColor Green
Write-Host ""

# Upload new Python modules
Write-Host "Uploading Python modules..." -ForegroundColor Yellow
scp src/uq_methods.py "${remoteUser}@${remoteHost}:${remotePath}/src/"
scp src/swag.py "${remoteUser}@${remoteHost}:${remotePath}/src/"
scp src/train_baseline.py "${remoteUser}@${remoteHost}:${remotePath}/src/"
scp src/train_mc_dropout.py "${remoteUser}@${remoteHost}:${remotePath}/src/"
scp src/train_ensemble_member.py "${remoteUser}@${remoteHost}:${remotePath}/src/"
scp src/train_swag.py "${remoteUser}@${remoteHost}:${remotePath}/src/"
scp src/evaluate_uq.py "${remoteUser}@${remoteHost}:${remotePath}/src/"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python modules uploaded" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to upload Python modules" -ForegroundColor Red
    exit 1
}

# Upload SLURM job scripts
Write-Host "Uploading SLURM job scripts..." -ForegroundColor Yellow
scp scripts/train_baseline.sbatch "${remoteUser}@${remoteHost}:${remotePath}/scripts/"
scp scripts/train_mc_dropout.sbatch "${remoteUser}@${remoteHost}:${remotePath}/scripts/"
scp scripts/train_ensemble.sbatch "${remoteUser}@${remoteHost}:${remotePath}/scripts/"
scp scripts/train_swag.sbatch "${remoteUser}@${remoteHost}:${remotePath}/scripts/"
scp scripts/evaluate_uq.sbatch "${remoteUser}@${remoteHost}:${remotePath}/scripts/"
scp scripts/run_all_experiments.sh "${remoteUser}@${remoteHost}:${remotePath}/scripts/"
scp scripts/monitor_jobs.sh "${remoteUser}@${remoteHost}:${remotePath}/scripts/"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ SLURM scripts uploaded" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to upload SLURM scripts" -ForegroundColor Red
    exit 1
}

# Upload documentation
Write-Host "Uploading documentation..." -ForegroundColor Yellow
scp UQ_EXPERIMENTS_GUIDE.md "${remoteUser}@${remoteHost}:${remotePath}/"
scp SWAG_GUIDE.md "${remoteUser}@${remoteHost}:${remotePath}/"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Documentation uploaded" -ForegroundColor Green
} else {
    Write-Host "WARNING: Failed to upload documentation (non-critical)" -ForegroundColor Yellow
}

# Make scripts executable
Write-Host "Making scripts executable..." -ForegroundColor Yellow
ssh $remoteUser@$remoteHost "chmod +x $remotePath/scripts/*.sh $remotePath/scripts/*.sbatch"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Scripts are now executable" -ForegroundColor Green
} else {
    Write-Host "WARNING: Failed to make scripts executable" -ForegroundColor Yellow
}

# Verify files
Write-Host ""
Write-Host "Verifying uploaded files..." -ForegroundColor Yellow
ssh $remoteUser@$remoteHost "cd $remotePath && ls -lh src/*.py scripts/*.sbatch scripts/*.sh *.md | tail -15"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✓ UPLOAD COMPLETE!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. SSH to Amarel: ssh hpl14@amarel.rutgers.edu" -ForegroundColor White
Write-Host "2. Navigate: cd /scratch/hpl14/uq_capstone" -ForegroundColor White
Write-Host "3. Run experiments: bash scripts/run_all_experiments.sh" -ForegroundColor White
Write-Host "4. Monitor: squeue -u hpl14" -ForegroundColor White
Write-Host ""
Write-Host "Or run individual jobs:" -ForegroundColor Cyan
Write-Host "  sbatch scripts/train_baseline.sbatch" -ForegroundColor White
Write-Host "  sbatch scripts/train_mc_dropout.sbatch" -ForegroundColor White
Write-Host "  sbatch scripts/train_ensemble.sbatch" -ForegroundColor White
Write-Host ""
Write-Host "Estimated total time: 6-8 hours" -ForegroundColor Yellow
Write-Host ""

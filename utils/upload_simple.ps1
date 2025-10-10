# Simple Upload Script for hpl14
# Run each command manually in PowerShell

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Uploading to Amarel - Step by Step" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You will be prompted for your password multiple times." -ForegroundColor Yellow
Write-Host "Press Enter to continue..." -ForegroundColor Yellow
Read-Host

Write-Host ""
Write-Host "[1/7] Testing connection..." -ForegroundColor Green
ssh hpl14@amarel.rutgers.edu "echo 'Connection OK'"

Write-Host ""
Write-Host "[2/7] Creating directories..." -ForegroundColor Green
ssh hpl14@amarel.rutgers.edu "mkdir -p /scratch/hpl14/uq_capstone/{data/brats,scripts,src,envs,notebooks,runs}"

Write-Host ""
Write-Host "[3/7] Uploading data (~150MB, takes 3-5 min)..." -ForegroundColor Green
scp -r data\brats hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/data/

Write-Host ""
Write-Host "[4/7] Uploading scripts..." -ForegroundColor Green
scp scripts\*.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/
scp scripts\*.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/

Write-Host ""
Write-Host "[5/7] Uploading source code..." -ForegroundColor Green
scp src\*.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/

Write-Host ""
Write-Host "[6/7] Uploading environment files..." -ForegroundColor Green
scp envs\conda_env.yml hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/envs/
scp requirements.txt hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/

Write-Host ""
Write-Host "[7/7] Uploading notebooks..." -ForegroundColor Green
scp notebooks\*.ipynb hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/notebooks/

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DONE! Files uploaded to Amarel" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: SSH to Amarel" -ForegroundColor Yellow
Write-Host "   ssh hpl14@amarel.rutgers.edu" -ForegroundColor White
Write-Host ""
